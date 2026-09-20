"""Focused regressions for incremental LLM transport and live diagnostics."""

import importlib
import json
import sys
import threading
import time
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CUSTOM_NODES = PACKAGE_DIR.parent
COMFY_ROOT = CUSTOM_NODES.parent
for path in (str(COMFY_ROOT), str(CUSTOM_NODES)):
    if path not in sys.path:
        sys.path.insert(0, path)

llm = importlib.import_module("ComfyUI-MiniMaxH3-Myang.llm_service")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


class FakeResponse:
    def __init__(self, lines=(), body=b"", content_type="text/event-stream"):
        self._lines = list(lines)
        self._body = body
        self.headers = {"Content-Type": content_type}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return self._body

    def close(self):
        self.closed = True


class ScriptedResponse(FakeResponse):
    """Yield stream lines with delays and support close-unblocking."""

    def __init__(self, script):
        super().__init__(content_type="text/event-stream")
        self._script = list(script)
        self.release = threading.Event()
        self.after_terminal_requested = threading.Event()

    def __iter__(self):
        for delay, line in self._script:
            if delay == "block":
                self.after_terminal_requested.set()
                self.release.wait(2.0)
            elif delay:
                time.sleep(float(delay))
            if self.closed:
                return
            yield line

    def close(self):
        super().close()
        self.release.set()


def _state():
    return {
        "cancel": threading.Event(), "response": None, "diagnostics": {},
        "started_at": 1.0, "response_at": 0.0, "first_chunk_at": 0.0,
        "last_chunk_at": 0.0, "chunks": 0, "content_chars": 0,
        "reasoning_chars": 0, "finish_reason": "", "preview": "",
        "last_emit_at": 0.0,
    }


def test_openai_sse_aggregates_content_reasoning_and_finish():
    response = FakeResponse(lines=[
        b": keepalive\r\n", b"\r\n",
        b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\r\n', b"\r\n",
        b'data: {"choices":[{"delta":{"content":"hello "}}]}\n', b"\n",
        b'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]}\n', b"\n",
        b"data: [DONE]\n", b"\n",
    ])
    original = llm.urllib.request.urlopen
    llm.urllib.request.urlopen = lambda _req, timeout=120: response
    try:
        result = llm._http_post_json_blocking(
            "https://example.invalid/v1/chat/completions", {},
            {"stream": True}, 10, _state())
    finally:
        llm.urllib.request.urlopen = original
    choice = result["choices"][0]
    check(choice["message"]["content"] == "hello world", "SSE content was not aggregated")
    check(choice["message"]["reasoning_content"] == "think", "reasoning channel was lost")
    check(choice["finish_reason"] == "stop", "finish reason was lost")
    check(result["_myang_stream"]["chunks"] == 3, "chunk count is incorrect")


def test_sse_proxy_without_blank_event_delimiters_is_tolerated():
    response = FakeResponse(lines=[
        b'data: {"choices":[{"delta":{"content":"first"}}]}\n',
        b'data: {"choices":[{"delta":{"content":" second"},"finish_reason":"stop"}]}\n',
        b"data: [DONE]\n",
    ])
    original = llm.urllib.request.urlopen
    llm.urllib.request.urlopen = lambda _req, timeout=120: response
    try:
        result = llm._http_post_json_blocking(
            "https://example.invalid/v1/chat/completions", {},
            {"stream": True}, 10, _state())
    finally:
        llm.urllib.request.urlopen = original
    check(result["choices"][0]["message"]["content"] == "first second",
          "proxy SSE without blank delimiters was not parsed")


def test_stream_without_terminal_marker_discards_partial_content():
    response = FakeResponse(lines=[
        b'data: {"choices":[{"delta":{"content":"half a json"}}]}\n', b"\n",
    ])
    original = llm.urllib.request.urlopen
    llm.urllib.request.urlopen = lambda _req, timeout=120: response
    try:
        try:
            llm._http_post_json_blocking(
                "https://example.invalid/v1/chat/completions", {},
                {"stream": True}, 10, _state())
        except Exception as error:
            check(isinstance(error, llm.LLMStreamResponseError),
                  "partial stream raised the wrong error")
            check("半截结果已丢弃" in str(error), "partial stream was not explicitly discarded")
        else:
            raise AssertionError("unterminated partial stream was accepted")
    finally:
        llm.urllib.request.urlopen = original


def test_stream_true_accepts_provider_regular_json_response():
    body = json.dumps({"choices": [{"message": {"content": "regular"}}]}).encode()
    response = FakeResponse(body=body, content_type="application/json")
    original = llm.urllib.request.urlopen
    llm.urllib.request.urlopen = lambda _req, timeout=120: response
    try:
        result = llm._http_post_json_blocking(
            "https://example.invalid/v1/chat/completions", {},
            {"stream": True}, 10, _state())
    finally:
        llm.urllib.request.urlopen = original
    check(result["choices"][0]["message"]["content"] == "regular",
          "regular JSON compatibility path failed")


def test_http_stream_emits_live_diagnostics():
    response = FakeResponse(lines=[
        b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n', b"\n",
        b"data: [DONE]\n", b"\n",
    ])
    original_urlopen = llm.urllib.request.urlopen
    original_emit = llm._emit_llm_stream
    events = []
    llm.urllib.request.urlopen = lambda _req, timeout=120: response
    llm._emit_llm_stream = lambda diagnostic, phase, **updates: events.append((diagnostic, phase, updates))
    llm._http_diagnostic_context.value = {
        "call_id": "test", "service": "svc", "route": "route",
    }
    try:
        result = llm._http_post_json(
            "https://example.invalid/v1/chat/completions", {}, {"stream": True}, timeout=10)
    finally:
        llm.urllib.request.urlopen = original_urlopen
        llm._emit_llm_stream = original_emit
        try:
            del llm._http_diagnostic_context.value
        except AttributeError:
            pass
    phases = [phase for _diagnostic, phase, _updates in events]
    check(result["choices"][0]["message"]["content"] == "ok", "live transport failed")
    check("connecting" in phases and "connected" in phases and "stream_complete" in phases,
          "stream transport did not publish connection lifecycle")


def test_active_reasoning_stream_outlives_startup_timeout_and_completes():
    response = ScriptedResponse([
        (0, b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n'),
        (0, b"\n"),
        (0.28, b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n'),
        (0, b"\n"),
    ])
    original = llm.urllib.request.urlopen
    llm.urllib.request.urlopen = lambda _req, timeout=120: response
    started = time.monotonic()
    try:
        result = llm._http_post_json(
            "https://example.invalid/v1/chat/completions", {},
            {"stream": True}, timeout=0.12)
    finally:
        llm.urllib.request.urlopen = original
    elapsed = time.monotonic() - started
    check(elapsed >= 0.25, "active reasoning stream was still cut at the startup timeout")
    check(result["choices"][0]["message"]["content"] == "answer",
          "content after a long reasoning pause was lost")
    check(result["choices"][0]["message"]["reasoning_content"] == "thinking",
          "reasoning content was not preserved")


def test_protocol_only_stream_does_not_disable_startup_timeout():
    response = ScriptedResponse([
        (0, b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'),
        (0, b"\n"),
        ("block", b"\n"),
    ])
    original = llm.urllib.request.urlopen
    llm.urllib.request.urlopen = lambda _req, timeout=120: response
    started = time.monotonic()
    try:
        try:
            llm._http_post_json(
                "https://example.invalid/v1/chat/completions", {},
                {"stream": True}, timeout=0.12)
        except Exception as error:
            check(isinstance(error, llm.LLMRequestTimeoutError),
                  "protocol-only stream raised the wrong timeout error")
        else:
            raise AssertionError("protocol-only stream incorrectly disabled the startup timeout")
    finally:
        llm.urllib.request.urlopen = original
        response.close()
    check(time.monotonic() - started < 0.8,
          "startup timeout did not stop an idle protocol stream")


def test_finish_reason_completes_without_done_or_connection_close():
    response = ScriptedResponse([
        (0, b'data: {"choices":[{"delta":{"content":"complete"},"finish_reason":"stop"}]}\n'),
        (0, b"\n"),
        ("block", b"data: [DONE]\n"),
    ])
    original = llm.urllib.request.urlopen
    llm.urllib.request.urlopen = lambda _req, timeout=120: response
    started = time.monotonic()
    try:
        result = llm._http_post_json_blocking(
            "https://example.invalid/v1/chat/completions", {},
            {"stream": True}, 10, _state())
    finally:
        llm.urllib.request.urlopen = original
        response.close()
    check(result["choices"][0]["message"]["content"] == "complete",
          "finish_reason completion lost the final content")
    check(time.monotonic() - started < 0.5,
          "transport waited for DONE/EOF after receiving finish_reason")
    check(not response.after_terminal_requested.is_set(),
          "transport requested another stream line after finish_reason")


def test_active_reasoning_stream_remains_manually_cancellable():
    response = ScriptedResponse([
        (0, b'data: {"choices":[{"delta":{"reasoning_content":"still thinking"}}]}\n'),
        (0, b"\n"),
        ("block", b'data: {"choices":[{"delta":{"content":"too late"}}]}\n'),
    ])
    original = llm.urllib.request.urlopen
    captured = []

    def invoke():
        try:
            llm._http_post_json(
                "https://example.invalid/v1/chat/completions", {},
                {"stream": True}, timeout=0.12)
        except BaseException as error:
            captured.append(error)

    llm.urllib.request.urlopen = lambda _req, timeout=120: response
    caller = threading.Thread(target=invoke)
    try:
        caller.start()
        check(response.after_terminal_requested.wait(1.0),
              "active stream did not reach the cancellable reasoning wait")
        check(llm.cancel_active_http_requests() == 1,
              "manual stop did not target the active reasoning stream")
        caller.join(1.0)
        check(not caller.is_alive(), "manual stop left the active stream blocked")
        check(response.closed, "manual stop did not close the active HTTP response")
        check(captured and type(captured[0]).__name__ == "InterruptProcessingException",
              "active-stream stop did not propagate as a normal ComfyUI interruption")
    finally:
        response.close()
        caller.join(1.0)
        llm.urllib.request.urlopen = original


def test_diagnostics_do_not_broadcast_route_secrets():
    import server

    captured = []

    class Recorder:
        def send_sync(self, event, payload):
            captured.append((event, payload))

    previous = getattr(server.PromptServer, "instance", None)
    server.PromptServer.instance = Recorder()
    try:
        llm._emit_llm_stream({
            "call_id": "test", "api_key": "secret-key",
            "headers": {"Authorization": "Bearer secret-key"},
            "url": "https://private.example/v1",
        }, "route_error", message="Bearer secret-key api_key=another-secret")
    finally:
        server.PromptServer.instance = previous
    check(captured and captured[0][0] == "myh3_llm_stream", "diagnostic event was not emitted")
    payload = captured[0][1]
    serialised = json.dumps(payload, ensure_ascii=False)
    check("secret-key" not in serialised and "another-secret" not in serialised,
          "diagnostic event leaked a route secret")
    check("api_key" not in payload and "headers" not in payload and "url" not in payload,
          "diagnostic event retained an unsafe field")


def test_call_llm_falls_back_once_when_stream_is_unsupported():
    service = {
        "id": "svc", "name": "测试服务", "type": "openai_compatible",
        "route_strategy": "failover",
    }
    route = {"id": "route_1", "name": "线路 1", "base_url": "https://example.invalid/v1"}
    target = {
        "name": "model", "temperature": 0.7, "top_p": 0.9,
        "max_tokens": 0, "stream": True, "timeout": 30,
    }
    originals = {
        "find_service": llm._find_service,
        "find_model": llm._find_model,
        "routes": llm._ordered_enabled_routes,
        "post": llm._http_post_json,
        "emit": llm._emit_llm_stream,
    }
    payloads = []
    events = []

    def post(_url, _headers, payload, timeout=120):
        payloads.append(dict(payload))
        if len(payloads) == 1:
            raise RuntimeError("API error 400: streaming is unsupported")
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    llm._find_service = lambda _ref: service
    llm._find_model = lambda _svc, _model, _key: target
    llm._ordered_enabled_routes = lambda _svc, _lane: [route]
    llm._http_post_json = post
    llm._emit_llm_stream = lambda diagnostics, phase, **updates: events.append((phase, updates))
    try:
        check(llm.call_llm("svc/model", "system", "user") == "ok",
              "same-route non-stream fallback did not complete")
    finally:
        llm._find_service = originals["find_service"]
        llm._find_model = originals["find_model"]
        llm._ordered_enabled_routes = originals["routes"]
        llm._http_post_json = originals["post"]
        llm._emit_llm_stream = originals["emit"]
        llm._clear_route_runtime_state("svc")
    check([item["stream"] for item in payloads] == [True, False],
          "unsupported streaming did not fall back exactly once")
    check(any(phase == "stream_fallback" for phase, _updates in events),
          "fallback was not exposed to diagnostics")
    check(any(phase == "done" for phase, _updates in events),
          "successful completion was not exposed to diagnostics")


def test_model_config_defaults_streaming_and_preserves_timeout_override():
    old_model = {
        "name": "legacy", "is_default": True, "temperature": 0.7,
        "max_tokens": 0, "top_p": 0.9,
    }
    explicit = {
        "name": "explicit", "is_default": False, "temperature": 0.6,
        "max_tokens": 0, "top_p": 0.8, "stream": False, "timeout": 420,
    }
    cleaned = llm._clean_models([old_model, explicit], "测试模型")
    check(cleaned[0]["stream"] is True and cleaned[0]["timeout"] == 0,
          "legacy model did not migrate to streaming + automatic timeout")
    check(cleaned[1]["stream"] is False and cleaned[1]["timeout"] == 420,
          "explicit stream/timeout settings were not preserved")
    vision = llm._clean_models([old_model], "测试 VLM", stream_default=False)
    check(vision[0]["stream"] is False,
          "VLM model was accidentally opted into the LLM stream transport")


def test_frontend_exposes_stream_controls_and_live_status():
    settings = (PACKAGE_DIR / "web" / "minimax_h3_myang_agent_ui.js").read_text("utf-8")
    director = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    check('api.addEventListener("myh3_llm_stream"' in settings,
          "settings/Skill UI does not subscribe to live diagnostics")
    check('class="m-stream"' in settings and 'class="m-timeout"' in settings,
          "per-model streaming and timeout controls are missing")
    check("15秒快探→30秒复查" in settings and "活跃流等待完成" in settings,
          "settings UI does not explain two-round active-stream semantics")
    check('api.addEventListener("myh3_llm_stream"' in director,
          "Director fixed progress panel does not receive LLM diagnostics")


def main():
    tests = [
        test_openai_sse_aggregates_content_reasoning_and_finish,
        test_sse_proxy_without_blank_event_delimiters_is_tolerated,
        test_stream_without_terminal_marker_discards_partial_content,
        test_stream_true_accepts_provider_regular_json_response,
        test_http_stream_emits_live_diagnostics,
        test_active_reasoning_stream_outlives_startup_timeout_and_completes,
        test_protocol_only_stream_does_not_disable_startup_timeout,
        test_finish_reason_completes_without_done_or_connection_close,
        test_active_reasoning_stream_remains_manually_cancellable,
        test_diagnostics_do_not_broadcast_route_secrets,
        test_call_llm_falls_back_once_when_stream_is_unsupported,
        test_model_config_defaults_streaming_and_preserves_timeout_override,
        test_frontend_exposes_stream_controls_and_live_status,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} streaming diagnostics tests passed")


if __name__ == "__main__":
    main()
