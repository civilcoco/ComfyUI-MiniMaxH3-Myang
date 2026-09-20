"""Lightweight LLM/VLM client for MiniMax H3 Media Agent.

Replaces the prompt-assistant dependency with direct OpenAI-compatible API calls.
Reads the same config file format for backward compatibility.
"""

import base64
import copy
import io
import json
import logging
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from comfy import model_management as _comfy_model_management
except ImportError:  # standalone config/tests without a ComfyUI runtime
    _comfy_model_management = None

logger = logging.getLogger(__name__)

_config_cache: dict | None = None
_config_mtime_ns: int = 0
_config_source: Path | None = None

_SERVICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_GENERATED_ID_RE = re.compile(r"^service[_-]\d+$", re.IGNORECASE)
_SERVICE_TYPES = {"openai_compatible", "ollama"}
_ROUTE_STRATEGIES = {"round_robin", "failover"}
RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS = 65.0
ROUTE_TRANSIENT_COOLDOWN_SECONDS = 30.0
ROUTE_TIMEOUT_COOLDOWN_SECONDS = 45.0
ROUTE_AUTH_COOLDOWN_SECONDS = 300.0
# A provider can return `insufficient_quota` for a temporary workspace/model
# allocation.  Do not turn one such response into a six-hour circuit break.
# The public call wrapper confirms exhaustion only after repeated responses.
ROUTE_QUOTA_RETRY_COOLDOWN_SECONDS = 10.0
ROUTE_QUOTA_COOLDOWN_SECONDS = 21600.0
QUOTA_CONFIRMATION_FAILURES = 10
QUOTA_RETRY_WAIT_SECONDS = 10.0
DEFAULT_LLM_ROUTE_TIMEOUT_SECONDS = 90
DEFAULT_LLM_CALL_BUDGET_SECONDS = 180
MAX_LLM_CALL_BUDGET_SECONDS = 300
MAX_INLINE_COOLDOWN_WAIT_SECONDS = 70.0
MIN_ROUTE_ATTEMPT_SECONDS = 20
LLM_FAST_FIRST_OUTPUT_SECONDS = 15
LLM_FINAL_FIRST_OUTPUT_SECONDS = 30


class LLMRateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMQuotaError(RuntimeError):
    """The provider returned a quota/allocation response; it may be transient."""


class LLMQuotaConfirmedError(LLMQuotaError):
    """The same quota response was observed enough times to stop retrying."""


class LLMEmptyResponseError(RuntimeError):
    """The provider completed the request but returned no assistant content."""


class LLMStreamResponseError(RuntimeError):
    """A streaming response connected but could not be completed or decoded."""


class LLMRequestTimeoutError(RuntimeError):
    """Connection/first output (or a non-stream request) exceeded its deadline."""


class LLMRoutesCoolingError(RuntimeError):
    """Every configured route is temporarily unavailable."""

    def __init__(self, message: str, retry_after: float = 0.0, reasons: tuple[str, ...] = ()):
        super().__init__(message)
        self.retry_after = max(0.0, float(retry_after or 0.0))
        self.reasons = tuple(str(reason or "") for reason in reasons if reason)


_route_guard = threading.RLock()
_route_counters: dict[str, int] = {}
_route_cooldowns: dict[str, float] = {}
_route_runtime: dict[str, dict] = {}
_active_http_requests: dict[int, dict] = {}
_active_http_request_id = 0
_llm_call_id = 0
_http_diagnostic_context = threading.local()
# Backward-compatible names used by the local regression fixtures and older
# callers; all point at the same circuit-breaker state.
_rate_state_guard = _route_guard
_rate_locks: dict[str, threading.Lock] = {}
_rate_cooldowns = _route_cooldowns


def _route_state_key(service_id: str, route_id: str) -> str:
    return f"{service_id}/{route_id}"


def active_http_request_count() -> int:
    with _route_guard:
        return len(_active_http_requests)


def cancel_active_http_requests() -> int:
    """Stop waits owned by this package without restarting ComfyUI."""
    with _route_guard:
        active = list(_active_http_requests.values())
        for state in active:
            state["cancel"].set()
            response = state.get("response")
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
    if active:
        logger.info("H3-Myang: 已请求停止 %d 个 LLM/VLM 网络请求", len(active))
    return len(active)


def _next_llm_call_id() -> str:
    global _llm_call_id
    with _route_guard:
        _llm_call_id += 1
        return f"llm-{int(time.time() * 1000):x}-{_llm_call_id:x}"


def _execution_context_fields() -> dict:
    """Capture ComfyUI's current node before urllib moves into a worker thread."""
    try:
        from comfy_execution.utils import get_executing_context
        context = get_executing_context()
        if context is None:
            return {}
        return {
            "prompt_id": str(getattr(context, "prompt_id", "") or ""),
            "node_id": str(getattr(context, "node_id", "") or ""),
        }
    except Exception:
        return {}


def _emit_llm_stream(diagnostics: dict | None, phase: str, **updates) -> None:
    """Best-effort local event; diagnostics must never break an API request."""
    if not diagnostics:
        return
    payload = {
        key: value for key, value in diagnostics.items()
        if key in {
            "call_id", "prompt_id", "node_id", "service_id", "service",
            "model", "route_id", "route", "attempt", "streaming",
        }
    }
    payload.update(updates)
    payload["phase"] = str(phase or "")
    payload["timestamp"] = time.time()
    # Never allow an accidentally supplied credential to reach the browser.
    for secret_key in ("api_key", "authorization", "headers", "url"):
        payload.pop(secret_key, None)
    if payload.get("message"):
        payload["message"] = re.sub(
            r"(?i)(bearer\s+|api[_-]?key\s*[=:]\s*)[^\s,;]+",
            r"\1[redacted]", str(payload["message"]))[:500]
    try:
        from server import PromptServer
        instance = getattr(PromptServer, "instance", None)
        if instance is not None:
            instance.send_sync("myh3_llm_stream", payload)
    except Exception:
        logger.debug("H3-Myang: 无法推送 LLM 流式诊断事件", exc_info=True)


def _normalize_routes(service: dict) -> list[dict]:
    """Migrate legacy root URL/key into one stable editable route."""
    raw = service.get("routes")
    if not isinstance(raw, list) or not raw:
        raw = [{
            "id": "route_1", "name": "线路 1", "enabled": True,
            "base_url": service.get("base_url", ""),
            "api_key": service.get("api_key", ""),
        }]
    routes = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        route = copy.deepcopy(item)
        route["id"] = str(route.get("id") or f"route_{index}")
        route["name"] = str(route.get("name") or f"线路 {index}")
        route["enabled"] = bool(route.get("enabled", True))
        route["base_url"] = str(route.get("base_url") or "").strip().rstrip("/")
        route["api_key"] = str(route.get("api_key") or "")
        routes.append(route)
    return routes


def _rate_cooldown_remaining(key: str) -> float:
    with _route_guard:
        return max(0.0, float(_route_cooldowns.get(key, 0.0) or 0.0) - time.monotonic())


def _route_runtime_snapshot(key: str, enabled: bool = True) -> dict:
    remaining = _rate_cooldown_remaining(key)
    with _route_guard:
        raw = copy.deepcopy(_route_runtime.get(key) or {})
    reason = str(raw.get("reason") or "")
    if not enabled:
        status = "disabled"
    elif int(raw.get("active", 0) or 0) > 0:
        status = "active"
    elif reason == "quota_exhausted" and remaining > 0:
        status = "blocked"
    elif remaining > 0:
        status = "cooling"
    else:
        status = "ready"
    return {
        "status": status,
        "reason": reason,
        "cooldown_remaining": round(remaining, 1),
        "last_used_at": float(raw.get("last_used_at", 0.0) or 0.0),
        "successes": int(raw.get("successes", 0) or 0),
        "failures": int(raw.get("failures", 0) or 0),
    }


def _clear_route_runtime_state(service_id: str = "", route_id: str = "") -> int:
    prefix = f"{service_id}/" if service_id else ""
    target = f"{prefix}{route_id}" if route_id else ""
    removed = 0
    with _route_guard:
        for key in list(_route_runtime):
            if (target and key == target) or (not target and (not prefix or key.startswith(prefix))):
                _route_runtime.pop(key, None); _route_cooldowns.pop(key, None); removed += 1
    return removed

# The lowest ceiling seen across vision endpoints (glm-4v-flash: 1..1024).
# Descriptions are a couple hundred characters, so this is never the binding
# constraint on quality -- only on whether the request is accepted at all.
VLM_SAFE_MAX_TOKENS = 1024


def _config_path() -> Path | None:
    try:
        import folder_paths
        user_dir = Path(folder_paths.get_user_directory())
        return user_dir / "default" / "Myang_node" / "config" / "llm_services.json"
    except Exception:
        return None


def _legacy_config_path() -> Path | None:
    try:
        import folder_paths
        user_dir = Path(folder_paths.get_user_directory())
        return user_dir / "default" / "prompt-assistant" / "config" / "config.json"
    except Exception:
        return None


def _read_config_path() -> Path | None:
    own = _config_path()
    if own is not None and own.is_file():
        return own
    legacy = _legacy_config_path()
    if legacy is not None and legacy.is_file():
        return legacy
    return own


def _service_fingerprint(service: dict) -> str:
    comparable = {
        "name": str(service.get("name") or "").strip().casefold(),
        "type": str(service.get("type") or "openai_compatible").strip(),
        "base_url": str(service.get("base_url") or "").strip().rstrip("/"),
        "api_key": str(service.get("api_key") or ""),
        "llm_models": service.get("llm_models") or [],
        "vlm_models": service.get("vlm_models") or [],
    }
    return json.dumps(comparable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _prefer_service(left: dict, right: dict) -> dict:
    left_generated = bool(_GENERATED_ID_RE.fullmatch(str(left.get("id") or "")))
    right_generated = bool(_GENERATED_ID_RE.fullmatch(str(right.get("id") or "")))
    if left_generated != right_generated:
        return right if left_generated else left
    return left


def _normalize_config(config: dict) -> dict:
    normalized = copy.deepcopy(config) if isinstance(config, dict) else {}
    aliases = {
        str(key): str(value)
        for key, value in (normalized.get("service_aliases") or {}).items()
        if str(key).strip() and str(value).strip()
    }
    services: list[dict] = []
    fingerprint_indexes: dict[str, int] = {}
    for raw in normalized.get("model_services") or []:
        if not isinstance(raw, dict):
            continue
        service = copy.deepcopy(raw)
        service["id"] = str(service.get("id") or "").strip()
        service["name"] = str(service.get("name") or service["id"]).strip()
        service["type"] = str(service.get("type") or "openai_compatible").strip()
        service["llm_models"] = list(service.get("llm_models") or [])
        service["vlm_models"] = list(service.get("vlm_models") or [])
        fingerprint = _service_fingerprint(service)
        duplicate_index = fingerprint_indexes.get(fingerprint)
        if duplicate_index is None:
            fingerprint_indexes[fingerprint] = len(services)
            services.append(service)
            continue
        current = services[duplicate_index]
        preferred = _prefer_service(current, service)
        dropped = service if preferred is current else current
        if dropped.get("id") and preferred.get("id"):
            aliases[str(dropped["id"])] = str(preferred["id"])
        if dropped.get("name") and dropped.get("name") != preferred.get("name"):
            aliases[str(dropped["name"])] = str(preferred["id"])
        if preferred is service:
            services[duplicate_index] = service

    valid_ids = {str(service.get("id") or "") for service in services}
    aliases = {
        alias: target for alias, target in aliases.items()
        if target in valid_ids and alias not in valid_ids
    }
    normalized["model_services"] = services
    normalized["service_aliases"] = aliases
    return normalized


def _load_config() -> dict:
    global _config_cache, _config_mtime_ns, _config_source
    path = _read_config_path()
    if path is None or not path.is_file():
        return {"model_services": []}
    mtime_ns = path.stat().st_mtime_ns
    if (_config_cache is not None and path == _config_source
            and mtime_ns == _config_mtime_ns):
        return _config_cache
    try:
        _config_cache = _normalize_config(json.loads(path.read_text("utf-8")))
        _config_mtime_ns = mtime_ns
        _config_source = path
    except Exception as exc:
        logger.warning("LLM config load failed: %s", exc)
        _config_cache = {"model_services": []}
    return _config_cache


def _services() -> list[dict]:
    return _load_config().get("model_services", [])


def llm_service_options() -> list[str]:
    options = []
    for svc in _services():
        label = str(svc.get("name") or svc.get("id") or "").strip()
        for model in svc.get("llm_models", []):
            mname = model.get("name", "")
            if label and mname:
                options.append(f"{label} :: {mname}")
    return options if options else ["未配置 LLM 服务"]


def vlm_service_options() -> list[str]:
    options = ["off"]
    for svc in _services():
        label = str(svc.get("name") or svc.get("id") or "").strip()
        for model in svc.get("vlm_models", []):
            mname = model.get("name", "")
            if label and mname:
                options.append(f"{label} :: {mname}")
    return options


def _parse_service_model(service_str: str) -> tuple[str, str]:
    value = str(service_str or "").strip()
    if " :: " in value:
        parts = value.split(" :: ", 1)
        return parts[0].strip(), parts[1].strip()
    parts = value.split("/", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return parts[0].strip(), ""


def _find_service(service_ref: str) -> dict:
    reference = str(service_ref or "").strip()
    config = _load_config()
    services = config.get("model_services", [])
    for svc in services:
        if str(svc.get("id") or "") == reference:
            return svc
    alias_target = (config.get("service_aliases") or {}).get(reference)
    if alias_target:
        for svc in services:
            if str(svc.get("id") or "") == alias_target:
                return svc
    for svc in services:
        if str(svc.get("name") or "") == reference:
            return svc
    folded = reference.casefold()
    matches = [svc for svc in services if folded in {
        str(svc.get("id") or "").casefold(),
        str(svc.get("name") or "").casefold(),
    }]
    if len(matches) == 1:
        return matches[0]
    return {}


def public_config() -> dict:
    """Return editable service data without exposing stored API keys."""
    config = _load_config()
    services = []
    for raw in config.get("model_services", []):
        service = copy.deepcopy(raw)
        service_id = str(raw.get("id") or raw.get("name") or "service")
        service.pop("api_key", None)
        service["_original_id"] = str(raw.get("id") or "")
        service["api_key_configured"] = bool(raw.get("api_key"))
        service["api_key_action"] = "keep"
        public_routes = []
        for raw_route in _normalize_routes(raw):
            route = copy.deepcopy(raw_route)
            route.pop("api_key", None)
            route["_original_id"] = str(raw_route.get("id") or "")
            route["api_key_configured"] = bool(raw_route.get("api_key"))
            route["api_key_action"] = "keep"
            route["runtime"] = _route_runtime_snapshot(
                _route_state_key(service_id, str(raw_route.get("id") or "route")),
                enabled=bool(raw_route.get("enabled", True)),
            )
            public_routes.append(route)
        service["routes"] = public_routes
        route_states = [route["runtime"]["status"] for route in public_routes]
        service["runtime"] = {
            "enabled_routes": sum(
                1 for route in public_routes if route.get("enabled", True)),
            "ready_routes": sum(
                1 for status in route_states if status in {"ready", "active"}),
            "cooling_routes": route_states.count("cooling"),
            "active_routes": route_states.count("active"),
        }
        services.append(service)
    return {
        "services": services,
        "aliases": copy.deepcopy(config.get("service_aliases") or {}),
        "source": "myang" if _read_config_path() == _config_path() else "legacy",
    }


def _clean_models(
    raw_models: Any, label: str, *, stream_default: bool = True,
) -> list[dict]:
    if raw_models is None:
        return []
    if not isinstance(raw_models, list):
        raise ValueError(f"{label} 必须是列表")
    cleaned = []
    seen = set()
    default_index = None
    for index, raw in enumerate(raw_models):
        if not isinstance(raw, dict):
            raise ValueError(f"{label} 第 {index + 1} 项格式错误")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError(f"{label} 第 {index + 1} 项缺少模型名")
        folded = name.casefold()
        if folded in seen:
            raise ValueError(f"{label} 中模型名重复：{name}")
        seen.add(folded)
        try:
            temperature = float(raw.get("temperature", 0.7))
            # 0 means "do not send max_tokens".  That leaves the limit to the
            # provider/model instead of imposing an arbitrary 4096-token cap
            # on every newly configured model.
            max_tokens = int(raw.get("max_tokens", 0))
            top_p = float(raw.get("top_p", 0.9))
            timeout = int(raw.get("timeout", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} / {name} 的采样参数不是有效数字") from exc
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"{label} / {name} 的 temperature 必须在 0～2")
        if not 0 <= max_tokens <= 262144:
            raise ValueError(f"{label} / {name} 的 max_tokens 必须在 0～262144（0 表示不发送限制）")
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"{label} / {name} 的 top_p 必须在 0～1")
        if not 0 <= timeout <= 1800:
            raise ValueError(f"{label} / {name} 的 timeout 必须在 0～1800 秒（0 表示自动）")
        model = copy.deepcopy(raw)
        model.update({
            "name": name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "timeout": timeout,
            # Existing configurations did not have this field. Streaming is
            # the new default, while the UI keeps an explicit per-model switch.
            "stream": bool(raw.get("stream", stream_default)),
            "is_default": bool(raw.get("is_default")),
        })
        if model["is_default"] and default_index is None:
            default_index = index
        cleaned.append(model)
    if cleaned:
        if default_index is None:
            default_index = 0
        for index, model in enumerate(cleaned):
            model["is_default"] = index == default_index
    return cleaned


def _resolve_original_service(config: dict, original_id: str) -> dict:
    if not original_id:
        return {}
    for service in config.get("model_services", []):
        if str(service.get("id") or "") == original_id:
            return service
    target = (config.get("service_aliases") or {}).get(original_id)
    if target:
        for service in config.get("model_services", []):
            if str(service.get("id") or "") == target:
                return service
    return {}


def _prepare_routes(raw: dict, original: dict, display_name: str) -> list[dict]:
    """Validate route groups and apply per-route secret update actions."""
    raw_routes = raw.get("routes")
    if not isinstance(raw_routes, list) or not raw_routes:
        # Accept saves from an older browser tab that still posts only root
        # fields. This also makes the public API backward compatible.
        raw_routes = [{
            "id": "route_1",
            "_original_id": "route_1" if original else "",
            "name": "线路 1",
            "enabled": True,
            "base_url": raw.get("base_url", ""),
            "api_key": raw.get("api_key", ""),
            "api_key_action": raw.get("api_key_action", "keep"),
        }]

    original_routes = {
        str(route.get("id") or ""): route
        for route in _normalize_routes(original)
        if str(route.get("id") or "")
    } if original else {}
    routes: list[dict] = []
    ids: set[str] = set()
    for index, raw_route in enumerate(raw_routes):
        if not isinstance(raw_route, dict):
            raise ValueError(f"服务 {display_name} 的第 {index + 1} 条线路格式错误")
        route_id = str(raw_route.get("id") or "").strip()
        original_id = str(raw_route.get("_original_id") or "").strip()
        if not _SERVICE_ID_RE.fullmatch(route_id):
            raise ValueError(
                f"服务 {display_name} 的线路 ID“{route_id or '(空)'}”无效；"
                "只能使用英文字母、数字、点、下划线和连字符")
        if route_id.casefold() in ids:
            raise ValueError(f"服务 {display_name} 的线路 ID 重复：{route_id}")
        ids.add(route_id.casefold())

        original_route = original_routes.get(original_id) if original_id else None
        if original_id and original_route is None:
            raise ValueError(f"服务 {display_name} 找不到待编辑的原线路：{original_id}")
        if original_route and route_id != original_id:
            raise ValueError(
                f"线路 ID 是稳定标识，不能从 {original_id} 改成 {route_id}；"
                "请新建线路后再删除旧线路")

        route_name = str(raw_route.get("name") or f"线路 {index + 1}").strip()
        if not route_name:
            raise ValueError(f"服务 {display_name} 的线路 {route_id} 缺少名称")
        base_url = str(raw_route.get("base_url") or "").strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"服务 {display_name} / {route_name} 的 Base URL 必须以 http:// 或 https:// 开头")

        action = str(raw_route.get("api_key_action") or "keep")
        if action == "keep":
            api_key = str((original_route or {}).get("api_key") or "")
        elif action == "set":
            api_key = str(raw_route.get("api_key") or "").strip()
            if not api_key:
                raise ValueError(
                    f"服务 {display_name} / {route_name} 选择了更新 API Key，但没有填写新值")
        elif action == "clear":
            api_key = ""
        else:
            raise ValueError(f"服务 {display_name} / {route_name} 的 API Key 操作无效")

        route = copy.deepcopy(original_route) if original_route else {}
        route.update({
            "id": route_id,
            "name": route_name,
            "enabled": bool(raw_route.get("enabled", True)),
            "base_url": base_url,
            "api_key": api_key,
        })
        for private_key in ("_original_id", "api_key_action", "api_key_configured"):
            route.pop(private_key, None)
        routes.append(route)

    if not routes:
        raise ValueError(f"服务 {display_name} 至少需要一条 API 线路")
    if not any(route["enabled"] for route in routes):
        raise ValueError(f"服务 {display_name} 至少需要启用一条 API 线路")
    return routes


def _prepare_services(raw_services: Any, current: dict) -> tuple[list[dict], dict[str, str]]:
    if not isinstance(raw_services, list):
        raise ValueError("services 必须是列表")
    services = []
    ids = set()
    names = set()
    aliases = copy.deepcopy(current.get("service_aliases") or {})
    for index, raw in enumerate(raw_services):
        if not isinstance(raw, dict):
            raise ValueError(f"第 {index + 1} 个服务格式错误")
        service_id = str(raw.get("id") or "").strip()
        display_name = str(raw.get("name") or "").strip()
        original_id = str(raw.get("_original_id") or "").strip()
        if not _SERVICE_ID_RE.fullmatch(service_id):
            raise ValueError(
                f"服务 ID“{service_id or '(空)'}”无效；只能使用英文字母、数字、点、下划线和连字符")
        if not display_name:
            raise ValueError(f"服务 {service_id} 缺少显示名称")
        if service_id.casefold() in ids:
            raise ValueError(f"服务 ID 重复：{service_id}")
        if display_name.casefold() in names:
            raise ValueError(f"显示名称重复：{display_name}")
        ids.add(service_id.casefold())
        names.add(display_name.casefold())

        original = _resolve_original_service(current, original_id)
        if original_id and not original:
            raise ValueError(f"找不到待编辑的原服务：{original_id}")
        if original and service_id != str(original.get("id") or ""):
            raise ValueError(
                f"服务 ID 是工作流稳定标识，不能从 {original_id} 改成 {service_id}；请新建服务后再删除旧服务")

        service_type = str(raw.get("type") or "openai_compatible").strip()
        if service_type not in _SERVICE_TYPES:
            raise ValueError(f"服务 {display_name} 的类型无效：{service_type}")
        route_strategy = str(raw.get("route_strategy") or "round_robin").strip()
        if route_strategy not in _ROUTE_STRATEGIES:
            raise ValueError(f"服务 {display_name} 的路由策略无效：{route_strategy}")
        routes = _prepare_routes(raw, original, display_name)
        primary = routes[0]

        service = copy.deepcopy(original) if original else {}
        service.update({
            "id": service_id,
            "name": display_name,
            "type": service_type,
            "route_strategy": route_strategy,
            "routes": routes,
            "base_url": primary["base_url"],
            "api_key": primary["api_key"],
            "llm_models": _clean_models(
                raw.get("llm_models"), f"{display_name} 的 LLM 模型", stream_default=True),
            "vlm_models": _clean_models(
                raw.get("vlm_models"), f"{display_name} 的 VLM 模型", stream_default=False),
        })
        for private_key in ("_original_id", "api_key_action", "api_key_configured"):
            service.pop(private_key, None)
        if original and str(original.get("name") or "") != display_name:
            aliases[str(original.get("name") or "")] = service_id
        services.append(service)

    valid_ids = {service["id"] for service in services}
    aliases = {
        str(alias): str(target) for alias, target in aliases.items()
        if str(target) in valid_ids and str(alias) not in valid_ids
    }
    return services, aliases


def save_public_services(raw_services: Any) -> dict:
    """Validate and atomically replace Myang's complete service list."""
    global _config_cache, _config_mtime_ns, _config_source
    current = _load_config()
    services, aliases = _prepare_services(raw_services, current)
    config = copy.deepcopy(current)
    config["model_services"] = services
    config["service_aliases"] = aliases
    path = _config_path()
    if path is None:
        raise RuntimeError("无法确定 Myang_node LLM 配置路径")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2), "utf-8")
    os.replace(temporary, path)
    _config_cache = _normalize_config(config)
    _config_mtime_ns = path.stat().st_mtime_ns
    _config_source = path
    # URLs, keys or enabled flags may have changed. Stale circuit state must
    # not keep a newly repaired route disabled after the user saves it.
    _clear_route_runtime_state()
    return public_config()


def reset_route_runtime(service_ref: str = "", route_id: str = "") -> dict:
    """Manually release route cooldowns and return the refreshed public config."""
    service_id = ""
    if service_ref:
        service = _find_service(service_ref)
        if not service:
            raise ValueError(f"找不到 LLM 服务：{service_ref}")
        service_id = str(service.get("id") or "")
        if route_id and not any(
            str(route.get("id") or "") == route_id
            for route in _normalize_routes(service)
        ):
            raise ValueError(f"服务 {service_id} 找不到 API 线路：{route_id}")
    elif route_id:
        raise ValueError("清除单条线路状态时必须同时提供 service_id")
    cleared = _clear_route_runtime_state(service_id, str(route_id or ""))
    result = public_config()
    result.update({"success": True, "cleared": cleared})
    return result


def _build_chat_url(base_url: str, service_type: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if service_type == "ollama" and not base.endswith("/v1"):
        base += "/v1"
    return f"{base}/chat/completions"


def _find_model(svc: dict, model_name: str, key: str) -> dict | None:
    models = svc.get(key, [])
    target = None
    if model_name:
        target = next((m for m in models if m.get("name") == model_name), None)
    if not target:
        target = next((m for m in models if m.get("is_default")), models[0] if models else None)
    return target


def _ollama_unload(base_url: str, model: str):
    try:
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        url = f"{base}/api/generate"
        payload = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except Exception:
        pass


def _ordered_enabled_routes(svc: dict, lane: str) -> list[dict]:
    routes = [r for r in _normalize_routes(svc)
              if r.get("enabled", True) and r.get("base_url")]
    if not routes:
        raise ValueError(f"服务 {svc.get('name') or svc.get('id')} 没有可用 API 线路")
    if str(svc.get("route_strategy") or "round_robin") == "failover":
        return routes
    key = f"{svc.get('id') or svc.get('name')}|{lane}"
    with _route_guard:
        start = _route_counters.get(key, 0) % len(routes)
        _route_counters[key] = start + 1
    return routes[start:] + routes[:start]


def _failure_policy(error: BaseException) -> tuple[str, float]:
    if isinstance(error, LLMQuotaConfirmedError):
        return "quota_exhausted", ROUTE_QUOTA_COOLDOWN_SECONDS
    if isinstance(error, LLMQuotaError):
        return "quota_retry", ROUTE_QUOTA_RETRY_COOLDOWN_SECONDS
    if isinstance(error, LLMRateLimitError):
        return "rate_limit", RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS
    if isinstance(error, LLMEmptyResponseError):
        return "empty", ROUTE_TRANSIENT_COOLDOWN_SECONDS
    if isinstance(error, LLMStreamResponseError):
        return "stream", ROUTE_TRANSIENT_COOLDOWN_SECONDS
    if isinstance(error, LLMRequestTimeoutError):
        return "timeout", ROUTE_TIMEOUT_COOLDOWN_SECONDS
    if isinstance(error, LLMRoutesCoolingError):
        return "cooling", min(
            MAX_INLINE_COOLDOWN_WAIT_SECONDS,
            max(0.0, float(error.retry_after or 0.0)),
        )
    message = str(error or "").casefold()
    if any(token in message for token in (
            "insufficient_quota", "allocated quota exceeded",
            "workspace quota exceeded", "quota limit exceeded",
            "额度不足", "额度耗尽", "配额耗尽", "余额不足", "欠费")):
        return "quota_retry", ROUTE_QUOTA_RETRY_COOLDOWN_SECONDS
    if "timed out" in message or "timeout" in message:
        return "timeout", ROUTE_TIMEOUT_COOLDOWN_SECONDS
    if "connection failed" in message:
        return "connection", ROUTE_TRANSIENT_COOLDOWN_SECONDS
    match = re.search(r"api error\s+(\d{3})", message)
    if match and int(match.group(1)) in {401, 403}:
        return "auth", ROUTE_AUTH_COOLDOWN_SECONDS
    if match and int(match.group(1)) in {408, 500, 502, 503, 504}:
        return "server", ROUTE_TRANSIENT_COOLDOWN_SECONDS
    return "", 0.0


def _error_chain(error: BaseException):
    current: BaseException | None = error
    seen: set[int] = set()
    for _depth in range(6):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def error_reason(error: BaseException) -> str:
    """Return the first recognised provider/route failure through wrappers."""
    for current in _error_chain(error):
        if isinstance(current, LLMRoutesCoolingError) and current.reasons:
            if "quota_exhausted" in current.reasons:
                return "quota_exhausted"
            if "quota_retry" in current.reasons:
                return "quota_retry"
            if "rate_limit" in current.reasons:
                return "rate_limit"
            return "cooling"
        reason, _seconds = _failure_policy(current)
        if reason:
            return reason
    return ""


def is_rate_limit_error(error: BaseException) -> bool:
    return error_reason(error) == "rate_limit"


def is_quota_error(error: BaseException) -> bool:
    return error_reason(error) in {"quota_exhausted", "quota_retry"}


def is_service_unavailable_error(error: BaseException) -> bool:
    """True when smaller prompts or per-segment retries cannot fix the call."""
    return error_reason(error) in {
        "quota_exhausted", "quota_retry", "rate_limit", "cooling", "timeout",
        "connection", "stream", "auth", "server",
    }


def _mark_route_failure(service_id: str, route: dict, error: BaseException) -> None:
    key = _route_state_key(service_id, str(route.get("id") or "route"))
    reason, seconds = _failure_policy(error)
    with _route_guard:
        state = _route_runtime.setdefault(key, {})
        state["failures"] = int(state.get("failures", 0) or 0) + 1
        state["reason"] = "quota_retry" if (
            isinstance(error, LLMQuotaError)
            and not isinstance(error, LLMQuotaConfirmedError)
        ) else reason
        state["last_error"] = str(error)[:300]
        state["last_used_at"] = time.time()
        state["active"] = max(0, int(state.get("active", 0) or 0) - 1)
        if seconds:
            _route_cooldowns[key] = time.monotonic() + seconds
    logger.warning("API 线路暂时跳过 | %s / %s | %s %.1fs",
                   service_id, route.get("name") or route.get("id"), reason, seconds)


def _mark_route_slow_probe(service_id: str, route: dict, error: BaseException) -> None:
    """Record a 15-second slow start without cooling the route.

    The same route is intentionally eligible for the 30-second confirmation
    round in the current call.  A second-round timeout uses the normal failure
    path and circuit-breaker cooldown.
    """
    key = _route_state_key(service_id, str(route.get("id") or "route"))
    with _route_guard:
        state = _route_runtime.setdefault(key, {})
        state["slow_probes"] = int(state.get("slow_probes", 0) or 0) + 1
        state["last_probe_error"] = str(error)[:300]
        state["last_used_at"] = time.time()
        state["active"] = max(0, int(state.get("active", 0) or 0) - 1)
        state["reason"] = ""
        # A quick probe is not a circuit-breaker failure.  Keep this route
        # immediately available for the explicit confirmation round.
        _route_cooldowns.pop(key, None)
    logger.info(
        "API 线路 15 秒内未开始生成，保留到 30 秒复试 | %s / %s",
        service_id, route.get("name") or route.get("id"))


def _mark_route_success(service_id: str, route: dict) -> None:
    key = _route_state_key(service_id, str(route.get("id") or "route"))
    with _route_guard:
        state = _route_runtime.setdefault(key, {})
        state["successes"] = int(state.get("successes", 0) or 0) + 1
        state["reason"] = ""
        state.pop("quota_confirmed", None)
        state["last_used_at"] = time.time()
        state["active"] = max(0, int(state.get("active", 0) or 0) - 1)
        _route_cooldowns.pop(key, None)


def _mark_route_inactive(service_id: str, route: dict) -> None:
    key = _route_state_key(service_id, str(route.get("id") or "route"))
    with _route_guard:
        state = _route_runtime.setdefault(key, {})
        state["active"] = max(0, int(state.get("active", 0) or 0) - 1)


def _mark_route_active(service_id: str, route: dict) -> None:
    key = _route_state_key(service_id, str(route.get("id") or "route"))
    with _route_guard:
        state = _route_runtime.setdefault(key, {})
        state["active"] = int(state.get("active", 0) or 0) + 1
        state["last_used_at"] = time.time()


def _route_can_retry(error: BaseException) -> bool:
    return bool(_failure_policy(error)[0])


def _raise_processing_interrupted() -> None:
    if _comfy_model_management is not None:
        raise _comfy_model_management.InterruptProcessingException()
    raise RuntimeError("LLM request cancelled")


def _interruptible_cooldown_wait(seconds: float, service_id: str) -> None:
    """Wait for a short circuit-breaker cooldown and remain stoppable from UI."""
    global _active_http_request_id
    wait_seconds = max(0.0, float(seconds or 0.0))
    if wait_seconds <= 0:
        return
    state = {"cancel": threading.Event(), "response": None, "kind": "cooldown_wait"}
    with _route_guard:
        _active_http_request_id += 1
        request_id = _active_http_request_id
        _active_http_requests[request_id] = state
    logger.info("API 所有线路冷却中: %s | 等待 %.1f 秒后自动重试（可点击停止）",
                service_id, wait_seconds)
    deadline = time.monotonic() + wait_seconds
    try:
        while time.monotonic() < deadline:
            if state["cancel"].wait(min(0.1, max(0.0, deadline - time.monotonic()))):
                _raise_processing_interrupted()
            if _comfy_model_management is not None:
                _comfy_model_management.throw_exception_if_processing_interrupted()
    finally:
        with _route_guard:
            _active_http_requests.pop(request_id, None)


def _llm_timeout_policy(
    target: dict, prompt_chars: int, use_stream: bool = False,
) -> tuple[int, int]:
    """Return startup timeout and route-selection budget.

    A stream that has emitted meaningful reasoning/content is no longer bound
    by either value; non-stream requests keep the ordinary absolute timeout.
    """
    if use_stream:
        # Streaming LLMs use a deterministic route-count-aware 15/30 second
        # schedule in call_llm.  An explicit smaller value remains a valid
        # user override, while legacy 90/360/etc. values are safely capped.
        explicit = int(target.get("timeout", 0) or 0)
        ceiling = (min(explicit, LLM_FINAL_FIRST_OUTPUT_SECONDS)
                   if explicit > 0 else LLM_FINAL_FIRST_OUTPUT_SECONDS)
        return max(1, ceiling), DEFAULT_LLM_CALL_BUDGET_SECONDS
    explicit = int(target.get("timeout", 0) or 0)
    if explicit > 0:
        value = max(MIN_ROUTE_ATTEMPT_SECONDS, min(explicit, 1800))
        return value, value
    route_timeout = min(
        210,
        max(DEFAULT_LLM_ROUTE_TIMEOUT_SECONDS,
            DEFAULT_LLM_ROUTE_TIMEOUT_SECONDS + max(0, int(prompt_chars)) // 300),
    )
    call_budget = min(
        MAX_LLM_CALL_BUDGET_SECONDS,
        max(DEFAULT_LLM_CALL_BUDGET_SECONDS, route_timeout + 90),
    )
    return int(route_timeout), int(call_budget)


def _stream_text(value: Any) -> str:
    """Flatten OpenAI-compatible string/content-block deltas."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "".join(_stream_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "value", "output_text"):
            if key in value:
                return _stream_text(value.get(key))
    return ""


def _streaming_unsupported(error: BaseException) -> bool:
    message = str(error or "").casefold()
    if not any(code in message for code in (
            "api error 400", "api error 404", "api error 422", "api error 501")):
        return False
    return any(token in message for token in (
        "stream", "streaming", "text/event-stream", "sse", "不支持流式",
    ))


def _http_state_snapshot(state: dict) -> dict:
    now = time.monotonic()
    started = float(state.get("started_at", now) or now)
    first = float(state.get("first_chunk_at", 0.0) or 0.0)
    last = float(state.get("last_chunk_at", 0.0) or 0.0)
    response_at = float(state.get("response_at", 0.0) or 0.0)
    generation_started = float(state.get("generation_started_at", 0.0) or 0.0)
    return {
        "elapsed": round(max(0.0, now - started), 1),
        "connected_after": round(max(0.0, response_at - started), 1) if response_at else None,
        "first_chunk_after": round(max(0.0, first - started), 1) if first else None,
        "generation_started_after": round(
            max(0.0, generation_started - started), 1,
        ) if generation_started else None,
        "waiting_until_complete": bool(generation_started),
        "idle": round(max(0.0, now - last), 1) if last else None,
        "chunks": int(state.get("chunks", 0) or 0),
        "content_chars": int(state.get("content_chars", 0) or 0),
        "reasoning_chars": int(state.get("reasoning_chars", 0) or 0),
        "finish_reason": str(state.get("finish_reason") or ""),
        "preview": str(state.get("preview") or "")[-220:],
    }


def _emit_http_state(state: dict, phase: str, *, force: bool = False, **updates) -> None:
    now = time.monotonic()
    if not force and now - float(state.get("last_emit_at", 0.0) or 0.0) < 0.35:
        return
    state["last_emit_at"] = now
    payload = _http_state_snapshot(state)
    payload.update(updates)
    _emit_llm_stream(state.get("diagnostics"), phase, **payload)


def _stream_error(error_value: Any) -> None:
    body = json.dumps(error_value, ensure_ascii=False)[:500]
    lowered = body.casefold()
    if any(token in lowered for token in (
            "insufficient_quota", "allocated quota exceeded",
            "workspace quota exceeded", "quota limit exceeded",
            "配额", "额度", "欠费")):
        # Some SSE gateways omit the HTTP status in the event payload.  The
        # provider's quota code is still enough to enter the recoverable path.
        raise LLMQuotaError(f"API stream quota response: {body}")
    if "429" in lowered or "rate_limit" in lowered or "tpm" in lowered:
        raise LLMRateLimitError(f"API stream error 429: {body}")
    raise LLMStreamResponseError(f"API stream error: {body}")


def _relax_stream_socket_timeout(resp: Any) -> bool:
    """Let an active token stream finish; the outer loop still handles stop."""
    candidates = [
        getattr(resp, "_sock", None),
        getattr(getattr(resp, "fp", None), "_sock", None),
        getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None),
    ]
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        setter = getattr(candidate, "settimeout", None)
        if not callable(setter):
            continue
        try:
            setter(None)
            return True
        except Exception:
            logger.debug("H3-Myang: 无法解除活跃流的 socket 读取超时", exc_info=True)
    return False


def _read_streaming_response(resp: Any, state: dict) -> dict:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = ""
    parsed_events = 0
    malformed_events = 0
    done_seen = False

    def consume_data(data: str) -> bool:
        nonlocal finish_reason, parsed_events, malformed_events, done_seen
        if data.strip() == "[DONE]":
            done_seen = True
            return True
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            malformed_events += 1
            return False
        if not isinstance(chunk, dict):
            malformed_events += 1
            return False
        if chunk.get("error"):
            _stream_error(chunk.get("error"))
        parsed_events += 1
        now = time.monotonic()
        if not state.get("first_chunk_at"):
            state["first_chunk_at"] = now
            diagnostics = state.get("diagnostics") or {}
            logger.info(
                "LLM stream first chunk: %s / %s | %.1fs",
                diagnostics.get("service") or diagnostics.get("service_id") or "?",
                diagnostics.get("route") or diagnostics.get("route_id") or "?",
                max(0.0, now - float(state.get("started_at", now) or now)),
            )
        state["last_chunk_at"] = now
        state["chunks"] = parsed_events
        # 仅含 delta.role 等协议字段的数据块只能证明连接还活着，
        # 不能算作正文/思考已经开始，也不能解除连接与首包保护。
        content = ""
        reasoning = ""
        choices = chunk.get("choices") or []
        if choices and isinstance(choices[0], dict):
            choice = choices[0]
            if isinstance(choice.get("message"), dict) and "delta" not in choice:
                # A proxy ignored streaming and returned one ordinary OpenAI
                # completion without a useful Content-Type header.
                done_seen = True
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                delta = {}
            content = _stream_text(
                delta.get("content") if "content" in delta else choice.get("text"))
            reasoning = ""
            for key in ("reasoning_content", "reasoning", "reasoning_details"):
                reasoning += _stream_text(delta.get(key))
            if content:
                content_parts.append(content)
            if reasoning:
                reasoning_parts.append(reasoning)
            finish_reason = str(choice.get("finish_reason") or finish_reason or "")
            state["finish_reason"] = finish_reason
        elif isinstance(chunk.get("message"), dict):
            # Ollama /api/chat streams newline-delimited objects instead of
            # OpenAI choices, but uses the same assistant message fields.
            message = chunk.get("message") or {}
            content = _stream_text(message.get("content"))
            reasoning = _stream_text(
                message.get("thinking") or message.get("reasoning_content")
                or message.get("reasoning"))
            if content:
                content_parts.append(content)
            if reasoning:
                reasoning_parts.append(reasoning)
        if chunk.get("done"):
            finish_reason = str(chunk.get("done_reason") or "stop")
            state["finish_reason"] = finish_reason
        content_text = "".join(content_parts)
        reasoning_text = "".join(reasoning_parts)
        if (content or reasoning) and not state.get("generation_started_at"):
            state["generation_started_at"] = now
            state["socket_timeout_relaxed"] = _relax_stream_socket_timeout(resp)
            logger.info(
                "LLM stream generation started | active stream will wait for provider completion "
                "or manual stop | socket_timeout_relaxed=%s",
                state["socket_timeout_relaxed"],
            )
        state["content_chars"] = len(content_text)
        state["reasoning_chars"] = len(reasoning_text)
        preview = content_text or reasoning_text
        state["preview"] = re.sub(r"\s+", " ", preview[-220:]).strip()
        if reasoning_text and not content_text:
            phase = "reasoning"
        elif content_text:
            phase = "streaming"
        else:
            phase = "waiting_generation"
        _emit_http_state(state, phase)
        # 部分 OpenAI 兼容网关会保持 HTTP keep-alive，却不再发送
        # data: [DONE]。finish_reason（或 Ollama done=true）已经是可靠的
        # 协议结束标记，收到后应立即收口，不能继续傻等 EOF。
        return bool(done_seen or finish_reason or chunk.get("done"))

    pending_data: list[str] = []
    for raw_line in resp:
        if state["cancel"].is_set():
            raise RuntimeError("LLM request cancelled")
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.rstrip("\r\n")
        if not line:
            if pending_data:
                if consume_data("\n".join(pending_data)):
                    break
                pending_data.clear()
            continue
        if line.startswith(":") or line.startswith(("event:", "id:", "retry:")):
            continue
        if line.startswith("data:"):
            value = line[5:]
            value = value[1:] if value.startswith(" ") else value
            # OpenAI SSE normally has a blank event separator. A few reverse
            # proxies omit it, so a new complete JSON/DONE line must flush the
            # preceding event instead of being concatenated into invalid JSON.
            if pending_data and (value.strip() == "[DONE]" or value.lstrip().startswith("{")):
                if consume_data("\n".join(pending_data)):
                    break
                pending_data.clear()
            pending_data.append(value)
            continue
        # Ollama and a few OpenAI-compatible proxies use NDJSON rather than
        # SSE. Flush a pending SSE event, then consume this complete JSON line.
        if pending_data:
            if consume_data("\n".join(pending_data)):
                break
            pending_data.clear()
        if consume_data(line.strip()):
            break
    else:
        if pending_data:
            consume_data("\n".join(pending_data))

    if parsed_events <= 0:
        raise LLMStreamResponseError(
            "流式连接已结束，但没有收到可解析的数据块"
            + (f"（忽略 {malformed_events} 个损坏数据块）" if malformed_events else ""))
    if malformed_events:
        raise LLMStreamResponseError(
            f"流式响应包含 {malformed_events} 个无法解析的数据块；"
            "为避免把不完整 JSON 交给分镜解析器，本线路结果已丢弃")
    if not done_seen and not finish_reason:
        raise LLMStreamResponseError(
            "流式连接在服务端结束标记之前断开：已收到 "
            f"{parsed_events} 块、正文 {len(''.join(content_parts))} 字、"
            f"思考 {len(''.join(reasoning_parts))} 字；本线路的半截结果已丢弃")
    content_text = "".join(content_parts)
    reasoning_text = "".join(reasoning_parts)
    state["finish_reason"] = finish_reason
    _emit_http_state(state, "stream_complete", force=True, done_seen=done_seen)
    logger.info(
        "LLM stream complete | chunks=%d | content=%d chars | reasoning=%d chars | finish=%s",
        parsed_events, len(content_text), len(reasoning_text), finish_reason or "?",
    )
    return {
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content_text,
                "reasoning_content": reasoning_text,
            },
            "finish_reason": finish_reason,
        }],
        "_myang_stream": {
            "chunks": parsed_events,
            "malformed_chunks": malformed_events,
            "done_seen": done_seen,
            "content_chars": len(content_text),
            "reasoning_chars": len(reasoning_text),
        },
    }


def _http_post_json_blocking(
    url: str, headers: dict, payload: dict, timeout: int, state: dict,
) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with _route_guard:
                state["response"] = resp
                state["response_at"] = time.monotonic()
            _emit_http_state(state, "connected", force=True)
            if state["cancel"].is_set():
                raise RuntimeError("LLM request cancelled")
            if not bool(payload.get("stream")):
                raw = resp.read()
                state["first_chunk_at"] = time.monotonic()
                state["last_chunk_at"] = state["first_chunk_at"]
                return json.loads(raw.decode("utf-8"))

            headers_obj = getattr(resp, "headers", None)
            content_type = str(headers_obj.get("Content-Type", "") if headers_obj is not None else "").casefold()
            if "application/json" in content_type and "text/event-stream" not in content_type:
                # Some endpoints accept stream=true but deliberately return a
                # normal completion. Preserve compatibility without a retry.
                raw = resp.read()
                state["first_chunk_at"] = time.monotonic()
                state["last_chunk_at"] = state["first_chunk_at"]
                try:
                    result = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise LLMStreamResponseError(
                        "服务端声明返回 JSON，但正文无法解析；已切换下一线路") from exc
                _emit_http_state(state, "stream_ignored", force=True)
                return result
            return _read_streaming_response(resp, state)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        if int(exc.code) == 429:
            lowered = body.casefold()
            if any(token in lowered for token in (
                    "insufficient_quota", "allocated quota exceeded",
                    "workspace quota exceeded", "quota limit exceeded",
                    "配额", "额度", "欠费")):
                raise LLMQuotaError(f"API error 429: {body}") from exc
            raise LLMRateLimitError(f"API error 429: {body}") from exc
        raise RuntimeError(f"API error {exc.code}: {body}") from exc
    except (LLMRateLimitError, LLMQuotaError, LLMStreamResponseError):
        raise
    except TimeoutError as exc:
        raise LLMRequestTimeoutError(f"Request timed out after {int(timeout)}s") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection failed: {exc.reason}") from exc
    except Exception as exc:
        if state["cancel"].is_set():
            raise RuntimeError("LLM request cancelled") from exc
        if bool(payload.get("stream")) and int(state.get("chunks", 0) or 0) > 0:
            raise LLMStreamResponseError(
                "流式连接在接收中途断开：已收到 "
                f"{int(state.get('chunks', 0))} 块、正文 "
                f"{int(state.get('content_chars', 0))} 字、思考 "
                f"{int(state.get('reasoning_chars', 0))} 字；{exc}") from exc
        raise


def _http_post_json(url: str, headers: dict, payload: dict, timeout: int = 120) -> dict:
    """Run urllib in a cancellable wait and expose connection/stream progress."""
    global _active_http_request_id
    diagnostics = copy.deepcopy(getattr(_http_diagnostic_context, "value", {}) or {})
    state = {
        "cancel": threading.Event(), "response": None,
        "diagnostics": diagnostics, "started_at": time.monotonic(),
        "response_at": 0.0, "first_chunk_at": 0.0, "last_chunk_at": 0.0,
        "chunks": 0, "content_chars": 0, "reasoning_chars": 0,
        "finish_reason": "", "preview": "", "last_emit_at": 0.0,
        "generation_started_at": 0.0, "socket_timeout_relaxed": False,
        "nominal_timeout_announced": False,
    }
    outcome: queue.Queue = queue.Queue(maxsize=1)
    with _route_guard:
        _active_http_request_id += 1
        request_id = _active_http_request_id
        _active_http_requests[request_id] = state

    def run_request():
        try:
            outcome.put((True, _http_post_json_blocking(
                url, headers, payload, timeout, state)))
        except BaseException as error:  # transfer the worker failure verbatim
            outcome.put((False, error))

    completed = False
    deadline = time.monotonic() + max(0.1, float(timeout))
    next_heartbeat = time.monotonic()
    # Publish the initial phase before the worker can report a fast response;
    # this keeps the UI lifecycle ordered even with local/proxy endpoints.
    _emit_http_state(state, "connecting", force=True)
    threading.Thread(
        target=run_request, name=f"myh3-llm-{request_id}", daemon=True).start()
    try:
        while True:
            if state["cancel"].is_set():
                _emit_http_state(state, "cancelled", force=True)
                _raise_processing_interrupted()
            if _comfy_model_management is not None:
                _comfy_model_management.throw_exception_if_processing_interrupted()
            now = time.monotonic()
            active_stream = bool(
                payload.get("stream") and state.get("generation_started_at"))
            if now >= deadline and not active_stream:
                # 超时边界上先接收已经完成并入队的结果，避免恰好完成的
                # 请求被随后 close 掉并误报超时。
                try:
                    ok, value = outcome.get_nowait()
                except queue.Empty:
                    pass
                else:
                    completed = True
                    if state["cancel"].is_set():
                        _emit_http_state(state, "cancelled", force=True)
                        _raise_processing_interrupted()
                    if ok:
                        return value
                    raise value
                # The worker may have parsed the first meaningful token while
                # this thread checked the queue.  Re-read the shared state at
                # the boundary so that token wins over the startup deadline.
                if payload.get("stream") and state.get("generation_started_at"):
                    continue
                state["cancel"].set()
                response = state.get("response")
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass
                raise LLMRequestTimeoutError(
                    (
                        f"Request timed out after {max(1, int(float(timeout)))}s"
                        + ("（尚未建立连接）" if not state.get("response_at")
                           else "（已连接但未收到首个数据块）" if not state.get("first_chunk_at")
                           else "（已收到协议数据块，但尚无正文或思考内容；最后数据块距今 "
                                f"{_http_state_snapshot(state).get('idle')} 秒；"
                                f"正文 {int(state.get('content_chars', 0))} 字，"
                                f"思考 {int(state.get('reasoning_chars', 0))} 字）")
                    ))
            if (now >= deadline and active_stream
                    and not state.get("nominal_timeout_announced")):
                state["nominal_timeout_announced"] = True
                _emit_http_state(
                    state,
                    "reasoning" if state.get("reasoning_chars") and not state.get("content_chars")
                    else "streaming",
                    force=True,
                    message="已开始生成，取消累计超时；等待服务端完整结束或手动停止",
                )
                logger.info(
                    "LLM active stream exceeded the startup timeout; waiting for provider completion "
                    "or manual stop | content=%d | reasoning=%d",
                    int(state.get("content_chars", 0) or 0),
                    int(state.get("reasoning_chars", 0) or 0),
                )
            if now >= next_heartbeat:
                phase = (
                    "reasoning" if state.get("reasoning_chars") and not state.get("content_chars")
                    else "streaming" if state.get("generation_started_at")
                    else "waiting_generation" if state.get("first_chunk_at")
                    else "waiting_first_byte" if state.get("response_at")
                    else "connecting"
                )
                _emit_http_state(state, phase, force=True)
                next_heartbeat = now + 1.0
            try:
                ok, value = outcome.get(timeout=0.1)
            except queue.Empty:
                continue
            completed = True
            # Closing a live response during manual stop can make the worker
            # enqueue a socket/partial-stream error before this loop observes
            # the cancel flag.  User cancellation is authoritative and must be
            # reported as a normal ComfyUI interruption, not a route failure.
            if state["cancel"].is_set():
                _emit_http_state(state, "cancelled", force=True)
                _raise_processing_interrupted()
            if ok:
                return value
            raise value
    finally:
        if not completed:
            state["cancel"].set()
            response = state.get("response")
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
        with _route_guard:
            _active_http_requests.pop(request_id, None)


def _post_json_with_diagnostics(
    url: str, headers: dict, payload: dict, timeout: int, diagnostics: dict,
) -> dict:
    """Attach local-only metadata without changing the tested HTTP signature."""
    previous = getattr(_http_diagnostic_context, "value", None)
    _http_diagnostic_context.value = diagnostics
    try:
        return _http_post_json(url, headers, payload, timeout=timeout)
    finally:
        if previous is None:
            try:
                delattr(_http_diagnostic_context, "value")
            except AttributeError:
                pass
        else:
            _http_diagnostic_context.value = previous


def _stream_completion_text(result: Any) -> tuple[str, str, str, dict]:
    """Extract one completed assistant message or raise a route-local error."""
    choices = result.get("choices", []) if isinstance(result, dict) else []
    reasoning = ""
    finish = ""
    stream_meta = result.get("_myang_stream") or {} if isinstance(result, dict) else {}
    text = ""
    if choices:
        message = choices[0].get("message", {}) or {}
        text = _stream_text(message.get("content")).strip()
        reasoning = _stream_text(message.get("reasoning_content"))
        finish = str(choices[0].get("finish_reason") or "")
    if text:
        return text, reasoning, finish, stream_meta
    logger.warning(
        "LLM returned empty content | finish_reason=%s | 思考通道 %d 字 | %s",
        finish or "?", len(reasoning), json.dumps(result, ensure_ascii=False)[:200])
    if finish.casefold() == "length" and reasoning:
        explanation = "输出额度被思考通道耗尽，服务端在正文开始前因 length 停止"
    elif finish.casefold() == "length":
        explanation = "服务端因输出长度上限停止，但没有返回正文"
    elif int(stream_meta.get("chunks", 0) or 0) <= 0:
        explanation = "服务端完成请求但没有返回正文或可用流数据块"
    else:
        explanation = "流式请求结束但助手正文为空"
    raise LLMEmptyResponseError(
        explanation
        + f" | finish_reason={finish or '?'}"
        + f" | reasoning_chars={len(reasoning)}"
        + f" | stream_chunks={int(stream_meta.get('chunks', 0) or 0)}")


def _call_llm_stream_routes(
    *, service_id: str, service_label: str, service_type: str,
    target: dict, routes: list[dict], payload: dict, route_timeout: int,
    base_diagnostics: dict, call_started_at: float,
    ollama_auto_unload: bool,
) -> str:
    """Probe streaming routes quickly, then confirm only slow starters."""
    cap = max(1, min(int(route_timeout), LLM_FINAL_FIRST_OUTPUT_SECONDS))
    fast = min(LLM_FAST_FIRST_OUTPUT_SECONDS, cap)
    if len(routes) > 1 and fast < cap:
        rounds = [(1, 2, fast), (2, 2, cap)]
        strategy = f"多线路 {fast}s 快探 → {cap}s 复查"
    else:
        rounds = [(1, 1, cap)]
        strategy = f"单线路 {cap}s" if len(routes) == 1 else f"多线路每线 {cap}s"
    logger.info(
        "LLM call: %s/%s | %s | max_output=%s | stream=开 | "
        "流式首内容策略=%s | 开始生成后等待服务端结束",
        service_label, target.get("name", ""),
        f"{sum(len(str(message.get('content') or '')) for message in payload.get('messages', []))} chars",
        payload.get("max_tokens", "服务端决定"), strategy)

    last_error: BaseException | None = None
    retry_ids: set[str] | None = None
    attempt_no = 0
    for round_no, round_total, first_output_timeout in rounds:
        candidates: list[dict] = []
        cooling: list[tuple[dict, float, str]] = []
        for route in routes:
            route_id = str(route.get("id") or "route")
            if retry_ids is not None and route_id not in retry_ids:
                continue
            key = _route_state_key(service_id, route_id)
            remaining = _rate_cooldown_remaining(key)
            if remaining > 0:
                reason = str(_route_runtime_snapshot(key).get("reason") or "cooling")
                cooling.append((route, remaining, reason))
            else:
                candidates.append(route)

        if not candidates:
            if retry_ids is not None and last_error is not None:
                break
            earliest = min((item[1] for item in cooling), default=0.0)
            reasons = tuple(sorted({item[2] for item in cooling if item[2]}))
            _emit_llm_stream(
                base_diagnostics, "routes_unavailable",
                elapsed=round(time.monotonic() - call_started_at, 1),
                error_reason=",".join(reasons) or "cooling",
                wait_seconds=round(earliest, 1),
                message=(f"所有线路都在冷却；最早 {earliest:.1f} 秒后可重试，"
                         "本次不再原地等待"))
            raise LLMRoutesCoolingError(
                "所有 API 线路都在冷却，最早 %.1f 秒后可重试；本次不会原地等待" % earliest,
                retry_after=earliest, reasons=reasons)

        if round_no > 1:
            message = (
                f"第一轮 {fast} 秒快探均未开始生成；进入 {cap} 秒复查轮，"
                f"仅复查 {len(candidates)} 条慢启动线路")
            logger.info("H3-Myang: %s", message)
            _emit_llm_stream(
                base_diagnostics, "route_retry_round",
                elapsed=round(time.monotonic() - call_started_at, 1),
                route_round=round_no, route_rounds=round_total,
                timeout=first_output_timeout, route_count=len(candidates),
                message=message)

        next_retry_ids: set[str] = set()
        for route_index, route in enumerate(candidates, 1):
            route_id = str(route.get("id") or "route")
            route_name = str(route.get("name") or route_id)
            attempt_no += 1
            base_url = route.get("base_url", "")
            headers = {"Content-Type": "application/json"}
            if route.get("api_key"):
                headers["Authorization"] = f"Bearer {route['api_key']}"
            attempt_diagnostics = {
                **base_diagnostics,
                "route_id": route_id,
                "route": route_name,
                "attempt": attempt_no,
                "route_round": round_no,
                "route_rounds": round_total,
                "route_index": route_index,
                "route_count": len(candidates),
                "first_output_timeout": first_output_timeout,
            }
            logger.info(
                "API 线路%s: %s / %s | 第%d/%d轮 | 线路%d/%d | 首内容≤%ss",
                "快探" if round_no == 1 and round_total > 1 else "复查",
                service_label, route_name, round_no, round_total,
                route_index, len(candidates), first_output_timeout)
            if attempt_no > 1:
                _emit_llm_stream(
                    attempt_diagnostics, "route_switch",
                    elapsed=round(time.monotonic() - call_started_at, 1),
                    message=("正在切换探测线路" if round_no == 1
                             else "正在复查下一条慢启动线路"))
            _emit_llm_stream(
                attempt_diagnostics, "route_attempt",
                elapsed=round(time.monotonic() - call_started_at, 1),
                timeout=first_output_timeout,
                content_chars=0, reasoning_chars=0, finish_reason="")
            _mark_route_active(service_id, route)
            attempt_started_at = time.monotonic()
            try:
                request_payload = payload
                try:
                    result = _post_json_with_diagnostics(
                        _build_chat_url(base_url, service_type), headers,
                        request_payload, first_output_timeout, attempt_diagnostics)
                except Exception as stream_error:
                    if not _streaming_unsupported(stream_error):
                        raise
                    elapsed = time.monotonic() - attempt_started_at
                    fallback_remaining = max(1, int(first_output_timeout - elapsed))
                    logger.warning(
                        "H3-Myang: %s / %s 不支持流式返回，同线路改用普通整包；剩余 %ss",
                        service_label, route_name, fallback_remaining)
                    fallback_diagnostics = {**attempt_diagnostics, "streaming": False}
                    _emit_llm_stream(
                        fallback_diagnostics, "stream_fallback",
                        elapsed=round(time.monotonic() - call_started_at, 1),
                        message=("服务端不兼容流式传输，已在本轮剩余时间内切换普通整包；"
                                 "普通整包无法观察思考进度"))
                    request_payload = {**payload, "stream": False}
                    result = _post_json_with_diagnostics(
                        _build_chat_url(base_url, service_type), headers,
                        request_payload, fallback_remaining, fallback_diagnostics)

                text, reasoning, finish, stream_meta = _stream_completion_text(result)
                _mark_route_success(service_id, route)
                logger.info("API 路由命中: %s / %s", service_label, route_name)
                _emit_llm_stream(
                    attempt_diagnostics, "done",
                    elapsed=round(time.monotonic() - call_started_at, 1),
                    content_chars=len(text), reasoning_chars=len(reasoning),
                    chunks=int(stream_meta.get("chunks", 0) or 0),
                    finish_reason=finish,
                    preview=re.sub(r"\s+", " ", text[-220:]).strip())
                if service_type == "ollama" and ollama_auto_unload:
                    _ollama_unload(base_url, target.get("name", ""))
                return text
            except Exception as error:  # noqa: BLE001 - route-local failover
                if (type(error).__name__ == "InterruptProcessingException"
                        or str(error) == "LLM request cancelled"):
                    _mark_route_inactive(service_id, route)
                    _emit_llm_stream(
                        attempt_diagnostics, "cancelled",
                        elapsed=round(time.monotonic() - call_started_at, 1),
                        message="用户已停止当前 LLM 请求")
                    raise
                last_error = error
                reason = error_reason(error) or "request_error"
                quick_timeout = bool(
                    round_no < round_total and reason == "timeout")
                if quick_timeout:
                    _mark_route_slow_probe(service_id, route, error)
                    next_retry_ids.add(route_id)
                else:
                    _mark_route_failure(service_id, route, error)
                _emit_llm_stream(
                    attempt_diagnostics, "route_error",
                    elapsed=round(time.monotonic() - call_started_at, 1),
                    error_reason=reason,
                    message=(f"{fast} 秒内未开始生成，转试下一线路；稍后进入 {cap} 秒复查"
                             if quick_timeout else str(error)[:300]),
                    will_failover=bool(quick_timeout or _route_can_retry(error)),
                    will_retry_round=quick_timeout)
                if not _route_can_retry(error):
                    raise

        if next_retry_ids:
            retry_ids = next_retry_ids
            continue
        break

    if last_error is not None:
        _emit_llm_stream(
            base_diagnostics, "failed",
            elapsed=round(time.monotonic() - call_started_at, 1),
            error_reason=error_reason(last_error) or "request_error",
            message=str(last_error)[:300])
        raise last_error
    raise LLMRequestTimeoutError("所有流式线路均未开始有效生成")


def _call_llm_once(
    service_str: str,
    system_prompt: str,
    user_prompt: str,
    ollama_auto_unload: bool = False,
    seed: int = 0,
    max_tokens: int | None = None,
    wait_for_cooldown: bool = True,
) -> str:
    service_ref, model_name = _parse_service_model(service_str)
    svc = _find_service(service_ref)
    if not svc:
        raise ValueError(f"LLM service not found: {service_ref}")
    service_id = str(svc.get("id") or service_ref)
    service_label = str(svc.get("name") or service_id)

    target = _find_model(svc, model_name, "llm_models")
    if not target:
        raise ValueError(f"No LLM model in service {service_label}")

    service_type = svc.get("type", "openai_compatible")
    use_stream = bool(target.get("stream", True))
    payload = {
        "model": target.get("name", ""),
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "temperature": target.get("temperature", 0.7),
        "top_p": target.get("top_p", 0.9), "stream": use_stream,
    }
    output_limit = int(max_tokens) if max_tokens is not None else int(target.get("max_tokens", 0) or 0)
    if output_limit > 0:
        payload["max_tokens"] = output_limit
    route_timeout, call_budget = _llm_timeout_policy(
        target, len(system_prompt) + len(user_prompt), use_stream=use_stream)
    if not use_stream:
        logger.info(
            "LLM call: %s/%s | %s | max_output=%s | stream=关 | "
            "普通整包超时≤%ss | 路由窗口≤%ss",
            service_label, target.get("name", ""), f"{len(user_prompt)} chars",
            payload.get("max_tokens", "服务端决定"), route_timeout, call_budget)

    text = ""
    call_id = _next_llm_call_id()
    call_started_at = time.monotonic()
    base_diagnostics = {
        "call_id": call_id,
        "service_id": service_id,
        "service": service_label,
        "model": str(target.get("name") or ""),
        "streaming": use_stream,
        **_execution_context_fields(),
    }
    _emit_llm_stream(
        base_diagnostics, "queued", elapsed=0.0,
        content_chars=0, reasoning_chars=0, finish_reason="")
    last_error: BaseException | None = None
    try:
        routes = _ordered_enabled_routes(svc, "llm")
    except Exception as error:
        _emit_llm_stream(
            base_diagnostics, "failed", elapsed=round(time.monotonic() - call_started_at, 1),
            error_reason=error_reason(error) or "no_route", message=str(error)[:300])
        raise
    if use_stream:
        return _call_llm_stream_routes(
            service_id=service_id,
            service_label=service_label,
            service_type=service_type,
            target=target,
            routes=routes,
            payload=payload,
            route_timeout=route_timeout,
            base_diagnostics=base_diagnostics,
            call_started_at=call_started_at,
            ollama_auto_unload=ollama_auto_unload,
        )
    deadline = time.monotonic() + float(call_budget)
    waited_for_cooldown = False
    attempted_route_ids: set[str] = set()

    while True:
        ready: list[dict] = []
        cooling: list[tuple[dict, float, str]] = []
        for route in routes:
            route_key = _route_state_key(service_id, str(route.get("id") or "route"))
            remaining = _rate_cooldown_remaining(route_key)
            if remaining > 0:
                reason = str(_route_runtime_snapshot(route_key).get("reason") or "cooling")
                cooling.append((route, remaining, reason))
            else:
                ready.append(route)

        total_remaining = max(0.0, deadline - time.monotonic())
        if not ready:
            if not cooling:
                break
            earliest = min(item[1] for item in cooling)
            reasons = tuple(sorted({item[2] for item in cooling if item[2]}))
            if (
                wait_for_cooldown
                and
                not waited_for_cooldown
                and reasons
                and all(reason == "rate_limit" for reason in reasons)
                and earliest <= MAX_INLINE_COOLDOWN_WAIT_SECONDS
                and total_remaining >= earliest + MIN_ROUTE_ATTEMPT_SECONDS
            ):
                attempted_route_ids.clear()
                _emit_llm_stream(
                    base_diagnostics, "cooldown_wait",
                    elapsed=round(time.monotonic() - call_started_at, 1),
                    wait_seconds=round(earliest, 1),
                    message=f"所有线路限流，等待 {earliest:.1f} 秒后重试")
                try:
                    _interruptible_cooldown_wait(earliest + 0.05, service_label)
                except BaseException:
                    _emit_llm_stream(
                        base_diagnostics, "cancelled",
                        elapsed=round(time.monotonic() - call_started_at, 1),
                        message="用户在限流等待期间停止了 LLM 请求")
                    raise
                waited_for_cooldown = True
                continue
            _emit_llm_stream(
                base_diagnostics, "routes_unavailable",
                elapsed=round(time.monotonic() - call_started_at, 1),
                error_reason=",".join(reasons) or "cooling",
                wait_seconds=round(earliest, 1),
                message=f"所有线路都不可用；最早 {earliest:.1f} 秒后可重试")
            raise LLMRoutesCoolingError(
                "所有 API 线路都在冷却，最早 %.1f 秒后可重试；不会继续生成兜底分段" % earliest,
                retry_after=earliest,
                reasons=reasons,
            )

        for route in ready:
            route_id = str(route.get("id") or "route")
            # Never re-enter a route already attempted by this call. A slow
            # provider may cool down before the total budget expires; retrying
            # it here creates a long loop with no new information.
            if route_id in attempted_route_ids:
                continue
            attempted_route_ids.add(route_id)
            total_remaining = max(0.0, deadline - time.monotonic())
            if total_remaining < MIN_ROUTE_ATTEMPT_SECONDS:
                last_error = LLMRequestTimeoutError(
                    f"LLM 路由启动窗口已达到 {call_budget} 秒上限")
                break
            attempt_timeout = max(
                MIN_ROUTE_ATTEMPT_SECONDS,
                min(route_timeout, int(total_remaining)),
            )
            base_url = route.get("base_url", "")
            headers = {"Content-Type": "application/json"}
            if route.get("api_key"):
                headers["Authorization"] = f"Bearer {route['api_key']}"
            logger.info(
                "API 线路尝试: %s / %s | 连接/首包最多等待 %ss | "
                "路由启动窗口剩余 %.1fs%s",
                service_label, route.get("name") or route.get("id"),
                attempt_timeout, total_remaining,
                " | 流开始后不再累计超时" if use_stream else "")
            attempt_diagnostics = {
                **base_diagnostics,
                "route_id": route_id,
                "route": str(route.get("name") or route_id),
                "attempt": len(attempted_route_ids),
            }
            if len(attempted_route_ids) > 1:
                _emit_llm_stream(
                    attempt_diagnostics, "route_switch",
                    elapsed=round(time.monotonic() - call_started_at, 1),
                    message="上一条线路失败，正在切换到本线路")
            _emit_llm_stream(
                attempt_diagnostics, "route_attempt",
                elapsed=round(time.monotonic() - call_started_at, 1),
                timeout=attempt_timeout,
                remaining=round(total_remaining, 1),
                content_chars=0, reasoning_chars=0, finish_reason="")
            _mark_route_active(service_id, route)
            try:
                request_payload = payload
                try:
                    result = _post_json_with_diagnostics(
                        _build_chat_url(base_url, service_type), headers,
                        request_payload, attempt_timeout, attempt_diagnostics)
                except Exception as stream_error:
                    if use_stream and _streaming_unsupported(stream_error):
                        logger.warning(
                            "H3-Myang: %s / %s 不支持流式返回，同线路自动改用普通请求",
                            service_label, route.get("name") or route_id)
                        fallback_diagnostics = {
                            **attempt_diagnostics, "streaming": False,
                        }
                        _emit_llm_stream(
                            fallback_diagnostics, "stream_fallback",
                            elapsed=round(time.monotonic() - call_started_at, 1),
                            message="服务端不兼容流式传输，已在同一线路切换为普通请求；普通请求仍受整次超时保护")
                        request_payload = {**payload, "stream": False}
                        fallback_remaining = max(1, int(deadline - time.monotonic()))
                        result = _post_json_with_diagnostics(
                            _build_chat_url(base_url, service_type), headers,
                            request_payload, min(attempt_timeout, fallback_remaining),
                            fallback_diagnostics)
                    else:
                        raise
                choices = result.get("choices", []) if isinstance(result, dict) else []
                reasoning = ""
                finish = ""
                stream_meta = result.get("_myang_stream") or {} if isinstance(result, dict) else {}
                if choices:
                    message = choices[0].get("message", {}) or {}
                    text = _stream_text(message.get("content")).strip()
                    if not text:
                        reasoning = _stream_text(message.get("reasoning_content"))
                        finish = str(choices[0].get("finish_reason") or "")
                        logger.warning(
                            "LLM returned empty content | finish_reason=%s | 思考通道 %d 字 | %s",
                            finish or "?", len(reasoning),
                            json.dumps(result, ensure_ascii=False)[:200])
                if not text:
                    if finish.casefold() == "length" and reasoning:
                        empty_explanation = "输出额度被思考通道耗尽，服务端在正文开始前因 length 停止"
                    elif finish.casefold() == "length":
                        empty_explanation = "服务端因输出长度上限停止，但没有返回正文"
                    elif int(stream_meta.get("chunks", 0) or 0) <= 0:
                        empty_explanation = "服务端完成请求但没有返回正文或可用流数据块"
                    else:
                        empty_explanation = "流式请求结束但助手正文为空"
                    raise LLMEmptyResponseError(
                        empty_explanation +
                        f" | finish_reason={finish or '?'}"
                        f" | reasoning_chars={len(reasoning)}"
                        f" | stream_chunks={int(stream_meta.get('chunks', 0) or 0)}")
                _mark_route_success(service_id, route)
                logger.info("API 路由命中: %s / %s", service_label, route.get("name") or route.get("id"))
                finish = str(choices[0].get("finish_reason") or "") if choices else ""
                reasoning = _stream_text((choices[0].get("message") or {}).get("reasoning_content")) if choices else ""
                _emit_llm_stream(
                    attempt_diagnostics, "done",
                    elapsed=round(time.monotonic() - call_started_at, 1),
                    content_chars=len(text), reasoning_chars=len(reasoning),
                    chunks=int(stream_meta.get("chunks", 0) or 0),
                    finish_reason=finish,
                    preview=re.sub(r"\s+", " ", text[-220:]).strip())
                if service_type == "ollama" and ollama_auto_unload:
                    _ollama_unload(base_url, target.get("name", ""))
                return text
            except Exception as error:  # noqa: BLE001 - route-local failover
                if type(error).__name__ == "InterruptProcessingException" or str(error) == "LLM request cancelled":
                    _mark_route_inactive(service_id, route)
                    _emit_llm_stream(
                        attempt_diagnostics, "cancelled",
                        elapsed=round(time.monotonic() - call_started_at, 1),
                        message="用户已停止当前 LLM 请求")
                    raise
                last_error = error
                _mark_route_failure(service_id, route, error)
                reason = error_reason(error) or "request_error"
                _emit_llm_stream(
                    attempt_diagnostics, "route_error",
                    elapsed=round(time.monotonic() - call_started_at, 1),
                    error_reason=reason,
                    message=str(error)[:300],
                    will_failover=bool(_route_can_retry(error)))
                if not _route_can_retry(error):
                    raise

        if time.monotonic() >= deadline:
            break
        # If every route hit TPM in this call, wait for the earliest one once.
        # This is the useful middle ground between an instant all-fallback run
        # and repeatedly sleeping/retrying for many minutes.
        current_cooling = []
        for route in routes:
            key = _route_state_key(service_id, str(route.get("id") or "route"))
            remaining = _rate_cooldown_remaining(key)
            if remaining > 0:
                current_cooling.append((route, remaining))
        earliest = min((item[1] for item in current_cooling), default=0.0)
        total_remaining = max(0.0, deadline - time.monotonic())
        if (
            wait_for_cooldown
            and
            not waited_for_cooldown
            and error_reason(last_error or RuntimeError()) == "rate_limit"
            and earliest <= MAX_INLINE_COOLDOWN_WAIT_SECONDS
            and total_remaining >= earliest + MIN_ROUTE_ATTEMPT_SECONDS
        ):
            attempted_route_ids.clear()
            _emit_llm_stream(
                base_diagnostics, "cooldown_wait",
                elapsed=round(time.monotonic() - call_started_at, 1),
                wait_seconds=round(earliest, 1),
                message=f"线路触发 TPM 限流，等待 {earliest:.1f} 秒后重试")
            try:
                _interruptible_cooldown_wait(earliest + 0.05, service_label)
            except BaseException:
                _emit_llm_stream(
                    base_diagnostics, "cancelled",
                    elapsed=round(time.monotonic() - call_started_at, 1),
                    message="用户在限流等待期间停止了 LLM 请求")
                raise
            waited_for_cooldown = True
            continue
        break

    if isinstance(last_error, BaseException):
        _emit_llm_stream(
            base_diagnostics, "failed",
            elapsed=round(time.monotonic() - call_started_at, 1),
            error_reason=error_reason(last_error) or "request_error",
            message=str(last_error)[:300])
        raise last_error
    _emit_llm_stream(
        base_diagnostics, "failed",
        elapsed=round(time.monotonic() - call_started_at, 1),
        error_reason="timeout",
        message=f"路由启动窗口达到 {call_budget} 秒上限，未开始有效生成")
    raise LLMRequestTimeoutError(
        f"LLM 路由启动窗口已达到 {call_budget} 秒上限，未开始有效生成")


def call_llm_interactive(
    service_str: str,
    system_prompt: str,
    user_prompt: str,
    ollama_auto_unload: bool = False,
    seed: int = 0,
    max_tokens: int | None = None,
) -> str:
    """Run one user-triggered LLM task without long quota/cooldown retries.

    Route failover and the configured streaming policy remain active, but an
    interactive button should return a useful error instead of sleeping through
    circuit-breaker cooldowns or the workflow-oriented quota confirmation loop.
    """
    _release_confirmed_quota_for_model(service_str)
    return _call_llm_once(
        service_str,
        system_prompt,
        user_prompt,
        ollama_auto_unload,
        seed,
        max_tokens,
        wait_for_cooldown=False,
    )


def _promote_confirmed_quota_cooldown(service_str: str) -> None:
    """Open the long circuit only after the caller confirms repeated quota errors."""
    service_ref, model_name = _parse_service_model(service_str)
    svc = _find_service(service_ref)
    if not svc:
        return
    service_id = str(svc.get("id") or service_ref)
    now = time.monotonic()
    with _route_guard:
        for route in _normalize_routes(svc):
            route_id = str(route.get("id") or "route")
            key = _route_state_key(service_id, route_id)
            _route_cooldowns[key] = now + ROUTE_QUOTA_COOLDOWN_SECONDS
            state = _route_runtime.setdefault(key, {})
            state["reason"] = "quota_exhausted"
            state["quota_confirmed"] = True
            state["quota_model"] = str(model_name or "")
    logger.warning(
        "API 工作区额度连续确认失败 %d 次，线路进入 %.1f 小时冷却 | %s",
        QUOTA_CONFIRMATION_FAILURES, ROUTE_QUOTA_COOLDOWN_SECONDS / 3600.0,
        service_ref)


def _release_confirmed_quota_for_model(service_str: str) -> None:
    """A confirmed model quota must not block a different model in the service."""
    service_ref, model_name = _parse_service_model(service_str)
    selected_model = str(model_name or "").strip()
    if not selected_model:
        return
    svc = _find_service(service_ref)
    if not svc:
        return
    service_id = str(svc.get("id") or service_ref)
    with _route_guard:
        for route in _normalize_routes(svc):
            key = _route_state_key(service_id, str(route.get("id") or "route"))
            state = _route_runtime.get(key) or {}
            if (state.get("reason") == "quota_exhausted"
                    and state.get("quota_confirmed")
                    and str(state.get("quota_model") or "").strip()
                    and str(state.get("quota_model") or "").strip() != selected_model):
                _route_cooldowns.pop(key, None)
                state["reason"] = ""
                state.pop("quota_confirmed", None)
                state.pop("quota_model", None)
    logger.info("切换 LLM 模型，释放旧模型的额度确认锁 | %s", service_str)


def _release_quota_retry_cooldowns(service_str: str) -> None:
    """Let the outer confirmation loop own the ten-second retry delay."""
    service_ref, _model_name = _parse_service_model(service_str)
    svc = _find_service(service_ref)
    if not svc:
        return
    service_id = str(svc.get("id") or service_ref)
    with _route_guard:
        for route in _normalize_routes(svc):
            key = _route_state_key(service_id, str(route.get("id") or "route"))
            state = _route_runtime.get(key) or {}
            if (state.get("reason") == "quota_retry"
                    and not state.get("quota_confirmed")):
                _route_cooldowns.pop(key, None)


def call_llm(
    service_str: str,
    system_prompt: str,
    user_prompt: str,
    ollama_auto_unload: bool = False,
    seed: int = 0,
    max_tokens: int | None = None,
) -> str:
    """Call the provider, confirming quota errors before failing the workflow."""
    _release_confirmed_quota_for_model(service_str)
    quota_failures = 0
    while True:
        try:
            return _call_llm_once(
                service_str, system_prompt, user_prompt, ollama_auto_unload,
                seed, max_tokens)
        except Exception as error:
            if type(error).__name__ == "InterruptProcessingException":
                raise
            reason = error_reason(error)
            if reason == "quota_exhausted" or not is_quota_error(error):
                raise
            quota_failures += 1
            if quota_failures >= QUOTA_CONFIRMATION_FAILURES:
                _promote_confirmed_quota_cooldown(service_str)
                raise LLMQuotaConfirmedError(
                    "LLM 工作区/模型配额连续 %d 次返回 429，确认暂时不可用；"
                    "已停止本次请求。可切换模型或服务后继续使用已保存的 Agent 上下文。"
                    % quota_failures) from error
            logger.warning(
                "LLM 配额响应暂按临时异常处理 | 第 %d/%d 次 | %.1f 秒后重试 | %s",
                quota_failures, QUOTA_CONFIRMATION_FAILURES,
                QUOTA_RETRY_WAIT_SECONDS, service_str)
            _interruptible_cooldown_wait(
                QUOTA_RETRY_WAIT_SECONDS, str(service_str))
            _release_quota_retry_cooldowns(service_str)


def call_vlm(
    service_str: str,
    images_base64: list[str],
    prompt: str,
    ollama_auto_unload: bool = False,
    max_tokens: int = 1024,
) -> str:
    _release_confirmed_quota_for_model(service_str)
    service_ref, model_name = _parse_service_model(service_str)
    svc = _find_service(service_ref)
    if not svc:
        raise ValueError(f"VLM service not found: {service_ref}")
    service_id = str(svc.get("id") or service_ref)
    service_label = str(svc.get("name") or service_id)

    target = _find_model(svc, model_name, "vlm_models")
    if not target:
        raise ValueError(f"No VLM model in service {service_label}")

    service_type = svc.get("type", "openai_compatible")

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img_b64 in images_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    payload = {
        "model": target.get("name", ""),
        "messages": [{"role": "user", "content": content}],
        "temperature": target.get("temperature", 0.7),
        "max_tokens": int(max_tokens),
        "top_p": target.get("top_p", 0.9),
        "stream": False,
    }

    logger.info("VLM call: %s/%s | %d images | %s chars",
                service_label, target.get("name", ""), len(images_base64), len(prompt))

    result = None
    hit_base_url = ""
    last_error = None
    for route in _ordered_enabled_routes(svc, "vlm"):
        key = _route_state_key(service_id, str(route.get("id") or "route"))
        if _rate_cooldown_remaining(key) > 0:
            continue
        base_url = route.get("base_url", "")
        headers = {"Content-Type": "application/json"}
        if route.get("api_key"):
            headers["Authorization"] = f"Bearer {route['api_key']}"
        try:
            try:
                result = _http_post_json(
                    _build_chat_url(base_url, service_type), headers, payload,
                    timeout=int(target.get("timeout", 0) or 0) or 300)
            except RuntimeError as error:
                message = str(error)
                if not ("400" in message and "max_tokens" in message
                        and int(payload["max_tokens"]) > VLM_SAFE_MAX_TOKENS):
                    raise
                payload["max_tokens"] = VLM_SAFE_MAX_TOKENS
                result = _http_post_json(
                    _build_chat_url(base_url, service_type), headers, payload,
                    timeout=int(target.get("timeout", 0) or 0) or 300)
            _mark_route_success(service_id, route)
            hit_base_url = base_url
            logger.info("API 路由命中: %s / %s", service_label, route.get("name") or route.get("id"))
            break
        except Exception as error:  # noqa: BLE001
            last_error = error
            _mark_route_failure(service_id, route, error)
            if not _route_can_retry(error):
                raise
    if result is None:
        raise last_error or RuntimeError("没有可用的 VLM API 线路")

    if service_type == "ollama" and ollama_auto_unload:
        _ollama_unload(hit_base_url, target.get("name", ""))

    text = ""
    choices = result.get("choices", [])
    if choices:
        text = choices[0].get("message", {}).get("content", "")

    return text.strip()


def tensor_to_base64(tensor) -> str:
    """Convert image tensor [B,H,W,C] or [H,W,C] in [0,1] to base64 JPEG."""
    import torch

    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    img_data = tensor[0].clamp(0, 1).cpu()
    img_data = (img_data * 255).byte().numpy()

    from PIL import Image
    img = Image.fromarray(img_data, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
