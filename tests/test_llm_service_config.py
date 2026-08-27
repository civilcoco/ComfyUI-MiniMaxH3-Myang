"""Regression tests for Myang-owned LLM service identity and persistence."""

import importlib
import json
import sys
import tempfile
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


def model(name, default=False):
    return {
        "name": name,
        "is_default": default,
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 0.9,
    }


def service(service_id, name, key="secret", llm_models=None, vlm_models=None):
    return {
        "id": service_id,
        "name": name,
        "type": "openai_compatible",
        "base_url": "https://example.invalid/v1",
        "api_key": key,
        "llm_models": llm_models or [],
        "vlm_models": vlm_models or [],
    }


class ConfigSandbox:
    def __enter__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="myang_llm_config_")
        root = Path(self.temporary.name).resolve()
        self.own = root / "_llm_services_test.json"
        self.legacy = root / "_llm_legacy_test.json"
        self.old_config_path = llm._config_path
        self.old_legacy_path = llm._legacy_config_path
        llm._config_path = lambda: self.own
        llm._legacy_config_path = lambda: self.legacy
        self.invalidate()
        return self

    def invalidate(self):
        llm._config_cache = None
        llm._config_mtime_ns = 0
        llm._config_source = None
        llm._route_counters.clear()

    def write_legacy(self, config):
        self.legacy.write_text(json.dumps(config, ensure_ascii=False), "utf-8")
        self.invalidate()

    def __exit__(self, exc_type, exc, tb):
        llm._config_path = self.old_config_path
        llm._legacy_config_path = self.old_legacy_path
        self.invalidate()
        self.temporary.cleanup()


def test_display_names_and_old_selections_resolve():
    with ConfigSandbox() as box:
        box.write_legacy({"model_services": [
            service("zhipu", "智谱", llm_models=[model("glm", True)]),
        ]})
        check(llm.llm_service_options() == ["智谱 :: glm"],
              "node options did not use the display name")
        check(llm._find_service("zhipu")["name"] == "智谱", "legacy ID stopped resolving")
        check(llm._find_service("智谱")["id"] == "zhipu", "display name did not resolve")
        check(llm._parse_service_model("智谱 :: glm") == ("智谱", "glm"),
              "new option format did not parse")
        check(llm._parse_service_model("zhipu/glm") == ("zhipu", "glm"),
              "old option format did not parse")

        original_post = llm._http_post_json
        captured = {}
        llm._http_post_json = lambda url, headers, payload, timeout=120: (
            captured.update({"url": url, "payload": payload})
            or {"choices": [{"message": {"content": "ok"}}]})
        try:
            check(llm.call_llm("智谱/glm", "system", "user") == "ok",
                  "display-name selection did not execute")
            check(captured["payload"]["model"] == "glm",
                  "display-name selection reached the wrong model")
            check(captured["payload"]["max_tokens"] == 4096,
                  "an explicit positive output limit was not preserved")
        finally:
            llm._http_post_json = original_post


def test_zero_output_limit_is_saved_and_omitted_from_request():
    with ConfigSandbox() as box:
        unrestricted = model("glm", True)
        unrestricted["max_tokens"] = 0
        box.write_legacy({"model_services": [
            service("zhipu", "智谱", llm_models=[unrestricted]),
        ]})
        editable = llm.public_config()["services"]
        llm.save_public_services(editable)
        stored = json.loads(box.own.read_text("utf-8"))
        check(stored["model_services"][0]["llm_models"][0]["max_tokens"] == 0,
              "zero output limit was not accepted by config validation")

        captured = {}
        original_post = llm._http_post_json
        llm._http_post_json = lambda url, headers, payload, timeout=120: (
            captured.update({"payload": payload})
            or {"choices": [{"message": {"content": "ok"}}]})
        try:
            check(llm.call_llm("智谱/glm", "system", "user") == "ok",
                  "zero-limit model did not execute")
            check("max_tokens" not in captured["payload"],
                  "zero output limit was sent to the provider instead of omitted")
        finally:
            llm._http_post_json = original_post


def test_exact_duplicate_is_deduplicated_with_alias():
    with ConfigSandbox() as box:
        generated = service("service_534", "日日新", llm_models=[model("deepseek", True)])
        named = service("sensenova", "日日新", llm_models=[model("deepseek", True)])
        box.write_legacy({"model_services": [generated, named]})
        config = llm._load_config()
        check([item["id"] for item in config["model_services"]] == ["sensenova"],
              "exact duplicate was not collapsed to the readable ID")
        check(config["service_aliases"] == {"service_534": "sensenova"},
              "dropped ID was not retained as a workflow alias")
        check(llm._find_service("service_534")["id"] == "sensenova",
              "old duplicate ID no longer resolves")


def test_public_config_never_exposes_api_key():
    with ConfigSandbox() as box:
        box.write_legacy({"model_services": [service("zhipu", "智谱")]})
        public = llm.public_config()["services"][0]
        check("api_key" not in public, "API key leaked into the settings response")
        check(public["api_key_configured"] is True, "configured-key state was lost")
        check(len(public["routes"]) == 1,
              "legacy root URL/key was not migrated into one editable route")
        check("api_key" not in public["routes"][0],
              "route API key leaked into the settings response")
        check(public["routes"][0]["api_key_configured"] is True,
              "route configured-key state was lost")


def test_replace_really_deletes_and_preserves_keys():
    with ConfigSandbox() as box:
        box.write_legacy({"model_services": [
            service("keep", "保留", key="keep-secret", llm_models=[model("a", True)]),
            service("delete", "删除", key="delete-secret", llm_models=[model("b", True)]),
        ]})
        editable = llm.public_config()["services"][:1]
        saved = llm.save_public_services(editable)
        check(box.own.is_file(), "save did not migrate configuration into Myang_node")
        stored = json.loads(box.own.read_text("utf-8"))
        check([item["id"] for item in stored["model_services"]] == ["keep"],
              "deleted service was merged back into the saved file")
        check(stored["model_services"][0]["api_key"] == "keep-secret",
              "blank API-key editor erased the stored key")
        check(saved["source"] == "myang", "saved configuration still reports legacy ownership")


def test_validation_blocks_identity_and_name_collisions():
    with ConfigSandbox() as box:
        box.write_legacy({"model_services": [
            service("one", "服务一"), service("two", "服务二"),
        ]})
        editable = llm.public_config()["services"]
        editable[0]["id"] = "renamed"
        try:
            llm.save_public_services(editable)
        except ValueError as error:
            check("稳定标识" in str(error), "wrong changed-ID validation error")
        else:
            raise AssertionError("editing a stable service ID was accepted")

        editable = llm.public_config()["services"]
        editable[1]["name"] = editable[0]["name"]
        try:
            llm.save_public_services(editable)
        except ValueError as error:
            check("显示名称重复" in str(error), "wrong duplicate-name validation error")
        else:
            raise AssertionError("duplicate display names were accepted")


def test_route_keys_are_saved_independently_without_secret_echo():
    with ConfigSandbox() as box:
        routed = service("multi", "多线路", key="first-secret",
                         llm_models=[model("glm", True)])
        routed["route_strategy"] = "round_robin"
        routed["routes"] = [
            {"id": "primary", "name": "主线路", "enabled": True,
             "base_url": "https://one.invalid/v1", "api_key": "first-secret"},
            {"id": "backup", "name": "备用线路", "enabled": True,
             "base_url": "https://two.invalid/v1", "api_key": "second-secret"},
        ]
        box.write_legacy({"model_services": [routed]})
        editable = llm.public_config()["services"]
        editable[0]["routes"][0]["name"] = "主线路 A"
        editable[0]["routes"][1]["api_key_action"] = "set"
        editable[0]["routes"][1]["api_key"] = "replacement-secret"
        saved = llm.save_public_services(editable)

        stored = json.loads(box.own.read_text("utf-8"))["model_services"][0]
        check(stored["routes"][0]["api_key"] == "first-secret",
              "keeping one route erased its stored key")
        check(stored["routes"][1]["api_key"] == "replacement-secret",
              "updating one route did not save its new key")
        check(stored["api_key"] == "first-secret",
              "legacy root key no longer mirrors the primary route")
        check(all("api_key" not in route for route in saved["services"][0]["routes"]),
              "saved response echoed a route secret")


def test_round_robin_uses_each_enabled_route():
    with ConfigSandbox() as box:
        routed = service("multi", "多线路", llm_models=[model("glm", True)])
        routed["route_strategy"] = "round_robin"
        routed["routes"] = [
            {"id": "one", "name": "线路一", "enabled": True,
             "base_url": "https://one.invalid/v1", "api_key": "key-one"},
            {"id": "two", "name": "线路二", "enabled": True,
             "base_url": "https://two.invalid/v1", "api_key": "key-two"},
        ]
        box.write_legacy({"model_services": [routed]})
        calls = []
        original_post = llm._http_post_json
        llm._http_post_json = lambda url, headers, payload, timeout=120: (
            calls.append((url, headers.get("Authorization")))
            or {"choices": [{"message": {"content": "ok"}}]})
        try:
            llm.call_llm("多线路/glm", "system", "first")
            llm.call_llm("多线路/glm", "system", "second")
        finally:
            llm._http_post_json = original_post

        check(calls[0][0].startswith("https://one.invalid/")
              and calls[1][0].startswith("https://two.invalid/"),
              "round-robin did not rotate the starting route: %s" % (calls,))
        check(calls[0][1] == "Bearer key-one" and calls[1][1] == "Bearer key-two",
              "each route did not use its own API key")


def test_tpm_on_primary_fails_over_without_waiting():
    with ConfigSandbox() as box:
        routed = service("multi", "多线路", llm_models=[model("glm", True)])
        routed["route_strategy"] = "failover"
        routed["routes"] = [
            {"id": "primary", "name": "主线路", "enabled": True,
             "base_url": "https://one.invalid/v1", "api_key": "key-one"},
            {"id": "backup", "name": "备用线路", "enabled": True,
             "base_url": "https://two.invalid/v1", "api_key": "key-two"},
        ]
        box.write_legacy({"model_services": [routed]})
        calls = []
        sleeps = []
        original_post = llm._http_post_json
        original_sleep = llm.time.sleep

        def primary_limited(url, headers, payload, timeout=120):
            calls.append(url)
            if url.startswith("https://one.invalid/"):
                raise llm.LLMRateLimitError("API error 429: TPM exhausted", retry_after=65)
            return {"choices": [{"message": {"content": "backup-ok"}}]}

        llm._http_post_json = primary_limited
        llm.time.sleep = lambda seconds: sleeps.append(seconds)
        try:
            result = llm.call_llm("多线路/glm", "system", "user")
        finally:
            llm._http_post_json = original_post
            llm.time.sleep = original_sleep
            for key in list(llm._rate_cooldowns):
                if key.startswith("multi/glm/"):
                    llm._rate_cooldowns.pop(key, None)
                    llm._rate_locks.pop(key, None)

        check(result == "backup-ok", "backup route did not recover the request")
        check(len(calls) == 2 and calls[1].startswith("https://two.invalid/"),
              "429 did not switch directly to the backup route")
        check(not sleeps, "client waited for primary cooldown while backup was available")


if __name__ == "__main__":
    for test in (
        test_display_names_and_old_selections_resolve,
        test_zero_output_limit_is_saved_and_omitted_from_request,
        test_exact_duplicate_is_deduplicated_with_alias,
        test_public_config_never_exposes_api_key,
        test_replace_really_deletes_and_preserves_keys,
        test_validation_blocks_identity_and_name_collisions,
        test_route_keys_are_saved_independently_without_secret_echo,
        test_round_robin_uses_each_enabled_route,
        test_tpm_on_primary_fails_over_without_waiting,
    ):
        test()
        print("PASS", test.__name__)
