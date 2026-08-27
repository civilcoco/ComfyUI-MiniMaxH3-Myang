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
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

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
ROUTE_QUOTA_COOLDOWN_SECONDS = 21600.0


class LLMRateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMQuotaError(RuntimeError):
    """The key/workspace has no remaining quota; fail over immediately."""


_route_guard = threading.RLock()
_route_counters: dict[str, int] = {}
_route_cooldowns: dict[str, float] = {}
_route_runtime: dict[str, dict] = {}
# Backward-compatible names used by the local regression fixtures and older
# callers; all point at the same circuit-breaker state.
_rate_state_guard = _route_guard
_rate_locks: dict[str, threading.Lock] = {}
_rate_cooldowns = _route_cooldowns


def _route_state_key(service_id: str, route_id: str) -> str:
    return f"{service_id}/{route_id}"


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


def _clean_models(raw_models: Any, label: str) -> list[dict]:
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
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} / {name} 的采样参数不是有效数字") from exc
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"{label} / {name} 的 temperature 必须在 0～2")
        if not 0 <= max_tokens <= 262144:
            raise ValueError(f"{label} / {name} 的 max_tokens 必须在 0～262144（0 表示不发送限制）")
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"{label} / {name} 的 top_p 必须在 0～1")
        model = copy.deepcopy(raw)
        model.update({
            "name": name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
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
            "llm_models": _clean_models(raw.get("llm_models"), f"{display_name} 的 LLM 模型"),
            "vlm_models": _clean_models(raw.get("vlm_models"), f"{display_name} 的 VLM 模型"),
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
    if isinstance(error, LLMQuotaError):
        return "quota_exhausted", ROUTE_QUOTA_COOLDOWN_SECONDS
    if isinstance(error, LLMRateLimitError):
        return "rate_limit", RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS
    message = str(error or "").casefold()
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


def _mark_route_failure(service_id: str, route: dict, error: BaseException) -> None:
    key = _route_state_key(service_id, str(route.get("id") or "route"))
    reason, seconds = _failure_policy(error)
    with _route_guard:
        state = _route_runtime.setdefault(key, {})
        state["failures"] = int(state.get("failures", 0) or 0) + 1
        state["reason"] = reason
        state["last_error"] = str(error)[:300]
        state["last_used_at"] = time.time()
        if seconds:
            _route_cooldowns[key] = time.monotonic() + seconds
    logger.warning("API 线路暂时跳过 | %s / %s | %s %.1fs",
                   service_id, route.get("name") or route.get("id"), reason, seconds)


def _mark_route_success(service_id: str, route: dict) -> None:
    key = _route_state_key(service_id, str(route.get("id") or "route"))
    with _route_guard:
        state = _route_runtime.setdefault(key, {})
        state["successes"] = int(state.get("successes", 0) or 0) + 1
        state["reason"] = ""
        state["last_used_at"] = time.time()
        _route_cooldowns.pop(key, None)


def _route_can_retry(error: BaseException) -> bool:
    return bool(_failure_policy(error)[0])


def _http_post_json(url: str, headers: dict, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
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
    except TimeoutError as exc:
        raise RuntimeError(f"Request timed out after {int(timeout)}s") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection failed: {exc.reason}") from exc


def call_llm(
    service_str: str,
    system_prompt: str,
    user_prompt: str,
    ollama_auto_unload: bool = False,
    seed: int = 0,
    max_tokens: int | None = None,
) -> str:
    service_id, model_name = _parse_service_model(service_str)
    svc = _find_service(service_id)
    if not svc:
        raise ValueError(f"LLM service not found: {service_id}")

    target = _find_model(svc, model_name, "llm_models")
    if not target:
        raise ValueError(f"No LLM model in service {service_id}")

    service_type = svc.get("type", "openai_compatible")
    payload = {
        "model": target.get("name", ""),
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "temperature": target.get("temperature", 0.7),
        "top_p": target.get("top_p", 0.9), "stream": False,
    }
    output_limit = int(max_tokens) if max_tokens is not None else int(target.get("max_tokens", 0) or 0)
    if output_limit > 0:
        payload["max_tokens"] = output_limit
    logger.info("LLM call: %s/%s | %s | max_output=%s",
                service_id, target.get("name", ""), f"{len(user_prompt)} chars",
                payload.get("max_tokens", "服务端决定"))

    result = None
    last_error = None
    routes = _ordered_enabled_routes(svc, "llm")
    for route in routes:
        route_key = _route_state_key(service_id, str(route.get("id") or "route"))
        if _rate_cooldown_remaining(route_key) > 0:
            continue
        base_url = route.get("base_url", "")
        headers = {"Content-Type": "application/json"}
        if route.get("api_key"):
            headers["Authorization"] = f"Bearer {route['api_key']}"
        try:
            result = _http_post_json(
                _build_chat_url(base_url, service_type), headers, payload,
                timeout=int(target.get("timeout", 0) or 0) or 600)
            _mark_route_success(service_id, route)
            logger.info("API 路由命中: %s / %s", service_id, route.get("name") or route.get("id"))
            if service_type == "ollama" and ollama_auto_unload:
                _ollama_unload(base_url, target.get("name", ""))
            break
        except Exception as error:  # noqa: BLE001 - route-local failover
            last_error = error
            _mark_route_failure(service_id, route, error)
            if not _route_can_retry(error):
                raise
    if result is None:
        raise last_error or RuntimeError("没有可用的 API 线路；请检查线路状态或等待配额恢复")

    text = ""
    choices = result.get("choices", [])
    if choices:
        message = choices[0].get("message", {}) or {}
        text = message.get("content", "") or ""
        if not text:
            # A reasoning model can burn its whole budget in the thinking
            # channel and return success with no answer. Naming that here is
            # what lets the caller retry usefully instead of guessing.
            reasoning = str(message.get("reasoning_content") or "")
            finish = str(choices[0].get("finish_reason") or "")
            logger.warning(
                "LLM returned empty content | finish_reason=%s | 思考通道 %d 字 | %s",
                finish or "?", len(reasoning),
                json.dumps(result, ensure_ascii=False)[:200])
    elif not text:
        logger.warning("LLM returned empty: %s",
                       json.dumps(result, ensure_ascii=False)[:300])

    return text.strip()


def call_vlm(
    service_str: str,
    images_base64: list[str],
    prompt: str,
    ollama_auto_unload: bool = False,
    max_tokens: int = 1024,
) -> str:
    service_id, model_name = _parse_service_model(service_str)
    svc = _find_service(service_id)
    if not svc:
        raise ValueError(f"VLM service not found: {service_id}")

    target = _find_model(svc, model_name, "vlm_models")
    if not target:
        raise ValueError(f"No VLM model in service {service_id}")

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
                service_id, target.get("name", ""), len(images_base64), len(prompt))

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
            logger.info("API 路由命中: %s / %s", service_id, route.get("name") or route.get("id"))
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
