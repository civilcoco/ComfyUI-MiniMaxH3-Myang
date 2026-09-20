import importlib.util
import copy
import sys
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TEST_DIR.parent
CUSTOM_NODES_DIR = PACKAGE_DIR.parent
COMFY_DIR = CUSTOM_NODES_DIR.parent
for path in (str(COMFY_DIR), str(CUSTOM_NODES_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

# `director_templates` reads the user directory, and a test must never write into
# the real one. Only that call is redirected: everything else falls through to
# ComfyUI's own module, because this stub stays in `sys.modules` for the rest of
# the suite and a bare SimpleNamespace used to break every later test that wanted
# `get_output_directory` or `get_temp_directory`.
try:
    import folder_paths as _real_folder_paths
except Exception:  # ComfyUI not importable: the fallback simply has nothing.
    _real_folder_paths = None


class _TestFolderPaths:
    __name__ = "folder_paths"

    @staticmethod
    def get_user_directory():
        return str(TEST_DIR)

    def __getattr__(self, name):
        if _real_folder_paths is None:
            raise AttributeError(name)
        return getattr(_real_folder_paths, name)


sys.modules["folder_paths"] = _TestFolderPaths()

spec = importlib.util.spec_from_file_location(
    "myang_director_templates_test", PACKAGE_DIR / "director_templates.py")
templates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(templates)


def test_template_round_trip_preserves_title_as_metadata():
    item = templates._normalise({
        "name": "草莓挑战",
        "task_mode": "纯生成（不用参考视频）",
        "cards": [{
            "title": "发现草莓", "duration_seconds": 5.2,
            "transition": "开场", "prompt_mode": "fixed",
            "prompt": "integrated_multimodal_description: @图片1 看向草莓。",
            "materials": [{
                "slot_id": "hero", "mode": "input", "kind": "image",
                "label": "主角参考", "required": True,
            }],
        }],
        "global_materials": [], "settings_fixed": False,
    })
    assert item["cards"][0]["title"] == "发现草莓"
    assert "发现草莓" not in item["cards"][0]["prompt"]
    assert item["cards"][0]["materials"][0]["mode"] == "input"
    assert item["task_mode"] == "纯生成（不用参考视频）"


def test_action_template_removes_storyboard_title_without_touching_prompt():
    prompt = "=" * 80 + "\n《异世界邻居》日常篇 01\n动作迁移正文"
    item = templates._normalise({
        "name": "动作模板",
        "task_mode": "动作迁移（跟随参考视频）",
        "cards": [{
            "title": "=" * 80 + " 《异世界邻居》日常篇 01",
            "prompt_mode": "fixed", "prompt": prompt,
            "prompt_label": "所需替换的" + "=" * 80 + " 《异世界邻居》日常篇 01提示词",
        }, {
            "title": "不应保留的第二张卡", "prompt_mode": "fixed",
            "prompt": "第二段",
        }],
    })
    assert len(item["cards"]) == 1
    assert item["cards"][0]["title"] == ""
    assert item["cards"][0]["prompt"] == prompt
    assert item["cards"][0]["prompt_label"] == "动作迁移提示词"
    assert item["duration_policy"]["total"] == {"mode": "reference", "value": 0.0}


def test_legacy_template_defaults_to_fresh_generation():
    item = templates._normalise({
        "name": "旧模板",
        "cards": [{"prompt_mode": "fixed", "prompt": "test"}],
    })
    assert item["task_mode"] == "纯生成（不用参考视频）"
    assert item["duration_policy"] == {}


def test_v2_template_preserves_duration_and_interface_contract():
    item = templates._normalise({
        "version": 2,
        "name": "锁定接口",
        "cards": [{"prompt_mode": "fixed", "prompt": "test"}],
        "duration_policy": {
            "total": {"mode": "fixed", "value": 12},
            "segment": {"mode": "input", "value": 6},
        },
        "interface_locked": True,
    })
    assert item["duration_policy"]["total"] == {"mode": "fixed", "value": 12.0}
    assert item["duration_policy"]["segment"] == {"mode": "input", "value": 6.0}
    assert item["interface_locked"] is True


def test_template_preserves_named_inputs_and_per_setting_policy():
    item = templates._normalise({
        "version": 2,
        "name": "角色换背景",
        "cards": [{
            "prompt_mode": "fixed", "prompt": "test",
            "materials": [{
                "slot_id": "hero", "mode": "input", "kind": "image",
                "label": "原角色", "input_label": "所需替换的角色",
            }],
        }],
        "settings": {"resolution": "640P", "aspect_ratio": "9:16"},
        "settings_policy": {"resolution": "fixed", "aspect_ratio": "local"},
    })
    assert item["cards"][0]["materials"][0]["input_label"] == "所需替换的角色"
    assert item["settings_policy"] == {"resolution": "fixed", "aspect_ratio": "local"}
    assert item["settings_fixed"] is True


def test_action_template_preserves_current_reference_weight():
    item = templates._normalise({
        "version": 2,
        "name": "动作权重模板",
        "task_mode": "动作迁移（跟随参考视频）",
        "cards": [{
            "prompt_mode": "fixed", "prompt": "动作迁移",
            "materials": [{
                "slot_id": "motion", "mode": "input", "kind": "video",
                "role": "action", "label": "动作源",
                "reference_weight_mode": "manual", "reference_weight": 1.65,
            }],
        }],
    })
    material = item["cards"][0]["materials"][0]
    assert material["reference_weight_mode"] == "manual"
    assert material["reference_weight"] == 1.65


def test_template_crud_keeps_permanent_records_without_file_io():
    store = []
    original_load = templates.load_templates
    original_save = templates._save
    templates.load_templates = lambda _path=None: copy.deepcopy(store)

    def save(items, _path=None):
        store[:] = copy.deepcopy(items)

    templates._save = save
    try:
        created = templates.add_template({
            "name": "仓库测试", "cards": [{"prompt": "固定提示词"}],
        })
        assert len(store) == 1 and store[0]["id"] == created["id"]
        updated = templates.update_template(created["id"], {"description": "可浏览详情"})
        assert updated["description"] == "可浏览详情" and len(store) == 1
        assert templates.remove_template(created["id"]) is True
        assert store == []
    finally:
        templates.load_templates = original_load
        templates._save = original_save


def test_template_storage_uses_permanent_user_json_and_atomic_replace():
    source = (PACKAGE_DIR / "director_templates.py").read_text("utf-8")
    assert '"director_templates.json"' in source
    assert "folder_paths.get_user_directory()" in source
    assert "temporary.replace(target)" in source
    assert 'routes.get("/minimax-h3-myang/director-templates")' in source
    assert 'routes.post("/minimax-h3-myang/director-templates")' in source


def test_prompt_metadata_title_is_stripped_but_content_fields_remain():
    source = (PACKAGE_DIR / "nodes.py").read_text("utf-8")
    assert "_LEADING_CARD_TITLE" in source
    assert "分镜标题只属于导演台卡片元数据" in source
    assert 'seg["prompt"] = _strip_legacy_character_reference_heading(prompt_val)' in source


def test_template_panel_exposes_media_duration_lock_and_portability_controls():
    source = (PACKAGE_DIR / "web" / "h3_director_ui.js").read_text("utf-8")
    for required in (
            "固定目标总时长", "固定智能分段上限", "锁定为接口模板",
            "由动作参考视频自动解析", "下方可查看和修改",
            "templateMaterialChoice", "输入用途", "固定生成参数",
            "一采分辨率", "二采分辨率", "画面比例", "沿用本地节点设置",
            "导入", "导出", "AES-GCM", "closeOnBackdrop: false",
            "selectAll.indeterminate", "min-height:340px",
            "guardDialogKeys", "isTextEditingTarget", "嵌入固定素材",
            "buildEmbeddedTemplateDocument", "importEmbeddedTemplateAssets",
            "不使用模板", "renderGenerationSettingsPanel",
            "种子与一采步数仍保留在节点外",
            "TEMPLATE_BUNDLE_MAX_FILE_BYTES = 256 * 1024 * 1024",
            "openDirectorTemplateRepository", "本地模板仓库",
            "本地永久模板", "templateRepositoryFixedSettings",
            "__myangDirectorSyncs"):
        assert required in source, "template UI is missing %s" % required
    assert "template_contract" in source and "template_lock" in source


if __name__ == "__main__":
    test_template_round_trip_preserves_title_as_metadata()
    test_action_template_removes_storyboard_title_without_touching_prompt()
    test_legacy_template_defaults_to_fresh_generation()
    test_v2_template_preserves_duration_and_interface_contract()
    test_template_preserves_named_inputs_and_per_setting_policy()
    test_action_template_preserves_current_reference_weight()
    test_template_crud_keeps_permanent_records_without_file_io()
    test_template_storage_uses_permanent_user_json_and_atomic_replace()
    test_prompt_metadata_title_is_stripped_but_content_fields_remain()
    test_template_panel_exposes_media_duration_lock_and_portability_controls()
    print("PASS director templates and prompt title separation")
