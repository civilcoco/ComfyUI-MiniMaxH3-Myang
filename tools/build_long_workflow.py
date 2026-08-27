"""Retired explicit-slot workflow generator.

The old implementation cloned third-party Motion-Context nodes into every
segment.  Myang now uses one H3LongVideo node and expands native anchors at
runtime, so regenerating the explicit graph would reintroduce the dependency
and could discard user-added nodes.
"""


def build(*_args, **_kwargs):
    raise RuntimeError(
        "build_long_workflow.py 已停用：它会重新生成旧 Motion-Context 图。"
        "请使用 build_loop_workflow.py 的默认就地更新；不要加 --rebuild，"
        "否则会删除你手动添加的节点。")


def main():
    build()


if __name__ == "__main__":
    main()
