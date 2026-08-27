# 发布指南

当前源码已经整理为发布候选版。GitHub 仓库 Owner 与 Comfy Registry Publisher ID
均为 `civilcoco`，作者及 Publisher 显示名称为“沐阳Myang”。Publisher ID 会成为
Registry 中的永久标识，不要随意更换。

不要把 Registry 发布令牌、模型服务 API Key 或个人素材提交到仓库。首次手动发布
验证成功后，如需使用 GitHub Actions 自动发布，只把 `REGISTRY_ACCESS_TOKEN`
保存为 GitHub Actions Secret。

## 发布前检查

在节点包目录运行：

```powershell
python tools/release_audit.py --strict-metadata
node --experimental-default-type=module tests/test_storyboard_cards.mjs
node --experimental-default-type=module tests/test_progress_state.mjs
```

运行完整本地回归：

```powershell
pwsh tools/run_tests.ps1 `
  -ComfyRoot "D:\你的路径\ComfyUI" `
  -Python "D:\你的路径\python\python.exe"
```

Python 回归需要导入 ComfyUI 和 PyTorch，因此应使用实际 ComfyUI 安装环境自带
或正在使用的 Python，不能把它们当作完全独立的普通 Python 包测试。

## GitHub 与 Registry 发布顺序

1. 运行严格发布审计，确认不再出现身份占位符警告。
2. 在节点包目录初始化 Git，把默认分支设为 `main`。
3. 检查待提交文件，确认没有 `release-excluded`、模型权重、生成媒体、缓存、
   API Key、个人路径或私人素材。
4. 提交已经清洗的源码，并推送到 `pyproject.toml` 中填写的 GitHub 仓库。
5. 在全新克隆的仓库中重新运行发布审计和测试。
6. 在干净的 ComfyUI 环境中打开两个示例工作流，检查缺失节点提示和基础连线。
7. 创建带说明的 `v0.1.0` 标签，并发布 GitHub Release。
8. 使用 `comfy node publish` 完成第一次手动 Registry 发布。
9. 确认用户能够通过 Registry 或 ComfyUI-Manager 安装后，再考虑配置自动发布。

`.comfyignore` 会从 Registry 安装包中排除测试、研究记录、开发工具和 CI 配置，
同时保留许可证、第三方声明、运行代码、前端文件和示例工作流。

## 视频发布提醒

录制界面和工作流介绍不等于已经获得 H3 生成内容的全球展示许可。公开视频包含
H3 生成片段前，应重新阅读 `LEGAL.md` 和 MiniMax 最新许可，确认发布地区、素材权利、
人物授权、音乐授权和 AI 生成内容标识要求。
