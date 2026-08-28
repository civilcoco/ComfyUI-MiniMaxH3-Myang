# 发布指南

当前源码已经整理为发布候选版。GitHub 仓库 Owner 与 Comfy Registry Publisher ID
均为 `civilcoco`，作者及 Publisher 显示名称为“沐阳Myang”。Publisher ID 会成为
Registry 中的永久标识，不要随意更换。

不要把 Registry 发布令牌、模型服务 API Key 或个人素材提交到仓库。Registry API Key
只保存为 GitHub Actions Secret，名称固定为 `REGISTRY_ACCESS_TOKEN`。

## 推荐发布方式：GitHub Actions

本仓库已经提供 `.github/workflows/publish_action.yml`，不要求维护者在本机安装
`comfy-cli`。工作流只有 `workflow_dispatch` 手动入口，不会因为普通 push 自动发布。

第一次发布前，在网页完成以下操作：

1. 登录 [Comfy Registry](https://registry.comfy.org/nodes)，进入 Publisher
   `civilcoco`，创建一个 Registry Publishing API Key。
2. API Key 只会完整显示一次；保存到密码管理器，不要写进任何仓库文件或聊天截图。
3. 打开 GitHub 仓库：`Settings → Secrets and variables → Actions`。
4. 在 `Repository secrets` 下选择 `New repository secret`。
5. Name 填 `REGISTRY_ACCESS_TOKEN`，Secret 填刚创建的 Registry API Key。
6. 推送所有待发布修改并完成 GitHub Release 后，进入 GitHub 的 `Actions` 页面。
7. 选择 `Publish to Comfy Registry`，点击 `Run workflow`，分支选择 `main`。
8. 等待任务成功，再到 Registry 和 ComfyUI-Manager 中确认名称、版本、README 与安装内容。

如果 Action 提示找不到 Secret，先确认 Secret 建在当前仓库的 `Actions` secrets，且名称
完全等于 `REGISTRY_ACCESS_TOKEN`。不要把 Key 改成普通变量。

## 可选方式：本地 comfy-cli

官方 CLI 是单独的 Python 包，不会随 ComfyUI 自动出现。需要时可在 PowerShell 安装：

```powershell
python -m pip install --user comfy-cli
comfy --version
```

如果安装成功后仍提示找不到 `comfy`，关闭并重新打开 PowerShell；也可以直接找到用户级
Scripts 目录中的程序：

```powershell
$ComfyScripts = "$(python -m site --user-base)\Scripts"
& "$ComfyScripts\comfy.exe" --version
```

仓库的 `pyproject.toml` 已经填写完成，因此不要再运行 `comfy node init`。本地手动发布时，
在仓库根目录执行：

```powershell
comfy node publish
```

命令会提示输入 Publisher `civilcoco` 的 API Key。Windows 终端中建议右键粘贴；官方文档
提示 `Ctrl+V` 偶尔会在 Key 末尾附加不可见的 `\x16` 字符。

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
8. 在 GitHub Actions 中手动运行 `Publish to Comfy Registry`；也可以使用
   `comfy node publish` 作为本地备用方式。
9. 确认用户能够通过 Registry 或 ComfyUI-Manager 安装，并检查发布包内容。

`.comfyignore` 会从 Registry 安装包中排除测试、研究记录、开发工具和 CI 配置，
同时保留许可证、第三方声明、运行代码、前端文件和示例工作流。

官方说明：

- [Publishing Nodes](https://docs.comfy.org/registry/publishing)
- [Comfy CLI Getting Started](https://docs.comfy.org/comfy-cli/getting-started)

## 视频发布提醒

录制界面和工作流介绍不等于已经获得 H3 生成内容的全球展示许可。公开视频包含
H3 生成片段前，应重新阅读 `LEGAL.md` 和 MiniMax 最新许可，确认发布地区、素材权利、
人物授权、音乐授权和 AI 生成内容标识要求。
