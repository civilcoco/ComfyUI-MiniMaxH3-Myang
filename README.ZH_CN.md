<div align="center">

# ComfyUI-MiniMaxH3-Myang

MiniMax H3 长视频导演、分段生成、段间多关键帧锚定与音画接缝。

<a href="./README.md"><img src="https://img.shields.io/badge/🇬🇧_English-e9e9e9" alt="English"></a>
<a href="./README.ZH_CN.md"><img src="https://img.shields.io/badge/🇨🇳_中文简体-0b8cf5" alt="中文简体"></a>

作者与维护者：**沐阳Myang**<br>
Bilibili：[**沐阳Myang**](https://space.bilibili.com/506587111) · GitHub：[@civilcoco](https://github.com/civilcoco)

</div>

这个节点包把 MiniMax H3 的短片生成组织成长视频流程：按 H3 时间网格规划镜头，逐段
生成画面与声音，把上一段尾部作为下一段的时序上下文，裁掉重叠区后再完成音画拼接。

本包直接使用 ComfyUI 官方 MiniMax H3 接口。核心功能不依赖其他第三方自定义节点；
模型权重、LoRA 和演示素材不随仓库分发。

> [!IMPORTANT]
> 节点代码采用 **GPL-3.0-only**。段间锚点实现包含从
> [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
> 改编并继续开发的 GPL-3.0 代码。来源版本、修改范围和其他第三方声明见
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

> [!CAUTION]
> MiniMax H3 模型及其输出受单独的社区许可证约束，并可能包含地域限制。下载模型或公开
> 发布生成内容前，请阅读 [LEGAL.md](LEGAL.md) 和模型发布方的最新许可条款。

## 功能

- **导演台**：用一个节点组织纯生成、视频续写和动作迁移。
- **分镜与长视频**：手动编排镜头，或让 LLM 按总时长和单段时长拆分剧本。
- **段间音画续接**：从上一段的 temporal latent 提取上下文，在下一段开头建立多关键帧锚点。
- **素材管理**：支持公共素材、镜头专属素材、预览、编号和引用校验。
- **Media Agent**：可选使用 LLM/VLM 整理素材、理解画面并生成 H3 引用标签。
- **Turbo 调度**：加载 LightX2V Turbo LoRA，并校验任务类型、步数、scheduler 和 AV shift。
- **二采放大**：支持像素、Latent、神经 3D 和 NVIDIA RTX VSR 路径。
- **断点续跑**：动作迁移任务可以从指定分段继续，并使用前段音画建立接缝。

## 安装

需要一个包含官方 MiniMax H3 节点的较新版本 ComfyUI。将仓库克隆到
`ComfyUI/custom_nodes`：

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/civilcoco/ComfyUI-MiniMaxH3-Myang.git
```

重启 ComfyUI 后，在节点菜单的 `沐阳 H3` 分类中即可找到本包节点。如果浏览器中仍显示
安装前的界面，请强制刷新页面。

运行 H3 需要用户自行准备：

- MiniMax H3 diffusion model；
- Qwen text encoder；
- video VAE；
- audio VAE；
- 对应任务所需的可选 LoRA 或放大模型。

本包没有强制安装的 Python 第三方依赖。只有在 Media Agent 中选择本地 Whisper 转写时，
才需要单独安装 `openai-whisper`。

## 快速开始

推荐先打开：

[`example_workflows/Minimax_H3_Myang_Director_CN.json`](example_workflows/Minimax_H3_Myang_Director_CN.json)

1. 在“沐阳 H3 加载器”中选择 diffusion model、CLIP、video VAE 和 audio VAE。
2. 在导演台选择纯生成、视频续写或动作迁移。
3. 填写剧本，或者在手动分镜卡中逐镜头填写标题、提示词和时长。
4. 上传公共素材或镜头专属素材；需要指定素材时，在提示词中使用
   `@图片N`、`@视频N`、`@音频N`。
5. 第一次运行建议只生成两段，并使用 22 帧上下文检查画面、动作、口型和声音接缝。
6. 接缝稳定后，再增加分段或启用 Turbo、低 Sigma 精修和二采放大。

可拆线的完整示例：

[`example_workflows/Minimax_H3_Myang_LongVideo_CN.json`](example_workflows/Minimax_H3_Myang_LongVideo_CN.json)

这个示例展示了外围优化节点。VideoHelperSuite、SolAttn、Spectrum、EasyCache、
SageAttention、KJNodes/PreviewAny 等属于可选集成，不是本包的 Python 导入依赖。
缺少这些节点时，可以安装对应节点包，也可以删除或旁路相关优化链路。

两个示例都已清除私人素材路径、提示词、输出名和种子。打开后请重新选择自己的文件与模型。

## 导演台

“沐阳 H3 · 导演台（全功能）”提供两种镜头组织方式：

- **手动分镜**：每张镜头卡可以单独设置标题、提示词、时长和素材。时长自动吸附到 H3
  支持的 `17k+5` 帧网格；每个镜头最多可上传 9 张图片、3 个视频和 3 个音频。
- **Agent / 长剧本智能切分**：LLM 根据总时长、单段时长、素材清单和写作规则生成整条
  时间线。关闭 LLM 时，输入提示词进入本地分段流程，不消耗 Token。

导演台顶部的“公共素材”适合全片共享的角色图、场景图、参考视频和配乐；镜头卡中的素材
只属于对应镜头。镜头可以选择“仅本镜头”或“叠加全局素材”。工作流中保存的是
`ComfyUI/input` 文件引用，媒体内容不会写进工作流 JSON。

动作迁移任务可以在镜头卡中指定动作源。没有为镜头指定动作源时，使用导演台左侧的全局
`ref_video`。动作迁移和视频续写模式只接受一条直接参考视频；图片和音频素材可以照常引用。

### 素材引用

三类素材分别编号：

```text
@图片1  @图片2
@视频1  @视频2
@音频1  @音频2
```

Media Agent 会把这些标签转换为官方 H3 使用的 `<Picture N>`、`<Video N>` 和
`<Audio N>`，并检查标签是否对应已连接的素材。VLM 可以为图片和视频补充内容描述；
Whisper 可以为音频补充转写结果。

### LLM 服务

在 ComfyUI 设置页左侧进入 `Myang_node`，打开“LLM 服务设置”。面板支持：

- OpenAI-compatible API；
- Ollama；
- 同一服务配置多组 URL/API Key；
- 轮询或主线路优先；
- 超时、限流和服务错误后的线路冷却与切换。

API Key 不会通过配置读取接口返回到浏览器。编辑服务时将 Key 留空即可保留当前值。
配置保存在 ComfyUI 用户目录下的
`user/default/Myang_node/config/llm_services.json`。Media Agent 的 LLM/VLM 调用由本包
直接实现，不要求安装 Prompt Assistant。

## 段间锚点与接缝

```text
第 N 段 temporal latent
        ↓
提取尾部视频与音频上下文
        ↓
在第 N+1 段开头建立多关键帧锚点
        ↓
采样 → 同步裁剪重叠区 → 接缝淡化 → 合并
```

视频使用 24fps 时间线，音频按 H3 的 40Hz 时间网格定位。接缝节点在真实切点执行短波形
淡化，并把每段音频裁到对应画面时长。

使用可拆线节点时，`H3ScriptSplitter.overlap_frames` 必须和
`H3LongVideo.context_length` 保持一致。

| 上下文帧数 | temporal blocks | 建议用途 |
|---:|---:|---|
| 5 | 2 | 实验性短上下文，速度更快、约束更弱 |
| 22 | 7 | 推荐起点，约 0.92 秒上下文 |
| 39 | 12 | 更强的动作与构图连续性 |
| 56 | 17 | 最长上下文，代价和裁剪量最高 |

上下文会占用条件 token，也会从每个后续分段的交付画面中裁掉。请先用 22 帧完成两段
A/B，再根据素材表现测试其他长度。

智能切分会为分界点标记“承接”或“切镜”。两种模式都会使用段间锚点；“切镜”由新分段的
镜头描述完成，不等同于黑场、闪白或后期硬切。

## Turbo 与低 Sigma 精修

“沐阳 H3 · Turbo LoRA 联合音画加载调度”调用 ComfyUI 的 LoRA 加载器，并设置 H3
视频与音频的配套 shift。

| LightX2V 档位 | video/audio shift | 推理 NFE | 训练分辨率 |
|---|---:|---:|---|
| v1.0 8-step | 12 / 3 | 8 或 4，建议先用 8 | 544p 混合比例 |
| v1.0 4-step 768P | 6 / 3 | 4 | 1344×768 |
| v0.1 4-step | 12 / 3 | 4 | 544p 混合比例 |
| Ref2VA v0.1 4-step | 12 / 3 | 4 | 544p 混合比例 |

Turbo 使用固定 NFE 轨迹，需配合 `simple` scheduler、`denoise=1.0` 和 Euler 采样器。
低 Sigma 插点会改变该轨迹，因此两者不能同时启用。纯生成优先使用 FL2VA/T2VA 档位，
动作迁移优先使用 Ref2VA 档位。

参数依据：
[LightX2V 模型页](https://huggingface.co/lightx2v/Minimax-h3-Turbo) 和
[发布方 ComfyUI 工作流](https://github.com/ModelTC/Minimax-H3-Turbo/tree/main/example_workflows)。

不使用 Turbo 时，可以启用“低 Sigma 精修”，在采样轨迹的低噪声区间增加积分点。建议用
相同素材、seed 和参数比较“关闭”与“均衡”，再决定是否用于长任务。

## 二采放大

导演台和“沐阳 H3 · 二采放大设置”支持三种模式：

- 放大后二采；
- 同分辨率二采；
- 仅放大，不二采。

放大方式包括像素/VAE、bislerp Latent、神经 3D Latent 和 NVIDIA RTX VSR。长视频按段
处理，最终音频直接使用一采结果，只执行接缝和时长裁剪。二采模型应使用未挂 Turbo LoRA
的 Ref2VA 基模。

神经 3D 模式兼容 LBH-123-AI 发布的 24 通道 H3 Latent Upscaler 权重。权重需要由用户
下载并放入：

```text
ComfyUI/models/latent_upscale_models/
```

权重与说明：
[LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)

建议从 `fp16`、时间分块 `16` 开始；显存不足时改为 `8`，出现精度或色块问题时尝试
`fp32`。

## 主要节点

| 节点 | 用途 |
|---|---|
| 沐阳 H3 · 导演台（全功能） | 组织分镜、素材、生成、接缝和可选二采 |
| 沐阳 H3 加载器 | 配置 Ref2VA/FL2VA 模型、CLIP、video VAE 和 audio VAE |
| 沐阳 H3 条件（提示词 + 素材） | 生成官方 H3 条件并组织引用素材 |
| 沐阳 H3 · Media Agent | 素材预览、编号、LLM/VLM 整理和引用校验 |
| 沐阳 H3 · 分段计划 | 计算分段数量、帧数和重叠窗口 |
| 沐阳 H3 · 长视频（原生多关键帧） | 展开并执行多段采样链 |
| 沐阳 H3 · 任意位置关键帧 | 在指定时间位置添加关键帧，可串联使用 |
| 沐阳 H3 · 段间多关键帧 | 建立前后分段的 temporal latent 上下文 |
| 沐阳 H3 · 锚点同步裁剪 | 同步裁剪视频与音频锚点区间 |
| H3 接缝淡化 | 处理画面和波形切点，并对齐交付时长 |
| 沐阳 H3 · Turbo LoRA 联合音画加载调度 | 加载 LoRA 并校验 Turbo 参数组合 |
| 沐阳 H3 · 二采放大设置 | 配置长视频二采模式和参数 |
| 沐阳 H3 · 二采放大精修（像素路径） | 执行像素放大和低降噪重绘 |
| 沐阳 H3 · Latent 直接放大（极速双采） | 执行 Latent 路径放大 |
| 沐阳 H3 · 段间漂移校正 | 可选校正多段累积的亮度与色彩漂移 |

名称中带“内部”的节点由导演台或长视频节点自动使用，通常不需要手动连接。

## 兼容性与限制

- 已对 ComfyUI `v0.33.2` 和 `v0.34.0` 的 H3 layout 完成 CPU 结构回归；更新 ComfyUI
  后建议先跑两段短片，再开始长任务。
- 使用 temporal latent 续接时，各段分辨率必须一致。
- 越长的生成链越容易积累画质、音色、亮度和饱和度漂移。接缝连续不代表内容不会逐段退化。
- Turbo、缓存、注意力补丁和二采都会改变速度、显存或画质。排查问题时，先使用无外围优化
  的两段基线。
- 本包不要求安装 ComfyUI-H3-Motion-Context。若它也存在于 `custom_nodes`，建议同一次任务
  只运行一种 H3 段间锚点实现，并在切换工作流前重启 ComfyUI。
- CPU 测试可以验证图结构、时间坐标、引用顺序和裁剪长度，不能替代真实模型的画质与声音验收。

## 测试

在 PowerShell 中运行：

```powershell
pwsh tools\run_tests.ps1 -ComfyRoot D:\path\to\ComfyUI
```

测试覆盖官方首尾锚点等价、多关键帧、image/video/audio 引用顺序、5/22/39/56 帧时间块、
40Hz 音频网格、接缝裁剪、Media Agent、LLM 配置、Turbo 参数契约、导演台和二采链路。
如果系统安装了 Node.js，脚本还会运行前端结构测试和 LLM 服务面板测试。

## 来源与许可

- 本仓库代码许可证：[GPL-3.0-only](LICENSE)。
- 主要改编来源：
  [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)，
  GPL-3.0。
- 神经 Latent 放大运行时参考：
  [ComfyUI MiniMax H3 Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)，
  Apache-2.0。
- 完整版权、审计 revision、修改边界和可选集成说明：
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- 版本变化：[CHANGELOG.md](CHANGELOG.md)。

问题和可复现样例请提交到
[GitHub Issues](https://github.com/civilcoco/ComfyUI-MiniMaxH3-Myang/issues)。
