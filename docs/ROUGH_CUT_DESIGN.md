# 粗剪时间轴与智能创作助手设计

## 结论

方案可行，但应拆成两个产品边界：

1. 导演台负责生成和“把结果提交到所选区间”。
2. 粗剪工作区负责素材、轨道、I/O、预览和后续导出。

时间轴采用自有的轻量帧级 JSON，未来增加 OTIO 导入导出适配器。首版不嵌入完整 MLT、Shotcut 或 Remotion：它们要么是桌面 NLE 引擎/应用，要么会引入与当前 ComfyUI 插件不相称的依赖和许可边界。

## 分层

```text
导演台卡片
  └─ 打开全屏粗剪工作区
       ├─ 素材库 API（已登记根目录 + 相对路径）
       ├─ 帧级工程 JSON（tracks / clips / selection）
       └─ I/O 与边界约束
              ↓ queue
H3Director 动态子图
  ├─ 原有 Agent / 手动分镜 / 动作迁移 / 续写
  ├─ 单帧关键图或前后 MotionContext
  └─ 精确裁时 + 保存 AV + overwrite_selection
              ↓ websocket
粗剪工程写回 + 原生视频播放器
```

职责边界：

- `roughcut.py`：纯工程规范、验证、边界查询和覆盖编辑，不依赖 UI、HTTP 或 Torch。
- `roughcut_library.py`：本地素材库登记、扫描、探测和受控流式读取。
- `roughcut_nodes.py`：边界素材解码、精确裁时、AV 保存和执行完成事件。
- `h3_roughcut_ui.js`：全屏编辑器；节点内只显示开关、I/O 摘要和打开按钮。
- `director.py` / `nodes.py`：只接收窄化后的时长与约束，不拥有素材库或编辑器状态。

## 工程模型

权威时间单位为整数帧：

```json
{
  "format": "myang.roughcut",
  "version": 1,
  "settings": {"fps": 24, "width": 1920, "height": 1080},
  "selection": {
    "in_frame": 240,
    "out_frame": 480,
    "target_track": "video_1",
    "start_mode": "motion",
    "end_mode": "motion"
  },
  "tracks": [{
    "id": "video_1",
    "kind": "video",
    "clips": [{
      "timeline_start": 0,
      "timeline_end": 720,
      "source_in": 0,
      "source_out": 720,
      "source_fps": 24,
      "source": {"type": "library", "library_id": "lib_x", "relative_path": "shot.mp4"}
    }]
  }]
}
```

覆盖编辑不是 ripple edit。与 `[I,O)` 相交的片段最多拆成左、右两段，生成片段严格占据 `[I,O)`。

## 素材库安全

- 添加文件夹是唯一接受绝对路径的操作，并限制为本机请求。
- 工程和后续请求只携带 `library_id + relative_path`。
- 每次读取都重新 `resolve()`，确认目标仍位于登记根目录；目录穿越和指向库外的符号链接被拒绝。
- 输出素材只允许 `ComfyUI/output` 内的 `filename + subfolder`。
- 删除素材库仅删登记记录，不删磁盘文件。

## I/O 生成与精确时长

时间轴区间时长：

```text
duration_seconds = (out_frame - in_frame) / project_fps
```

H3 固定按 24fps 和 `17k+5` 帧网格生成，所以请求长度向上吸附。粗剪保存节点以区间秒数换算目标 24fps 帧数，统一裁画面和声音，再保存成 MP4。写回片段继续保留工程帧率上的 `[I,O)`，素材侧保存真实的 24fps 入出帧。

## 单帧与 MotionContext

单帧关键图使用 Myang 任意位置关键帧，而不是强制整段切到 FL2VA。这样它可以和现有 Ref2VA 图片、视频、音频素材条件并存，也允许“入点单帧 + 出点 MotionContext”等混合配置。

MotionContext 入点：

```text
读取 [I-C, I) → 编码为 C 帧上下文 → 钉在目标开头 → 只给新增区加噪
→ 采样 → 裁掉开头 C 帧
```

MotionContext 出点不能把 `[O,O+C)` 直接复制进可见结果。设计采用隐藏尾窗：

```text
可见目标结束帧 V
分配 H3 合法总帧 T >= V + C
把 [O,O+C) 锚到 V 开始的位置
V 后所有画面仅供模型规划进入后段，最终统一裁掉 [V,T)
```

这样成片最后一帧仍属于新生成区间；后段素材从自己的第 0 帧正常播放，不会重复或倒跳。尾窗始终使用完整的任意位置多帧条件；只有 O 点恰好落在 H3 VAE 的循环相位时，才额外把尾 latent 硬写入并设为零噪声。精确 I/O 的 O 点若不对相位，会自动跳过不安全的 latent 写入而保留多帧条件，避免错帧或执行报错。入点从时间轴起点开始，始终可以使用零噪声 latent 窗。

前后声音按各自在 40Hz 音频时间轴上的坐标定位。V1 视频内嵌音轨是默认来源；若 A1 在同一 I/O 边界覆盖了完整窗口，A1 优先。解码器产生的 codec padding 会按方向裁齐：入点保留最靠近 I 的尾部声音，出点保留最靠近 O 的头部声音。

生成完成事件不再无条件覆盖整份工程。若用户在长时间生成期间继续编辑时间轴，前端只把本次生成片段合并到运行开始时的 `[I,O)`，保留区间外的新编辑。

## 开源项目取舍

- [OpenTimelineIO](https://opentimelineio.readthedocs.io/en/latest/index.html) 是剪辑信息交换格式和 API，能表示 clips、tracks、transitions、markers，但不内嵌媒体也不是渲染器。适合作为第二阶段导入导出，不适合作为当前浏览器时间轴 UI。
- [MLT](https://mltframework.org/) 能管理和渲染多轨音视频工程，且支持序列化、转场和接近 FFmpeg 的格式覆盖。它适合未来专业导出后端，但把完整 MLT 运行时打包进 ComfyUI 插件过重。
- [Shotcut](https://github.com/mltframework/shotcut) 证明 MLT 路线成熟，但它是 Qt/MLT/FFmpeg 桌面应用，不是可直接嵌入节点的前端组件。
- [Olive](https://github.com/olive-editor/olive) 是可参考的 GPL 时间轴交互实现，但官方仍标注 alpha/高度不稳定，不应成为运行时依赖。

因此首版采用原生 DOM 时间轴 + ComfyUI 自带 PyAV/视频类型；第二阶段再评估 OTIO 交换和 MLT/FFmpeg 渲染适配器。

## 图片模型与智能助手

### 模式级导演台状态

导演台的“公共”只在当前生成任务模式内共享。版本 4 的时间线 JSON 使用
`modes[task_mode]` 分桶，每个桶独立保存 `shots`、`global_assets`、
`plan_snapshot` 和 `storyboard_metadata`。动作迁移、纯生成、视频续写切换时，
前端先保存旧桶再加载新桶；后端按 `task_mode` 选择同一个桶。旧版没有 `modes`
的工作流只按首次打开时的模式读取，避免把旧数据复制到所有模式。

能力注册表建议：

```text
prompt.generate
image.generate.general      -> Krea 2 Turbo
image.generate.anime        -> Anima
image.edit                  -> 用户的 MiniMax H3 图片编辑工作流
image.compose.multi_ref     -> OmniGen2（后续候选）
video.generate              -> MiniMax H3
video.bridge                -> MiniMax H3 + timeline contexts
timeline.insert             -> deterministic rough-cut commit
```

[Krea 2 官方仓库](https://github.com/krea-ai/krea-2)明确 RAW 是可微调底座、Turbo 是 8 步文生图模型，适合生成而不是默认编辑器。图片编辑不再路由到 Qwen，而是固定复用用户现有的 MiniMax H3 图片编辑工作流；Anima 与 Krea 2 只作为文生图工作流候选，不能互相串用。

首版智能助手使用普通 Python 状态机与严格 JSON Schema：

```text
用户请求 → 意图计划 → 本机能力/许可/显存校验 → 预置工作流编译
→ ComfyUI /prompt → /ws 进度 → /history 结果 → 时间轴提交
```

ComfyUI 已提供 `/prompt`、`/ws`、`/object_info`、`/models` 和 `/history/{prompt_id}`，无需先加入 MCP 或第三方 Agent 框架。只有在需要持久断点、人审和复杂循环后，才评估 LangGraph 一类状态图。

## 风险

- MotionContext 尾窗是双边约束实验，需用真实运动、对白和声场样例验证；失败时允许退回单帧或关闭出点约束。
- 浏览器媒体元素不是最终多轨合成器；第一阶段的轨道预览不等同最终导出。
- 大素材库的缩略图和波形必须异步缓存，不能在一次目录扫描中解码所有文件。
- 图片/视频模型许可必须成为能力元数据；不能仅凭模型已安装就自动选择。
