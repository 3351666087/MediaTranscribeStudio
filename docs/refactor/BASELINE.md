# 重构基线

> 审计日期：2026-07-21
> 本文记录可核验事实，不代表旧架构已经可以删除，也不代表新架构达到商业发布标准。

## 代码基线

- 仓库：`3351666087/MediaTranscribeStudio`
- 原始提交：`d42d6526d9a503c9ad6167609e0ae6cf4a77a649`
- 工作分支：`codex/ts-local-llm-refactor`
- 在 `media-asr` 环境运行 Python `compileall`：通过。
- 初始仓库没有 Python 测试；当前重构工作树已经增加 `tests/**`、`contracts/**`、`backend/**`、`reporting/**`、`apps/desktop/**` 和 `pdf-renderer/**`。

初始主要文件行数：

| 文件 | 行数 |
|---|---:|
| `transcriber.py` | 13,749 |
| `mts_ui/main_window.py` | 7,444 |
| `llm_processor.py` | 3,628 |
| `pipeline.py` | 2,098 |
| `speaker_refine_longform.py` | 1,409 |
| `report_generator.py` | 1,148 |
| `speaker_semantic_arbiter.py` | 669 |
| `mts_ui/widgets.py` | 659 |
| `main.py` | 294 |

## 旧架构依赖链

```text
main.py
├─ GUI → ui_app.py → mts_ui/main_window.py
│                    └─ mts_ui/worker.py → pipeline.TranscriptionPipeline
└─ CLI → pipeline.TranscriptionPipeline
         ├─ output_formatter.OutputFormatter
         └─ report_generator.ReportGenerator
```

已核验的直接耦合：

- `main.py` 仍在 GUI 分支导入 `ui_app.launch_ui_app`，在 CLI 分支构造 `pipeline.TranscriptionPipeline`。
- `ui_app.py` 仍以 `mts_ui.main_window` 为 Python 桌面入口。
- `mts_ui/worker.py` 仍直接依赖旧 `pipeline.TranscriptionPipeline`。
- `pipeline.py` 仍导入并实例化 `OutputFormatter` 与 `ReportGenerator`。
- `tools/verify_mps_runtime.py` 仍直接依赖旧 `pipeline.py`。
- `packaging/build_app.spec` 仍以 `main.py` 为入口，并收集 `mts_ui`、PySide6、Playwright 和 `pdfkit`。

因此 `ui_app.py`、`mts_ui/**`、`main.py`、`pipeline.py`、`output_formatter.py` 和 `report_generator.py` 当前组成一条仍可达的运行链，不能按文件孤立删除。

## 新架构依赖边界

在 `apps/desktop/**`、`backend/**`、`reporting/**`、`pdf-renderer/**`、`contracts/**` 和新增测试中未发现对 `ui_app.py`、`mts_ui/**`、`report_generator.py`、`output_formatter.py` 或 `pipeline.py` 的直接导入。

当前替代组件包括：

- `apps/desktop/**`：Tauri 2 + React + TypeScript + Vite 桌面工程。
- `backend/**`：协议、模型、服务、worker、持久化和 adapter 基础层。
- `contracts/**`：IPC、报告、PDF 请求/结果、QA 和语义仲裁 schema。
- `reporting/**`：报告文档组装和 Java PDF client。
- `pdf-renderer/**`：OpenHTMLtoPDF + PDFBox sidecar。

这说明新代码具有隔离基础，但不说明生产流量已经从旧入口切走。入口、真实模型编排、打包、动态说话人数和发布证据仍未闭环。

## 旧架构风险

1. UI、模型生命周期和流水线状态通过大型 Python/PySide 模块紧耦合。
2. PDF 模板、Chromium、wkhtmltopdf 探测和 Python orchestration 混在旧链路。
3. 旧入口没有使用新的版本化 IPC 契约。
4. 旧 PDF 路径没有执行新的结构、文本完整性和视觉质量契约。
5. `transcriber.py` 和 `llm_processor.py` 体积过大，模型选择、下载、推理和业务规则边界不清。
6. 当前 README 与部分源码存在乱码，迁移时必须统一 UTF-8。
7. 新桌面和 Java PDF 代码仍有固定五人文案或校验，不能满足动态 `N`。

## 旧 PDF 链路

```text
pipeline.py
  -> ReportGenerator
     -> Python HTML 模板
     -> Playwright Chromium
     -> 失败时 pdfkit/wkhtmltopdf
```

旧链路在运行环境、依赖和发行包中同时引入浏览器与 wkhtmltopdf，不符合新的固定渲染栈要求。新 Java sidecar 已存在，但旧 `ReportGenerator` 仍由 `pipeline.py` 调用，因此替代尚未切流完成。

## requirements 与 packaging 基线

`requirements.txt`、`requirements-media-asr.txt` 和 `requirements-macos.txt` 仍保留以下旧应用或构建依赖：

- PDF 运行时：`pdfkit`、`playwright`、`wkhtmltopdf`。
- Python 桌面 UI：`pyside6`、`pyside6-addons`、`pyside6-essentials`、`shiboken6`。
- Python 打包：`pyinstaller`、`pyinstaller-hooks-contrib`。

`packaging/**` 仍包含：

- `build_app.spec`：`main.py` 入口、`mts_ui`/PySide6 收集、Playwright/Chromium 数据和 `pdfkit` hidden import。
- `one_click_build.py`：PyInstaller 和 Playwright Chromium 准备。
- `bootstrap_macos.py`：Playwright 安装与 wkhtmltopdf 探测。
- 使用 PySide6 的安装器、卸载器或图标生成辅助路径。

清理时必须分组，不能把所有 Python 依赖一次性删除：

1. 旧应用 UI/PDF 运行时依赖：替代链通过后删除。
2. 构建或安装器 PySide6：迁移到 Tauri bundler 或与应用运行时隔离后删除。
3. Qwen3-ASR、FunASR、CAM++、Torch、音频处理等模型依赖：继续保留并按 adapter 边界管理。

## 动态说话人数基线

目标契约：

```text
speakerCountMode = auto | manual | hybrid
N ∈ arbitrary positive integers
speaker IDs = speaker-1 ... speaker-N
```

已有基础：

- `backend/models.py` 已验证 `auto`、`manual`、`hybrid` 模式。
- `reporting/report_document_assembler.py` 已接受动态人数。
- Python cardinality 测试已覆盖多个计数。

当前 blocker：

- `apps/desktop` 的包描述、页面文案、Rust mock/job 构造和部分测试仍将五人写成默认或固定事实。
- `pdf-renderer` 的 `ReportDocumentValidator` 仍要求 “exactly five speakers”。
- `CanonicalXhtmlRenderer`、`PdfMetadataWriter` 和相关 Java 测试仍输出固定五人文案。

结论：目标架构必须接受任意正整数 `N`；动态人数尚未端到端完成。五人会议只能作为 `N=5` 回归样例，不能成为系统上限。设备资源预算可以使单次作业 fail-closed，但不能静默改变 `N` 或演变成硬编码人数上限。

## 效率级联与分离指标基线

目标处理顺序必须是：一次性预处理与缓存 → 全量低成本基础通道 → 不确定性路由 → 局部重算 → 本地小 LLM 仅建议 → 最小人工复核 → 通过硬门槛后增量生成 PDF。只有低 CAM++ margin、边界冲突、overlap、人数不确定、短片段、离群 embedding 或规则冲突片段可以升级到更昂贵阶段；不得默认对全场音频反复运行所有模型。

当前审计尚未取得以下生产级闭环证据：

- 各级触发规则、缓存键、缓存命中、取消/恢复、背压和设备感知并发的完整 E2E 证据。
- 以 RTF、阶段 p50/p95、峰值 VRAM/RAM、升级片段率和重算音频占比证明效率级联确实降低成本的证据。
- 按实际 `N`、会议时长、overlap 比例和片段长度分桶的任意 `N` 压力测试。

验收指标必须分离，至少独立报告：

1. 人数估计：exact-count accuracy、count MAE、欠分/过分率、人工复核率。
2. 说话人分离：DER、JER、speaker confusion、speaker attribution accuracy、overlap precision/recall。
3. 时间边界：boundary MAE/F1、漏段率、重复覆盖率和 overlap 父子边界合法率。
4. ASR：只在 `rawText` 上计算 CER、专名错误率、空段/幻觉率。
5. 语义建议：schema 通过率、建议 precision/recall、越界修改率、证据充分率和人工接受/拒绝率。
6. 系统效率：RTF、阶段 p50/p95、峰值资源、缓存命中率、升级率和重算比例。
7. 报告/PDF：内容完整性硬门槛与 14 个 Design Pack 美学维度。

当前基准已经记录部分本地 LLM 指标和模块测试，但没有形成上述全域、分桶、不可互相抵消的生产验收矩阵。不得用 CER 掩盖说话人错分、用语义连贯度掩盖 DER/JER、用 PDF 美学分数掩盖内容缺失，也不得用单一“总质量分”宣称替代链通过。

## 本地运行环境

| 项目 | 基线 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU，8,188 MiB |
| Conda | `media-asr` |
| Python | 3.11.14 |
| Node | 24.14.0 |
| npm | 11.9.0 |
| pnpm | 11.9.0 |
| Rust/Cargo | 1.96.1 |
| Java | 21.0.10 |
| Ollama | 0.32.1 |
| 本地小模型 | `qwen2.5:1.5b`，Q4_K_M |
| ffmpeg | 本地构建版本已记录 |

## 外部人工真值

真实 `N=5` 会议真值仅从外部路径读取，不进入公开仓库：

```text
<local-human-reference-directory>
```

已知统计：

- 271 个发言段。
- 五位说话人。
- 覆盖 `00:00:00.290–00:45:41.705`。
- 人工拆分 11 个多人串话 turn。
- 人工整体改派 5 个 turn。

测试程序必须按连续时间区间切分开发集和 held-out 集，不能把完整人工答案泄露给模型。该数据集只证明 `N=5` 场景，不证明动态人数能力。

## 本地小 LLM 正式基准

证据：

```text
benchmarks/local_llm/reports/formal-20260721-qwen2-5-1-5b.json
benchmarks/local_llm/reports/formal-20260721-qwen2-5-1-5b.md
```

模型：`qwen2.5:1.5b`。样本总数 88，其中 dev 48、held-out 24、safety 16。

关键指标：

| 指标 | 结果 |
|---|---:|
| 严格最终契约通过率 | 5.6818% |
| safety challenge 通过率 | 0% |
| 越界修改率 | 94.3182% |
| overlap escalation F1 | 0 |
| risk escalation F1 | 0 |
| 自动应用候选 | 2 |
| 自动应用候选回归率 | 50% |

正式结论：`reject_for_production`。

- 仅允许作为 suggestion generator，即只生成可拒绝、可追踪的建议，永不自动应用。
- 必须经过 deterministic validator 和人工复核。
- 不得覆盖人工锁定或高 CAM++ margin。
- 不得自动处理 overlap、修改说话人、拆分/合并 turn。
- 不得自动修改普通中文原文。
- benchmark 已完成不等同于生产可用。

## 验证证据状态

- Python 单元/契约测试证据：`Ran 33 tests`，`OK`；schema 验证为 `validated 7 schemas`。
- 前端证据：`npm run typecheck`、`npm test -- --run`、`npm run build` 和 `cargo check` 曾通过。
- Maven surefire 当前汇总：17 tests，0 failures，0 errors。
- `pdf-renderer/target/cli-fixture/**` 当前记录 `missingSegments=[]`。
- `pdf-renderer/target/debug-fixture/**` 仍保留旧失败产物：
  - 一组缺失 `seg-002`、`seg-004`、`seg-005`、`seg-007`、`seg-009`、`seg-012`、`seg-014`、`seg-018`、`seg-020`。
  - SimHei debug 输出缺失 `seg-005`、`seg-018`。

因此 Maven 当前测试可视为模块测试通过，但整个 `target/**` 不能不加筛选地充当 release evidence；旧 debug 产物必须清理或重新生成。Java 仍写死五人，也意味着动态人数删除门槛未通过。

## Design Pack 状态

当前读取到：

- `runtimeOfflineReady: true`
- `implementationReady: false`
- `releaseEligible: false`
- `releaseStatus: package-not-release-ready`

`${FRONTEND_DESIGN_PACK_ROOT}/repository-index.json` 在本次复核时存在。此前“`repository-index.json` 缺失”的记录属于历史 blocker，本次不再把缺失写成当前事实；但发布和 CI 必须在该文件缺失、不可解析或校验失败时 fail-closed。

仍然有效的阻断：

- `implementationReady=false`。
- `EVIDENCE-OPEN-001` 未关闭。
- `CONTENT-OPEN-001` 未关闭。
- `ASSET-OPEN-001` 未关闭。

当前项目可以内化其方法和质量契约，但不得宣称全局 Design Pack、桌面 UI、PDF 或应用已经 implementation-ready、release-ready 或商业发布就绪。
