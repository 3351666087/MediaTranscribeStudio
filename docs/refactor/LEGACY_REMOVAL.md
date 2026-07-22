# Legacy Removal 执行审计

> 审计日期：2026-07-21
> 状态：文档审计完成，旧架构尚未达到删除门槛。
> 本次范围：只更新 `docs/refactor/**`；不删除代码、不修改 requirements、不修改 packaging，也不修改根 `README.md`。

## 执行结论

旧架构目前不能直接删除。`main.py`、`ui_app.py`、`mts_ui/**`、`pipeline.py`、`output_formatter.py` 与 `report_generator.py` 仍组成可达运行链；`tools/verify_mps_runtime.py` 和 `packaging/build_app.spec` 也仍依赖旧入口或旧 pipeline。新 `apps/desktop`、`backend`、`reporting`、`contracts` 与 `pdf-renderer` 已建立隔离基础，但尚未完成真实生产切流、动态说话人数闭环、requirements/packaging 迁移和发布门槛。

删除策略必须是“先切流、再验证、最后删除”，不能按文件逐个孤立移除。

## 当前依赖图

```text
main.py
├─ GUI → ui_app.py → mts_ui/main_window.py
│                    └─ mts_ui/worker.py → pipeline.TranscriptionPipeline
└─ CLI → pipeline.TranscriptionPipeline
         ├─ output_formatter.OutputFormatter
         └─ report_generator.ReportGenerator

tools/verify_mps_runtime.py
└─ pipeline.py

packaging/build_app.spec
├─ main.py
├─ mts_ui/**
├─ PySide6
├─ Playwright / Chromium
└─ pdfkit
```

目标替代边界：

```text
apps/desktop (React + TypeScript)
└─ apps/desktop/src-tauri (Tauri 2 / Rust)
   └─ versioned JSONL IPC
      └─ backend/**
         ├─ model/audio adapters
         ├─ Dynamic-N meeting orchestration
         ├─ contracts/**
         └─ reporting/report_document_assembler.py
            └─ reporting/java_pdf_client.py
               └─ pdf-renderer (OpenHTMLtoPDF + PDFBox)
```

本次检查未在 `apps/desktop/**`、`backend/**`、`reporting/**`、`contracts/**`、`pdf-renderer/**` 和新增测试中发现对上述旧模块的直接导入。这只是代码隔离证据，不等于入口、打包或真实作业已经切换。

## 文件级保留与删除决策

| 对象 | 当前保留理由 | 最终删除或缩减理由 | 删除前必须通过的门槛 | 当前决定 |
|---|---|---|---|---|
| `ui_app.py` | `main.py` GUI 分支仍导入 `launch_ui_app`；当前 Python GUI 仍可达 | Tauri 成为唯一桌面入口后，该转发层没有独立职责 | Tauri 启动、IPC、异常恢复、日志、取消和升级回归通过；`rg` 不再发现生产引用 | 保留，切流后删除 |
| `mts_ui/**` | 包含现行 PySide6 主窗口、worker 和 UI 组件；worker 仍构造旧 pipeline | React/TypeScript 已是目标 UI，双 UI 会造成行为、状态和依赖漂移 | 动态 `N` UI、人工纠错、局部试听、PDF QA、可访问性、错误状态和安装包 E2E 通过 | 保留，替代后整目录删除 |
| `main.py` | 仍同时承载旧 CLI 和 GUI 入口 | 目标入口应由 Tauri/Rust 与新 backend CLI 明确分工，避免双栈路由 | Tauri 是唯一桌面入口；`python -m backend` 或等价新 CLI 覆盖无头运行；所有脚本和打包入口切换 | 先缩减为兼容警告入口，稳定窗口后删除 |
| `pipeline.py` | 当前 CLI、PySide worker 和 MPS 验证工具仍直接依赖；仍编排格式化和报告生成 | 新 backend 应以版本化契约、adapter 和可测试服务替代大型旧 pipeline | 真实 Qwen3-ASR、FunASR、CAM++、Dynamic-N 解码、复核、持久化、取消/重试和 sidecar 调用全部走新链 | 保留，生产切流后删除 |
| `output_formatter.py` | 仍输出 TXT、JSON 和批处理摘要，`pipeline.py` 直接实例化 | 输出职责应迁入版本化 transcript/report document 与专用 exporter，避免旧 schema 继续扩散 | 新 exporter 对文件名、编码、字段、排序、原子写入和批量摘要做到等价或有版本化迁移说明 | 保留；职责迁移后删除或拆成独立 exporter |
| `report_generator.py` | 旧 pipeline 仍调用；保留 Python HTML、Playwright 和 pdfkit/wkhtmltopdf 回退 | 与“Java sidecar 是唯一 PDF 渲染器”的目标冲突，且引入双渲染栈和浏览器依赖 | 新 backend 只调用 Java sidecar；动态 `N`、文本完整性、字体、结构、页面 PNG、manifest、QA 与回滚全部通过 | 保留，Java 链路切流后删除 |
| `requirements*.txt` | 同时承载模型、旧 UI/PDF 和构建依赖，贸然删会破坏 ASR/CAM++/FunASR 环境 | 旧 PySide6、Playwright、pdfkit、wkhtmltopdf 与 PyInstaller 依赖应退出应用运行时 | 先按 runtime/build/model 分组；安装器迁移；依赖图与冷启动测试证明无生产引用 | 分组精简，禁止整文件粗暴重写 |
| `packaging/**` | 当前安装器、PyInstaller spec、Chromium 准备和 macOS 路径仍服务旧发行方式 | 目标应由 Tauri bundler 管理桌面发行，Python 只打包 backend/model runtime | Windows 安装、升级、卸载、路径含空格、离线资源、模型目录、Java sidecar 和回滚安装验证通过 | 保留，按文件迁移后清理 |
| `tools/verify_mps_runtime.py` | 仍验证旧 pipeline 的 macOS/MPS 运行链 | 旧 pipeline 删除后会成为悬挂依赖 | 改为调用新 backend adapter/service；保留等价 MPS smoke test | 先迁移，再解除旧依赖 |

`transcriber.py`、`llm_processor.py`、声纹和语义仲裁相关旧模块不在本轮直接删除清单中。它们包含仍可能被新 adapter 复用的模型能力；只有建立清晰 adapter、调用图和真实结果等价证据后，才能另行决定拆分或删除。

## requirements 与 packaging 清理规则

### 必须删除的旧应用依赖

只有在引用归零和替代链验证完成后，才删除：

- `pdfkit`
- `playwright`
- Playwright Chromium 下载、探测和打包逻辑
- `wkhtmltopdf` 安装、探测和回退逻辑
- Python 桌面运行时中的 `pyside6`、`pyside6-addons`、`pyside6-essentials`、`shiboken6`
- 仅服务旧应用入口的 `pyinstaller`、`pyinstaller-hooks-contrib`

### 不能随旧 UI 一起删除的模型依赖

Qwen3-ASR、FunASR、CAM++、Torch、音频解码、VAD、声纹和推理后端依赖属于 Python 模型运行时。它们必须按照实际 adapter 引用和平台矩阵管理，不能因为删除 PySide6 或旧 PDF 就一并移除。

### packaging 删除门槛

- Tauri bundler 能安装桌面壳、新 backend、Java sidecar、字体、schema 和离线资源。
- 首次启动、模型发现、模型缺失提示、无网络运行、升级和卸载均通过。
- 安装路径含中文、空格和非管理员目录时通过。
- 不再打包 `mts_ui`、Playwright、Chromium、pdfkit 或 wkhtmltopdf。
- 不再以旧 `main.py` 作为 GUI executable 入口。
- macOS/MPS 若仍受支持，必须保留迁移后的 smoke test，而不是直接删除平台验证。

## 动态说话人数删除门槛

固定五人必须从产品约束降级为 `N=5` 回归样例。目标架构必须接受任意正整数 `N`；设备资源预算只能触发可恢复错误或人工分批处理，不能静默截断、合并角色或形成固定人数上限。任何旧代码删除前，新链必须满足：

1. `speakerCountMode` 支持 `auto`、`manual`、`hybrid`。
2. `N` 为任意正整数；资源预算必须可配置、可解释，不能硬编码为五或其他人数上限。
3. canonical ID 连续为 `speaker-1` 到 `speaker-N`。
4. `speakerCount`、speaker set、CAM++ score vector、speaker profile、segment speaker 映射的 cardinality 完全一致。
5. `manual` 模式严格满足用户指定的 `N`。
6. `hybrid` 模式严格满足 `minSpeakers <= N <= maxSpeakers`。
7. `auto` 模式人数不确定时进入人工复核并 fail-closed，禁止静默猜测、截断、合并或新增角色。
8. overlap/串话拆分必须保留父子时间边界、最小时长、文本来源和人工锁定。
9. UI 轨道、Rust job、backend、报告 JSON、Java validator、XHTML、PDF 元数据、图例和测试 fixture 全部由动态 speaker set 驱动。
10. 最小回归矩阵覆盖 `N=1/2/5/8/13`，并包含零发言 speaker、短会、长会、串话、人数不确定和人工锁定场景。

当前状态不满足第 9 项：桌面端、Rust mock、Java validator/renderer/metadata 和部分测试仍固定五人。因此不得勾选动态人数端到端完成，也不得以 Maven 全绿作为旧 PDF 链删除依据。

## 效率级联与分离指标删除门槛

旧 pipeline 删除前，新 backend 必须证明按以下成本级联执行，而不是把全部模型无差别跑遍全场：

1. 一次性音频预处理、VAD、FunASR 时间边界和基础特征可缓存、可校验、可恢复。
2. 全量基础通道只运行 Qwen3-ASR 基础转写、CAM++ 首轮打分和确定性约束。
3. 只有低 margin、边界冲突、overlap、人数不确定、短片段、离群 embedding 或规则冲突片段进入昂贵复核。
4. 重分段、重嵌入、候选 ASR 和上下文扩大只作用于入队片段。
5. 本地小 LLM 只生成结构化建议，永不自动应用。
6. 人工只复核仍有冲突或高影响的最小证据包；人工锁定后的重算应为增量式。
7. 每一级具备版本化缓存键、触发/退出原因、耗时与资源遥测、有界队列、背压、取消和恢复。

删除 PR 必须提供相互分离、不可互相抵消的指标报告：

| 指标域 | 最低证据 |
|---|---|
| 人数估计 | exact-count accuracy、count MAE、欠分/过分率、人工复核率，按实际 `N` 分桶 |
| 说话人分离 | DER、JER、speaker confusion、speaker attribution accuracy、overlap precision/recall |
| 时间边界 | boundary MAE/F1、漏段率、重复覆盖率、父子 overlap 边界合法率 |
| ASR | 基于 `rawText` 的 CER、专名错误率、空段/幻觉率 |
| 语义建议 | schema 通过率、precision/recall、越界修改率、证据充分率、人工接受/拒绝率 |
| 效率 | RTF、阶段 p50/p95、峰值 VRAM/RAM、缓存命中率、升级片段率、重算音频占比 |
| PDF | segment/text/timestamp/speaker 完整性硬门槛与 14 个 Design Pack 维度 |

任何一项硬门槛失败都停止删除。DER/JER 不能被 CER 或语义连贯度抵消，`rawText` CER 不能用优化后的文本回填，内容完整性不能被 PDF 美学分数抵消；单一“总质量分”不能作为旧架构删除依据。

## PDF 替代与 Design Pack 门槛

### Java sidecar 必须成为唯一渲染链

- `reporting/report_document_assembler.py` 生成版本化报告文档。
- `reporting/java_pdf_client.py` 只通过结构化请求调用 `pdf-renderer`。
- `pdf-renderer` 固定使用 OpenHTMLtoPDF `1.0.10` 与 PDFBox `2.0.30`。
- 输出至少包括 XHTML/HTML、PDF、逐页 PNG、联系表、artifact manifest、`pdf-quality-report.json` 和 `repair-queue.json`。
- PDFBox 检查的 segment ID、文本、页数、时间戳和 speaker set 必须与输入一致。
- 中文字体必须嵌入；禁止运行时网络、远程字体、CDN、遥测、脚本和远程图片。
- 任一硬门槛失败时，不允许用美学分数抵消。

### Design Pack 当前事实

`${FRONTEND_DESIGN_PACK_ROOT}/repository-index.json` 当前存在。“`repository-index.json` 缺失”只能记录为历史 blocker；系统仍必须在索引缺失、不可解析或校验失败时 fail-closed。

当前有效状态：

```text
runtimeOfflineReady = true
implementationReady = false
releaseEligible = false
releaseStatus = package-not-release-ready
```

仍未关闭：

- `EVIDENCE-OPEN-001`
- `CONTENT-OPEN-001`
- `ASSET-OPEN-001`

因此当前可以内化 Design Pack 的检查方法和稳定维度 ID，但不能宣称桌面 UI、PDF 或应用 implementation-ready、release-ready 或商业发布就绪。`implementationReady=false` 是旧链最终删除和商业发布之前必须显式处理的 blocker。

## 本地 LLM 生产门控

正式基准模型为 `qwen2.5:1.5b`，结论为：

```text
reject_for_production
```

关键证据：

- 样本 88：dev 48、held-out 24、safety 16。
- 严格最终契约通过率：`5.6818%`。
- safety challenge 通过率：`0%`。
- 越界修改率：`94.3182%`。
- overlap escalation F1：`0`。
- risk escalation F1：`0`。
- 自动应用候选：2；其回归率为 `50%`。

当前允许：

- 作为 suggestion generator 生成结构化建议；模型权限严格为“仅建议、永不自动应用”。
- 经过 schema、deterministic validator 和人工复核后展示建议。
- 保存建议、证据代码、修改前后文本与拒绝原因。

当前禁止：

- 自动修改说话人。
- 自动拆分或合并 turn。
- 自动处理 overlap。
- 自动修改普通中文原文。
- 覆盖人工锁定或高 CAM++ margin。
- 把模型自报置信度当作自动应用依据。

在更强模型通过预先登记的契约、安全、越界修改、风险升级和回归阈值之前，旧架构删除不得以“小 LLM 可以接管人工语义复核”为前提。

## 分阶段可执行清单

### 阶段 A：冻结调用图和兼容契约

- [x] 记录旧入口、UI、pipeline、formatter、PDF、工具和 packaging 调用图。
- [x] 记录新目录对旧模块无直接导入。
- [x] 固化版本化 IPC、report document 和 PDF request/result schema。
- [ ] 为旧 CLI、TXT/JSON 输出和报告命名建立兼容样例与迁移说明。
- [ ] 为所有将删除模块建立生产调用计数或静态引用归零检查。

### 阶段 B：切换桌面与 CLI 入口

- [ ] 让 Tauri/Rust 只启动新 backend，不再调用 `ui_app.py` 或 `mts_ui/**`。
- [ ] 让无头 CLI 只调用新 backend service。
- [ ] 将 `tools/verify_mps_runtime.py` 迁移到新 adapter/service。
- [ ] 让 `main.py` 先变为带明确弃用提示的薄兼容入口。
- [ ] 连续发布窗口内验证没有回退到旧 GUI 或旧 pipeline。

### 阶段 C：切换模型与 Dynamic-N 编排

- [ ] 新 backend 接入真实 Qwen3-ASR-1.7B、FunASR 和 CAM++ adapter。
- [ ] 实现 Dynamic-N 全局约束解码和人工复核队列。
- [ ] 完成自动、手动、混合人数模式。
- [ ] 通过 `N=1/2/5/8/13`、更大 `N` 资源压力和人数不确定场景，证明资源失败不会静默改变 `N`。
- [ ] 实现并验证“基础通道 → 不确定性路由 → 局部重算 → 仅建议语义层 → 最小人工复核”的效率级联。
- [ ] 分别产出人数估计、DER/JER、边界、`rawText` CER、语义建议安全性和效率报告；禁止用综合分替代分域门槛。
- [ ] 证明取消、恢复、失败重试和持久化不损坏审计轨迹。

### 阶段 D：切换输出与 PDF

- [ ] 将 TXT、JSON、批量摘要职责迁入版本化 exporter。
- [ ] Python backend 只调用 Java sidecar。
- [ ] Java validator、renderer、metadata 和 tests 全部改为动态 `N`。
- [ ] 每个 segment ID、原文、时间戳和 speaker set 在 PDFBox 检查中完整一致。
- [ ] Design Pack 硬门槛、14 维度、截图证据、SHA-256 与 repairQueue 在真实报告上通过。
- [ ] 清理或重新生成 `pdf-renderer/target/debug-fixture/**` 的旧失败产物。

### 阶段 E：迁移 requirements 与 packaging

- [ ] 拆分模型 runtime、backend runtime、Java sidecar 和桌面构建依赖。
- [ ] Tauri bundler 完成 Windows 安装、升级、卸载和离线运行验证。
- [ ] 移除 PySide6 应用运行时依赖。
- [ ] 移除 Playwright、Chromium、pdfkit 和 wkhtmltopdf。
- [ ] 移除旧应用 PyInstaller spec 或仅保留有明确独立用途的构建工具。
- [ ] 验证安装产物不包含 `mts_ui`、旧 PDF 引擎或旧 GUI 入口。

### 阶段 F：最终删除

- [ ] 删除 `ui_app.py`。
- [ ] 删除 `mts_ui/**`。
- [ ] 删除 `report_generator.py`。
- [ ] 删除或拆分后删除 `output_formatter.py`。
- [ ] 删除 `pipeline.py`。
- [ ] 删除或最终移除 `main.py` 兼容入口。
- [ ] 删除迁移后无用途的 requirements 项和 packaging 文件。
- [ ] 重新执行全部测试、安装包 E2E、动态人数回归、PDF QA 和离线检查。
- [ ] 更新迁移文档并记录回滚版本。

本次阶段 F 全部保持未勾选；本次没有删除任何代码。

## 删除前验证命令基线

以下命令用于形成删除 PR 的最低本地证据；执行时应在仓库根目录运行，并保留日志：

```powershell
conda run -n media-asr python -m compileall backend reporting tests
conda run -n media-asr python contracts\validate_contracts.py
conda run -n media-asr python -m unittest discover -s tests -p "test_*.py"

Push-Location apps\desktop
npm run typecheck
npm test -- --run
npm run build
cargo check --manifest-path src-tauri\Cargo.toml
Pop-Location

mvn -f pdf-renderer\pom.xml test

rg -n "ui_app|mts_ui|TranscriptionPipeline|OutputFormatter|ReportGenerator" `
  main.py tools packaging apps backend reporting tests
rg -n "five speakers|five-speaker|五人|speakerCount.?[:=].?5" `
  apps backend reporting contracts pdf-renderer tests
rg -n "playwright|pdfkit|wkhtmltopdf|PySide6|PyInstaller" `
  requirements*.txt packaging apps backend reporting
```

通过标准不是“命令退出码为零”这么简单：

- 旧符号搜索只能命中文档、迁移测试或明确兼容层，不能命中生产入口。
- 固定五人搜索只能命中 `N=5` 回归 fixture 或历史说明。
- 旧 UI/PDF/build 依赖搜索在最终删除阶段必须不再命中应用运行时和发行路径。
- Maven 测试必须包含动态人数，而不只是固定五人 fixture。
- `pdf-renderer/target/**` 只能引用本轮生成且与报告 manifest 匹配的证据。

## Fail-closed 与回滚条件

出现以下任一情况时停止删除并恢复到上一个已验证版本：

- 新 backend 无法完成真实模型作业，或只能依赖 mock。
- 动态 `N` cardinality invariant 失败。
- 效率级联退化为对全场无差别运行昂贵模型，或无法说明升级、缓存和重算范围。
- 人数估计、说话人分离、边界、ASR、语义建议、效率或 PDF 指标被合并分数掩盖。
- 自动人数不确定但系统仍生成最终报告。
- 人工锁定被自动流程覆盖。
- 报告缺 segment、文本、时间戳、说话人或字体。
- Java sidecar 不可用时静默回退到旧 Python/浏览器 PDF。
- Design Pack 索引缺失、不可解析、校验失败，或仍不满足项目要求却被标记为发布就绪。
- 本地 LLM 的越界修改被自动应用。
- Tauri 安装包仍依赖旧 PySide6 GUI、Playwright/Chromium 或 wkhtmltopdf。

回滚必须保留 transcript document、作业事件、人工锁定、模型版本、渲染请求、QA 报告和 artifact hash；禁止只回滚可执行文件而丢失审计数据。

## 当前审计判定

- 旧 UI、旧 pipeline 和旧 Python PDF：**保留，尚不可删除**。
- 新架构直接导入旧模块：**未发现**。
- 生产入口切流：**未完成**。
- 动态说话人数：**目标为任意正整数 `N`；有 Python 基础，端到端未完成**。
- 效率级联：**目标与删除门槛已定义，尚无完整生产 E2E 证据**。
- 分离指标：**验收域已定义，尚未形成全域分桶生产报告**。
- Java PDF 动态人数：**未完成，仍有固定五人 blocker**。
- requirements/packaging 清理：**未开始最终删除**。
- Design Pack：索引当前存在；`implementationReady=false` 仍阻断。
- 本地小 LLM：`qwen2.5:1.5b` 为 `reject_for_production`，仅可建议、永不自动应用。
- 本次代码删除：**无**。
