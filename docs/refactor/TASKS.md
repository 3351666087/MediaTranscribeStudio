# MediaTranscribeStudio 重构任务清单

> 执行日期：2026-07-21
> 分支：`codex/ts-local-llm-refactor`
> 原则：先建立可验证替代链路，再删除旧实现；每项只有在本地证据通过后才能勾选。
> 状态说明：`[x]` 表示审计证据已经存在，不表示整个重构、动态人数或商业发布已经完成。

## 0. 基线与约束

- [x] 克隆并定位仓库，记录原始提交和工作分支。
- [x] 读取 `frontend-design-pack-global` 的入口规则、QA 契约和评估器。
- [x] 记录硬件、Conda、Node、Rust、Java、ffmpeg 和 Ollama 基线。
- [x] 运行 Python `compileall` 基线。
- [x] 记录旧 UI、旧 PDF、ASR、LLM 和说话人模块规模。
- [x] 核验旧入口依赖链：`main.py`、`ui_app.py`、`mts_ui/**`、`pipeline.py`、`output_formatter.py` 与 `report_generator.py` 当前仍可达。
- [x] 核验新 `apps/desktop`、`backend`、`reporting`、`contracts` 与 `pdf-renderer` 未直接导入上述旧模块。
- [x] 按旧 UI/PDF 运行时、构建/安装器和模型运行时分类审计 requirements/packaging。
- [x] 核验 Design Pack 的 `repository-index.json` 当前存在；索引缺失仅保留为 fail-closed 条件和历史 blocker。
- [x] 记录当前 Design Pack blocker：`implementationReady=false`、`releaseEligible=false`，且三个 blocking open question 未关闭。
- [x] 建立新的版本化 IPC、转写文档和 PDF QA 契约。

## 1. 本地小 LLM 可行性

- [x] 建立不泄露真实会议内容的基准测试框架。
- [x] 支持通过环境变量读取外部人工真值。
- [x] 用 `qwen2.5:1.5b` 跑开发集、连续时间 held-out 集与 safety challenge。
- [x] 记录严格契约、中文字保真、角色/拆分、越界修改、风险升级、回归与速度指标。
- [x] 形成正式判定：`qwen2.5:1.5b` 为 `reject_for_production`。
- [x] 将当前模型限制为 suggestion generator（仅建议、永不自动应用），并要求 deterministic validator 与人工复核。
- [x] 根据 8 GB 显存实测更强的 `qwen3.5:4b` Q4_K_M 候选，并生成不含会议原文的正式报告。
- [x] 审计 `qwen3.5:4b` 正式 88 样本结果：契约有效率 0.784、越界文本修改率 0.205、auto-apply 候选回归率 0.135，结论为 `reject_for_production`。
- [ ] 找到达到预先登记生产阈值的本地模型。
- [ ] 允许任一本地小 LLM 自动修改说话人、overlap、turn 结构、普通中文原文或人工锁定结果。

## 2. 动态人数中文会议流水线

- [x] 定义 `speakerCountMode = auto | manual | hybrid` 和任意正整数 `N` 的领域契约；固定人数只能作为 fixture。
- [x] 在 `backend/models.py` 建立动态人数校验基础，并用 Python cardinality tests 覆盖多个人数。
- [x] 在 `reporting/report_document_assembler.py` 接受动态 speaker set。
- [ ] 让新 `backend/**` 成为真实生产入口并完全替代旧 `pipeline.py`。
- [ ] 固化 Qwen3-ASR-1.7B、FunASR 时间边界、CAM++ 主声纹通道与 ERes2NetV2 难例复核通道的适配器边界。
- [ ] 增加 overlap/串话候选和局部音频复核任务。
- [ ] 实现 Dynamic-N 全局约束解码，保证 speaker set、score vector、profile 与 segment 映射基数一致。
- [ ] 完成自动检测、手动指定、混合上下界和人数不确定时 fail-closed 的端到端流程。
- [ ] 实现问答关系、轮次一致性、称谓、议题和相邻声纹的语义仲裁。
- [ ] 保存机器原文、语义优化文、变更原因、置信度和审计轨迹。
- [ ] 保留确定性回退路径：小 LLM 不可用时不破坏 ASR 与声纹结果。
- [ ] 实现效率级联：一次性预处理/缓存、全量基础通道、不确定性路由、局部重算、仅建议语义层、最小人工复核和增量报告。
- [ ] 为每一级记录触发原因、输入范围、缓存键、耗时、峰值资源、输出置信度和退出原因，并实现有界队列、背压、取消与恢复。
- [ ] 将人数估计、DER/JER/speaker confusion、时间边界、`rawText` CER、语义建议安全性、系统效率和 PDF 质量拆成不可互相抵消的指标域。
- [ ] 消除桌面端、Rust mock、Java validator/renderer/metadata 和测试中的固定五人约束。
- [ ] 通过至少 `N=1/2/5/8/13` 的回归矩阵，并对更大 `N` 做资源压力测试；五人仅作为 `N=5` fixture，任何资源失败都不得静默改变 `N`。

## 3. TypeScript 桌面端

- [x] 建立 Tauri 2 + React + TypeScript + Vite 工程。
- [x] 建立可爱、高颜值、离线优先的视觉 token 与组件基础。
- [x] 复用远端仓库的 `scene.png` 背景资产。
- [ ] 实现项目创建、文件选择、模型状态、运行进度和日志。
- [ ] 实现按动态 `N` 生成的说话人轨道、置信度、串话候选、局部试听和人工纠错。
- [ ] 实现逐字稿预览、PDF QA、repairQueue 和证据浏览。
- [ ] 完成键盘、焦点、缩放、窄屏、字体失败、图片失败和脚本失败状态。
- [x] 留存 `npm run typecheck`、`npm test -- --run`、`npm run build` 与 `cargo check` 通过证据。
- [ ] 通过完整 lint、component、E2E、Rust fmt、clippy 和 Rust tests 门槛。
- [ ] 清除 `index.html`、`package.json`、Rust job/mock 和 React tests 中的固定五人文案或数据。
- [ ] TypeScript/Rust 替代入口通过后删除 `ui_app.py` 与 `mts_ui/`。
- [ ] 清除应用运行依赖中的 PySide6；安装器若保留 Python UI，必须单独标注并迁移。

## 4. Java PDF 渲染

- [x] 建立独立 `pdf-renderer` Maven 模块。
- [x] 固定 OpenHTMLtoPDF `1.0.10`。
- [x] 固定 Apache PDFBox `2.0.30`。
- [x] 定义转写报告 JSON 输入契约。
- [x] 建立离线 XHTML/CSS、字体、页眉页脚、元数据和 PDFBox 检查的基础实现。
- [x] 建立 HTML、PDF、逐页 PNG、联系表、manifest、QA 报告和 repairQueue 的基础产物链。
- [x] 当前 Maven surefire 汇总为 17 tests、0 failures、0 errors。
- [ ] 修复 Java validator、XHTML 文案、PDF 元数据和 fixtures 中的固定五人约束。
- [ ] 为动态 `N` 建立 Java validator/render/metadata 和金样回归。
- [ ] Python 后端只调用 Java sidecar，不再生成或渲染 PDF。
- [ ] 替代链路通过后删除 `report_generator.py`。
- [ ] 删除 Playwright、pdfkit、wkhtmltopdf 依赖、探测和打包代码。

## 5. 内化 Design Pack PDF QA

- [x] 建立不可由分数抵消的 PDF 硬门槛基础契约。
- [x] 建立全部 14 个美学维度的稳定 ID 与基础评估器。
- [x] 建立分数低于 85 时失败的质量判定基础。
- [x] 建立 `repairQueue`、manifest 与 SHA-256 证据基础。
- [ ] 第二轮起强制截图重拍、测试重跑、完整重评分和高优先级回归检查。
- [ ] 证明 `repairQueue` 在真实动态人数报告上稳定、确定且优先级正确。
- [ ] 最多自动修复五轮，之后报告 `blocked`。
- [ ] 通过测试证明运行时网络、远程字体、CDN、遥测和远程图片全部被拒绝。
- [ ] 关闭 `EVIDENCE-OPEN-001`、`CONTENT-OPEN-001` 与 `ASSET-OPEN-001`。
- [ ] 等待全局 Design Pack 变为 `implementationReady=true` 并完成项目专属证据审查。
- [ ] 达到 `releaseEligible=true`；当前不得宣称 implementation-ready、release-ready 或商业发布就绪。

## 6. 验证与交付

- [x] Python 单元/契约测试通过：33 tests，schema 验证 7 个。
- [ ] TypeScript lint、typecheck、unit、component 和 E2E 全部通过。
- [ ] Rust fmt、clippy 和 tests 全部通过。
- [ ] Maven tests 与动态人数 Java PDF 金样测试全部通过。
- [ ] 用真实动态人数会议进行端到端回归，不提交会议原文。
- [ ] 使用真实五人会议作为 `N=5` 回归样例，而不是系统人数上限。
- [ ] 为人数估计、说话人分离、边界、ASR、语义建议、效率和 PDF 分别登记生产阈值并生成分桶报告；禁止只给综合分。
- [ ] 证明效率级联只升级不确定片段，并报告 RTF、阶段 p50/p95、峰值 VRAM/RAM、缓存命中率、升级率和重算比例。
- [ ] 对每页 PDF 做结构、文本、字体、裁切、空白、对比度和视觉证据检查。
- [ ] 清理或重新生成 `pdf-renderer/target/debug-fixture/**` 的旧失败产物。
- [ ] 完成 Windows 开发与打包说明。
- [ ] 更新 README、架构、模型卡、隐私、许可证和迁移文档。
- [ ] 按 `LEGACY_REMOVAL.md` 顺序完成入口切流、requirements/packaging 清理和旧代码删除。
- [ ] 提交源码并报告仍然存在的外部阻断。
