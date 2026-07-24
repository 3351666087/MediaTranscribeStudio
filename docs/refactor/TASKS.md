# MediaTranscribeStudio 重构任务清单

> 开始日期：2026-07-21
> 当前迭代：2026-07-24
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

## 7. 2026-07-23 产品化冲刺清单

> 本节是当前 `/goal` 的执行看板。只有代码、聚焦测试和必要的真实媒体证据全部通过后才勾选；发现质量回归时必须重新取消勾选。

### 7.1 全媒体原生接入与批处理

- [x] 将扩展名降级为提示，所有安全的本地路径都交给 FFmpeg/FFprobe 内容探测决定是否可处理。
- [x] Windows 原生文件选择器支持多选，原生目录选择器支持用户选择输出目录。
- [ ] 支持未知扩展名、无扩展名、大小写混合扩展名和 FFmpeg 可解码的所有音视频容器/编码。
- [x] 支持连续多批拖拽追加、Windows 路径大小写不敏感去重和首次出现顺序稳定。
- [ ] 支持批量任务逐项失败隔离、逐项修改、移除、重试、取消与并发上限。
- [x] 创建任务正常路径不暴露手写 BCP-47 标签。
- [x] 通过媒体能力、原生选择器、TaskCreator 聚焦测试和 TypeScript typecheck。
- [x] 通过真实未知扩展名、无扩展名、音频和视频探测验收；同一组真实 MOV/M4A 字节以原始名称、`.content-probe` 未知扩展名和无扩展名共 6 个案例通过本机 FFprobe/FFmpeg 内容探测，证据写入 `.runtime_cache/evidence/real-media/probe.v1.json`。

### 7.2 任意人数和高质量角色分离

- [x] 完成未知人数真实长音频 r10 诊断；明确拒绝“96 个粗窗口覆盖约 61.5 分钟”的结果进入成品链路。
- [x] 定位当前主要问题：长 VAD/ASR 窗口包含多轮说话、单窗口单角色赋值、全段零置信度、复核项过多、逐片 ERes2NetV2 延迟过高。
- [ ] 在最终聚类前加入基于 FunASR 字词/句子时间、停顿、标点和多尺度声学子窗的细粒度 turn candidate。
- [ ] 在长窗口内部检测说话人变化点，禁止把明显混合说话窗口直接绑定为单一角色。
- [ ] 对短 turn candidate 执行 CAM++ 嵌入、Dynamic-N 聚类、时间平滑、邻接合并和过分裂/欠分裂校正。
- [ ] 将 ERes2NetV2 改为模型常驻、批量推理和仅难例升级，消除逐片约 40 秒级开销。
- [ ] 增强 overlap/change detector；本地可用时提供受控 pyannote 回退，并保留离线和许可证证据。
- [ ] 声学候选稳定后再执行问答关系、称谓、轮次和上下文语义仲裁。
- [ ] 为原文语义修正保存 evidence、理由、置信度和 human lock；禁止静默覆盖不可变 raw ASR。
- [ ] 自动人数不确定时 fail closed；支持手动人数、混合上下界和任意正整数人数。
- [ ] 重跑未知人数真实 M4A；长媒体必须使用覆盖开头、中段、结尾、随机有声段和声学变化点的分层窗口，整段终态与窗口结果共同按人数稳定性、turn 粒度、speaker confusion、审查量和 RTF 分域验收。
- [ ] 重跑真实 MOV 的 auto 与 manual=5；只有人工确认包含五位不同说话人的窗口或整段真值才能用于 `manual=5` 质量验收，固定取前 60 秒仅可作为媒体接入/执行烟雾测试，不能证明五人分离质量。

### 7.3 字幕、视频写回与事务式发布

- [x] SRT、WebVTT、ASS sidecar 使用同一不可变 transcript document 和输出配方。
- [x] ASS 支持确定性大规模说话人配色、用户覆盖色、禁用角色色和字幕样式 DIY。
- [x] Soft-mux 与 burn-in 始终使用高保真 ASS carrier，不以 SRT 作为视频写回载体。
- [ ] 提供 YouTube 级字幕排版：安全区、行宽、断句、阅读速度、描边/阴影、角色色、屏幕分辨率适配和无障碍选项。
- [x] 每个视频输出先进入隔离区，完成代表帧视觉 QA 后再原子发布。
- [x] 事务失败、取消、QA 不通过或 manifest 持久化失败时回滚本事务拥有的全部产物。
- [x] 发布 manifest 绑定 recipe、customization、完整 execution plan、源媒体和每个客户产物哈希。
- [x] rerender 复用通过验证的字幕/视频产物；任何篡改都必须 fail closed。
- [x] 通过字幕、输出编排、代表帧 QA、输出发布和 WorkerService 全套测试。
- [ ] 用真实视频分别验收 sidecar、soft-mux 和 burn-in 的同步、字体、颜色、清晰度和兼容性。

### 7.4 报告、字体和 DIY

- [ ] PDF 仅通过 OpenHTMLtoPDF + PDFBox 构建、检查和生成元数据。
- [ ] 用户可选择报告款式、页面尺寸/边距、字体组合、字号密度、封面、页眉页脚、角色色和时间戳格式。
- [ ] 用户可选择字幕主题、字体、字号、位置、安全区、每行字数、最多行数、角色标签、颜色、描边、阴影和动画强度。
- [ ] 用户可选择交付模式、容器、命名模板、输出目录结构和是否保留中间证据。
- [ ] Design Pack PDF 棞查成为运行时硬门槛：分数、硬门槛、14 个视觉维度、repairQueue 和证据哈希全部持久化。
- [ ] PDF 检验详情默认折叠，界面以圆圈加数字显示分数，不使用夸张大字。
- [ ] 通过动态人数、长文本、多语言、字体缺失和窄页面 PDF 金样回归。

### 7.5 本地业务模型与完整业务覆盖

- [ ] 翻译执行分块、上下文重叠、术语一致性检查和完整性对账，禁止大段英文静默漏译。
- [ ] 润色不得删除事实、数字、角色或时间边界；所有改动可追踪并可恢复原文。
- [ ] 总结覆盖全部输入分块，并校验议题、行动项、决定和说话人引用。
- [ ] 本地小模型失败、越界或输出不完整时 fail closed，并回退到未经修改的已验证内容。
- [ ] 找到达到登记阈值的本地模型，或明确保持“仅建议、人工确认”的产品策略。
- [ ] 原仓库翻译、润色、总结、字幕、报告和导出业务全部迁移后才允许删除旧实现。

### 7.6 原生 TypeScript UI 与 Design Pack

- [ ] Tauri/React/TypeScript 原生多级界面覆盖创建、队列、运行、复核、编辑、导出和设置。
- [ ] 亮/暗 PNG 全屏背景、磨砂玻璃、可爱风、远端 GIF 桌宠和 Emil Kowalski/Apple 风格动效完成。
- [ ] 动效支持 reduced motion；界面通过键盘、焦点、缩放、320 px、WCAG 2.2 AA 和屏幕阅读器检查。
- [ ] 拖拽区域有明确提示、进入/离开/放下动画和批量文件反馈。
- [ ] 滚动条、左侧圆角容器、弹窗、多级导航和亮暗主题在真实 Tauri 截图中兼容。
- [ ] 翻译目标语言使用简洁的人类语言选择器；高级标签仅在必要时渐进披露。
- [ ] Design Pack 的结构、色彩、动效、字体和视觉校验全部通过，最终分数不低于 85。

### 7.7 发布、清理和主分支

- [x] 建立 fail-closed Ultimate parity/cutover 检查器，绑定当前 Git HEAD、外部证据根、证据时效、字节数、SHA-256、真实媒体分域指标、Design Pack facets 和独立授权。
- [x] 建立事务式 Windows/Tauri release、install、upgrade、rollback、recover、uninstall 框架及开发 fixture 回归；这不等同于真实签名 native bundle 已完成。
- [ ] 完成 Windows native bundle、运行时依赖、FFmpeg、Java、模型和字体证据。
- [ ] 删除 Python 前端、Python PDF 渲染和所有已被新架构替代的旧代码、依赖与打包入口。
- [ ] 英文 README、全球化架构说明、隐私、模型卡、许可证和迁移指南完成。
- [ ] 完整 parity 清单、Python/TS/Rust/Java/E2E/真实媒体/Design Pack 门禁全部通过。
- [ ] 对每个独立通过的产品切片小步 commit 并 push。
- [ ] 所有旧业务迁移且最终门禁通过后覆盖 `main`。
- [ ] 最终成品通过验收后结束 `/goal`。

### 7.8 当前本地执行清单（逐项验证、逐项勾选）

> 本节是本轮工作的本地 Markdown 看板。每一项必须有直接测试或产物证据；任何后续回归都要取消对应勾选。
> 固定取用户媒体前 60 秒仅属于流水线烟雾测试；除非该窗口有独立说话人真值，否则不得用于人数、说话人分离或五人会议验收。

- [x] Dynamic-N v7 使用 SHA-256 精确保留掩码；覆盖精确目标大小、human lock 全保留、10 次运行超过 5 个唯一掩码、输入置换不变和重复调用确定性。
- [x] Dynamic-N v7 恢复近声纹严格校正门槛，并验证 `N=8` 的稳定性、覆盖率、目标差和 `CLOSE_VOICE_RESIDUAL_COLLAPSE` 路径。
- [x] Dynamic-N v7 低预算搜索保留声学 leader/anchor，错误锚点场景直接形成 `12/13/14` 局部括号，资源不足场景区分自适应预算受限与真实资源截断。
- [x] Dynamic-N v7 缓存解析器拒绝旧版本、缺失字段、非有限数、频率/支持冲突和搜索审计冲突；正确缓存键上的畸形条目会重算而不是复用。
- [x] Dynamic-N v7 聚焦门禁：`91 passed`（两个 Dynamic-N 模块加生产流水线畸形聚类缓存重算测试）。
- [x] 独立复核 production CAM++/ERes2NetV2/pyannote 生命周期、OOM 和配置迁移切片；相关 Dynamic-N/production 门禁共 `193 passed, 48 subtests passed`，`compileall` 与定向 `git diff --check` 通过。
- [x] 独立复核并提交 Ultimate parity 原始字节、Git attributes/filter、RFC3339 授权时间顺序、manifest 决策字节 TOCTOU、隐藏 index flags 与授权合取加固切片；`49 passed`，manifest validation、`py_compile`、定向 `git diff --check` 通过，两轮修复后独立复审结论为“无阻断项”。
- [x] 本地通过 TypeScript 多任务 UI 的 lint、typecheck、unit/component 和 production build；`npm run lint`、`npm run typecheck`、`npm test -- --run`（31 files / 375 tests）与 `npm run build` 全部通过，并补充精确 job ID 切换、非当前任务取消、批量部分失败隔离和 Tauri/mock 严格运行时契约覆盖。
- [ ] 通过 Design Pack 结构、色彩、动效、字体、可访问性和 PDF 视觉门禁。
- [ ] 未知人数真实 `20260723_123047(1).m4a` 完成整段终态、全时段分层抽样、自动人数、角色分离、人工真值和分域性能验收。
- [ ] 真实 MOV 完成全时段分层抽样，并仅在人工确认五人覆盖的窗口/整段上运行 auto 与 manual=5，对比人数稳定性、speaker confusion、审查量、RTF、RAM/VRAM 和缓存指标。
- [ ] 完成 Java-only PDF、SRT/WebVTT/ASS、soft-mux、burn-in 和代表帧视觉 QA。
- [ ] 原业务全部迁移、旧 Python UI/PDF/业务入口删除、最终 parity 通过后覆盖 `main` 并结束 `/goal`。

### 7.9 全球多源、多语种样本与严谨抽样

> 质量结论不得只来自两个用户文件、两个公开数据集、每种语言一个短样本、合成语音或固定开头片段。样本库必须覆盖全球语言、口音、人数、媒体类型和真实录音条件，并为每个可评分结论保存可追溯真值。

- [ ] 建立全球公开样本源清单；每个远程样本固定来源 URL、数据集/仓库版本、许可证、署名要求、原始 SHA-256、下载时间和本地派生关系，来源或许可证不明确的样本禁止进入门禁。
- [ ] 建立分层规模门槛：快速回归集至少 60 个真实短窗口；完整发布矩阵至少 300 个真实窗口、12 个相互独立的数据集、30 种语言和 12 个地区，其中真实多人会话至少 100 个窗口、12 种语言；当前 22 条单人语音和 7 条英语多人窗口只算首批引导样本，不得宣称全球质量覆盖完成。
- [ ] 每个关键语言、人数和声学场景至少包含 3 个不同原始录音，并尽量来自至少 2 个独立数据集；报告同时给出逐样本、逐数据集和等权分桶结果，禁止让单一数据集或其大量相关切片主导总结果。
- [x] 首批单人语音覆盖东亚、南亚、东南亚、欧洲、非洲、中东、北美、拉丁美洲、大洋洲等 11 个地区和 22 种语言；每条同时保存原生文本、规范化评分文本和 `development`/`regression`/`held-out` 角色。
- [x] 当前矩阵覆盖单人及 `N=2/3/4/5/8/13`；其中真实多人覆盖 `N=2/3/4/5/8`，派生压力矩阵覆盖 `N=2/3/5/8/13`，`N=5` 仅是一个分桶，构建器不会因系统资源限制静默改写真值人数。
- [ ] 真实多人会话扩展到普通话、粤语、日语、韩语、英语、西班牙语、法语、德语、阿拉伯语、印地语、葡萄牙语等不同语系与代码切换场景；每种语言都必须有真实 turn/overlap 真值，不能用单人拼接样本替代多人质量门禁。
- [ ] 覆盖清晰近讲、远场会议、电话带宽、车内/户外、音乐/电视背景、混响、强噪声、不同年龄/性别音色、地区口音、代码切换、快速轮换、打断、串话和重叠说话，并区分真实录音、真实录音派生扰动与合成压力样本。
- [ ] 覆盖 WAV、FLAC、MP3、M4A/AAC、OGG/Opus、MOV、MP4、MKV/WebM，以及未知扩展名和无扩展名；扩展名仅作提示，最终以本机 FFmpeg 内容探测和解码结果为准。
- [ ] 单个迭代样本默认控制在 10–90 秒；长媒体采用全时段分层抽样（开头、中段、结尾、固定随机种子有声窗口、声学变化点/重叠候选），并保存窗口选择算法、种子、源时间戳与覆盖率；固定前 60 秒只能做接入烟雾测试，禁止用于说话人数或分离质量结论。
- [ ] 基于参考 RTTM/turn 标注选择 `N=5` 等人数窗口：先在整段真值中定位确实覆盖目标人数的候选，再按预先登记的随机规则取样；窗口选择不得查看模型得分，避免为了得到好结果而挑片。找不到满足条件的窗口时必须更换源样本，不能把人工人数强设为真值。
- [ ] 按指标保存和校验真值资格：人数指标要求 speaker set，DER/JER、speaker confusion 和边界指标要求 turn/overlap 标注，WER/CER 要求覆盖完整音频窗口的逐字稿；某项真值缺失时只禁用该项评分，不得用另一项真值替代或宣称通过。
- [ ] 建立互斥的开发集、回归集和 held-out 集；必须先按数据集、原始录音和说话人分组再切分，同一录音、同一说话人或任何派生/转码/扰动版本不得跨集合泄漏，调参后必须重新跑从未参与阈值选择的 held-out。
- [ ] 每个样本分别报告人数绝对误差、DER、JER、speaker confusion、边界误差、WER/CER、字幕同步、人工复核量、RTF、阶段 p50/p95、峰值 RAM/VRAM、缓存命中率、升级率和重算比例；禁止用综合分抵消任一失败域。
- [x] 下载并验证首批全球单人真实语音：MInDS-14 与 FLEURS 共 22 条、22 种语言、11 个地区，均规范化为 PCM s16le/16 kHz/单声道，时长 2.56–37.08 秒，解析与双层 SHA-256 逐项验证 `22/22`、`failedCases: []`；固定 revision、源定位、许可证/署名、下载时间、原生/评分文本与落盘哈希保存在 `.runtime_cache/sample-library/global/global-sample-library.resolved.v1.json` 和 `ATTRIBUTION.md`，二进制未提交 Git。
- [x] 构建并验证 Dynamic-N 派生压力矩阵：顺序样本覆盖 `N=2/3/5/8`，重叠样本覆盖 `N=2/3/5/8/13`，共 9 条、10.89–79.67 秒；逐项保存并验证 speaker set、turn、overlap、源样本哈希和派生音频哈希，重叠样本明确 `asr=false`，不得用于 WER/CER。
- [x] 下载并验证首批真实多人分离样本：固定 revision 的 AMI 覆盖 `N=2/3/4`，VoxConverse 覆盖 `N=2/3/5/8`，共 7 条英语 90 秒窗口；每条均保存真实 speaker set、turn/overlap 真值、窗口源时间戳、独立音频/标注哈希并允许 DER/JER 评分；因无覆盖窗口的逐字稿，全部明确 `asr=false`。这只证明首批工具链可用，不满足多语种或大规模质量门禁。
- [x] 修复生产 worker 意外异常诊断：内部 stderr 记录 `jobId`、stage、异常类型和完整 traceback，公开 JSONL 只返回脱敏 `INTERNAL_ERROR`，不泄露异常消息或本机路径；`tests/test_service.py` 聚焦回归 `16 passed`。
- [x] 定位并修复 `voxconverse-dev-row035-n2` 的二级说话人模型 `AssertionError`：ERes2NetV2 经 ModelScope CPU 校验构造后再迁移内部 embedding model 到 MPS，并对 MPS 不可用或迁移失败返回结构化错误；生产 runner 聚焦回归 `42 passed, 9 subtests passed`，真实案例恢复到 `review.required`，自动人数预期/实际均为 2。
- [x] 完成真实 VoxConverse `N=2` 的 auto、manual=2 与 hybrid `[1,3]`/prior=2 功能对照；三种模式得到相同的 8 段和相同 DER/JER，说明本案例的人数策略不是当前质量瓶颈。manual/hybrid 的 RTF 受缓存命中影响，只作为本次运行证据，不作为冷启动性能结论。
- [x] 修复 change-boundary 规划中“低置信复核候选阻止强自动边界”的缺陷，保留两个过近强候选时 fail closed 的约束；CAM++ 缓存版本升级为 `2.1.0`，聚焦回归 `23 passed`。真实 `N=2` 重跑从 8 段增至 11 段，边界 p50 误差从 9360 ms 降至 2060 ms。
- [ ] 真实 `N=2` 整体质量门禁仍未通过：canonical Pyannote 多轨证据已使生产结果达到 DER `0.036183971`、JER `0.041564942`、speaker confusion `0.006092896`、overlap F1 `0.930924004`，但边界 p95/最大误差仍为 `16983 / 19240 ms`，待复核项仍有 23 个；本次 RTF `0.013802365` 来自 95% 缓存命中的热运行，不能替代冷启动性能结论。不得用说话人指标改善抵消边界、复核量或冷启动性能失败。
- [x] 下载并逐文件校验非 gated、CC BY 4.0 的 `pyannote-community/speaker-diarization-community-1` 离线镜像，固定 revision `8a527374977391da736e0daaef26855d949d9685`；10 个业务文件约 32 MB，本地 `Pipeline.from_pretrained(...)` 加载成功。官方 `pyannote/speaker-diarization-community-1` 仍为 gated，当前本机未登录，未匿名绕过授权。
- [x] 用内存 waveform 完成 Pyannote 4 对真实 VoxConverse 的最小推理和完整 90 秒独立评分：自动人数为 2，DER `0.036183971`、JER `0.041564942`、speaker confusion `0.006092896`、overlap F1 `0.930924004`、RTF `0.571457244`；生产 fallback 已输出精确 `speakerTurns`/`overlapIntervals`，并在 overlap 阶段后释放模型资源。
- [x] 完成 Pyannote 匿名轨道到 canonical `speaker-N` 的严格证据链验收：共享校验器逐项核对全局时长加权声学矩阵、Hungarian 一对一映射、最优/次优 margin、dominance、canonical turns、human lock、人数变化和 blocker；伪造权重、时长、margin、dominance 或 blocker 均 fail closed。真实 `N=2` 重跑自动人数为 2，11 段含 4 条可追溯 speaker revision 和 27 条 canonical turn，报告中 4 份 `speakerMapping` 均无悬空引用；Java renderer `3.0.0` 生成 3 页真实报告，质量分 `97.87`、13 个硬门槛全部通过。证据位于 `.runtime_cache/sample-library/global/results/real-n2-pyannote-map-v2/quality-report.v1.json` 与 `.runtime_cache/outputs/global-sample-library/real-n2-pyannote-map-report-direct-lxgw/`。
- [x] 修复 PDF 字体配置漂移：生产配置、Pyannote 评估配置和示例统一使用 renderer 唯一允许的内置 `LXGW WenKai`，配置加载阶段提前拒绝 `MTS CJK` 等无效请求值；更新后的 Pyannote 严格生产 preflight 全部通过。
- [x] 隔离 Pyannote 4 生产运行时：主 `media-asr` 环境移除 Pyannote 4 并固定 NumPy `1.26.4`、torchmetrics `0.11.4`，独立 `media-pyannote` 环境固定 Pyannote Audio `4.0.4`、NumPy `2.4.6`、Torch/Torchaudio `2.10.0`；两边 `pip check` 均通过，主环境无法导入 `pyannote.audio`，严格 preflight 的 22 项检查全部通过且 Pyannote import 明确来自 `executables.pyannotePython`。真实 VoxConverse 90 秒生产重跑在主进程不加载 Pyannote 的情况下完成，人数和 DER/JER/confusion/overlap 与隔离前 canonical 结果一致，证据位于 `.runtime_cache/sample-library/global/results/real-n2-pyannote-isolated-v2/quality-report.v1.json` 与 `.runtime_cache/outputs/global-sample-library/real-n2-pyannote-isolated-v2/`；冷启动流水线 RTF `1.057901456`、峰值 RAM `1311.78125 MB`，边界与复核量仍按独立未通过项处理。
- [ ] 复核或替换 `minds_en_us_034` 的 ASR 参考：当前参考只覆盖 joint-account 句而音频后半段仍有明显语音，必须先确认数据集漏标或 ASR 幻觉，未经确认不得用其 `WER/CER=1.5333` 调参或判定门禁失败。
- [ ] 在真实 `N=2` 达到分域阈值后扩展到真实 `N=3/5/8` 与派生 `N=13`，输出人数误差、DER/JER、confusion、overlap、边界、RTF、人工复核量和资源分桶；任一失败域不得由其他指标抵消。
- [ ] 在全球样本矩阵上完成自动人数、手动人数、hybrid 上下界、字幕三格式、真实视频 soft-mux/burn-in、Java PDF 与 Design Pack 全链路回归，并按语言、地区、人数和声学场景生成分桶报告。
