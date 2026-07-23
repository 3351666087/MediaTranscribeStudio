# MediaTranscribeStudio 重构任务清单

> 开始日期：2026-07-21
> 当前迭代：2026-07-23
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
- [ ] 通过真实未知扩展名、无扩展名、音频和视频探测验收。

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
- [ ] 重跑未知人数真实 M4A，并以人数稳定性、turn 粒度、speaker confusion、审查量和 RTF 分域验收。
- [ ] 重跑真实 MOV 的 auto 与 manual=5，并证明五人只是测试样例而不是上限。

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

- [x] Dynamic-N v7 使用 SHA-256 精确保留掩码；覆盖精确目标大小、human lock 全保留、10 次运行超过 5 个唯一掩码、输入置换不变和重复调用确定性。
- [x] Dynamic-N v7 恢复近声纹严格校正门槛，并验证 `N=8` 的稳定性、覆盖率、目标差和 `CLOSE_VOICE_RESIDUAL_COLLAPSE` 路径。
- [x] Dynamic-N v7 低预算搜索保留声学 leader/anchor，错误锚点场景直接形成 `12/13/14` 局部括号，资源不足场景区分自适应预算受限与真实资源截断。
- [x] Dynamic-N v7 缓存解析器拒绝旧版本、缺失字段、非有限数、频率/支持冲突和搜索审计冲突；正确缓存键上的畸形条目会重算而不是复用。
- [x] Dynamic-N v7 聚焦门禁：`91 passed`（两个 Dynamic-N 模块加生产流水线畸形聚类缓存重算测试）。
- [x] 独立复核 production CAM++/ERes2NetV2/pyannote 生命周期、OOM 和配置迁移切片；相关 Dynamic-N/production 门禁共 `193 passed, 48 subtests passed`，`compileall` 与定向 `git diff --check` 通过。
- [x] 独立复核并提交 Ultimate parity 原始字节、Git attributes/filter、RFC3339 授权时间顺序、manifest 决策字节 TOCTOU、隐藏 index flags 与授权合取加固切片；`49 passed`，manifest validation、`py_compile`、定向 `git diff --check` 通过，两轮修复后独立复审结论为“无阻断项”。
- [x] 本地通过 TypeScript 多任务 UI 的 lint、typecheck、unit/component 和 production build；`npm run lint`、`npm run typecheck`、`npm test -- --run`（31 files / 375 tests）与 `npm run build` 全部通过，并补充精确 job ID 切换、非当前任务取消、批量部分失败隔离和 Tauri/mock 严格运行时契约覆盖。
- [ ] 通过 Design Pack 结构、色彩、动效、字体、可访问性和 PDF 视觉门禁。
- [ ] 未知人数真实 `20260723_123047(1).m4a` 完成自动人数、角色分离、人工作证和分域性能验收。
- [ ] 真实 MOV 完成 auto 与 manual=5，对比人数稳定性、speaker confusion、审查量、RTF、RAM/VRAM 和缓存指标。
- [ ] 完成 Java-only PDF、SRT/WebVTT/ASS、soft-mux、burn-in 和代表帧视觉 QA。
- [ ] 原业务全部迁移、旧 Python UI/PDF/业务入口删除、最终 parity 通过后覆盖 `main` 并结束 `/goal`。
