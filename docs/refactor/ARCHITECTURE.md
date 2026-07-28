# 目标架构

> 审计日期：2026-07-28
> 状态：目标架构已经建立部分替代组件，但旧入口、旧流水线、旧 PDF 链路与固定五人约束仍未完成退役。

> 2026-07-25 模型架构更新：旗舰模型、专用挑战者、1600+ 语言长尾路径、托管前沿/私有服务器/M4 三档部署和晋级门禁见 [`FLAGSHIP_SPEECH_ARCHITECTURE.md`](./FLAGSHIP_SPEECH_ARCHITECTURE.md)。最新决策以该文件为准：Precision-2 是逐次授权上传时的托管 diarization 前沿候选，Community-1/VBx 是默认离线 Dynamic-N 主权威，Qwen3-ASR-1.7B 是支持集内主 ASR；CAM++、ERes2NetV2、Sortformer 和其他 ASR 均先作为挑战者或审计证据，只有 held-out 胜出后才能接管分桶。

## 设计目标

- 完全离线处理中文会议，不上传音频、逐字稿、说话人声纹或人工真值。
- 说话人数支持自动检测、手动指定和混合约束；目标契约接受任意正整数 `N`，不把五人会议或任何固定常量写成系统上限。
- ASR、时间边界、声纹、局部音频复核和上下文语义互相提供证据，但任何单一模型都不能无约束修改说话人或中文原文。
- 强制语义仲裁只在前级模型生成的不可变候选格中选择或重排人数、时间线、speaker、语言 span 和文本 candidate ID；候选格必须覆盖待修复域，LLM 不得自由发明候选。
- 采用“便宜证据先行、只对不确定片段升级”的效率级联；禁止默认对整段会议重复运行最昂贵模型。
- 说话人分离、人数估计、时间边界、ASR、语义建议、性能和 PDF 质量分别计量，禁止用单一综合分数掩盖任一硬失败。
- Python 只承担模型、音频、领域编排和 sidecar 调用，不再承担桌面 UI 或 PDF 渲染。
- React + TypeScript UI、Tauri 2/Rust 桌面壳、Python 后端、Java PDF sidecar 通过版本化 JSON/JSONL 契约解耦。
- 旧实现只有在新链路通过等价性、完整性、动态说话人数、离线性和回滚验证后才能删除。

## 组件图

```mermaid
flowchart LR
  UI["React + TypeScript UI"] <-->|"typed events"| TAURI["Tauri 2 / Rust"]
  TAURI <-->|"JSONL IPC v1"| API["Python backend API"]
  API --> PIPE["Meeting pipeline"]
  PIPE --> CONTENT["VAD + lexical speech gate"]
  PIPE --> DIAR["Community-1 segmentation + VBx Dynamic-N"]
  DIAR --> FUSION["Speaker count / overlap evidence fusion"]
  PIPE --> LID["Windowed multilingual LID"]
  LID --> ROUTER["Language + uncertainty router"]
  ROUTER --> ASR["Qwen3-ASR-1.7B primary"]
  ROUTER --> SPECIALIST["Whisper / Parakeet / Canary challengers"]
  ROUTER --> OMNI["Omnilingual 7B long-tail fallback"]
  ASR --> ALIGN["Forced alignment / segment fallback"]
  SPECIALIST --> ALIGN
  OMNI --> ALIGN
  PIPE --> VERIFY["CAM++ / ERes2NetV2 audit evidence"]
  PIPE --> REVIEW["Local audio review"]
  CONTENT --> LATTICE["Immutable cross-model candidate lattice"]
  DECODER --> LATTICE
  ALIGN --> LATTICE
  VERIFY --> LATTICE
  PIPE --> LLM["Mandatory local semantic arbitrator"]
  LATTICE --> LLM
  FUSION --> DECODER["Dynamic-N constrained decoder"]
  REVIEW --> DECODER
  LLM --> VALIDATOR["Candidate-ID validator + deterministic composer"]
  VALIDATOR --> DOC["Versioned transcript document"]
  DOC --> PDF["Java PDF sidecar"]
  PDF --> HTML["Offline XHTML"]
  PDF --> OUTPUT["PDF + page PNGs + contact sheet"]
  OUTPUT --> QA["PDF structure and visual QA"]
  QA --> REPORT["quality report + repairQueue"]
  REPORT --> UI
```

## 动态说话人数契约

### 模式

`speakerCountMode` 必须是以下之一：

- `auto`：由声纹聚类和全局解码估计人数。
- `manual`：用户指定任意正整数 `N`。
- `hybrid`：自动估计结合用户先验、`minSpeakers` 和 `maxSpeakers`。

`N` 的领域约束为任意正整数。产品可以基于设备能力声明可解释的单次作业资源预算，但该预算不是产品人数语义上限；资源不足时必须给出可恢复错误或转入人工分批处理，禁止静默截断、合并角色或退化为固定人数。

### 规范化标识

- canonical speaker ID 必须连续：`speaker-1`、`speaker-2`、…、`speaker-N`。
- `speakerCount`、speaker ID 集、CAM++ score vector、speaker profile 和 segment 映射的基数必须一致。
- 手动模式必须满足 `speakerCount == N`。
- 混合模式必须满足 `minSpeakers <= N <= maxSpeakers`。
- 自动模式无法可靠确定 `N` 时必须进入人工复核，禁止静默猜人数。
- 五人真实会议与合成 fixture 只作为 `N=5` 回归样例；最小回归矩阵至少覆盖 `N=1/2/5/8/13`。

### 证据与全局解码

每个候选发言段保留：

1. CAM++ 到全部 `N` 个全局 speaker centroid 的相似度和 margin。
2. diarization 后验、FunASR 时间边界和 VAD 置信度。
3. 当前 turn 与相邻 turn 的时间、重叠、停顿和不能自相重叠约束。
4. 问句—回答、称谓、角色职责、话题连续性和自我指称特征。
5. 局部音频重跑、人工试听和人工锁定的复核结果。
6. 语义模型的结构化建议、证据代码和适用范围。

全局解码必须：

- 输出与确定后的 `N` 完全一致的角色集合。
- 尊重人工锁定，不得被自动流程覆盖。
- 尊重高 CAM++ margin；没有更强声学或人工证据时不得仅凭语言风格重分配。
- 将低置信度、冲突证据、人数不确定和 overlap 候选送入 review queue。
- 对多人串话拆分验证父子时间边界包含关系、最小时长和文本来源。
- 在任何 cardinality invariant 失败时 fail-closed，不生成“看似成功”的最终报告。

## 强制语义候选格

本地 LLM 是最终质量链的必经仲裁器，但它不是声学模型，也不是无约束文本生成器。前级模型必须先构建哈希绑定的不可变候选格：

1. `speech disposition`：可转写人声或无可转写人声候选及 VAD/lexical evidence。
2. `speaker cardinality/timeline`：Community-1/VBx、MOSS、受控 Pyannote/CAM++ 投影和其他已登记挑战者给出的完整人数与时间线 candidate。
3. `speaker assignment`：每个稳定 turn 的 acoustic top-K canonical speaker candidate。
4. `language span`：支持集语言、开放集 `und`、逐段/逐词语言和代码切换边界 candidate。
5. `text`：provider 原生 N-best、token 时间、强制对齐和术语证据；没有真实候选时禁止内容词修复。

`semantic-candidate-lattice.v1` 已实现上述五域的统一表示、源媒体/transcript/model revision/payload/candidate/group/lattice 多层 SHA-256 绑定、运行时全量重建和 `available / partial / candidate-domain-unavailable` 状态；`semantic-candidate-lattice-v9` 会把有界候选摘要送入旧逐段强制语义请求，并把完整格绑定到兼容工件。旧 `v8` 工件只读兼容，但未绑定的旧 N-best 即使自报可修改，也不再能授权内容词变化。

LLM 对**可验证候选空间**拥有完整终态仲裁权限：可选择人声 disposition、人数/整段时间线、turn split/merge、角色归属、语言 span、代码切换边界和文本候选，也可要求针对缺失域执行有界局部重算或挑战模型补候选；不得用固定的“禁止改人数/边界/语言”规则削弱语义校准。硬边界只保护源媒体、原始 ASR、人工锁和证据身份：LLM 只能输出 candidate ID、排序、abstention、补候选请求和证据引用，不能把未执行的模型结果或自由生成的 speaker、语言、文本、时间点伪装成已有证据。确定性 composer 重算全部哈希后组合终态；一个域只有单一候选时必须触发候选生成或明确 `candidate-domain-unavailable`，不能声称该域已由语义层修复。任何自动应用权限只由同一版本组合在统一后语义 held-out 的各硬域结果授予，不由单条禁止规则或模型自报 confidence 决定。

`semantic-job-arbitration.v1` 现已把高权限落实为完整 job 协议：每个候选组必须恰好执行一次“精确排序全部 eligible candidate ID”或“请求域匹配的有界 challenger”，空域也必须请求补算；候选 ID、group/domain/scope、evidence ref、request kind、状态和计数均由运行时重建。`semantic-composition.v1` 只在补算请求为零时执行，重算以所选 candidate 为 current 的新格，并生成隔离的 speech disposition、完整人数/时间线、逐段角色、语言和 N-best 文本终态；完整时间线可以改变人数并携带 split/merge，跨域语言/文本冲突、时间线不支持的角色、人工锁变化和任意哈希篡改均 fail closed。旧逐段 suggestion runner 仍只读兼容，不再代表高权限终态。

生产 worker 现已接入可恢复的有界组合循环：每个 transcript hash 使用独立目录原子持久化初始 lattice、逐轮 arbitration、candidate generation、扩展 lattice 和 composition，重复启动必须逐件验证后复用，最多三轮仍未可组合就 fail closed。配置了本地 Community-1 的生产图会注册 canonical voice-activity、全媒体 Pyannote regular/exclusive 时间线、时间线主导的 speaker assignment、同一规范音频上的 Qwen3-ASR re-decode/LID 和 provider candidate-set；没有独立 diarizer 时不伪造时间线 challenger，也不启用高权限组合。`final-adjudicated-transcript 1.2` 直接绑定最终 lattice/arbitration/composition，并把被选中的完整时间线、角色、语言和文本投影给翻译、字幕、报告和发布，持久化源 transcript 仍不可变；人工复核后必须按新 transcript hash 重新运行组合。

仍未完成的是 MOSS 等更多生产 timeline handler、真实生产配置下的新闭环复跑、仲裁吞吐优化和广泛 held-out 晋级。AISHELL-4 `N=5` 的单例真实结果只证明文本域小幅改善，speaker 指标不变；生产接入本身不构成任意人数、任意语言或单人/多人代码切换质量证明。

候选格和 LLM 组合只以强制语义后的完整 `speaker + language span + time + finalText` 终态晋级。前级模型指标用于候选召回、路由和诊断，不单独决定发布；终态人数、DER/JER、边界、cp/tcp/SA-WER/CER、语言/切换、overlap、事实、复核量和资源域仍不可互相抵消。

## 效率级联

生产链按成本由低到高升级，所有中间产物以输入 hash、模型版本、参数和契约版本为缓存键：

1. **一次性预处理**：音频解码、重采样、VAD、Community-1 segmentation/overlap/turn proposals 和基础特征只生成一次；失败恢复不得无理由重算已验证产物。
2. **全量旗舰通道**：Community-1/VBx 生成整段 regular/exclusive Dynamic-N 时间线；Qwen3-ASR-1.7B 对支持集内候选段完成基础转写。两者按 stage 单次加载并批内复用，不调用本地 LLM，也不默认全场运行所有挑战模型。`modelResidency=stage` 面向统一内存边缘机，`worker` 只面向已验证容量充足的服务器 worker。
3. **不确定性路由**：仅将语言、边界、overlap、人数后验、短片段、离群 embedding 或模型冲突片段送入更高成本复核。
4. **定向重算与挑战者验证**：只对入队片段执行局部重分段、Whisper/Parakeet/Canary/Omnilingual ASR 候选、CAM++/ERes2NetV2 声纹审计或上下文扩大；挑战者只有在目标分桶 held-out 晋级后才能接管主结果。
5. **强制语义仲裁**：每个有人声作业都经过本地 LLM；低风险段可以批量 abstain，高风险段在不可变候选格中选择或重排 candidate ID。人数、边界、语言或文本候选缺失时先按域触发有界挑战模型、局部重算或上下文扩展，穷尽已登记生成器后才进入最小人工复核；不能用自由生成冒充未运行的候选。
6. **人工复核**：只展示仍有冲突或高影响的最小证据包；人工锁定结果进入后续增量解码，避免全局无差别返工。
7. **增量报告**：只有版本化 transcript document 通过硬门槛后才调用 Java sidecar；内容未变时复用已验证 artifact，内容变更时只重建受影响报告版本。

每一级必须记录触发原因、输入范围、耗时、缓存命中、CPU/GPU/显存峰值、输出置信度和退出原因。调度器必须具备有界队列、背压、取消、恢复和设备感知并发；超预算时 fail-closed，不得以降低说话人数或覆盖人工结果换取完成。

## 分离指标与验收

所有指标按层独立报告，并按会议时长、实际 `N`、overlap 比例、片段长度和设备分桶：

| 指标域 | 必须单独报告 | 不允许的替代 |
|---|---|---|
| 人数估计 | exact-count accuracy、count MAE、欠分/过分率、人工复核率 | 不得用 segment accuracy 代替人数正确性 |
| 说话人分离 | DER、JER、speaker confusion、speaker attribution accuracy、overlap recall/precision | 不得用 ASR CER 或语义连贯度抵消错分角色 |
| 时间边界 | boundary MAE/F1、漏段率、重复覆盖率、父子 overlap 边界合法率 | 不得把“文本看起来完整”当作边界通过 |
| ASR | `rawText` CER、专名错误率、空段/幻觉率 | 不得用 `normalizedText` 或 `displayText` 回填后成绩冒充原始 ASR |
| 语义建议 | schema 通过率、建议 precision/recall、越界修改率、证据充分率、人工接受/拒绝率 | 不得把模型自报置信度当质量指标 |
| 系统效率 | RTF、阶段 p50/p95、峰值 VRAM/RAM、缓存命中率、升级片段率、重算音频占比 | 不得只报总耗时而隐藏昂贵阶段或失败重试 |
| 报告/PDF | segment/text/timestamp/speaker 完整性、字体与结构硬门槛、14 个 Design Pack 维度 | 美学分数不得抵消内容缺失或角色错误 |

删除旧架构或发布前，必须分别满足预先登记的各域阈值和硬门槛。可以展示只读总览，但不得把这些指标压成一个可相互抵消的“总质量分”。

## 本地小 LLM 生产门控

`qwen2.5:1.5b` 与 `qwen3.5:4b` 的正式本地基准结论均为 `reject_for_production`。其中 `qwen3.5:4b` 在 88 个脱敏样本上的最终契约有效率为 `0.784`、越界文本修改率为 `0.205`、auto-apply 候选回归率为 `0.135`；安全挑战契约有效率仅为 `0.625`，未达到预登记门槛，因此它已退出生产 allowlist 并从本机 Ollama 卸载，只保留不可变报告作为历史证据。`qwen3.5:9b` 已安装并成为默认高能力候选。当前 `semantic-candidate-state-v8` 在 AISHELL-4 `N=5`、Liva `en/sw N=3` 和 Liva `en/tl N=5` 三个有部分真值的真实开发诊断中合计 `25/25 abstain`、0 建议、0 应用，所有可评分终态指标均无变化；它没有产生回退，也没有证明能弥补基模。现有协议又禁止新 speaker、turn split/merge、边界变化和语言变化，因此在候选格扩展前结构上无法修复这些域。当前 9B 仍是生产必经但 fail-closed 的 semantic arbitrator，即**仅建议、永不自动应用**：

`v9` 已移除“这些域原则上不可修改”的架构限制，但没有伪造尚不存在的候选。三份真实 transcript 的候选格审计均通过 source/transcript/producer/payload/derived-state 篡改拒绝；每份只有 `speaker-assignment` 域可选，另外四域均为单候选。因此当前不重跑无解的 9B，而先补真实跨模型候选和 job-level composer。完成后是否自动应用只看统一后语义 held-out 的最终质量与回退，不永久绑定在 suggestion-only 策略上。

- 输出必须经过 JSON/schema 和 deterministic validator。
- 模型自报置信度不作为自动应用依据。
- 不得自由生成说话人、文本、语言或时间边界；语义仲裁只能选择受哈希绑定的候选 ID，并在尚未通过 held-out 前提交复核建议。
- 不得覆盖人工锁定或高 CAM++ margin。
- 普通中文原文不得由该模型自动改写；仅可提出标点、语气词、口吃和机械重复清理建议。
- 所有建议必须可拒绝、可追踪，并保留修改前后文本和理由。

正式报告同时证明 `qwen3.5:4b` 在关闭 thinking 后能稳定返回 JSON，但“JSON 可解析”不等于“修改安全”。其说话人字段和 turn 拆分字段本来就被输出 schema 禁止，因此该评测不能被解释为说话人分离准确率。说话人角色候选仍只由声学证据、全局约束、局部音频复核和人工锁定产生；LLM 只能对这些候选做语义重排。

只有重新基准达到预先登记的契约、安全、越界修改、风险升级和回归阈值后，才能重新讨论自动应用；完成 benchmark 本身不代表生产可行。

## 中文原文语义优化

系统保存三层文本：

- `rawText`：ASR 原始输出，只读。
- `normalizedText`：标点、断句、常见口误和确定性术语修正。
- `displayText`：经审核的中文逐字稿展示文本。

允许的改动：

- 修正有声学、术语表或人工证据支持的错字、同音字、专有名词和断句。
- 删除纯语气词、口吃和机械重复，但必须保留变更审计。
- 拆分有时间边界和音频证据支持的多人串话。

禁止的改动：

- 翻译。
- 文风润色、总结、补写、改变态度或改变事实强度。
- 添加原音频中不存在的事实。
- 用小 LLM 猜测不确定内容后直接覆盖原文。

## PDF 架构

```text
pdf-renderer/
  pom.xml
  src/main/java/
    app/              CLI 与作业编排
    contract/         JSON 输入输出
    render/           OpenHTMLtoPDF
    inspect/          PDFBox 结构检查
    qa/               硬门槛、14 维度、repairQueue
  src/main/resources/
    templates/
    fonts/
    schemas/
```

固定渲染依赖：

- `com.openhtmltopdf:openhtmltopdf-pdfbox:1.0.10`
- `org.apache.pdfbox:pdfbox:2.0.30`

Python 端只：

1. 写入版本化转写 JSON。
2. 启动 Java sidecar。
3. 消费结构化结果和进度事件。

当前 Java validator、XHTML 文案、PDF 元数据和部分测试仍写死五人，因此 Maven 测试全绿不能视为动态人数替代链已经完成。

## 内化的 PDF 质量循环

### 硬门槛

- PDF 可被 PDFBox 打开，页数大于零。
- 全部发言文本可提取，segment ID 和发言段数量与输入完全一致。
- 时间戳单调、有界、格式合法，并与输入毫秒值一致。
- 输入 speaker set 中全部 `N` 位角色均存在图例、标签和非颜色区分，且输出不得新增、漏掉或合并角色。
- 中文字体嵌入，不依赖远程字体。
- 没有空白页、裁切、内容越界或不可打印对象。
- 不含远程 URL、脚本、遥测或网络依赖。
- 每页 PNG、联系表和测试证据存在且 SHA-256 可验证。

任一硬门槛失败时，分数和视觉精致度不能抵消失败。

### 14 个美学维度

沿用全局 Design Pack 的稳定 ID：

1. `AESTHETIC-COHERENCE`
2. `AESTHETIC-DISTINCTION`
3. `AESTHETIC-REFINEMENT`
4. `AESTHETIC-PROPORTION`
5. `AESTHETIC-HIERARCHY`
6. `AESTHETIC-TYPOGRAPHY`
7. `AESTHETIC-COLOR-RELATIONSHIPS`
8. `AESTHETIC-RHYTHM`
9. `AESTHETIC-DENSITY`
10. `AESTHETIC-RESTRAINT`
11. `AESTHETIC-REAL-CONTENT-STRESS`
12. `AESTHETIC-FONT-FAILURE`
13. `AESTHETIC-IMAGE-FAILURE`
14. `AESTHETIC-SCRIPT-FAILURE`

### 循环规则

- 分数必须至少为 85。
- 第一轮也必须有本地结构测试和页面截图证据。
- 第二轮起必须重拍截图、重跑测试、完整重评分并检查更高优先级回归。
- `repairQueue` 按严重度、硬门槛、维度权重和稳定 ID 排序。
- 最多五轮；第五轮后仍不合格则状态为 `blocked`，不能宣称已通过。

## Design Pack 发布门控

- `frontend-design-pack-global/repository-index.json` 在本次复核时已经存在；此前记录的“文件缺失”是历史 blocker，不得继续写成当前事实。
- 系统仍必须把该索引缺失、不可解析或校验失败视为 fail-closed 条件。
- 当前全局包保持 `runtimeOfflineReady=true`、`implementationReady=false`。
- `EVIDENCE-OPEN-001`、`CONTENT-OPEN-001`、`ASSET-OPEN-001` 仍是 blocking open questions。
- 在 `implementationReady=true`、阻断问题关闭并生成项目专属证据前，禁止宣称 UI、PDF 或应用商业发布就绪。
