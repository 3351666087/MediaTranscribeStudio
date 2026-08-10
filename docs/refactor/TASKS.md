# MediaTranscribeStudio 当前产品交付清单

> 生效日期：2026-08-07
> 当前平台：Windows 11，RTX 3060 12 GB，约 32 GB RAM
> 仓库：`D:\MediaTranscribeStudio`，分支 `main`
> 大文件位置：模型、缓存、数据集和评测产物统一放在 `D:`
> 历史清单：`docs/refactor/TASKS_HISTORY_2026-08-01.md`

## 完成规则

- `[x]` 只表示该行已有可复算的本地证据；代码存在、模型能加载或单个样本成功都不能代替产品验收。
- 最终目标是完整产品结果：`声纹/说话人 + 转写 + 语义裁决 + 字幕 + OpenHTMLtoPDF 报告`。
- 自动协议、schema 和安全检查继续用于定位缺陷，但不再单独决定模型或产品是否达标。
- 最终质量由 Codex 在看不到参考答案和模型身份的盲审包上逐例裁决；解盲后再计算客观指标并记录分歧。
- development 用于发现和修复问题；冻结的 held-out 只在方案确定后运行，不把 held-out 结果回灌调参。
- 不用单一平均分掩盖严重缺陷。每例记录 `blocker / major / minor / pass`、理由、证据和建议动作。
- `production` 是可随时替换的部署角色，不享有保护；challenger 在真实产品盲审中胜出即可顶替，旧 production 降为 challenger/rollback。
- 胜出方案完成回滚验证前不卸载旧权重；最终交付前再用 held-out 确认当时的 production。
- 每完成一行立即补充证据并勾选，不在最后集中补勾。

## 0. 当前基线

- [x] 将仓库切到 `main` 并确认 Windows 工作副本位于 `D:\MediaTranscribeStudio`。
- [x] 盘点硬件：RTX 3060 12 GB、约 32 GB RAM；确认 WSL 15 GB 不是 Windows 原生推理上限。
- [x] 将 Ollama 模型目录固定到 `D:\models\ollama`，Windows 原生服务使用 `127.0.0.1:11434`。
- [x] 实测 Qwen3.6 27B Q4_K_M 可用 CPU/GPU 混合运行，峰值显存约 11.9 GB，并可显式释放。
- [x] 实测 Qwen3.6 35B-A3B Q4_K_M 可用 CPU/GPU 混合运行，峰值显存约 11.8 GB，并可显式释放。
- [x] 将旧的 416 行迭代清单迁入历史档案，当前文件成为唯一执行看板。
- [x] 修复 `run_real_semantic_acceptance.py` 的 Windows `resource` 崩溃，使用真实 `peak_wset` 或明确 unavailable；聚焦回归 `30 passed`。
- [x] 生成一次当前代码、环境、模型、数据和磁盘余量的统一基线报告并绑定 SHA-256；证据 `D:\mts-eval\baseline\product-development-baseline-20260809-v1.json`，file SHA-256 `c9829782b26e1d13b3c6a61dae6176bb2a2f536f64434b1474a284126b1d5a12`，canonical SHA-256 `22eee32835496fc554709781c71cbd93492bd403783f7d9911c5c5279630897d`。

## 1. 本地模型选型

- [x] 将盲审冠军 `qwen3.5:27b-q4_K_M`（digest `7653528ba5cba4dd8e19da24aaddc7f4d0b5ecd93571c0825dfd4137958ec06e`）固定为当前 production 默认；保留 `qwen3.5:9b` 作为可回滚 challenger，production 无保护期，可被证据更好的 challenger 随时顶替。
- [x] 下载并固定 `qwen3.6:27b-q4_K_M`，manifest SHA-256 为 `a50eda8ed977ab48a12431878896b27ffd5cef552c17af3317d9623b939a7f1e`。
- [x] 下载并固定 `qwen3.6:35b-a3b-q4_K_M`，manifest SHA-256 为 `07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522`。
- [x] 完成最新开放权重候选的一手来源调研表：版本日期、许可、架构、active parameters、上下文、Q4 体积和本机可运行性；证据见 `docs/refactor/LOCAL_MODEL_RESEARCH_2026-08-07.md`。
- [x] 将 Qwen3.6 27B/35B 作为 challenger 写入 `local-model-registry.json`；当前 9B role 暂未变更，但无晋级保护，registry 聚焦回归 `13 passed`。
- [x] 用相同 development 产品样本比较 Qwen3.5 9B/27B/35B、Qwen3.6 27B/35B 与 GLM-4.7-Flash；22 例覆盖 10 种语言，6 个候选均使用同一冻结 document/lattice。证据 `D:\mts-eval\semantic-model-comparison\multilingual22-six-model-blind-20260809-r1\benchmark.v1.json`，canonical SHA-256 `a742731ff2b44b3791036d665f459ce1a0b00f795f9e429333410cbc800322d4`。
- [x] 同批比较记录语义质量、严格结构化输出、冷/热调用延迟、RAM/VRAM、失败恢复与显式资源释放；6 个候选均执行完成、运行 digest 匹配且释放后不再驻留，指标仅作诊断，不替代人工裁决。证据同上，stable comparison SHA-256 `57044b5e5844b192d3c019f296129a22cc72fc86739ba773cbf0c02df1c1a33c`。
- [x] Codex 在模型身份封存的同批盲审中逐例裁决后确定唯一优胜者 `qwen3.5:27b-q4_K_M`（`12 pass / 3 minor / 7 major / 0 blocker`）；自动 accuracy 未参与排名。解盲比较 `D:\mts-eval\semantic-model-comparison\multilingual22-six-model-blind-20260809-r1\comparison\codex-semantic-model-comparison.v1.json`，file SHA-256 `89367afd34a998b5b1f7df98ee8b51577d403853bbf9f557bf6895ac799ea65e`，canonical SHA-256 `491d0bf75ee53d5d67ce8b1dc76f0fd267874d558f2fec380511e4e443f25f19`。
- [x] 用 v13 同一冻结 22 例、10 语言输入完成四个 30B 级候选的第二轮身份封存盲审；`22 x 4 = 88` 项均逐项覆盖，封存前未读取 identity vault。唯一优胜者仍为 `qwen3.5:27b-q4_K_M`（`13 pass / 6 minor / 3 major / 0 blocker`）；证据目录 `D:\mts-eval\semantic-model-comparison\multilingual22-four-30b-blind-20260810-v13-r2`，benchmark canonical SHA-256 `a84abad8f640cefd22ef2562ab498579c2e21e8adff8ac0accc37938c09bfc9c`，sealed review canonical SHA-256 `101de196eaa977a884322b3d909de356bbd69363fdd0e60946845e8a896b635a`，comparison canonical SHA-256 `cc873de474dcdde7db9b28d5a8b37e9c05868ec9dde0798b19c77d2e52671aa2`。
- [x] 将 v13 的 3 个 major 固化为可见 development 回归，并用真实 `qwen3.5:27b-q4_K_M`、`semantic-job-candidate-arbitration-v14`、`multilingual-fidelity-v2` 复跑；阿语仅请求相邻两段 ASR N-best，法语仅请求时间线与 `9/9` 个说话人归属，中文仅请求时间线与 `3/3` 个说话人归属，三例均无异常且请求集合精确。Codex 另行逐段复核语义、时间连续性和请求域，不把 exact-match 当最终判断，也不以文字连续性代替声学身份结论。运行报告 `D:\mts-eval\semantic-model-comparison\semantic-v14-three-fixture-qwen35-27b-real-20260810-r2\run-report.json` SHA-256 `3660a1e8d324c4eff2834e855af0f6ea3714f69c276090ab7bfd7381cbc4fad1`；人工复核 `codex-post-run-semantic-audit.v1.json` SHA-256 `7579839b41f06b279d96c3fe42c4fa19cb7f578a07b3ec364bb6f68478a8e186`；聚焦回归 `154 passed, 1 skipped`。
- [x] 用收紧后的 `semantic-job-candidate-arbitration-v15` 在同一 6 条可见 development major 回归上重跑真实 `qwen3.5:27b-q4_K_M`（digest `7653528ba5cba4dd8e19da24aaddc7f4d0b5ecd93571c0825dfd4137958ec06e`）；pipeline 请求集合 `6/6` 精确，模型原生 `3/6` 精确，结构 guard 只补齐 10 个局部 speaker-assignment 声学证据缺口。Codex 逐例复核原始响应、最终请求域和 guard 增量后判为 `targeted-regression-pass`，但明确不授权晋级，也不代替完整 22 例竞争或产品链验收。证据目录 `D:\mts-eval\semantic-model-comparison\semantic-v15-six-fixture-qwen35-27b-real-20260810-r3`；运行报告/人工审计 file SHA-256 为 `147820116b44281ca7d58c3618c85846ac5e82183bad6d7bab267417fffd5160` / `515097b8372fce4d247503937fb1e38cd06e95786bd0a4784696c8102aadda54`。
- [x] 用同一 v15 六例 development 筛选固定 `gemma4:26b`（digest `5571076f3d70050487b26b341705799e0ab29b808164f90d20d4cf84f699d251`）；模型六例均未请求任何 bounded candidate evidence，pipeline/model-native 均为 `0/6` 精确。Codex 审计结论为 `development-fail`，因此在进入完整 22 例前淘汰且未获晋级授权；运行目录 `D:\mts-eval\semantic-model-comparison\semantic-v15-six-fixture-gemma4-26b-real-20260810-r2`，运行报告/人工审计 file SHA-256 为 `f78638a31779fe0efbcc6ed6f93d5e05fbac19c297c3cbbdb7e71c781cfcf4bc` / `9d87c8521c131f245c804c71b51570aa811611b72e382b027f2085779464ef10`。
- [x] 完成 `gemma4:31b` 新 challenger 的官方身份、本机可运行性与同协议 v16 development 筛选：Google HF revision `842da3794eaa0b77d5f08bae87a17459d91ff475` 报告 Apache-2.0，Ollama manifest digest `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`、Q4_K_M 模型层 `19,868,969,920` bytes；Windows 原生 Ollama 以约 `9.05 GB` 模型显存加主机内存混合驻留通过 8K 严格 JSON 探针并可显式释放。相同 runner/code、六例、可见 document/lattice、初始 prompt/schema 和推理参数下，Codex 语义审计为 Gemma `0/6`、当前冠军 `qwen3.5:27b-q4_K_M` `5/6`；Gemma 因零 evidence request 在进入冻结 22 例前淘汰，不替换 production，Qwen 的印尼语 `bus kruaster` ASR N-best 漏请求仍未闭环。本项不属于 held-out 或最终验收；challenge receipt、Gemma run/audit、Qwen run/audit file SHA-256 依次为 `63cb23275f97e7d0dbc3f1403e68fff3faac308b91a1ac02aaf0a9cf60d1ffb6`、`2fcf16d545091521611ca44b481a8505f7b9b59c2864f75cf9cc02879985e768` / `cbca5fe7e32d4b810b26245d3fc1ffe7e521a4d476432f0b1be899ede927d4f3`、`f667c3b9c48aabcd5fa3b24e415e2e60600bd5d8f40057699883ef49e67d64e3` / `8f72cebe5f2ed3299a8141099841e2b1e68e6e7ea7974a295c06410d326e2178`。
- [x] 用同一 v15 六例 development 筛选固定 `magistral:24b`（digest `27bcbbf6d32417d3a8a12d5eb9bd55fda6e5591807e2b02aaaf4423899925afc`）；pipeline `2/6`、模型原生 `1/6` 精确，阿语/日语漏 ASR、印尼语漏 lexical ASR，西语误请求 speaker-assignment。Codex 审计结论为 `development-fail`，因此在进入完整 22 例前淘汰且不替换 production；运行目录 `D:\mts-eval\semantic-model-comparison\semantic-v15-six-fixture-magistral-24b-real-20260810-r1`，运行报告/人工审计 file SHA-256 为 `50bee6f96ea71202479c7d8c7cb3412367edf807af1dd1991359281f26594753` / `4e0c6dde7c7f8e1b7f71a618db2d9da709ed35443180ca46fa8b5505db2c8b5a`。
- [ ] 用冻结 held-out 复核唯一优胜者，不通过则保持现有生产默认并继续迭代。
- [x] 27B 盲审胜出后以 CAS 立即替换 production，保留 9B rollback，并完成真实回滚和再晋级；production 不设 margin、冷却期或保护期。收据位于 `D:\mts-eval\semantic-model-comparison\multilingual22-six-model-blind-20260809-r1\promotion`；当前 production config SHA-256 `1ac8816eb9dcb3b4167b4506df8d77003d2f76cc951e23e886a26f1c8bdb49aa`，rollback config SHA-256 `c8ce354b30a937c8fe7dd3c8748de551a1c5f22ef62aec8f7b4cbb0fe1e03ea7`。

## 2. 真实样本矩阵

- [x] 冻结无真值泄漏的中文语义 development/held-out 清单：96 例，72/24，四类和两语料均衡。
- [x] 保留 AliMeeting、AISHELL-4、AMI、VoxConverse 和现有全球样本库的真实音频及来源证据。
- [x] 为 CN-Celeb v2 固定官方 OpenSLR 来源、CC BY-SA 4.0、长度、ETag 和完整性要求。
- [x] 完成 CN-Celeb v2 的 512 MiB 独立分块下载、逐块 SHA-256、拼接、gzip/tar 全扫描和安全解包；证据 `D:\mts-eval\source-archives\CN-Celeb\cn-celeb_v2.archive-evidence.json`，archive SHA-256 `d6b3986fd23f613865a978f44170585b055646b1710814cf3091f7ca6492285f`，canonical SHA-256 `9d7d6c497babfa6558ba5b57f072a9ed8b7e9ae03014e2d1bf48594535e3882a`。
- [x] 冻结 CN-Celeb speaker-disjoint development/held-out 清单，阈值只在 development 拟合；`development.trials.json` canonical SHA-256 `cc2043df3f06e4208748dbfc3aea4fee77c93b551edd4f3543d22701c269bb6c`，`held-out.trials.json` canonical SHA-256 `b23c01c77fb2dd67bb832ac7b2028d2d35d8c94004e45eda43e10a675c3341a3`。
- [x] 冻结 FLEURS 多语言 development/held-out 样本 `36/36`，源 revision、viewer/parquet 回退、音频哈希和 split 均可复算，且无跨 split 音频泄漏；证据 `D:\mts-eval\fleurs-multilingual-frozen-20260810\fleurs-multilingual-frozen.v1.json`，file SHA-256 `30325f83451be94a82a49263ed2a59f93473b874749235ff7a287096283c021c`，canonical SHA-256 `712f0e69929b8b05bd2dd14160857b7b3dab06c91f6ba0b179417430c2447a82`。
- [x] 冻结真实 VoxConverse `N=13` development/held-out 证据，11 个 case、失败 `0`；development 取 300 秒且 13 人最短发言 `4.44 秒`，held-out 取 120 秒且最短发言 `0.84 秒`，不保存 transcript 或 speaker identity。证据 `D:\mts-eval\voxconverse-real-diarization-v3-n13\freeze-receipt.v1.json`，file SHA-256 `e3c11a81688460a8be6176ab975ac190b247b847bbc6cc141460c15f3da12ddf`，canonical SHA-256 `c1d9e444d4855c3400719e438ac852b0dffb2843770dfa22aba68d4f1239d37a`。
- [x] 冻结真实静音场景的 development/held-out：固定 Wikimedia Commons TU Delft 安静自习室现场录音页面 revision `751123960`、CC BY 3.0 和源资产 SHA-256 `047c704947a9a9d96486b0d32b13665edf8e781b9d12c886df042daf3f0cff91`，选取不重叠的 `125–140 秒` / `195–210 秒`；两段经频谱人工复核且 FSMN-VAD 均返回空语音区间，派生媒体 SHA-256 跨 split 不重复。冻结 manifest `D:\mts-eval\voice-activity-v2\voice-activity-samples.resolved.v1.json` file SHA-256 `5070f423d5656bd8bf437784da18f8a205834377b80dc57516a81d12bb9dc4f3`；r10 矩阵已确认 silence `1/1` 且 `crossSplitDisjoint=true`。
- [x] 冻结真实纯音乐 development/held-out：development 使用美国海军乐队演奏的 `Amar Sonar Bangla` 公共领域器乐录音（Commons revision `910318431`，源资产 SHA-256 `44c9f87c60d07db92ad31e5e40c972012e041caf0ad2d886d2a94f55c539c6b5`），held-out 保留 P.S. Mukherjee 公共领域 Hindustani 器乐录音；r10 矩阵已确认 music `1/1`、跨 split 媒体 SHA-256 不重复。矩阵 `D:\mts-eval\product-matrix-v1\truth-redacted-product-matrix.r10.v1.json` 共 125 个 case、19/22 分桶完整，file/canonical SHA-256 为 `ddf98e8c7a029a3a13b2262f29f209ccb472b8a8834612ea76ebc68ada0e521a` / `f416d3c0557ef5a6f03b59edf86dacfa59b1a0ad01678cf9651bae62d0096c9a`；相关聚焦回归 `9 passed`，总体矩阵仍因普通话 development、远场 development 和长媒体 held-out 三项缺口保持未勾选。
- [x] 冻结独立真实长媒体 held-out：Wikimedia Commons `Wikimania 2006 press conference` 多人现场录音长 `1860.622667 秒`，页面 revision `1151277181`、CC BY 2.5、源 SHA-256 `a6ef986e061c7cf8faad1d2282f952aa2ef94a2b8df98176e8cd07adbf46da1a`；8 个固定 60 秒窗口覆盖开头、中段、结尾、2 个 seeded-active-random 和 3 个 acoustic-change，选择不使用模型分数。resolved manifest `D:\mts-eval\long-media-wikimania2006-heldout-20260810\resolved\long-media-samples.resolved.v1.json` SHA-256 `e86a05ba73fc2bfc7ae1b560ed30ea4807e0d5f602d6d8a299894fbe8518854b`；与 JFK development 跨 split 源媒体不重复。r14 产品矩阵共 20/22 分桶完整，file/canonical SHA-256 `c71eade054811755b089b3b5b02c4065a79d53eae354f4b6097c0b2a8587b16d` / `6ad1d18c3c9f4d026c706bb04cbeb85a81f119251da0ff20fc2d54f445053145`，只剩普通话和远场 development 两项缺口；聚焦回归 `16 passed`。该 held-out 无 RTTM，不冒充说话人变化或重叠真值。
- [x] 补齐真实普通话与远场 development：官方 OpenSLR SLR119 `Train_Ali_far` 归档 `78,639,309,701` bytes、147 个 512 MiB 分块、archive SHA-256 `98a5b2840704c7fb2ab3fdc7e5c63c25bd453528e0e32443938279bae8e200e7`、ETag `"B5C1C3F463D4D0393241F7A11C3E303F-7500"`、CRC64 `7416259116226311466`；固定 `R0003_M0046` 的官方 TextGrid 和 8-channel far WAV，生成 1 个 `N=4`、`10.08 秒`、4 个真实重叠区间的 development case。freeze manifest `D:\mts-eval\alimeeting-train-development-v1\global-real-diarization.resolved.v1.json` file/canonical SHA-256 `9eb4945ceaddcf7b1e7fac5d824a28a1e077fe1690f60e973b0d9e05e15786f1` / `e3b064a43ae9aed6240982f8f07b508e78f5d75ade14d1a3166b3d3a24f6a900`；与 Eval held-out 的 archive、session、原始音频和派生媒体四层隔离均为 `true`。r15 产品矩阵 `D:\mts-eval\product-matrix-v1\truth-redacted-product-matrix.r15.v1.json` 共 134 cases、22/22 分桶完整、0 gap，file/canonical SHA-256 `6366b9ea40b33c5c7dcc33bda8481920bb99012f3a5a26873947213f9bab8e49` / `9e80fc7ddd72b3675e874194a6140958f34bf26dc4c6b68d5d04a63826c44288`；相关聚焦回归 `46 passed`。该证据只证明样本矩阵与录音隔离，不替代声纹、ASR、语义或最终产品质量验收。
- [ ] 建立产品矩阵：普通话、粤语、英语、西语、日语、韩语、阿拉伯语、印地语及真实代码切换。
- [ ] 覆盖 `N=1/2/3/5/8/13`，并加入真实重叠、远场、电话、噪声、静音、音乐和长媒体。
- [ ] 每个分桶至少包含可分离的 development 与 held-out；记录许可、说话人/录音隔离和参考真值来源。
- [ ] 缺少合法真实样本的分桶先补公开许可样本；合成样本只作压力测试，不冒充真实质量证据。
- [ ] 为每个样本冻结源媒体 SHA-256、期望人数、语言、时间轴/文字真值可用性和允许的用途。

## 3. 声纹与说话人识别

- [x] 在 AMI development 上比较 ReDimNet2-B6-LM 与 W2V-BERT2；ReDimNet2 当前 EER 3.125%，优于 5.46875%。
- [x] 保留 CAM++、ERes2NetV2 small/wide 与 ReDimNet2-B6-LM 作为中文/跨域候选。
- [x] 在 CN-Celeb development 上同协议比较 CAM++、ERes2NetV2 small/wide 和 ReDimNet2；四份报告位于 `D:\mts-eval\gates\CN-Celeb-v2\reports`，file SHA-256 依次为 `8f4c826e0b979cc7b3b64f2155aa955feb7edf9e18e0a678a9a9a38ee38e8164`、`0ecce52ba6de91fe0fcfcc76234001886096b4442780f1447e889ff6b217ff87`、`f9c86eb55b566c6f8fa8d3d8122a35c906353a33fe7470f0d155c6d0d4ec47e0`、`b3bf3b4c15a021c19d4622f5c163a4ce67d400d85a4d624ae7805163494b9a94`。
- [x] 冻结每个候选的阈值、校准器、模型 revision、预处理和 embedding SHA-256；四候选（CAM++、ERes2NetV2 small/wide、WeSpeaker ReDimNet2）的完整冻结证据汇总于 `D:\mts-eval\gates\CN-Celeb-v2\reports\speaker-final-acceptance-v4.json`，file SHA-256 `98908d009326436f398ac487cedb19bc183fb1d14e6207bafb577cf8a48accb7`，canonical SHA-256 `0153b48e5b23b44f31625471cf2ffd9a83478dc0cd561ba8bed76e534c33b5e8`；`winnerEvidence.remainingFreezeGaps=[]`。
- [x] 在 speaker-disjoint held-out 上报告 EER、minDCF、AUC、FAR/FRR、校准误差和分桶结果；同一 v4 报告覆盖四候选、768 trials、same/cross-genre 与 32 个 genre-pair 分桶，胜者 `eres2netv2-wide` 的 EER/minDCF/AUC/FAR/FRR/Brier/ECE10 分别为 `0.0546875/0.002760416666666667/0.9910007052951388/0.08333333333333333/0.0390625/0.03950270849158043/0.029797319643014217`。
- [ ] 在完整 diarization 链路报告人数、DER/JER、speaker confusion、overlap 漏说话人、边界和复核量。
- [ ] 由 Codex 试听盲化的争议片段并裁决同人/异人、角色连续性和明显的 split/merge 错误。
- [ ] 将胜出声纹模型接入生产难例路由，保留已验证的回滚模型和人工锁优先级。

## 4. 端到端产品运行

- [x] 用冻结 ASCEND `ascend-held-test-00919` 完成一次 production 27B/v13 真实 held-out 全链：语义两轮完成，两个公开 review item 由 Codex 明确接受后队列归零，干净 `job.completed`，生成并发布 PDF/TXT/JSON/SRT/WebVTT/ASS，源音频哈希不变。独立 verifier 复算 30 个文件、PDF `97.87/100`、13 个硬门禁全部通过；证据 `D:\mts-eval\product-runs\semantic-heldout-ascend6-20260809\winner-full-chain-00919-manual-v13-r10\verification\product-full-chain-verification.v1.json`，file SHA-256 `34be31764b02c1f421ebe7c1d8e5a0bcd53254f23c2b50530b60dbe7cbca94db`，canonical SHA-256 `8348f14dc440d38c65a740aa94b37a3bda32ee9a79cd7cfef7d456e897285cf8`。本项只证明该冻结 case 的完整产品链，不代替剩余 held-out 矩阵。
- [x] 修复真实 r9 暴露的媒体/输出配方晚失败：可信 media probe 判定音频后立即预编译输出配方，`audio-only + burn-in` 现在在 transcription、27B 语义和 review 前 fail fast，保持原错误码；有效视频配方不回归，相关聚焦回归 `45 passed`。
- [ ] 固定一个可重复命令，从本地媒体一次生成 transcript、语义结果、SRT/WebVTT/ASS、视频写回和 PDF。
- [ ] 先跑小型分层 development 批次，覆盖每种语言、人数和声学场景，确认所有产物链都能完成。
- [ ] 再跑大规模 development 批次，所有候选使用相同源媒体、切分、声学证据和输出设置。
- [ ] 长媒体必须覆盖开头、中段、结尾、随机有声段、说话人变化点和重叠片段。
- [ ] 字幕逐例检查时间同步、断句、阅读速度、安全区、角色色、字体、软封装和烧录一致性。
- [ ] PDF 只由 OpenHTMLtoPDF + PDFBox 生成和检查；逐页验证中文字体、裁切、重叠、空白、目录和元数据。
- [ ] 每个失败可重试、取消和恢复；失败不得覆盖源媒体、人工锁或已经验证的客户产物。
- [ ] 记录冷/热 RTF、阶段 p50/p95、峰值 RAM/VRAM、缓存命中、升级率和重算比例。

## 5. Codex 盲审与语义裁决

- [x] 生成不含模型名、参考答案和自动评分的语义模型盲审包；候选顺序随机且保持可复算映射。v13 包的 reviewer packet set SHA-256 为 `cfddccc595fd956ecd52e5e90b769e429f373a4e3103cde6cb031df409f476cc`，封存时 `identityVaultRead=false`。
- [ ] 为每例同时展示可播放音频、说话人时间轴、原始 ASR、候选终稿、字幕截图和 PDF 页面。
- [ ] Codex 逐例判断：谁在说、内容是否忠实、语义修正是否必要、是否引入幻觉、字幕和报告是否可交付。
- [x] v13 语义盲审的 88 项候选均记录 `blocker/major/minor/pass`、自然语言理由、定位时间和建议动作，不使用模型自报 confidence；sealed review file/canonical SHA-256 为 `47f8c85316efce32e96ef50b90e7ea6f2a6b5036fe5e5ab19f2d24765b7c65bb` / `101de196eaa977a884322b3d909de356bbd69363fdd0e60946845e8a896b635a`。
- [x] v13 解盲后按严重缺陷和人工 preferred share 比较四个 30B 级候选，自动指标不参与排名；`qwen3.5:27b-q4_K_M` 为唯一胜出者，comparison canonical SHA-256 `cc873de474dcdde7db9b28d5a8b37e9c05868ec9dde0798b19c77d2e52671aa2`。
- [ ] 把所有 blocker/major 缺陷转成可复现回归样本或测试，再修改生产实现。
- [ ] 修复后重新生成盲审包；旧裁决保留，不覆盖失败历史。
- [ ] 只有 Codex 盲审和客观证据均支持时才登记“该分桶达标”；不外推到未测语言或人数。

## 6. 产品缺陷闭环

- [x] 将 v13 四模型盲审中的全部 blocker/major 汇总为 14 条可定位缺陷（`2 blocker / 12 major`；ASR lexical 6、language/ASR scope 3、speaker continuity 5），并为冠军的 3 条 major 绑定真实 case 与推荐回归入口；审计 `D:\mts-eval\semantic-model-comparison\multilingual22-four-30b-blind-20260810-v13-r2\review\v13-major-blocker-defect-audit.v3.json`，file/canonical SHA-256 `a6a2905105a43b9ff4ef192fdac5ac12eb4863c5544cee59009c4e4a16a3673d` / `4b5424dacb01257a2eb14b97631c88534a6f298374da9b4fb24ef1347a4b5dd3`。此项只完成缺陷盘点，不表示 14 条均已修复。
- [ ] 修复真实样本暴露的说话人数估计、split/merge、overlap 和角色漂移问题。
- [ ] 修复真实 ASR/语义暴露的漏字、幻觉、专名/数字破坏、代码切换和跨段上下文问题。
- [ ] 修复字幕的同步、换行、可读性、角色色、字体和视频兼容问题。
- [ ] 修复 PDF 的字体、版式、分页、元数据、repairQueue 和动态人数问题。
- [ ] 完成桌面端项目创建、批处理、进度、试听、人工纠错、证据浏览和失败恢复工作流。
- [ ] 每个修复先通过聚焦测试，再重跑对应真实 development 分桶和盲审。
- [ ] 清除所有仍影响 Windows 产品路径的固定五人、固定单语言和旧入口依赖。

## 7. 最终交付

- [x] 以 Tauri 原生路径替代 macOS 的旧 PyInstaller 默认打包入口：支持 `x86_64`、`aarch64`、universal2 的 `.app`/DMG 与 development-only 确定性 ZIP、无权重 runtime overlay、用户目录初始化、模型管理、完整性 manifest 与显式 unsigned-development 模式；独立 verifier 会重算全账本、按 `Info.plist` 验证主 Mach-O、要求 universal2 同时含 `arm64`/`x86_64`、安全解 ZIP/transport tar、只读挂载 DMG、复核 app/runtime/bootstrap/LaunchServices 启动，并支持 Developer ID 与显式 App+DMG notarization/stapling。GitHub workflow 会在同架构 runner 下载 tar+SHA-256 后再次校验 checksum、安全解包、原生信任、bootstrap 和 LaunchServices；cross-host plan 必须随提交重新生成，禁止把输入已漂移的旧 plan 称为当前证据。2026-08-10 r7 审计 plan `D:\mts-release\macos\macos-release-plan-universal2-r7.json` 为 100 个运行时文件、`37,371,072` bytes、无模型权重，SHA-256 `1b67594982971582f8f1bc57ab468b0eb1d253c23a1d18633fe7c87d6154eb34`。本项只证明实现与交叉主机计划，不冒充已生成或下载验证了 Mach-O/DMG。
- [ ] 在 Apple 原生 runner 执行 universal2 候选构建，验证 `.app`/DMG 安装与启动、运行时 bootstrap、签名身份、notarization/stapling、升级和回滚；ZIP 只保留 development 测试，正式发布需提供 Developer ID 与 App Store Connect API 凭据。
- [ ] 将当前 macOS native shell candidate 补成可用离线产品：分别提供 `osx-arm64`/`osx-64` 的可重定位 Python/PyTorch runtime、FFmpeg/FFprobe、Java 17 与受支持模型下载链，并把首次初始化/模型状态接入桌面 UI。当前 universal2 包不含这些架构相关依赖，`requirements-media-asr.txt` 又是 Windows conda export，因此不得宣称开箱即用转录；Apple Silicon 可继续建设当前主栈，Intel 仍需冻结 legacy 依赖组合或降级支持，不能宣称与 arm64 等价。
- [x] 完成 Linux Tauri 原生 x86_64 候选路径：AppImage、deb、rpm 均以最新 npm/Cargo lock 真实构建并独立解包，三包逐字节核验同一份 `usr/lib/MediaTranscribe Studio/mts-runtime`，共 100 个文件、`37,371,547` bytes、无模型权重；manifest SHA-256 `c48666f3054255205212d6d961a308e856d2f4e8f7027abcf9109371b72042a1`，AppImage/deb/rpm SHA-256 分别为 `2539c27f487235260f893edcc8020a3c8911a2c5a6065dcd6df655e5f32dffc5` / `a252bf906ba2cb01cfb2ebd3ed0d4257a85ea65c48eb292460da5cf7c408fd83` / `c840fdd35099dfe33f97a114be58ac8268d154be710f77fa25a1f1d5f060c7eb`，证据目录 `D:\mts-release\linux-audit-r4\release`。三种解包形态均在 WSLg 真启动出 `1440x920` 窗口；包内 bootstrap 真运行、`ProductionConfig.load()` 解析成功，Linux 聚焦回归 `9 passed`，npm audit 为 0。CI 将 x64/arm64 映射到原生 runner，并以 tar+SHA-256 保留 AppImage 可执行位后做往返账本复验。本项不冒充已包含 Python/PyTorch、模型或端到端推理验收。
- [ ] 完成 Linux x64 包管理器实际安装、升级、卸载和回滚验收，并在原生 arm64 runner 构建及启动对应候选；当前三种 x64 包只完成独立解包与启动。
- [x] 完成 Windows Tauri 原生打包实现与离线验证：runtime overlay、MSI/NSIS 收集、确定性 portable ZIP、D 盘 target 支持、配置清洗、`worker-python.path` 和 signed/unsigned workflow；production lifecycle 强制外部固定发布者指纹，detached CMS 签名锚定完整 manifest 与 Python/JAR/config 账本，缺签名、缺信任锚、直接篡改已签名 PE 或替换 payload 后重算普通哈希均 fail closed。Windows Python/PowerShell 聚焦回归 `29 passed`，workflow 与五份 PowerShell 脚本/模块解析、manifest schema 均通过。当前只证明脚本和伪目标收集，不声称已生成真实 PE/MSI/NSIS。
- [ ] 在 `windows-2022` runner 完成 x64/arm64 真实 Tauri 编译、MSI/NSIS 解包启动、Authenticode 签名、升级/卸载和回滚验收。
- [x] 统一 worker 运行时定位：支持显式 `MTS_RUNTIME_ROOT`、安装包旁运行时、Tauri macOS/Linux `resource_dir()/mts-runtime`、用户配置与 Finder/AppImage 环境；Rust 全量 lib 回归 `93 passed`，包括 Linux/macOS 资源路径和 AppImage `PYTHONHOME/PYTHONPATH` 清理。
- [x] 恢复桌面依赖锁的发布健康度：仅更新 `brace-expansion`、`nanoid`、`postcss` 三个传递性构建依赖，`npm audit --audit-level=moderate` 为 0；`npm run lint`、31 个 Vitest 文件共 `377 passed`、TypeScript 检查与 Vite production build 均通过。
- [ ] 冻结唯一生产模型组合、阈值、路由、prompt、registry、lifecycle、配置和全部 digest。
- [ ] 在未参与调参的 held-out 上运行完整声纹、转写、语义、字幕和 PDF 链路。
- [ ] Codex 完成 held-out 盲审并逐例签署结论；不得把自动协议通过替代该裁决。
- [ ] Python、TypeScript、Rust、Maven、契约、E2E、打包和离线 preflight 全部通过。
- [ ] 在 Windows 桌面应用完成真实媒体验收：导入、运行、复核、导出、重试、取消和恢复。
- [ ] 演练生产模型不可用、显存不足、磁盘不足、进程中断和产物篡改的回滚路径。
- [ ] 更新 README、架构、模型卡、数据许可、隐私、Windows 安装/使用和迁移文档。
- [ ] 只有胜出配置和 rollback 均验证后，才卸载被正式淘汰且无引用的旧模型。
- [ ] 生成最终验收索引，列出每项任务的证据路径、SHA-256、命令、结果和已知限制。
- [ ] 全部任务勾选后交付；在此之前保持 `/goal` active。
