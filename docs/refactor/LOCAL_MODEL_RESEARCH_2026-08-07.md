# Local Semantic Model Research - 2026-08-07

This note freezes the evidence used to choose local semantic-arbitration
challengers for MediaTranscribeStudio on the current Windows host:

- Intel Core i9-11900K, 8 physical cores and 16 logical processors
- 34,145,546,240 bytes of RAM (about 31.8 GiB visible)
- NVIDIA GeForce RTX 3060 with 12 GiB VRAM
- Ollama 0.32.6 with its model store on `D:/models/ollama`

The research question is deliberately narrower than "which model is best in
general". The selected model must produce stable multilingual
semantic-arbitration contracts under the project's frozen development protocol
while leaving enough RAM and VRAM for the speech pipeline.

## Evidence Policy

Official model cards, source repositories, licenses, and Ollama OCI manifests
establish identity, architecture, context, packaging, and deployment
feasibility. Vendor benchmark charts and scores from different harnesses do
not establish promotion. They are not comparable with the project's Chinese
meeting-arbitration task and are not used as ranking evidence.

Promotion requires one fixed prompt and sampling contract, one frozen
development set, schema-valid output, and the same latency and memory probes
for every candidate. Held-out labels must not be inspected or fitted. The
previous semantic-v1 results are excluded because their reference labels leaked
into prompts. A model remains a challenger until the corrected protocol passes.
`production` is only the current deployment status, not a protected incumbent:
any challenger that passes and wins the common gates may replace it without an
extra incumbent margin.

## Upstream Facts

| Model | Published | License | Total / active parameters | Native context | Primary official evidence |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 31B IT | 2026-03-11 | Apache-2.0 reported by the official model API; linked Gemma 4 terms apply | 31.273B dense / 31.273B | 262,144 | [Google model card at pinned revision](https://huggingface.co/google/gemma-4-31B-it/tree/842da3794eaa0b77d5f08bae87a17459d91ff475), [Google Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license) |
| Qwen3.6 27B | 2026-04-22 | Apache-2.0 | 27B dense / 27B | 262,144 | [Qwen3.6 source](https://github.com/QwenLM/Qwen3.6), [27B model card](https://modelscope.cn/models/Qwen/Qwen3.6-27B) |
| Qwen3.6 35B-A3B | 2026-04-16 | Apache-2.0 | 35B / 3B | 262,144 | [Qwen3.6 source](https://github.com/QwenLM/Qwen3.6), [35B-A3B model card](https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B) |
| GLM-4.7-Flash | 2026-01-19 | MIT | 30B / 3B | 202,752 | [official model card](https://modelscope.cn/models/ZhipuAI/GLM-4.7-Flash), [official deployment source](https://github.com/zai-org/GLM-4.5) |
| Mistral Small 3.2 24B Instruct 2506 | 2025-06-20 | Apache-2.0 | 24B dense / 24B | 131,072 | [official model card](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) |
| Magistral Small 2506 | 2025-06-10 | Apache-2.0 | 24B dense / 24B | 128K family limit; official guidance recommends at most about 40K | [Mistral release](https://mistral.ai/news/magistral/), [official model card](https://huggingface.co/mistralai/Magistral-Small-2506) |
| gpt-oss-20b | 2025-08-05 | Apache-2.0 | 21B / 3.6B | 131,072 | [OpenAI source and model card](https://github.com/openai/gpt-oss), [official weights](https://huggingface.co/openai/gpt-oss-20b) |

Qwen3.6 35B-A3B declares 256 experts with 8 routed experts plus 1 shared
expert active per token. Its small active-parameter count lowers compute, but
does not mean that only 3B parameters need to reside in RAM/VRAM. The same
residency distinction applies to all MoE candidates.

The official Google-owned Hugging Face repository for Gemma 4 31B IT was
queried at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. The API reported
`apache-2.0`, linked the Gemma 4 license above, and reported 31,273,088,876
BF16 parameters. This establishes upstream identity and stated terms. It does
not establish byte-level derivation of Ollama's quantized conversion from the
official BF16 files.

## Immutable Ollama Packages

The manifest SHA-256 values below are hashes of the exact v2 manifest bodies.
The model-layer SHA-256 and byte counts come from those manifests, not from the
rounded sizes on catalog pages.

| Tag | Encoding | Manifest SHA-256 | Model-layer SHA-256 | Model-layer bytes | GiB | Role |
| --- | --- | --- | --- | ---: | ---: | --- |
| `gemma4:31b` | Q4_K_M | `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7` | `280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313` | 19,868,969,920 | 18.50 | rejected challenger after v16 development challenge |
| `qwen3.6:27b-q4_K_M` | Q4_K_M | `a50eda8ed977ab48a12431878896b27ffd5cef552c17af3317d9623b939a7f1e` | `83c54730a5fea8a0958598c01617c1419c431e93b33bacf980b49a420c798926` | 17,420,420,832 | 16.22 | formal challenger |
| `qwen3.6:35b-a3b-q4_K_M` | Q4_K_M | `07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522` | `f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d` | 23,938,321,664 | 22.29 | formal challenger, constrained |
| `glm-4.7-flash:q4_K_M` | Q4_K_M | `4475827791a269b02c8ec49b1c3bc1abb5846bacf3fae015b75d33986322d8f6` | `9eba2761cf0b88b8bc11a065a7b5b47f1b13ce820e8e492cb1010b450f9ec950` | 19,019,269,280 | 17.71 | formal independent challenger |
| `mistral-small3.2:24b-instruct-2506-q4_K_M` | Q4_K_M | `5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b` | `41a5b0c36a28a3a0480ce2e4007d3a21e3298be70e2b9a103960581412997dca` | 15,177,369,888 | 14.14 | diagnostic only |
| `magistral:24b-small-2506-q4_K_M` | Q4_K_M | `27bcbbf6d32417d3a8a12d5eb9bd55fda6e5591807e2b02aaaf4423899925afc` | `641615e9986bc8687f936cd87c586bdd92d338172c4180963080e48b8e84ec36` | 14,333,907,488 | 13.35 | reasoning diagnostic only |
| `gpt-oss:20b` | native MXFP4 | `17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7` | `e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb` | 13,793,422,144 | 12.85 | structured-output diagnostic only |

The corresponding immutable manifests are available from the official Ollama
registry:

- [Gemma 4 31B Q4_K_M](https://registry.ollama.ai/v2/library/gemma4/manifests/31b)
- [Qwen3.6 27B Q4_K_M](https://registry.ollama.ai/v2/library/qwen3.6/manifests/27b-q4_K_M)
- [Qwen3.6 35B-A3B Q4_K_M](https://registry.ollama.ai/v2/library/qwen3.6/manifests/35b-a3b-q4_K_M)
- [GLM-4.7-Flash Q4_K_M](https://registry.ollama.ai/v2/library/glm-4.7-flash/manifests/q4_K_M)
- [Mistral Small 3.2 Q4_K_M](https://registry.ollama.ai/v2/library/mistral-small3.2/manifests/24b-instruct-2506-q4_K_M)
- [Magistral Small Q4_K_M](https://registry.ollama.ai/v2/library/magistral/manifests/24b-small-2506-q4_K_M)
- [gpt-oss 20B](https://registry.ollama.ai/v2/library/gpt-oss/manifests/20b)

The two installed Qwen3.6 aliases were independently verified against their
local OCI manifests. Their model blob names, file sizes, and full-file SHA-256
digests agree. Both manifests also bind the same independently verified
Apache-2.0 license layer:
`5f3a3c817e78f5b8a4ad2d2c458a3e4b2cce470d6c12642c4eeb12cb8a9bf51d`.

## Host Fit

All semantic development probes use an 8,192-token context. The native context
figures above are capability metadata, not a reason to allocate a 128K or 256K
KV cache on this host.

`qwen3.6:27b-q4_K_M` is the first formal challenger. Its 8K cold resource probe
took 52.312 seconds in total, including 31.258 seconds to load. Ollama reported
9,780,204,664 bytes of model VRAM, aggregate process RSS peaked at
8,530,677,760 bytes, and whole-GPU use peaked at 11,888 MiB. Stopping the model
emptied `/api/ps`, reduced whole-GPU use to 1,209 MiB, and increased free
physical RAM from 9,516,032 KiB to 17,592,808 KiB. The probe exhausted its
output budget in a thinking trace, so its empty response is not semantic
evidence; the benchmark must disable thinking explicitly or apply one identical
thinking policy to every candidate.

`qwen3.6:35b-a3b-q4_K_M` is runnable but materially tighter. Its 8K cold probe
took 52.353 seconds, including 47.528 seconds to load. Ollama split execution at
about 57 percent CPU and 43 percent GPU and reported 9,917,599,578 bytes of
model VRAM. Windows free physical memory fell to about 5,034,968 KiB and the
pagefile reached about 2,275 MiB. It passed the functional probe and released
cleanly, but it must retain a hard memory gate and must not be tested at its
native 262K context on this machine.

`gemma4:31b` is also runnable on this host. Its strict-JSON 8K cold probe took
54.373 seconds, including 42.572 seconds to load, and returned
`{"ok":true,"language":"zh-CN"}`. Ollama split the dense model across host
and GPU memory, offloading 25 of 60 layers. Rounded external observations were
21.46 GB Ollama residency, 9.05 GB model VRAM residency, 11.31 GB whole-GPU
use, and 3.84 GB free Windows physical memory while loaded. Explicit release
then left `/api/ps` empty. This proves that a 31B dense Q4_K_M model can run by
Windows-native mixed residency; WSL's 15 GB assignment is not this path's
memory ceiling. The challenge receipt is
`D:\mts-eval\model-challenges\gemma4-31b-20260810\challenge-receipt.v1.json`
(file SHA-256
`63cb23275f97e7d0dbc3f1403e68fff3faac308b91a1ac02aaf0a9cf60d1ffb6`),
and the strict-JSON response file SHA-256 is
`1cebc13ab717066c006d4d8f379abbe4a9911774bc98c8eaae8884b0e5e46f61`.

`glm-4.7-flash:q4_K_M` is the formal independent-architecture challenger. Its
Chinese/English model card, MIT license, 3B active MoE design, and 17.71 GiB
package fit the host envelope. It still needs the same corrected development
protocol before any promotion decision.

Mistral Small 3.2, Magistral Small, and gpt-oss-20b are diagnostic candidates,
not automatic downloads. Mistral Small 3.2 is a useful cross-family JSON and
instruction-following control but has no task-specific Chinese arbitration
evidence. Magistral's long reasoning traces can consume the fixed output budget.
gpt-oss uses the Harmony protocol and has no official Chinese-specific evidence.
They should be downloaded only if the three formal candidates cannot explain a
development failure.

## Gemma 4 31B Development Challenge - 2026-08-10

Gemma and the current `qwen3.5:27b-q4_K_M` champion were run on the same six
visible multilingual development regressions with
`semantic-job-candidate-arbitration-v16`, `multilingual-fidelity-v2`, 32,768
context tokens, 4,096 output tokens, batch size 8, temperature 0, and top-p
0.1. The fixture names and source hashes, generated visible documents and
lattices, initial prompts, response schemas, runner, semantic code snapshot,
and inference parameters matched. Hidden references were unavailable; Codex
reviewed visible text, timestamp order, and raw responses, and did not use
automatic exact match as the final judgment.

Gemma completed all 16 calls but made zero evidence requests and passed `0/6`
cases in the semantic audit. It missed all bounded lexical repairs and directly
committed unsupported one-speaker identities in the Indonesian, French, and
Mandarin continuity cases. Qwen completed the same 16 calls, made 24 native
bounded requests with no host-guard additions, and passed `5/6`. Its remaining
failure is not closed: the Indonesian case still omits ASR N-best for the
malformed `bus kruaster` lexical slot even though its speaker-evidence request
set is complete.

Gemma evidence is under
`D:\mts-eval\semantic-model-comparison\semantic-v16-six-fixture-gemma4-31b-real-20260810-r1`;
its run-report/audit file SHA-256 values are
`2fcf16d545091521611ca44b481a8505f7b9b59c2864f75cf9cc02879985e768` /
`cbca5fe7e32d4b810b26245d3fc1ffe7e521a4d476432f0b1be899ede927d4f3`
and canonical SHA-256 values are
`d8b4afdd1ee4a31db6221a662180efd0c1fb05815613875c6024c1be963b5e4d` /
`e64c5c1d00fdbeeb1dc5adf395a089eb17db30b6bfa5d6ee6d2bb48bcc32f78e`.
The Qwen control is under
`D:\mts-eval\semantic-model-comparison\semantic-v16-six-fixture-qwen3.5-27b-q4_K_M-real-20260810-r1`;
its corresponding file SHA-256 values are
`f667c3b9c48aabcd5fa3b24e415e2e60600bd5d8f40057699883ef49e67d64e3` /
`8f72cebe5f2ed3299a8141099841e2b1e68e6e7ea7974a295c06410d326e2178`.

The development verdict is therefore to reject Gemma 4 31B before spending the
frozen 22-case blind-review budget. It does not enter that competition, does
not replace production, and does not count as held-out or final product
acceptance. Qwen3.5 27B remains the replaceable production champion while its
Indonesian lexical regression remains open.

## Exclusions

Q8 packages do not provide acceptable RAM, VRAM, and KV-cache margin on this
32 GiB / 12 GiB host:

| Tag | Manifest SHA-256 | Model-layer bytes | GiB | Decision |
| --- | --- | ---: | ---: | --- |
| `qwen3.6:27b-q8_0` | `cd0210c667bffa98ad702668d05fda1f340bcbb0a2c769bd389670d19ad1441b` | 29,970,380,512 | 27.91 | exclude |
| `qwen3.6:35b-a3b-q8_0` | `0218f872e86baa9c7610509f27db36a7bc52eea7afee24688f81ca74ffcb6c77` | 38,696,955,136 | 36.04 | exclude; model layer alone exceeds host RAM |
| `glm-4.7-flash:q8_0` | `4420340fd3190fc6f5da59d85aeff8e74c8c401679b067c5d00df16efb98104d` | 31,843,415,200 | 29.66 | exclude |

[Mistral Small 4](https://mistral.ai/news/mistral-small-4/) is a 119B-total,
6B-active, 256K MoE model released on 2026-03-16. The official Ollama library
does not expose a local package, and its full residency is outside this host's
envelope. It is excluded even though its active parameter count is small.

The official Ollama pages for [GLM-5.1](https://ollama.com/library/glm-5.1) and
[GLM-5.2](https://ollama.com/library/glm-5.2) expose cloud-only tags at the time
of this audit. With no local manifest to pin and verify, they are excluded from
the local candidate set.

Qwen3.5 35B-A3B remains a same-family regression control. Qwen3.5 27B is the
current, replaceable production champion and Qwen3.5 9B remains its rollback.
The Gemma 4 31B development rejection changes neither role.

## Evaluation Order

1. Run `qwen3.6:27b-q4_K_M` on the corrected frozen development protocol.
2. Run the already installed `glm-4.7-flash:q4_K_M` with identical settings.
3. Run `qwen3.6:35b-a3b-q4_K_M` only with the 8K resource gate enabled.
4. Use diagnostic candidates only to investigate an unresolved failure mode.
5. Promote nothing until schema, quality, latency, RAM, VRAM, and held-out
   non-regression gates all pass without held-out threshold or prompt fitting.
