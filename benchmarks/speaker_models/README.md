# Speaker Model Candidate Benchmarks

This directory keeps research candidates separate from production defaults.
A candidate may be promoted only after the same frozen, truth-qualified cases
show a non-regression in every applicable final-state metric and resource gate.

## Initial blind matrix

| Candidate | Role | License | Local artifact |
|---|---|---|---|
| CAM++ Chinese common | current primary embedding baseline | Apache-2.0 | `D:/models/funasr/campplus` |
| ERes2NetV2 Chinese common | current secondary baseline | Apache-2.0 | `D:/models/modelscope/eres2netv2` |
| ERes2NetV2 w24s4ep4 | Chinese-domain challenger | Apache-2.0 | `D:/models/speaker-candidates/eres2netv2-wide` |
| ReDimNet2-B6-LM | cross-domain challenger | Apache-2.0 | `D:/models/wespeaker/redimnet2-B6-LM` |
| Community-1 | diarization baseline | CC BY 4.0 | `D:/models/pyannote/speaker-diarization-community-1` |
| DiariZen WavLM Large s80 | research ceiling only | CC BY-NC 4.0 | not installed |

`modelscope-candidates.lock.json` pins the ModelScope inventory used by the
existing fail-closed downloader. ReDimNet2 is pinned to Hugging Face revision
`e34354de2429a45894905bd58c24c22250485b9a`; its `avg_model.pt` SHA-256 is
`b9314cd0184d3823c70a2518d354397bf049832f90fb1d7584ff6c0b0d8b152a`.

Published benchmark numbers are discovery evidence, not local promotion
evidence. The production decision must use recording- and speaker-isolated
development/regression/held-out partitions and separately report speaker
count, DER/JER, confusion, overlap, attributed text, review load, RTF, RAM,
VRAM, and cache behavior. Missing truth disables only that metric and can never
be interpreted as a pass.
