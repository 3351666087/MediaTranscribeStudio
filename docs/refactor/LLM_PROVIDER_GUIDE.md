# LLM Provider Profiles

The production default is pinned to the current semantic champion:

- Ollama: `qwen3.5:27b-q4_K_M`
- digest: `sha256:7653528ba5cba4dd8e19da24aaddc7f4d0b5ecd93571c0825dfd4137958ec06e`
- configured speaker verifier candidate: `eres2netv2-w24s4ep4` (wide ERes2NetV2
  deployment slot; registry status remains `challenger` until full diarization
  acceptance, so this is not a quality-promotion claim)

The preset catalog in `configs/llm-provider-presets.v1.json` covers Ollama,
Hugging Face Router, OpenAI-compatible relays, Anthropic, Google Gemini, and
the common Chinese and international gateways. Any OpenAI-compatible gateway
can be used by entering its endpoint and model id; no vendor-specific code is
needed for that route.

Remote profiles are opt-in. A remote endpoint must use HTTPS, while an Ollama
loopback endpoint may use HTTP. API keys are referenced by environment-variable
name (for example `OPENAI_API_KEY`) and are never written to arbitration
artifacts, manifests, logs, or release files. A proxy is explicit and is
validated with the same URL rules.

## Downloading models

Use the standard CLI from the repository root:

```text
runtime/media-asr/python.exe -m tools.model_manager list
runtime/media-asr/python.exe -m tools.model_manager pull --provider ollama --model qwen3.5:27b-q4_K_M --model-root D:\models
runtime/media-asr/python.exe -m tools.model_manager pull --provider huggingface --model Qwen/Qwen3-ASR-1.7B --revision <40-char-commit> --model-root D:\models --token-env HF_TOKEN
```

`MTS_MODEL_ROOT` overrides the default location. On this workstation the
default resolves to `D:\models` (or `/mnt/d/models` from WSL). Downloads use
argument arrays rather than a shell and emit a receipt containing only hashes
and the key variable name.

The existing `production.config.json` remains an offline, digest-pinned profile.
To enable a remote provider, copy the `llm` example block from the production
configuration template, set `offline` to `false`, choose a remote provider,
and provide the key through the configured environment variable. The worker
still validates every model response against the same semantic contracts and
keeps the fail-closed evidence boundary.
