# Portable runtime bootstrap

The release contains application code, provider presets, the reviewed model
registry, configuration templates, contracts, and the PDF renderer. It does
not contain model weights or API keys. The current native overlays also do not
contain Python, FFmpeg, Java, Ollama, or vendor download CLIs; bind verified
external executables before treating the shell package as a transcription
runtime.

Initialize non-secret paths after installation:

```powershell
.\bootstrap\Initialize-MtsRuntime.ps1
```

Use the bundled remote profile when an online provider is explicitly desired:

```powershell
.\bootstrap\Initialize-MtsRuntime.ps1 `
  -ConfigurationTemplate "production.config.remote.example.json"
```

On Linux and macOS the equivalent entry points are `Initialize-MtsRuntime.sh`
and `Manage-MtsModels.sh`. On macOS, the scripts live below
`Contents/Resources/mts-runtime/bootstrap` inside the app bundle and default
to `~/Library/Application Support/MediaTranscribeStudio`; on Linux they use
`${XDG_DATA_HOME:-~/.local/share}/MediaTranscribeStudio`. Use `--model-root`
to place weights on another volume. Both scripts also accept `--dry-run`.
When no pyannote interpreter is supplied, the POSIX initializer disables that
fallback coherently instead of retaining a Windows `.exe` path. macOS configs
start with CPU/float32 devices rather than the Windows CUDA defaults.

The default model root is selected in this order: `MTS_MODEL_ROOT`,
`D:\models` when drive D is available, then the application data directory.
Pass `-ModelRoot` to select another location. Add
`-PersistUserEnvironment` only when the current Windows user should keep the
resolved runtime, configuration, and model paths across launches.

List registered models or download one through the packaged model manager:

```powershell
.\bootstrap\Manage-MtsModels.ps1 list
.\bootstrap\Manage-MtsModels.ps1 pull --provider ollama --model qwen3.5:27b-q4_K_M
.\bootstrap\Manage-MtsModels.ps1 pull --provider huggingface --model <repo> --revision <commit> --token-env HF_TOKEN
```

Tokens are read from environment variables by the vendor CLI. Do not place API
keys in `production.config.json`, the catalog, command arguments, or release
artifacts. The portable model catalog is
`configs\model-catalog.v1.json`; local and remote provider templates are in
`configs\llm-provider-presets.v1.json`; a no-key Anthropic example is provided
as `production.config.remote.example.json`. Custom OpenAI-compatible relays use
the same remote-explicit network policy.
