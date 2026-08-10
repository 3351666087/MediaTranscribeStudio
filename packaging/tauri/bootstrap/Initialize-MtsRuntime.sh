#!/usr/bin/env bash
set -euo pipefail

# Initialize the user-owned portion of a portable MediaTranscribe Studio
# runtime.  This script is intentionally dependency-light: only Python 3 is
# required to bind the JSON template, and no model or credential is copied.

usage() {
  cat >&2 <<'EOF'
Usage: Initialize-MtsRuntime.sh [options] [template]

Options:
  --app-root PATH             runtime payload root (default: script parent)
  --data-root PATH            user data directory
  --model-root PATH           model directory (may be on another volume)
  --worker-python PATH        Python used by the worker
  --pyannote-python PATH      optional isolated pyannote Python
  --production-config PATH    destination production config
  --template PATH             JSON configuration template
  --dry-run                   print the resolved plan without writing files
  -h, --help                  show this message
EOF
}

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
app_root="${MTS_RUNTIME_ROOT:-$(cd -P "$script_dir/.." && pwd)}"
data_root="${MTS_DATA_ROOT:-}"
model_root="${MTS_MODEL_ROOT:-}"
worker_python="${MTS_WORKER_PYTHON:-}"
pyannote_python="${MTS_PYANNOTE_PYTHON:-}"
production_config="${MTS_PRODUCTION_CONFIG:-}"
template=""
dry_run=0
platform_name="${MTS_PLATFORM_OVERRIDE:-$(uname -s 2>/dev/null || printf '%s' Unknown)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app-root) [[ $# -ge 2 ]] || { usage; exit 2; }; app_root="$2"; shift 2 ;;
    --data-root) [[ $# -ge 2 ]] || { usage; exit 2; }; data_root="$2"; shift 2 ;;
    --model-root) [[ $# -ge 2 ]] || { usage; exit 2; }; model_root="$2"; shift 2 ;;
    --worker-python) [[ $# -ge 2 ]] || { usage; exit 2; }; worker_python="$2"; shift 2 ;;
    --pyannote-python) [[ $# -ge 2 ]] || { usage; exit 2; }; pyannote_python="$2"; shift 2 ;;
    --production-config) [[ $# -ge 2 ]] || { usage; exit 2; }; production_config="$2"; shift 2 ;;
    --template) [[ $# -ge 2 ]] || { usage; exit 2; }; template="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage >&1; exit 0 ;;
    --) shift; break ;;
    -*) usage; exit 2 ;;
    *)
      # Keep the old positional-template form working for existing installers.
      [[ -z "$template" ]] || { usage; exit 2; }
      template="$1"
      shift
      ;;
  esac
done
[[ $# -eq 0 ]] || { usage; exit 2; }

app_root="$(cd -P "$app_root" && pwd)"
# When invoked from the source checkout (rather than from an installed
# `Contents/Resources/mts-runtime` payload), the script lives one extra level
# below the project root.  Accept both layouts so CI can exercise the exact
# bootstrap entrypoint.
if [[ ! -f "$app_root/production.config.example.json" && -f "$app_root/../../production.config.example.json" ]]; then
  app_root="$(cd -P "$app_root/../.." && pwd)"
fi
if [[ ! -f "$app_root/backend/worker.py" ]]; then
  printf '%s\n' '{"ok":false,"action":"InitializeMtsRuntime","error":"runtime root must contain backend/worker.py"}' >&2
  exit 1
fi
if [[ -z "$data_root" ]]; then
  case "$(uname -s 2>/dev/null || printf '%s' Unknown)" in
    Darwin)
      data_root="${XDG_DATA_HOME:-$HOME/Library/Application Support}/MediaTranscribeStudio"
      ;;
    *)
      data_root="${XDG_DATA_HOME:-$HOME/.local/share}/MediaTranscribeStudio"
      ;;
  esac
fi
if [[ -z "$model_root" ]]; then
  model_root="$data_root/models"
fi
if [[ -z "$production_config" ]]; then
  production_config="$data_root/config/production.config.json"
fi
if [[ -z "$template" ]]; then
  template="$app_root/production.config.example.json"
elif [[ "$template" != /* ]]; then
  template="$app_root/$template"
fi

normalize_path() {
  local value="$1"
  if [[ "$value" != /* ]]; then
    value="$PWD/$value"
  fi
  local parent
  local base
  parent="$(dirname "$value")"
  base="$(basename "$value")"
  if [[ -d "$parent" ]]; then
    printf '%s/%s' "$(cd -P "$parent" && pwd -P)" "$base"
  else
    # Do not require a not-yet-created data/config parent to exist, especially
    # during --dry-run.  mkdir -p below will create it for a real run.
    printf '%s' "$value"
  fi
}

data_root="$(normalize_path "$data_root")"
model_root="$(normalize_path "$model_root")"
production_config="$(normalize_path "$production_config")"

if [[ ! -f "$template" ]]; then
  printf '{"ok":false,"action":"InitializeMtsRuntime","error":"configuration template does not exist: %s"}\n' \
    "${template//\/\\}" >&2
  exit 1
fi

if [[ -z "$worker_python" ]]; then
  for candidate in \
    "$app_root/runtime/media-asr/bin/python" \
    "$app_root/runtime/media-asr/python" \
    "$app_root/runtime/media-asr/python3"; do
    if [[ -x "$candidate" ]]; then worker_python="$candidate"; break; fi
  done
fi
if [[ -z "$worker_python" ]]; then
  worker_python="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$pyannote_python" ]]; then
  for candidate in \
    "$app_root/runtime/pyannote/bin/python" \
    "$app_root/runtime/pyannote/python3" \
    "$app_root/runtime/pyannote/python"; do
    if [[ -x "$candidate" ]]; then pyannote_python="$candidate"; break; fi
  done
fi

config_exists=0
[[ -f "$production_config" ]] && config_exists=1
python_helper="$worker_python"
if [[ -z "$python_helper" || ! -x "$python_helper" ]]; then
  python_helper="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$python_helper" ]]; then
  printf '%s\n' '{"ok":false,"action":"InitializeMtsRuntime","error":"Python 3 is required to bind the production configuration"}' >&2
  exit 1
fi

if (( ! dry_run )); then
  mkdir -p "$data_root" "$model_root" "$data_root/inputs" "$data_root/exports" "$data_root/cache" "$(dirname "$production_config")"
  if (( ! config_exists )); then
    tmp_config="$(mktemp "$(dirname "$production_config")/.production.config.XXXXXX")"
    trap 'rm -f -- "$tmp_config"' EXIT
    "$python_helper" - "$template" "$tmp_config" "$data_root" "$model_root" "$app_root" "$pyannote_python" "$platform_name" <<'PY'
import json
import os
import sys
from pathlib import Path

template, destination, data_root, model_root, app_root, pyannote_python, platform_name = sys.argv[1:]
document = json.loads(Path(template).read_text(encoding="utf-8"))

def portable(value):
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    lower = normalized.casefold()
    if lower.startswith("d:/models/"):
        return str(Path(model_root, normalized[len("D:/models/"):])).replace("\\", "/")
    if lower.startswith("models/"):
        return str(Path(model_root, normalized[len("models/"):])).replace("\\", "/")
    if lower.startswith("d:/downloads"):
        return str(Path(data_root, "inputs")).replace("\\", "/")
    if lower == "inputs" or lower.startswith("inputs/"):
        suffix = normalized[len("inputs/"):] if lower.startswith("inputs/") else ""
        return str(Path(data_root, "inputs", suffix)).replace("\\", "/")
    if lower.startswith("d:/desktop/"):
        suffix = normalized.split("/", 3)[-1]
        return str(Path(data_root, "cache" if "cache" in suffix.casefold() else "exports")).replace("\\", "/")
    if lower == "exports" or lower.startswith("exports/"):
        suffix = normalized[len("exports/"):] if lower.startswith("exports/") else ""
        return str(Path(data_root, "exports", suffix)).replace("\\", "/")
    if lower == "cache" or lower.startswith("cache/"):
        suffix = normalized[len("cache/"):] if lower.startswith("cache/") else ""
        return str(Path(data_root, "cache", suffix)).replace("\\", "/")
    if lower.startswith("d:/mediatranscribestudio/runtime/"):
        suffix = normalized[len("D:/MediaTranscribeStudio/runtime/"):]
        return str(Path(app_root, "runtime", suffix)).replace("\\", "/")
    if lower.startswith("runtime/"):
        return str(Path(app_root, normalized)).replace("\\", "/")
    return value

document = portable(document)
paths = document.setdefault("paths", {})
paths["allowedInputRoots"] = [str(Path(data_root, "inputs")).replace("\\", "/")]
paths["allowedOutputRoot"] = str(Path(data_root, "exports")).replace("\\", "/")
paths["cacheRoot"] = str(Path(data_root, "cache")).replace("\\", "/")
executables = document.setdefault("executables", {})
jar = Path(app_root, "pdf-renderer", "target", "pdf-renderer.jar")
if jar.is_file():
    executables["pdfRendererJar"] = str(jar).replace("\\", "/")
if pyannote_python:
    executables["pyannotePython"] = pyannote_python
else:
    # Never retain the Windows template's runtime/pyannote/python.exe hint when
    # an installed POSIX runtime has no dedicated pyannote interpreter.
    executables.pop("pyannotePython", None)
    models = document.setdefault("models", {})
    models["pyannote"] = None
    speaker = document.setdefault("speaker", {})
    speaker["pyannoteMode"] = "disabled"
    speaker["overlapRecoveryMode"] = "disabled"
if platform_name == "Darwin":
    # The portable template is tuned for the Windows CUDA production host.
    # A Mac must start from a compatible fail-closed CPU profile; operators can
    # opt into a separately validated MPS profile later.
    runtime = document.setdefault("runtime", {})
    runtime.update(
        {
            "vadDevice": "cpu",
            "asrDevice": "cpu",
            "asrDtype": "float32",
            "camPlusDevice": "cpu",
            "eres2netDevice": "cpu",
            "pyannoteDevice": "cpu",
        }
    )
Path(destination).write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
    chmod 600 "$tmp_config"
    mv -f -- "$tmp_config" "$production_config"
    trap - EXIT
  fi
fi

worker_python_hint="$(dirname "$production_config")/worker-python.path"
if (( ! dry_run )) && [[ "$worker_python" == /* && -x "$worker_python" ]]; then
  tmp_hint="$(mktemp "$(dirname "$production_config")/.worker-python.XXXXXX")"
  trap 'rm -f -- "$tmp_hint"' EXIT
  printf '%s\n' "$worker_python" > "$tmp_hint"
  chmod 600 "$tmp_hint"
  mv -f -- "$tmp_hint" "$worker_python_hint"
  trap - EXIT
fi

"$python_helper" - "$app_root" "$data_root" "$model_root" "$production_config" "$worker_python" "$pyannote_python" "$template" "$worker_python_hint" "$dry_run" "$config_exists" <<'PY'
import json
import os
import sys

keys = ("appRoot", "dataRoot", "modelRoot", "productionConfig", "workerPython", "pyannotePython", "configurationTemplate", "workerPythonHint")
values = dict(zip(keys, sys.argv[1:9]))
values["dryRun"] = sys.argv[9] == "1"
values["bundledModelArtifacts"] = False
values["configPresent"] = os.path.isfile(values["productionConfig"])
values["configCreated"] = not values["dryRun"] and values["configPresent"] and sys.argv[10] == "0"
print(json.dumps({"ok": True, "action": "InitializeMtsRuntime", **values}, ensure_ascii=True))
PY
