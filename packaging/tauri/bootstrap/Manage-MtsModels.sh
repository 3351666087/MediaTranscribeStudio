#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: Manage-MtsModels.sh <list|pull> [model-manager options]

The manager stores model artifacts below MTS_MODEL_ROOT (or the user data
directory's models folder). Credentials are read by the vendor CLI from an
environment variable; they are never accepted as command-line values.
EOF
}

script_dir="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
app_root="${MTS_RUNTIME_ROOT:-$(cd -P "$script_dir/.." && pwd)}"
# Support direct invocation from the source checkout as well as the installed
# runtime payload layout.
if [[ ! -f "$app_root/tools/model_manager.py" && -f "$app_root/../../tools/model_manager.py" ]]; then
  app_root="$(cd -P "$app_root/../.." && pwd)"
fi
data_root="${MTS_DATA_ROOT:-}"
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
model_root="${MTS_MODEL_ROOT:-$data_root/models}"
manager="$app_root/tools/model_manager.py"
catalog="$app_root/configs/model-catalog.v1.json"

if [[ ! -f "$manager" ]]; then
  printf '%s\n' '{"ok":false,"action":"ManageMtsModels","error":"model manager is missing from the runtime payload"}' >&2
  exit 1
fi
if [[ ! -f "$catalog" ]]; then
  printf '%s\n' '{"ok":false,"action":"ManageMtsModels","error":"portable model catalog is missing from the runtime payload"}' >&2
  exit 1
fi

python_bin="${MTS_WORKER_PYTHON:-}"
if [[ -z "$python_bin" && -f "$data_root/config/worker-python.path" ]]; then
  IFS= read -r python_bin < "$data_root/config/worker-python.path" || true
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
  for candidate in \
    "$app_root/runtime/media-asr/bin/python" \
    "$app_root/runtime/media-asr/python" \
    "$app_root/runtime/media-asr/python3"; do
    if [[ -x "$candidate" ]]; then python_bin="$candidate"; break; fi
  done
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
  python_bin="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$python_bin" ]]; then
  printf '%s\n' '{"ok":false,"action":"ManageMtsModels","error":"Python 3 is required to manage models"}' >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- list
fi
case "$1" in
  list)
    shift
    exec "$python_bin" "$manager" list --registry "$catalog" "$@"
    ;;
  pull)
    shift
    has_model_root=0
    for argument in "$@"; do
      if [[ "$argument" == "--model-root" || "$argument" == --model-root=* ]]; then
        has_model_root=1
        break
      fi
    done
    if (( ! has_model_root )); then
      set -- --model-root "$model_root" "$@"
    fi
    exec "$python_bin" "$manager" pull "$@"
    ;;
  -h|--help)
    usage >&1
    exit 0
    ;;
  *)
    usage
    exit 2
    ;;
esac
