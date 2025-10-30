#!/usr/bin/env bash

# Project environment activator (bash/zsh friendly)
# Usage: source /Users/philchan/Desktop/GLBX-20250926-XC4BWUHD9D/env.sh

# Absolute project path (prefer absolute per workspace note)
export PROJECT_ROOT="/Users/philchan/Desktop/GLBX-20250926-XC4BWUHD9D"

# Prefer Eastern timezone for analyses
export TZ="America/New_York"

# Ensure project root is on PYTHONPATH
if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$PROJECT_ROOT"
else
  case ":$PYTHONPATH:" in
    *":$PROJECT_ROOT:"*) ;; # already present
    *) export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" ;;
  esac
fi

# Activate venv if present; also put its bin first on PATH
VENV_DIR="$PROJECT_ROOT/venv"
if [[ -d "$VENV_DIR" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
  export PATH="$VENV_DIR/bin:$PATH"
else
  echo "[env.sh] Warning: venv not found at $VENV_DIR. Create one with: python3 -m venv $VENV_DIR" >&2
fi

# Helpful aliases
alias py="$PROJECT_ROOT/venv/bin/python"
alias pip="$PROJECT_ROOT/venv/bin/pip"

echo "[env.sh] Project activated. Python: $(command -v python)"
echo "[env.sh] PYTHONPATH includes: $PROJECT_ROOT"


