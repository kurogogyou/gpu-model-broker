#!/bin/bash
# Convenience: source this from anywhere to activate the broker venv.
# Mirrors brain repo's activate_venv.sh convention; venv lives at /opt/fast/
# per the fateburn-filesystem-anchor centralized-venvs decision.
VENV_DIR="/opt/fast/venvs/gpu-broker"
SRC_DIR="/opt/brain/src/gpu-broker"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"
    echo "✓ gpu-broker venv activated: $VENV_DIR"
    echo "  PYTHONPATH: $SRC_DIR"
    echo "  Run: python -m broker  (or systemctl --user start gpu-broker)"
else
    echo "Error: venv not found at $VENV_DIR"
    echo "Create it: python3 -m venv $VENV_DIR && $VENV_DIR/bin/pip install -r $SRC_DIR/requirements.txt"
    return 1 2>/dev/null || exit 1
fi
