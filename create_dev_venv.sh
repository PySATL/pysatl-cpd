#!/bin/bash


if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    if python -c "import sys; print(sys.version_info[0])" | grep -q 3; then
        PYTHON_CMD="python"
    else
        echo "Python 3 is required, but only Python 2 is found."
        exit 1
    fi
else
    echo "Python is not installed."
    exit 1
fi
echo "Using $PYTHON_CMD for installing .venv"
$PYTHON_CMD -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install --with dev
deactivate
