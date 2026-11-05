#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ -f .venv/Scripts/python.exe ]]; then
    python_cmd=".venv/Scripts/python.exe"
elif [[ -x .venv/bin/python ]]; then
    python_cmd=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    python_cmd="python"
else
    echo "Error: No project virtual environment or active Python was found." >&2
    exit 1
fi

if [[ ! -f streamlit_app.py ]]; then
    echo "Error: streamlit_app.py is missing from the project folder." >&2
    exit 1
fi

if ! "$python_cmd" -c 'import streamlit' >/dev/null 2>&1; then
    echo "Error: Streamlit is unavailable in $python_cmd. Use the configured project environment." >&2
    exit 1
fi

echo "Open http://localhost:8501 in your browser."
echo "This runs locally on your computer. Press Ctrl+C to stop."
exec "$python_cmd" -m streamlit run streamlit_app.py --server.address localhost --server.port 8501
