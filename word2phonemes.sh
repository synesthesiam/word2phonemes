#!/usr/bin/env bash
DIR="$( cd "$( dirname "$0" )" && pwd )"

venv_path="$DIR/.venv"
if [[ ! -d "$venv_path" ]]; then
    # Create new environment
    echo 'Creating virtual environment'
    python3 -m venv "$venv_path"
    source "$venv_path/bin/activate"
    python3 -m pip install requirements.txt
else
    # Activate environment
    source "$venv_path/bin/activate"
fi

python3 -m word2phonemes "$@"
