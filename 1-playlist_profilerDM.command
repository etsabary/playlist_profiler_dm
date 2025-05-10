#!/bin/bash
# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Extract the base name of the script (without extension)
BASE_NAME="$(basename "$0" .command)"

# Form the corresponding Python script name
PYTHON_SCRIPT="${BASE_NAME}.py"

# Navigate to the script directory
cd "$SCRIPT_DIR"

# Check if the Python script exists, and run it
if [[ -f "$PYTHON_SCRIPT" ]]; then
    python3 "$PYTHON_SCRIPT"
else
    echo "Error: Script '$PYTHON_SCRIPT' not found in '$SCRIPT_DIR'."
    exit 1
fi
