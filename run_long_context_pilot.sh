#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${script_directory}"

mode="${1:-run}"
if [[ "${#}" -gt 1 || ("${mode}" != "run" && "${mode}" != "--setup-only") ]]; then
    echo "usage: ./run_long_context_pilot.sh [--setup-only]" >&2
    exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH." >&2
    echo "Install uv, then rerun this script; do not install dependencies with system Python." >&2
    exit 1
fi

echo "Installing the QKVS command with its pinned runtime dependencies..."
uv tool install --force --editable --python 3.13 "${script_directory}"

qkvs_executable="$(uv tool dir --bin)/qkvs"
if [[ ! -x "${qkvs_executable}" ]]; then
    echo "uv installed QKVS but did not expose the qkvs executable." >&2
    exit 1
fi

"${qkvs_executable}" validate --config configs/heatfield_learning_study.json

if [[ "${mode}" == "--setup-only" ]]; then
    echo "QKVS setup and configuration validation completed; no experiment was started."
    exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null; then
    echo "NVIDIA driver health check failed; the pilot was not started." >&2
    exit 1
fi

exec "${qkvs_executable}" run \
    --config configs/heatfield_learning_study.json \
    --device cuda
