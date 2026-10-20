#!/usr/bin/env bash

set -euo pipefail

repository_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${repository_directory}"

if [[ "${#}" -lt 1 || "${#}" -gt 3 ]]; then
    echo "usage: ./run.sh <configuration.json> [snapshot-feasibility.csv]" >&2
    echo "       ./run.sh <configuration.json> --confidence <confidence.csv>" >&2
    exit 2
fi

configuration="${1}"
if [[ ! -f "${configuration}" ]]; then
    echo "configuration not found: ${configuration}" >&2
    exit 1
fi

if command -v qkvs >/dev/null 2>&1; then
    qkvs_executable="$(command -v qkvs)"
elif [[ -x "${HOME}/.local/bin/qkvs" ]]; then
    qkvs_executable="${HOME}/.local/bin/qkvs"
else
    echo "QKVS is not installed; run ./bootstrap.sh first." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to read the execution device from the configuration." >&2
    exit 1
fi

device="$(python3 -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    configuration = json.load(stream)
devices = configuration.get("execution", {}).get("allowed_devices")
if not isinstance(devices, list) or len(devices) != 1:
    raise SystemExit(
        "configuration must declare exactly one execution.allowed_devices entry"
    )
print(devices[0])
' "${configuration}")"

if [[ "${device}" == "cuda" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null; then
        echo "NVIDIA driver health check failed; execution was not started." >&2
        exit 1
    fi
elif [[ "${device}" != "cpu" ]]; then
    echo "unsupported execution device: ${device}" >&2
    exit 1
fi

if [[ "${#}" -eq 3 ]]; then
    if [[ "${2}" != "--confidence" ]]; then
        echo "unsupported run mode: ${2}" >&2
        exit 2
    fi
    arguments=(
        snapshot-confidence
        --config "${configuration}"
        --device "${device}"
        --output "${3}"
    )
else
    arguments=(run --config "${configuration}" --device "${device}")
    if [[ "${#}" -eq 2 ]]; then
        arguments+=(--snapshot-csv "${2}")
    fi
fi
exec "${qkvs_executable}" "${arguments[@]}"
