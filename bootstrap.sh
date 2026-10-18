#!/usr/bin/env bash

set -euo pipefail

repository_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${repository_directory}"

if [[ "${#}" -ne 0 ]]; then
    echo "usage: ./bootstrap.sh" >&2
    exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to bootstrap QKVS but was not found on PATH." >&2
    exit 1
fi

echo "Installing the QKVS command with its pinned runtime dependencies..."
uv tool install --force --editable --python 3.14 "${repository_directory}"

qkvs_executable="$(uv tool dir --bin)/qkvs"
if [[ ! -x "${qkvs_executable}" ]]; then
    echo "uv installed QKVS but did not expose the qkvs executable." >&2
    exit 1
fi

"${qkvs_executable}" --help >/dev/null

echo "QKVS is ready at ${qkvs_executable}."
echo "Run ./run.sh <configuration.json> to execute or resume a configuration."
