#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

[ -d ext/sparse_autoencoder ] || git clone --depth 1 \
  https://github.com/openai/sparse_autoencoder.git ext/sparse_autoencoder
[ -d .venv ] || python3 -m venv .venv

.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps -e ext/sparse_autoencoder

echo "ok: source .venv/bin/activate"
