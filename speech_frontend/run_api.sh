#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m speech_frontend.api \
  --host "${SPEECH_FRONTEND_HOST:-127.0.0.1}" \
  --port "${SPEECH_FRONTEND_PORT:-8092}" \
  --max-concurrency "${SPEECH_FRONTEND_CONCURRENCY:-2}"
