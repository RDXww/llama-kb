#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# llama-kb launcher
#
# Starts (optionally) llama.cpp, the RAG proxy, and the terminal chat client.
#
#   ./scripts/start.sh                 # start and enter the chat
#   ./scripts/start.sh --no-chat       # services only
#   ./scripts/start.sh --rebuild       # rebuild the index first
#   ./scripts/start.sh --no-build      # use the existing index
#   ./scripts/start.sh --model PATH    # model file (or set KB_MODEL)
#
# Attach to an already-running OpenAI-compatible backend instead of starting
# llama.cpp:
#   KB_LLAMA_URL=http://127.0.0.1:11434 ./scripts/start.sh --no-llama
#
# Common environment variables:
#   KB_MODEL            path to a .gguf model (omit in attach mode)
#   KB_NGL              GPU layers for llama-server (auto-detected, default 0)
#   KB_CTX              context size (default 8192)
#   KB_LLAMA_PORT       llama-server port (default 8080)
#   KB_PROXY_PORT       RAG proxy port (default 8081)
#   KB_DATA_DIR         data root (default ~/.llama-kb)
#   KB_DOCS_DIR         documents (default $KB_DATA_DIR/docs)
#   KB_CATALOG          catalog JSON (default $KB_DATA_DIR/catalog.json)
#   KB_PYTHON           Python interpreter (default python3)
#   KB_LLAMA_EXTRA_ARGS extra flags for llama-server
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${KB_PYTHON:-python3}"
DATA_DIR="${KB_DATA_DIR:-$HOME/.llama-kb}"
DOCS_DIR="${KB_DOCS_DIR:-$DATA_DIR/docs}"
CATALOG="${KB_CATALOG:-$DATA_DIR/catalog.json}"
DB_DIR="${KB_DB_DIR:-$DATA_DIR/index}"
LOG_DIR="${KB_LOG_DIR:-$DATA_DIR/logs}"
LLAMA_PORT="${KB_LLAMA_PORT:-8080}"
PROXY_PORT="${KB_PROXY_PORT:-8081}"
LLAMA_URL="${KB_LLAMA_URL:-http://127.0.0.1:$LLAMA_PORT}"
MODEL="${KB_MODEL:-}"
NGL="${KB_NGL:-}"
CTX="${KB_CTX:-8192}"
REBUILD=0
NO_BUILD=0
NO_CHAT=0
NO_LLAMA=0
LLAMA_PID=""
KB_PID=""

usage() {
    awk 'NR > 1 { if ($0 ~ /^#/) { sub(/^# ?/, ""); print } else { exit } }' "$0"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --docs) DOCS_DIR="${2:-}"; shift 2 ;;
        --catalog) CATALOG="${2:-}"; shift 2 ;;
        --rebuild) REBUILD=1; shift ;;
        --no-build) NO_BUILD=1; shift ;;
        --no-chat) NO_CHAT=1; shift ;;
        --no-llama) NO_LLAMA=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1 (use --help)" >&2; exit 2 ;;
    esac
done

mkdir -p "$DATA_DIR" "$LOG_DIR" "$DB_DIR"
LLAMA_LOG="$LOG_DIR/llama-server.log"
KB_LOG="$LOG_DIR/kb-server.log"
BUILD_LOG="$LOG_DIR/kb-build.log"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export KB_LLAMA_URL="$LLAMA_URL"
export KB_PROXY_HOST="${KB_PROXY_HOST:-127.0.0.1}"
export KB_PROXY_PORT="$PROXY_PORT"
export KB_DATA_DIR="$DATA_DIR"
export KB_DOCS_DIR="$DOCS_DIR"
export KB_CATALOG="$CATALOG"
export KB_DB_DIR="$DB_DIR"
export KB_LOG_DIR="$LOG_DIR"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export TOKENIZERS_PARALLELISM=false

port_in_use() {
    if command -v ss &>/dev/null; then
        ss -tln 2>/dev/null | grep -qE ":$1[[:space:]]"
    elif command -v lsof &>/dev/null; then
        lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
    else
        "$PYTHON" - "$1" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket()
s.settimeout(0.3)
raise SystemExit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
    fi
}

cleanup() {
    local code=$?
    trap - EXIT INT TERM
    if [ -n "$KB_PID" ] && kill -0 "$KB_PID" 2>/dev/null; then
        kill "$KB_PID" 2>/dev/null || true
    fi
    if [ -n "$LLAMA_PID" ] && kill -0 "$LLAMA_PID" 2>/dev/null; then
        kill "$LLAMA_PID" 2>/dev/null || true
    fi
    [ -n "$KB_PID" ] && wait "$KB_PID" 2>/dev/null || true
    [ -n "$LLAMA_PID" ] && wait "$LLAMA_PID" 2>/dev/null || true
    echo ""
    echo "llama-kb stopped."
    exit "$code"
}
trap cleanup EXIT INT TERM

wait_http() {
    local url="$1" seconds="$2" pid="$3" i
    for ((i = 0; i < seconds; i++)); do
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

# ---------- preflight ----------
echo "=== checking environment ==="
if ! command -v curl &>/dev/null; then
    echo "error: required command not found: curl" >&2
    exit 1
fi
if ! command -v "$PYTHON" &>/dev/null; then
    echo "error: Python not found: $PYTHON (set KB_PYTHON)" >&2
    exit 1
fi
if ! "$PYTHON" -c "import chromadb, sentence_transformers, pypdf" >/dev/null 2>&1; then
    echo "error: missing Python dependencies." >&2
    echo "  pip install -e \"$ROOT\"" >&2
    exit 1
fi
if ! "$PYTHON" -c "import llama_kb" >/dev/null 2>&1; then
    echo "error: cannot import llama_kb from $ROOT." >&2
    exit 1
fi

# ---------- first-run demo documents ----------
if [ ! -d "$DOCS_DIR" ] || [ -z "$(find "$DOCS_DIR" -type f -print -quit 2>/dev/null)" ]; then
    if [ -d "$ROOT/examples/docs" ] && [ "${KB_NO_SEED:-0}" != "1" ]; then
        mkdir -p "$DOCS_DIR"
        for sample in "$ROOT"/examples/docs/*; do
            name="$(basename "$sample")"
            [ -e "$DOCS_DIR/$name" ] || cp -R "$sample" "$DOCS_DIR/"
        done
        [ -f "$CATALOG" ] || cp "$ROOT/examples/catalog.json" "$CATALOG"
        echo "note: no documents found; seeded demo documents into $DOCS_DIR"
        echo "      replace them with your own files under $DOCS_DIR"
    fi
fi
if [ ! -f "$CATALOG" ] && [ -f "$ROOT/examples/catalog.json" ]; then
    cp "$ROOT/examples/catalog.json" "$CATALOG"
    echo "note: no catalog found; copied the demo catalog to $CATALOG"
fi

# ---------- backend ----------
START_LLAMA=0
if [ "$NO_LLAMA" -eq 1 ]; then
    if ! wait_http "$LLAMA_URL/v1/models" 3 ""; then
        echo "error: --no-llama given but $LLAMA_URL/v1/models is not reachable." >&2
        exit 1
    fi
    echo "Using existing OpenAI-compatible backend at $LLAMA_URL"
elif [ -n "$MODEL" ]; then
    if [ ! -f "$MODEL" ]; then
        echo "error: model file not found: $MODEL" >&2
        exit 1
    fi
    START_LLAMA=1
elif wait_http "$LLAMA_URL/v1/models" 2 ""; then
    echo "Found a running backend at $LLAMA_URL; attaching."
else
    cat >&2 <<EOF
error: no model specified.

Set KB_MODEL to a .gguf file to start llama.cpp:

  KB_MODEL=/path/to/model.gguf ./scripts/start.sh

Or point llama-kb at an already-running OpenAI-compatible backend:

  KB_LLAMA_URL=http://127.0.0.1:11434 ./scripts/start.sh --no-llama
EOF
    exit 1
fi

if [ "$START_LLAMA" -eq 0 ] && port_in_use "$PROXY_PORT"; then
    echo "error: proxy port $PROXY_PORT is already in use." >&2
    exit 1
fi
if [ "$START_LLAMA" -eq 1 ]; then
    for cmd in llama-server; do
        command -v "$cmd" &>/dev/null || {
            echo "error: $cmd not found in PATH (install llama.cpp)." >&2
            exit 1
        }
    done
    for port in "$LLAMA_PORT" "$PROXY_PORT"; do
        if port_in_use "$port"; then
            echo "error: port $port is already in use." >&2
            exit 1
        fi
    done
fi

# ---------- embedding model ----------
echo "=== checking embedding model ==="
if ! "$PYTHON" -m llama_kb.cli fetch-model --check >/dev/null 2>&1; then
    echo "First run: downloading the embedding model (about 1.2 GB) ..."
    if ! "$PYTHON" -m llama_kb.cli fetch-model 2>&1 | tee "$LOG_DIR/embed-fetch.log"; then
        echo "error: embedding model download failed." >&2
        exit 1
    fi
else
    echo "Embedding model ready."
fi

# ---------- index ----------
if [ "$NO_BUILD" -eq 0 ]; then
    echo "=== updating index ==="
    BUILD_ARGS=(build --if-needed)
    if [ "$REBUILD" -eq 1 ]; then
        BUILD_ARGS=(build --rebuild)
        echo "note: a full rebuild re-embeds every document."
    fi
    BUILD_ENV=(env)
    if "$PYTHON" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
    then
        if [ -z "${KB_EMBED_DEVICE:-}" ]; then
            echo "Build device: CUDA (float16)."
            BUILD_ENV=(env KB_EMBED_DEVICE=cuda KB_EMBED_DTYPE=float16)
            export KB_BUILD_BATCH="${KB_BUILD_BATCH:-16}"
        fi
    else
        echo "No CUDA detected; building on CPU (slow for large corpora)." >&2
        if [ "${KB_ALLOW_CPU_BUILD:-1}" = "0" ]; then
            echo "error: KB_ALLOW_CPU_BUILD=0; aborting CPU build." >&2
            exit 1
        fi
    fi
    if ! "${BUILD_ENV[@]}" "$PYTHON" -m llama_kb.cli "${BUILD_ARGS[@]}" 2>&1 | tee "$BUILD_LOG"; then
        echo "error: index build failed; see $BUILD_LOG" >&2
        exit 1
    fi
else
    echo "=== skipping index build (--no-build) ==="
fi

# Serving uses CPU by default so the embedding model does not compete with the
# LLM for VRAM. Override with KB_SERVE_EMBED_DEVICE / KB_SERVE_EMBED_DTYPE.
export KB_EMBED_DEVICE="${KB_SERVE_EMBED_DEVICE:-cpu}"
export KB_EMBED_DTYPE="${KB_SERVE_EMBED_DTYPE:-float32}"

CHUNKS="$("$PYTHON" - <<'PY' 2>/dev/null || true
from llama_kb.config import Config
from llama_kb.store import KBStore
print(KBStore(Config.from_env()).count())
PY
)"
if [ "${CHUNKS:-0}" = "0" ]; then
    echo "error: the index is empty. Add documents to $DOCS_DIR and rerun." >&2
    exit 1
fi
echo "Indexed chunks: $CHUNKS"

# ---------- llama.cpp ----------
if [ "$START_LLAMA" -eq 1 ]; then
    if [ -z "$NGL" ]; then
        if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
            NGL=99
        else
            NGL=0
        fi
    fi
    echo "=== starting llama-server ($MODEL) ==="
    LLAMA_ARGS=(
        -m "$MODEL"
        -ngl "$NGL"
        -c "$CTX"
        -np 1
        --host 127.0.0.1
        --port "$LLAMA_PORT"
        -a "${KB_ALIAS:-llama-kb}"
    )
    # Use the model's chat template when the installed llama.cpp supports it.
    if llama-server --help 2>&1 | grep -q -- "--jinja"; then
        LLAMA_ARGS+=(--jinja)
    fi
    # shellcheck disable=SC2086
    llama-server "${LLAMA_ARGS[@]}" ${KB_LLAMA_EXTRA_ARGS:-} \
        > "$LLAMA_LOG" 2>&1 &
    LLAMA_PID=$!
    echo "llama-server PID: $LLAMA_PID (log: $LLAMA_LOG)"
    echo "waiting for the model ..."
    if ! wait_http "http://127.0.0.1:$LLAMA_PORT/health" 300 "$LLAMA_PID"; then
        echo "error: llama-server did not become ready; last log lines:" >&2
        tail -30 "$LLAMA_LOG" >&2 || true
        exit 1
    fi
    echo "llama-server ready."
fi

# ---------- RAG proxy ----------
echo "=== starting RAG proxy on port $PROXY_PORT ==="
"$PYTHON" -m llama_kb.server --host "$KB_PROXY_HOST" --port "$PROXY_PORT" \
    > "$KB_LOG" 2>&1 &
KB_PID=$!
echo "RAG proxy PID: $KB_PID (log: $KB_LOG)"
echo "waiting for embeddings and the index ..."
if ! wait_http "http://127.0.0.1:$PROXY_PORT/kb/health" 900 "$KB_PID"; then
    echo "error: RAG proxy did not become ready; last log lines:" >&2
    tail -30 "$KB_LOG" >&2 || true
    exit 1
fi

echo ""
echo "=== ready ==="
echo "  backend:  $LLAMA_URL"
echo "  proxy:    http://127.0.0.1:$PROXY_PORT"
echo "  chunks:   $CHUNKS"
echo "  logs:     $LOG_DIR"
echo ""

if [ "$NO_CHAT" -eq 1 ]; then
    echo "Services are running (--no-chat). Press Ctrl-C to stop."
    wait "$KB_PID"
fi

"$PYTHON" -m llama_kb.chat --base-url "http://127.0.0.1:$PROXY_PORT"
