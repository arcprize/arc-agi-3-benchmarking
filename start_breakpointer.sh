#!/bin/bash
# Build the breakpoint UI and start the breakpoint server (HTTP + WebSocket)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BREAKPOINTER_DIR="${SCRIPT_DIR}/breakpointer"
BREAKPOINTER_DIST="${BREAKPOINTER_DIR}/dist"

# Set default ports if not specified
HTTP_PORT=8080
WS_PORT=8765

# Check if user has provided --http-port or --ws-port in $@
EXTRA_ARGS=()
USE_HTTP_PORT_DEFAULT=true
USE_WS_PORT_DEFAULT=true
USE_STATIC_DIR_DEFAULT=true
for arg in "$@"; do
  if [[ "$arg" == --http-port* ]]; then
    USE_HTTP_PORT_DEFAULT=false
  fi
  if [[ "$arg" == --ws-port* ]]; then
    USE_WS_PORT_DEFAULT=false
  fi
  if [[ "$arg" == --static-dir* ]]; then
    USE_STATIC_DIR_DEFAULT=false
  fi
  EXTRA_ARGS+=("$arg")
done

if $USE_HTTP_PORT_DEFAULT; then
  EXTRA_ARGS+=(--http-port "$HTTP_PORT")
fi
if $USE_WS_PORT_DEFAULT; then
  EXTRA_ARGS+=(--ws-port "$WS_PORT")
fi
if $USE_STATIC_DIR_DEFAULT; then
  EXTRA_ARGS+=(--static-dir "$BREAKPOINTER_DIST")
fi

# Pick a python interpreter. This script is often run in a non-interactive shell,
# so relying on a `python` shim can fail if pyenv isn't initialized.
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if command -v pyenv >/dev/null 2>&1; then
    PYTHON_BIN="$(pyenv which python 2>/dev/null || true)"
  fi
fi
if [ -z "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python 2>/dev/null || true)"
fi
if [ -z "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python3 2>/dev/null || true)"
fi
if [ -z "${PYTHON_BIN}" ]; then
  echo "ERROR: Could not find a Python interpreter. Set PYTHON_BIN or configure pyenv." >&2
  exit 1
fi

# Build UI (served by the Python HTTP server at :$HTTP_PORT)
NEEDS_REBUILD=false
if [ ! -d "$BREAKPOINTER_DIST" ]; then
  NEEDS_REBUILD=true
else
  # Rebuild if any UI source file is newer than the built index.html
  if [ -f "$BREAKPOINTER_DIST/index.html" ]; then
    if find "$BREAKPOINTER_DIR/src" "$BREAKPOINTER_DIR/public" \
      "$BREAKPOINTER_DIR/index.html" "$BREAKPOINTER_DIR/vite.config.ts" "$BREAKPOINTER_DIR/package.json" \
      -type f -newer "$BREAKPOINTER_DIST/index.html" -print -quit 2>/dev/null | grep -q .; then
      NEEDS_REBUILD=true
    fi
  else
    NEEDS_REBUILD=true
  fi
fi

if $NEEDS_REBUILD; then
  echo "Building breakpoint UI..."
  cd "${BREAKPOINTER_DIR}"
  npm run build
  cd "${SCRIPT_DIR}"
else
  echo "Breakpoint UI build is up-to-date."
fi

# Start the Python breakpoint server in the background
echo "Starting breakpoint server (backend)..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/run_breakpoint_server.py" "${EXTRA_ARGS[@]}" &
SERVER_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $SERVER_PID 2>/dev/null || true
    exit
}

# Trap Ctrl-C and cleanup
trap cleanup INT TERM

echo ""
echo "Breakpoint server running at: http://localhost:$HTTP_PORT"
echo "WebSocket server running at: ws://localhost:$WS_PORT/ws"
echo "Press Ctrl-C to stop"
echo ""

# Wait for both processes
wait

