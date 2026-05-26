#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$ENGINE_DIR/build"

echo "=== CFR Engine Build ==="
echo "Source: $ENGINE_DIR"
echo "Build:  $BUILD_DIR"
echo

# ── LibTorch (regret network inference) ──────────────────────────────────────
TORCH_PREFIX=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || echo "")
if [ -n "$TORCH_PREFIX" ]; then
    echo "LibTorch prefix: $TORCH_PREFIX"
    CMAKE_EXTRA="-DCMAKE_PREFIX_PATH=$TORCH_PREFIX"
else
    echo "LibTorch not found — model-based traversal disabled"
    CMAKE_EXTRA=""
fi

# ── ONNX Runtime (blueprint strategy inference) ───────────────────────────────
# Auto-detect from environment or common install locations.
ORT_ROOT="${ONNXRUNTIME_ROOT:-}"

if [ -z "$ORT_ROOT" ]; then
    for candidate in \
        "/opt/homebrew" \
        "/usr/local" \
        "/usr/local/opt/onnxruntime" \
        "/usr"
    do
        # Check either header location variant
        if [ -f "$candidate/include/onnxruntime/core/session/onnxruntime_cxx_api.h" ] || \
           [ -f "$candidate/include/onnxruntime_cxx_api.h" ]; then
            ORT_ROOT="$candidate"
            break
        fi
    done
fi

if [ -n "$ORT_ROOT" ]; then
    echo "ONNX Runtime root: $ORT_ROOT"
    CMAKE_EXTRA="$CMAKE_EXTRA -DONNXRUNTIME_ROOT=$ORT_ROOT"
else
    echo "ONNX Runtime not found — blueprint C++ inference disabled"
    echo "  macOS:  brew install onnxruntime"
    echo "  Custom: export ONNXRUNTIME_ROOT=/path/to/ort && bash build.sh"
fi

echo

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$ENGINE_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    $CMAKE_EXTRA

NPROC=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)
make -j"$NPROC"

echo
echo "=== Copying .so to cpp_engine/ ==="
SO_FILE=$(find . -name "cfr_engine*.so" | head -1)
if [ -n "$SO_FILE" ]; then
    cp "$SO_FILE" "$ENGINE_DIR/"
    echo "Copied: $SO_FILE → $ENGINE_DIR/"
else
    echo "WARNING: .so not found"
fi

echo
echo "=== Build complete ==="