#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$ENGINE_DIR/build"

echo "=== CFR Engine Build ==="
echo "Source: $ENGINE_DIR"
echo "Build:  $BUILD_DIR"
echo

# Locate LibTorch via installed PyTorch
TORCH_PREFIX=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || echo "")
if [ -n "$TORCH_PREFIX" ]; then
    echo "LibTorch prefix: $TORCH_PREFIX"
    CMAKE_EXTRA="-DCMAKE_PREFIX_PATH=$TORCH_PREFIX"
else
    echo "LibTorch not found — building without model-based traversal"
    CMAKE_EXTRA=""
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$ENGINE_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    $CMAKE_EXTRA

# macOS: sysctl -n hw.logicalcpu; Linux: nproc
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