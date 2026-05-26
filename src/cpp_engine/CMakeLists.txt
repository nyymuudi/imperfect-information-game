cmake_minimum_required(VERSION 3.18)
project(cfr_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ── Optional CUDA ─────────────────────────────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_compile_definitions(CFR_CUDA_AVAILABLE)
    message(STATUS "CUDA found")
else()
    message(STATUS "CUDA not found — CPU-only")
endif()

# ── Optional LibTorch ─────────────────────────────────────────────────────────
# Used for: regret network inference during MCCFR traversal (hot path).
# Locate: python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
# Or set: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch QUIET)
if(Torch_FOUND)
    add_compile_definitions(CFR_TORCH_AVAILABLE)
    message(STATUS "LibTorch found: ${TORCH_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    message(STATUS "LibTorch not found — model-based traversal disabled")
    message(STATUS "  To enable: cmake -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') ..")
endif()

# ── Optional ONNX Runtime ─────────────────────────────────────────────────────
# Used for: blueprint strategy network inference (subgame solving).
# The strategy network is exported as ONNX from blueprint.py.
#
# Install (macOS):  brew install onnxruntime
# Install (Linux):  apt install libonnxruntime-dev
#                   OR download from https://github.com/microsoft/onnxruntime/releases
# Override:         cmake -DONNXRUNTIME_ROOT=/path/to/ort ..

set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH
    "ONNX Runtime root (auto-detected from homebrew if empty)")

# Search paths: user override → homebrew Apple Silicon → homebrew Intel → system
set(_ORT_SEARCH_PATHS
    "${ONNXRUNTIME_ROOT}"
    "/opt/homebrew"
    "/usr/local"
    "/usr/local/opt/onnxruntime"
    "/usr"
)

set(ONNXRUNTIME_INCLUDE_DIR "")
set(ONNXRUNTIME_LIB         "")

foreach(_ORT_PATH ${_ORT_SEARCH_PATHS})
    # Header location varies by install method
    foreach(_INC_SUFFIX
        "include/onnxruntime/core/session"
        "include/onnxruntime"
        "include"
    )
        if(EXISTS "${_ORT_PATH}/${_INC_SUFFIX}/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_INCLUDE_DIR "${_ORT_PATH}/${_INC_SUFFIX}")
            break()
        endif()
    endforeach()

    if(ONNXRUNTIME_INCLUDE_DIR)
        find_library(ONNXRUNTIME_LIB onnxruntime
            HINTS "${_ORT_PATH}/lib"
            NO_DEFAULT_PATH
        )
        if(ONNXRUNTIME_LIB)
            break()
        endif()
        set(ONNXRUNTIME_INCLUDE_DIR "")
    endif()
endforeach()

if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE_DIR)
    add_compile_definitions(CFR_ORT_AVAILABLE)
    message(STATUS "ONNX Runtime found:")
    message(STATUS "  Library: ${ONNXRUNTIME_LIB}")
    message(STATUS "  Include: ${ONNXRUNTIME_INCLUDE_DIR}")
else()
    message(STATUS "ONNX Runtime not found — blueprint C++ inference disabled")
    message(STATUS "  macOS:  brew install onnxruntime")
    message(STATUS "  Linux:  see https://github.com/microsoft/onnxruntime/releases")
    message(STATUS "  Custom: cmake -DONNXRUNTIME_ROOT=/path/to/ort ..")
endif()

# ── pybind11 ──────────────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# ── Sources ───────────────────────────────────────────────────────────────────
set(CPP_SOURCES
    src/leduc_game.cpp
    src/mccfr.cpp
    src/nlhe_game.cpp
    src/nlhe_mccfr.cpp
    src/torch_model.cpp
    src/bindings.cpp
)

if(CMAKE_CUDA_COMPILER)
    list(APPEND CPP_SOURCES cuda/reservoir_buffer.cu)
endif()

pybind11_add_module(cfr_engine ${CPP_SOURCES})
target_include_directories(cfr_engine PRIVATE include)
target_compile_options(cfr_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
)

if(Torch_FOUND)
    target_link_libraries(cfr_engine PRIVATE ${TORCH_LIBRARIES})
endif()

if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE_DIR)
    target_include_directories(cfr_engine PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(cfr_engine PRIVATE ${ONNXRUNTIME_LIB})
endif()

# ── Test binary ───────────────────────────────────────────────────────────────
add_executable(test_engine
    tests/test_game.cpp
    src/leduc_game.cpp src/mccfr.cpp
    src/nlhe_game.cpp  src/nlhe_mccfr.cpp
    src/torch_model.cpp
)
target_include_directories(test_engine PRIVATE include)
if(Torch_FOUND)
    target_link_libraries(test_engine PRIVATE ${TORCH_LIBRARIES})
endif()
if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE_DIR)
    target_include_directories(test_engine PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(test_engine PRIVATE ${ONNXRUNTIME_LIB})
endif()

install(TARGETS cfr_engine DESTINATION .)