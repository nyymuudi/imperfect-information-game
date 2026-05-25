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
# Locate via installed PyTorch: python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
# Or set manually: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch QUIET)
if(Torch_FOUND)
    add_compile_definitions(CFR_TORCH_AVAILABLE)
    message(STATUS "LibTorch found: ${TORCH_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    message(STATUS "LibTorch not found — model-based traversal disabled")
    message(STATUS "  To enable: cmake -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') ..")
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

install(TARGETS cfr_engine DESTINATION .)