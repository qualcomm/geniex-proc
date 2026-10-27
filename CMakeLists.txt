# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.10)
project(geniex-proc VERSION 0.1.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # Used for VSCode intellisense to highlight errors

option(GENIEXPROC_INSTALL          "Install geniex-proc"                    OFF)
option(GENIEXPROC_BUILD_SHARED_LIBS "Build geniex-proc libraries as shared (.dll/.so) instead of static (.lib/.a)" OFF)
option(GENIEXPROC_BUILD_TESTS      "Build and register CTest unit tests (uses GoogleTest via FetchContent)" OFF)

# Library type — controlled by GENIEXPROC_BUILD_SHARED_LIBS
if(GENIEXPROC_BUILD_SHARED_LIBS)
    set(_GENIEXPROC_LIB_TYPE SHARED)
    message(STATUS "geniex-proc: Building as shared libraries")
else()
    set(_GENIEXPROC_LIB_TYPE STATIC)
endif()

# Modality flags — enable a modality to compile all processors for that modality.
# ENABLE_VISION: stb image loading/resizing + all VLM processor code (Qwen2-VL, Qwen3-VL, ...)
# ENABLE_AUDIO:  libsndfile/soxr/fftw3 + all Omni processor code (Qwen2.5-Omni, Gemma-3N, ...)
# ENABLE_VIDEO:  decord/ffmpeg + video processor code
option(GENIEXPROC_ENABLE_VISION "Build image processing + all VLM processors (stb)"                OFF)
option(GENIEXPROC_ENABLE_AUDIO  "Build audio processing + all Omni processors (libsndfile, soxr, fftw3)" OFF)
option(GENIEXPROC_ENABLE_VIDEO  "Build video processing (decord, ffmpeg)"                             OFF)
option(GENIEXPROC_ENABLE_OPENCL "Build OpenCL image preprocessing"                       OFF)
option(GENIEXPROC_ENABLE_HTP "Build Hexagon HTP image preprocessing"                         OFF)

include(GNUInstallDirs)
macro(geniexproc_install_htp_cleanup)
    if(IS_ABSOLUTE "${CMAKE_INSTALL_BINDIR}")
        set(_geniexproc_htp_install_dir "\$ENV{DESTDIR}${CMAKE_INSTALL_BINDIR}/geniex-proc-htp")
    else()
        set(_geniexproc_htp_install_dir "\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/geniex-proc-htp")
    endif()
    install(CODE
        "set(_geniexproc_htp_dir \"${_geniexproc_htp_install_dir}\")
        file(GLOB _geniexproc_htp_skels \"\${_geniexproc_htp_dir}/libgeniex-proc-htp-v*.so\")
        file(REMOVE \${_geniexproc_htp_skels}
            \"\${_geniexproc_htp_dir}/libgeniex-proc-htp.cat\"
            \"\${_geniexproc_htp_dir}/libgeniex-proc-htp.inf\"
            \"\${_geniexproc_htp_dir}/libcdsprpc.dll\"
        )"
        COMPONENT geniex-proc-htp
    )
endmacro()

if(GENIEXPROC_ENABLE_OPENCL OR GENIEXPROC_ENABLE_HTP)
    if(NOT GENIEXPROC_ENABLE_VISION)
        message(FATAL_ERROR "Device preprocessing requires GENIEXPROC_ENABLE_VISION")
    endif()
endif()
if(GENIEXPROC_ENABLE_OPENCL)
    if(NOT OpenCL_ROOT AND DEFINED ENV{OPENCL_SDK_ROOT})
        set(OpenCL_ROOT "$ENV{OPENCL_SDK_ROOT}")
    endif()
    if(POLICY CMP0074)
        cmake_policy(SET CMP0074 NEW)
    endif()
    find_package(OpenCL REQUIRED)
endif()
if(GENIEXPROC_ENABLE_HTP)
    enable_language(C)
    if(NOT HEXAGON_SDK_ROOT AND DEFINED ENV{HEXAGON_SDK_ROOT})
        set(HEXAGON_SDK_ROOT "$ENV{HEXAGON_SDK_ROOT}")
    endif()
    if(NOT HEXAGON_TOOLS_ROOT AND DEFINED ENV{HEXAGON_TOOLS_ROOT})
        set(HEXAGON_TOOLS_ROOT "$ENV{HEXAGON_TOOLS_ROOT}")
    endif()
    add_subdirectory(src/vision/npu)
endif()

# Keep tokenizers-cpp and xtensor always static regardless of our shared flag
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)

# place `include_directories` before dependencies to avoid headers naming conflicts
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party
)

find_package(OpenBLAS QUIET)
if(OpenBLAS_FOUND)
    message(STATUS "geniex-proc: OpenBLAS found")
endif()

add_subdirectory(third-party/xtl)
add_subdirectory(third-party/xtensor)
add_subdirectory(third-party/xtensor-io)
add_subdirectory(third-party/xtensor-blas)

if(OpenBLAS_FOUND)
    target_compile_definitions(xtensor-blas INTERFACE WITH_OPENBLAS)
    if(TARGET OpenBLAS::OpenBLAS)
        target_link_libraries(xtensor-blas INTERFACE OpenBLAS::OpenBLAS)
    else()
        target_link_libraries(xtensor-blas INTERFACE -lopenblas)
    endif()
endif()

add_subdirectory(third-party/tokenizers-cpp)
include_directories(
    third-party/tokenizers-cpp/include
)

# minja is header-only. We avoid add_subdirectory(third-party/minja) because
# its CMakeLists FetchContents nlohmann/json (we already vendor v3.12.0 at
# include/json.hpp, re-exported as <nlohmann/json.hpp>) and builds gtest /
# fuzztest / examples we don't need. Linked PRIVATELY below so neither minja
# nor nlohmann headers leak through geniex-proc's public include surface.
add_library(geniexproc_minja INTERFACE)
target_include_directories(geniexproc_minja INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/minja/include
)
target_compile_features(geniexproc_minja INTERFACE cxx_std_17)

# ─────────────────────────────────────────────────────────────────────────────
# Helper macro: configure common properties shared by all three libs
# ─────────────────────────────────────────────────────────────────────────────
macro(geniexproc_configure_target TARGET_NAME)
    # Use explicit API exports only; do not auto-export every symbol.
    if(GENIEXPROC_BUILD_SHARED_LIBS)
        target_compile_definitions(${TARGET_NAME} PRIVATE GENIEXPROC_BUILD)
        target_compile_definitions(${TARGET_NAME} PUBLIC GENIEXPROC_SHARED)

        if(NOT WIN32)
            set_target_properties(${TARGET_NAME} PROPERTIES
                CXX_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN YES
            )
        endif()
    endif()
    target_include_directories(${TARGET_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/geniex-sampling
        ${CMAKE_CURRENT_SOURCE_DIR}/third-party/tokenizers-cpp/include
    )
endmacro()

# ─────────────────────────────────────────────────────────────────────────────
# geniex-proc  —  core: tokenizer + sampler (always built)
# ─────────────────────────────────────────────────────────────────────────────
message(STATUS "geniex-proc: Building core library (${_GENIEXPROC_LIB_TYPE})")
add_library(geniex-proc ${_GENIEXPROC_LIB_TYPE}
    src/tokenizer/tokenizer.cpp
    src/tokenizer/tokenizer_config.cpp
    src/sampler/sampler.cpp
    src/geniex-sampling/geniex-sampling.cpp
    src/geniex-sampling/geniex-grammar.cpp
    src/geniex-sampling/sampling-vocab.cpp
    src/internal/token_bytes.cpp
)
geniexproc_configure_target(geniex-proc)
target_link_libraries(geniex-proc PUBLIC
    tokenizers_cpp
    xtl
    xtensor
    xtensor-io
    xtensor-blas
)
# Private — minja and nlohmann must not appear in geniex-proc's public headers.
target_link_libraries(geniex-proc PRIVATE geniexproc_minja)

# ─────────────────────────────────────────────────────────────────────────────
# geniex-proc-vision  —  image processing
# Built only when GENIEXPROC_ENABLE_VISION is ON.
# Links geniex-proc so consumers only need to link this one target.
# ─────────────────────────────────────────────────────────────────────────────
if(GENIEXPROC_ENABLE_VISION)
    message(STATUS "geniex-proc: Vision support enabled — building geniex-proc-vision (stb, ${_GENIEXPROC_LIB_TYPE})")
    add_library(geniex-proc-vision ${_GENIEXPROC_LIB_TYPE}
        src/vision/vision.cpp
        src/vision/qwen2vl.cpp
        src/vision/preprocess_device.cpp
        src/vision/cpu/image_ops.cpp
        src/processors/processor.cpp
        src/processors/qwen2vl.cpp
        src/processors/gemma4.cpp
        src/processors/internvl.cpp
    )
    geniexproc_configure_target(geniex-proc-vision)
    if(GENIEXPROC_ENABLE_OPENCL)
        target_sources(geniex-proc-vision PRIVATE
            src/vision/gpu/gpu_runtime.cpp
            src/vision/gpu/image_ops.cpp
            src/vision/gpu/qwen2vl_patches.cpp
        )
        target_link_libraries(geniex-proc-vision PRIVATE OpenCL::OpenCL)
        target_compile_definitions(geniex-proc-vision PRIVATE GENIEXPROC_ENABLE_OPENCL)
    endif()
    if(GENIEXPROC_ENABLE_HTP)
        target_sources(geniex-proc-vision PRIVATE
            src/vision/npu/npu_runtime.cpp
            src/vision/npu/image_ops.cpp
            src/vision/npu/qwen2vl_patches.cpp
        )
        target_link_libraries(geniex-proc-vision PRIVATE geniex_proc_htp_runtime)
        target_compile_definitions(geniex-proc-vision PRIVATE GENIEXPROC_ENABLE_HTP)
    endif()
    target_include_directories(geniex-proc-vision PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third-party/stb
    )
    # Propagate the flag so consumers can #ifdef on it
    target_compile_definitions(geniex-proc-vision PUBLIC GENIEXPROC_ENABLE_VISION)
    target_link_libraries(geniex-proc-vision PUBLIC geniex-proc)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# geniex-proc-audio  —  audio processing
# Built only when GENIEXPROC_ENABLE_AUDIO is ON.
# Links geniex-proc so consumers only need to link this one target.
# ─────────────────────────────────────────────────────────────────────────────
if(GENIEXPROC_ENABLE_AUDIO)
    message(STATUS "geniex-proc: Audio support enabled — building geniex-proc-audio (${_GENIEXPROC_LIB_TYPE})")
    add_library(geniex-proc-audio ${_GENIEXPROC_LIB_TYPE}
        # src/audio/audio.cpp
        # src/processors/qwen25omni.cpp
    )
    geniexproc_configure_target(geniex-proc-audio)
    target_compile_definitions(geniex-proc-audio PUBLIC GENIEXPROC_ENABLE_AUDIO)
    target_link_libraries(geniex-proc-audio PUBLIC geniex-proc)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Install
# ─────────────────────────────────────────────────────────────────────────────
if(NOT GENIEXPROC_ENABLE_HTP)
    geniexproc_install_htp_cleanup()
endif()

if(GENIEXPROC_INSTALL)
    set(_INSTALL_TARGETS geniex-proc tokenizers_cpp xtl xtensor xtensor-io xtensor-blas)
    if(GENIEXPROC_ENABLE_VISION)
        list(APPEND _INSTALL_TARGETS geniex-proc-vision)
    endif()
    if(GENIEXPROC_ENABLE_AUDIO)
        list(APPEND _INSTALL_TARGETS geniex-proc-audio)
    endif()

    install(TARGETS ${_INSTALL_TARGETS}
        EXPORT geniex-procTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )

    install(DIRECTORY include/ DESTINATION include)

    install(EXPORT geniex-procTargets
        FILE geniex-procTargets.cmake
        DESTINATION lib/cmake/geniex-proc
    )
endif(GENIEXPROC_INSTALL)

# ─────────────────────────────────────────────────────────────────────────────
# Tests  —  CTest suite (opt-in via GENIEXPROC_BUILD_TESTS)
# ─────────────────────────────────────────────────────────────────────────────
if(GENIEXPROC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
