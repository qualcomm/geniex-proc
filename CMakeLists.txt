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

# Keep tokenizers-cpp and xtensor always static regardless of our shared flag
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)

# place `include_directories` before dependencies to avoid headers naming conflicts
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

find_package(OpenBLAS QUIET)
if(OpenBLAS_FOUND)
    message(STATUS "geniex-proc: OpenBLAS found")
endif()

add_subdirectory(third_party/xtl)
add_subdirectory(third_party/xtensor)
add_subdirectory(third_party/xtensor-io)
add_subdirectory(third_party/xtensor-blas)

if(OpenBLAS_FOUND)
    target_compile_definitions(xtensor-blas INTERFACE WITH_OPENBLAS)
    if(TARGET OpenBLAS::OpenBLAS)
        target_link_libraries(xtensor-blas INTERFACE OpenBLAS::OpenBLAS)
    else()
        target_link_libraries(xtensor-blas INTERFACE -lopenblas)
    endif()
endif()

# Normalize CMAKE_SYSTEM_PROCESSOR so tokenizers-cpp picks the correct Rust
# target triple. The SDK toolchain sets it to lowercase "arm64" but
# tokenizers-cpp only checks for "ARM64" or "aarch64".
string(TOUPPER "${CMAKE_SYSTEM_PROCESSOR}" _proc_upper)
if(_proc_upper STREQUAL "ARM64")
    set(CMAKE_SYSTEM_PROCESSOR "ARM64")
endif()
unset(_proc_upper)

# sentencepiece sets CMAKE_POSITION_INDEPENDENT_CODE ON when SPM_ENABLE_SHARED
# is ON, which emits -fPIC — unsupported on Windows targets. Force it off.
if(WIN32)
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "Disable sentencepiece shared lib on Windows" FORCE)
endif()

# When building with Clang targeting *-windows-msvc, CMake's MSVC variable is
# false (it is only true for cl.exe). tokenizers-cpp uses MSVC to decide
# between tokenizers_c.lib and libtokenizers_c.a, but Rust on Windows always
# produces a .lib regardless of compiler.
#
# We cannot override TOKENIZERS_RUST_LIB via cache because tokenizers-cpp uses
# a normal set() which shadows cache values in its own scope. Instead, we
# generate a cargo wrapper script at configure time. The wrapper invokes the
# real cargo with all forwarded arguments, then copies tokenizers_c.lib to
# libtokenizers_c.a so the tokenizers-cpp copy step finds the file it expects.
if(WIN32 AND NOT MSVC)
    find_program(_real_cargo cargo)
    set(_cargo_wrapper "${CMAKE_CURRENT_BINARY_DIR}/cargo_wrapper.cmake")
    file(WRITE "${_cargo_wrapper}" [=[
# Auto-generated cargo wrapper — forwards all args to real cargo, then
# aliases the Windows .lib output as .a for tokenizers-cpp compatibility.
#
# Invoked as: cmake -P script.cmake -- build --target ... --release
# CMAKE_ARGV0=cmake  CMAKE_ARGV1=-P  CMAKE_ARGV2=script  CMAKE_ARGV3=--
# CMAKE_ARGV4..N = actual cargo args.  CMAKE_ARGC = total count.
set(CARGO_ARGS)
set(i 4)  # skip cmake, -P, script path, --
while(i LESS CMAKE_ARGC)
    list(APPEND CARGO_ARGS "${CMAKE_ARGV${i}}")
    math(EXPR i "${i}+1")
endwhile()

execute_process(
    COMMAND "@_real_cargo@" ${CARGO_ARGS}
    RESULT_VARIABLE _ret
)
if(NOT _ret EQUAL 0)
    message(FATAL_ERROR "cargo failed with exit code ${_ret}")
endif()

# Locate the built .lib and create a .a alias beside it.
set(_cargo_target_dir "$ENV{CARGO_TARGET_DIR}")
file(GLOB_RECURSE _libs "${_cargo_target_dir}/**/tokenizers_c.lib")
foreach(_lib IN LISTS _libs)
    get_filename_component(_dir "${_lib}" DIRECTORY)
    set(_a "${_dir}/libtokenizers_c.a")
    if(NOT EXISTS "${_a}")
        file(COPY_FILE "${_lib}" "${_a}")
    endif()
endforeach()
]=])
    # Substitute the real cargo path into the script.
    file(READ "${_cargo_wrapper}" _wrapper_content)
    string(REPLACE "@_real_cargo@" "${_real_cargo}" _wrapper_content "${_wrapper_content}")
    file(WRITE "${_cargo_wrapper}" "${_wrapper_content}")
    unset(_wrapper_content)

    # Point tokenizers-cpp at our wrapper instead of cargo directly.
    set(CARGO_EXECUTABLE
        "${CMAKE_COMMAND}" "-P" "${_cargo_wrapper}" "--"
        CACHE STRING "Cargo wrapper for Windows Clang builds" FORCE)
    unset(_cargo_wrapper)
    unset(_real_cargo)
endif()

# sentencepiece/src/CMakeLists.txt hardcodes "-fPIC" and "-pthread" into
# CMAKE_CXX_FLAGS unconditionally (line ~256). Clang targeting
# arm64-pc-windows-msvc rejects -fPIC as a hard error. We patch the file
# at configure time to guard those flags with an if(NOT WIN32) check.
# The patch is idempotent — it only applies when the original string is present.
if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_spm_src_cmake
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp/sentencepiece/src/CMakeLists.txt")
    file(READ "${_spm_src_cmake}" _spm_content)
    string(FIND "${_spm_content}" [=[set(CMAKE_CXX_FLAGS "-O3 -Wall -fPIC ${CMAKE_CXX_FLAGS}")]=] _spm_pos)
    if(_spm_pos GREATER_EQUAL 0)
        string(REPLACE
            [=[set(CMAKE_CXX_FLAGS "-O3 -Wall -fPIC ${CMAKE_CXX_FLAGS}")]=]
            [=[if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "-O3 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "-O3 -Wall ${CMAKE_CXX_FLAGS}")
endif()]=]
            _spm_content "${_spm_content}")
        file(WRITE "${_spm_src_cmake}" "${_spm_content}")
        message(STATUS "geniex-proc: Patched sentencepiece CMakeLists.txt to remove -fPIC/-pthread on Windows")
    endif()
    unset(_spm_content)
    unset(_spm_src_cmake)
    unset(_spm_pos)
endif()

add_subdirectory(third_party/tokenizers-cpp)
include_directories(
    third_party/tokenizers-cpp/include
)

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
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp/include
    )
endmacro()

# ─────────────────────────────────────────────────────────────────────────────
# geniex-proc  —  core: tokenizer + sampler (always built)
# ─────────────────────────────────────────────────────────────────────────────
message(STATUS "geniex-proc: Building core library (${_GENIEXPROC_LIB_TYPE})")
add_library(geniex-proc ${_GENIEXPROC_LIB_TYPE}
    src/tokenizer/tokenizer.cpp
    src/sampler/sampler.cpp
    src/geniex-sampling/geniex-sampling.cpp
    src/geniex-sampling/geniex-grammar.cpp
    src/geniex-sampling/sampling-vocab.cpp
)
geniexproc_configure_target(geniex-proc)
target_link_libraries(geniex-proc PUBLIC
    tokenizers_cpp
    xtl
    xtensor
    xtensor-io
    xtensor-blas
)

# ─────────────────────────────────────────────────────────────────────────────
# geniex-proc-vision  —  image processing
# Built only when GENIEXPROC_ENABLE_VISION is ON.
# Links geniex-proc so consumers only need to link this one target.
# ─────────────────────────────────────────────────────────────────────────────
if(GENIEXPROC_ENABLE_VISION)
    message(STATUS "geniex-proc: Vision support enabled — building geniex-proc-vision (stb, ${_GENIEXPROC_LIB_TYPE})")
    add_library(geniex-proc-vision ${_GENIEXPROC_LIB_TYPE}
        src/vision/vision.cpp
        src/processors/processor.cpp
        src/processors/qwen2vl.cpp
    )
    geniexproc_configure_target(geniex-proc-vision)
    target_include_directories(geniex-proc-vision PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
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
    # TODO: link libsndfile, soxr, fftw3 here when audio sources are added
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Install
# ─────────────────────────────────────────────────────────────────────────────
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
