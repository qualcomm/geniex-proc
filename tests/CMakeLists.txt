# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

include(CTest)
include(FetchContent)

# ── GoogleTest ───────────────────────────────────────────────────────────────
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
# Use shared CRT on MSVC to match parent project
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# We don't want gtest leaking install() targets
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(_DEFAULT_TOKENIZER_URL
    "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer.json")
set(_DEFAULT_TOKENIZER_FILE "${CMAKE_CURRENT_BINARY_DIR}/fixtures/tokenizer.json")

set(GENIEXPROC_TEST_TOKENIZER "${_DEFAULT_TOKENIZER_FILE}"
    CACHE FILEPATH "tokenizer.json used by tokenizer unit tests")

if(NOT EXISTS "${GENIEXPROC_TEST_TOKENIZER}")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
    message(STATUS "Downloading test tokenizer: ${_DEFAULT_TOKENIZER_URL}")
    file(DOWNLOAD
        "${_DEFAULT_TOKENIZER_URL}"
        "${_DEFAULT_TOKENIZER_FILE}"
        SHOW_PROGRESS
        STATUS _DL_STATUS
        TIMEOUT 60)
    list(GET _DL_STATUS 0 _DL_CODE)
    if(NOT _DL_CODE EQUAL 0)
        list(GET _DL_STATUS 1 _DL_MSG)
        message(WARNING
            "Failed to download test tokenizer (${_DL_MSG}). "
            "Tokenizer tests will report the file missing at run time. "
            "Download manually or pass -DGENIEXPROC_TEST_TOKENIZER=<path>.")
    endif()
endif()

# Companion fixture for the chat-template tests. Same model, same repo, so a
# missing download leaves only the chat-template suite skipping at runtime.
set(_DEFAULT_TOKENIZER_CONFIG_URL
    "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer_config.json")
set(_DEFAULT_TOKENIZER_CONFIG_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/tokenizer_config.json")

set(GENIEXPROC_TEST_TOKENIZER_CONFIG "${_DEFAULT_TOKENIZER_CONFIG_FILE}"
    CACHE FILEPATH "tokenizer_config.json used by chat-template unit tests")

if(NOT EXISTS "${GENIEXPROC_TEST_TOKENIZER_CONFIG}")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
    message(STATUS "Downloading test tokenizer_config: ${_DEFAULT_TOKENIZER_CONFIG_URL}")
    file(DOWNLOAD
        "${_DEFAULT_TOKENIZER_CONFIG_URL}"
        "${_DEFAULT_TOKENIZER_CONFIG_FILE}"
        SHOW_PROGRESS
        STATUS _DL_STATUS
        TIMEOUT 60)
    list(GET _DL_STATUS 0 _DL_CODE)
    if(NOT _DL_CODE EQUAL 0)
        list(GET _DL_STATUS 1 _DL_MSG)
        message(WARNING
            "Failed to download test tokenizer_config (${_DL_MSG}). "
            "Chat-template tests will skip at run time. "
            "Download manually or pass -DGENIEXPROC_TEST_TOKENIZER_CONFIG=<path>.")
    endif()
endif()

# ── Helper: add one test binary ──────────────────────────────────────────────
function(geniexproc_add_test NAME)
    add_executable(${NAME} ${ARGN})
    target_link_libraries(${NAME} PRIVATE
        geniex-proc
        GTest::gtest_main
    )
    target_compile_definitions(${NAME} PRIVATE
        GENIEXPROC_TEST_TOKENIZER_PATH="${GENIEXPROC_TEST_TOKENIZER}"
        GENIEXPROC_TEST_TOKENIZER_CONFIG_PATH="${GENIEXPROC_TEST_TOKENIZER_CONFIG}"
    )
    set_target_properties(${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME ${NAME} COMMAND ${NAME})
    set_tests_properties(${NAME} PROPERTIES
        TIMEOUT 60
        LABELS  "unit"
    )
endfunction()

# ── Tests ────────────────────────────────────────────────────────────────────
geniexproc_add_test(sampler   sampler.cpp)
geniexproc_add_test(tokenizer tokenizer.cpp)
geniexproc_add_test(token_bytes token_bytes.cpp)

if(GENIEXPROC_ENABLE_VISION)
    geniexproc_add_test(vision
        vision.cpp
        ${CMAKE_SOURCE_DIR}/src/vision/vision.cpp
    )
    target_include_directories(vision PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/third-party/stb
    )

    geniexproc_add_test(processor processor.cpp)
    target_link_libraries(processor PRIVATE geniex-proc-vision)
    target_include_directories(processor PRIVATE
        ${CMAKE_SOURCE_DIR}/third-party/stb
    )
endif()
