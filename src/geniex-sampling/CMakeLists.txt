# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: Apache-2.0

set(GENIEX_SAMPLING_SRCS
    geniex-sampling.cpp
    geniex-grammar.cpp
    sampling-vocab.cpp
)

if (GENIEXPROC_BUILD_SHARED_LIB)
    add_library(geniex-sampling SHARED ${GENIEX_SAMPLING_SRCS})
    target_compile_definitions(geniex-sampling PRIVATE GENIEXPROC_BUILD)
    target_compile_definitions(geniex-sampling PUBLIC GENIEXPROC_SHARED)
    if(WIN32)
        set_target_properties(geniex-sampling PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON
        )
    endif()
else()
    add_library(geniex-sampling STATIC ${GENIEX_SAMPLING_SRCS})
endif()

target_link_libraries(geniex-sampling PUBLIC tokenizers_cpp)

target_include_directories(geniex-sampling
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)