# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.22.2)
project(geniex-proc-htp-skel C)

include("${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake")

include_directories(
    "${HEXAGON_SDK_ROOT}/incs"
    "${HEXAGON_SDK_ROOT}/incs/stddef"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
)

set(_GENIEXPROC_HTP_SKEL geniex-proc-htp-${DSP_VERSION})
set(_GENIEXPROC_HTP_SKEL_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/geniex_proc_htp_skel.c")
add_library(${_GENIEXPROC_HTP_SKEL} SHARED
    geniex_proc_htp.c
    resize_kernel.c
    "${_GENIEXPROC_HTP_SKEL_SOURCE}"
)

# hexagon_fun.cmake adds the generated DSP skeleton to this target.
build_idl("${CMAKE_CURRENT_SOURCE_DIR}/../geniex_proc_htp.idl" ${_GENIEXPROC_HTP_SKEL})

install(TARGETS ${_GENIEXPROC_HTP_SKEL}
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
