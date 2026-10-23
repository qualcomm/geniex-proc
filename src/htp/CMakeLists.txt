# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

if(NOT DEFINED HEXAGON_SDK_ROOT OR "${HEXAGON_SDK_ROOT}" STREQUAL "")
    message(FATAL_ERROR "GENIEXPROC_ENABLE_HTP requires HEXAGON_SDK_ROOT")
endif()
if(NOT DEFINED HEXAGON_TOOLS_ROOT OR "${HEXAGON_TOOLS_ROOT}" STREQUAL "")
    message(FATAL_ERROR "GENIEXPROC_ENABLE_HTP requires HEXAGON_TOOLS_ROOT")
endif()

file(TO_CMAKE_PATH "${HEXAGON_SDK_ROOT}" HEXAGON_SDK_ROOT)
file(TO_CMAKE_PATH "${HEXAGON_TOOLS_ROOT}" HEXAGON_TOOLS_ROOT)
if(NOT IS_DIRECTORY "${HEXAGON_SDK_ROOT}")
    message(FATAL_ERROR "HEXAGON_SDK_ROOT is not a Hexagon SDK directory: ${HEXAGON_SDK_ROOT}")
endif()
if(NOT IS_DIRECTORY "${HEXAGON_TOOLS_ROOT}")
    message(FATAL_ERROR "HEXAGON_TOOLS_ROOT is not a Hexagon tools directory: ${HEXAGON_TOOLS_ROOT}")
endif()

include(ExternalProject)

set(_GENIEXPROC_HTP_DSP_ARCHS v73 v81)
set(_GENIEXPROC_HTP_IDL "${CMAKE_CURRENT_SOURCE_DIR}/geniex_proc_htp.idl")
set(_GENIEXPROC_HTP_IDL_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/idl")
set(_GENIEXPROC_HTP_HEADER "${_GENIEXPROC_HTP_IDL_OUT_DIR}/geniex_proc_htp.h")
set(_GENIEXPROC_HTP_STUB "${_GENIEXPROC_HTP_IDL_OUT_DIR}/geniex_proc_htp_stub.c")
set(_GENIEXPROC_HTP_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${_GENIEXPROC_HTP_IDL_OUT_DIR}" "${_GENIEXPROC_HTP_BIN_DIR}")

if(WIN32)
    set(_GENIEXPROC_HTP_QAIC "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/bin/qaic.exe")
else()
    set(_GENIEXPROC_HTP_QAIC "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/bin/qaic")
endif()
if(NOT EXISTS "${_GENIEXPROC_HTP_QAIC}")
    message(FATAL_ERROR "QAIC was not found at ${_GENIEXPROC_HTP_QAIC}")
endif()

add_custom_command(
    OUTPUT "${_GENIEXPROC_HTP_HEADER}" "${_GENIEXPROC_HTP_STUB}"
    COMMAND "${_GENIEXPROC_HTP_QAIC}"
        -mdll -o "${_GENIEXPROC_HTP_IDL_OUT_DIR}"
        "-I${HEXAGON_SDK_ROOT}/incs"
        "-I${HEXAGON_SDK_ROOT}/incs/stddef"
        "${_GENIEXPROC_HTP_IDL}"
    DEPENDS
        "${_GENIEXPROC_HTP_IDL}"
        "${HEXAGON_SDK_ROOT}/incs/stddef/AEEStdDef.idl"
        "${HEXAGON_SDK_ROOT}/incs/remote.idl"
        "${_GENIEXPROC_HTP_QAIC}"
    VERBATIM
)
add_custom_target(geniex_proc_htp_idl
    DEPENDS "${_GENIEXPROC_HTP_HEADER}" "${_GENIEXPROC_HTP_STUB}")

set(_GENIEXPROC_HTP_DSP_TARGETS)
set(_GENIEXPROC_HTP_ARCH_NUMBERS)
foreach(_arch IN LISTS _GENIEXPROC_HTP_DSP_ARCHS)
    string(REGEX REPLACE "^v" "" _arch_number "${_arch}")
    list(APPEND _GENIEXPROC_HTP_ARCH_NUMBERS "${_arch_number}")
    set(_target "geniex-proc-htp-${_arch}")
    list(APPEND _GENIEXPROC_HTP_DSP_TARGETS "${_target}")
    ExternalProject_Add(${_target}
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dsp"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/dsp-${_arch}"
        BUILD_ALWAYS TRUE
        BUILD_BYPRODUCTS "${_GENIEXPROC_HTP_BIN_DIR}/libgeniex-proc-htp-${_arch}.so"
        CMAKE_ARGS
            "-DCMAKE_BUILD_TYPE=Release"
            "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${HEXAGON_SDK_ROOT}/build/cmake/hexagon_toolchain.cmake"
            "-DHEXAGON_SDK_ROOT:PATH=${HEXAGON_SDK_ROOT}"
            "-DHEXAGON_TOOLS_ROOT:PATH=${HEXAGON_TOOLS_ROOT}"
            "-DDSP_VERSION=${_arch}"
            "-DPREBUILT_LIB_DIR=toolv19_${_arch}"
            "-DCMAKE_INSTALL_LIBDIR=${_GENIEXPROC_HTP_BIN_DIR}"
    )
endforeach()
string(REPLACE ";" "," _GENIEXPROC_HTP_ARCH_DEFINITION "${_GENIEXPROC_HTP_ARCH_NUMBERS}")

add_library(geniex_proc_htp_runtime STATIC
    "${_GENIEXPROC_HTP_STUB}"
    "${CMAKE_CURRENT_SOURCE_DIR}/htp_runtime.cpp"
)
add_dependencies(geniex_proc_htp_runtime geniex_proc_htp_idl ${_GENIEXPROC_HTP_DSP_TARGETS})
target_include_directories(geniex_proc_htp_runtime PUBLIC
    "${HEXAGON_SDK_ROOT}/incs"
    "${HEXAGON_SDK_ROOT}/incs/stddef"
    "${_GENIEXPROC_HTP_IDL_OUT_DIR}"
)
target_compile_definitions(geniex_proc_htp_runtime PRIVATE
    GENIEXPROC_HTP_ARCHS=${_GENIEXPROC_HTP_ARCH_DEFINITION}
)
if(WIN32)
    target_compile_definitions(geniex_proc_htp_runtime PUBLIC _WIN32_WINNT=0x0A00)
endif()

set(GENIEXPROC_HTP_CERT "$ENV{HEXAGON_HTP_CERT}" CACHE FILEPATH
    "Optional Windows signing certificate for the HTP DSP skels")
set(GENIEXPROC_HTP_CERT_PASS "" CACHE STRING "Optional password for GENIEXPROC_HTP_CERT")
set(GENIEXPROC_HTP_CERT_THUMBPRINT "" CACHE STRING
    "Optional LocalMachine/My signing certificate thumbprint (no PFX export needed)")
if(GENIEXPROC_HTP_CERT AND GENIEXPROC_HTP_CERT_THUMBPRINT)
    message(FATAL_ERROR "Choose GENIEXPROC_HTP_CERT or GENIEXPROC_HTP_CERT_THUMBPRINT, not both")
endif()
set(GENIEXPROC_HTP_CDSPRPC_DLL "" CACHE FILEPATH
    "Optional real libcdsprpc.dll to copy beside HTP runtime binaries")

if(WIN32)
    set(_GENIEXPROC_HTP_CDSPRPC_DEF "${CMAKE_CURRENT_SOURCE_DIR}/cdsprpc.def")
    set(_GENIEXPROC_HTP_CDSPRPC_LIB "${_GENIEXPROC_HTP_BIN_DIR}/cdsprpc.lib")
    add_custom_command(
        OUTPUT "${_GENIEXPROC_HTP_CDSPRPC_LIB}"
        COMMAND lib /DEF:${_GENIEXPROC_HTP_CDSPRPC_DEF}
            /OUT:${_GENIEXPROC_HTP_CDSPRPC_LIB} /MACHINE:ARM64
        DEPENDS "${_GENIEXPROC_HTP_CDSPRPC_DEF}"
        COMMENT "Generating cdsprpc import library"
        VERBATIM
    )
    add_custom_target(geniex_proc_htp_cdsprpc_implib DEPENDS "${_GENIEXPROC_HTP_CDSPRPC_LIB}")
    target_link_libraries(geniex_proc_htp_runtime PUBLIC "${_GENIEXPROC_HTP_CDSPRPC_LIB}")
    add_dependencies(geniex_proc_htp_runtime geniex_proc_htp_cdsprpc_implib)

    if(GENIEXPROC_HTP_CDSPRPC_DLL)
        if(NOT EXISTS "${GENIEXPROC_HTP_CDSPRPC_DLL}")
            message(FATAL_ERROR "GENIEXPROC_HTP_CDSPRPC_DLL does not exist: ${GENIEXPROC_HTP_CDSPRPC_DLL}")
        endif()
        add_custom_command(
            OUTPUT "${_GENIEXPROC_HTP_BIN_DIR}/libcdsprpc.dll"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${GENIEXPROC_HTP_CDSPRPC_DLL}" "${_GENIEXPROC_HTP_BIN_DIR}/libcdsprpc.dll"
            DEPENDS "${GENIEXPROC_HTP_CDSPRPC_DLL}"
            VERBATIM
        )
        add_custom_target(geniex_proc_htp_cdsprpc_dll DEPENDS "${_GENIEXPROC_HTP_BIN_DIR}/libcdsprpc.dll")
        add_dependencies(geniex_proc_htp_runtime geniex_proc_htp_cdsprpc_dll)
    endif()

    if(GENIEXPROC_HTP_CERT OR GENIEXPROC_HTP_CERT_THUMBPRINT)
        if(GENIEXPROC_HTP_CERT AND NOT EXISTS "${GENIEXPROC_HTP_CERT}")
            message(FATAL_ERROR "GENIEXPROC_HTP_CERT does not exist: ${GENIEXPROC_HTP_CERT}")
        endif()
        file(TO_CMAKE_PATH "$ENV{WINDOWS_SDK_BIN}/arm64" _GENIEXPROC_HTP_WINSDK_ARM64)
        file(TO_CMAKE_PATH "$ENV{WINDOWS_SDK_BIN}/x86" _GENIEXPROC_HTP_WINSDK_X86)
        find_program(_GENIEXPROC_HTP_INF2CAT NAMES inf2cat.exe Inf2Cat.exe
            PATHS ${_GENIEXPROC_HTP_WINSDK_X86} ${_GENIEXPROC_HTP_WINSDK_ARM64} REQUIRED)
        find_program(_GENIEXPROC_HTP_SIGNTOOL NAMES signtool.exe
            PATHS ${_GENIEXPROC_HTP_WINSDK_ARM64} ${_GENIEXPROC_HTP_WINSDK_X86} REQUIRED)

        set(_GENIEXPROC_HTP_SIGN_DIR "${CMAKE_CURRENT_BINARY_DIR}/sign")
        file(MAKE_DIRECTORY "${_GENIEXPROC_HTP_SIGN_DIR}")
        set(GENIEXPROC_HTP_SOURCE_FILES "")
        set(GENIEXPROC_HTP_COPY_FILES "")
        set(_GENIEXPROC_HTP_COPY_COMMANDS)
        foreach(_arch IN LISTS _GENIEXPROC_HTP_DSP_ARCHS)
            set(_name "libgeniex-proc-htp-${_arch}.so")
            string(APPEND GENIEXPROC_HTP_SOURCE_FILES "${_name} = 1\n")
            string(APPEND GENIEXPROC_HTP_COPY_FILES "${_name},,,0x10\n")
            list(APPEND _GENIEXPROC_HTP_COPY_COMMANDS COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_GENIEXPROC_HTP_BIN_DIR}/${_name}" "${_GENIEXPROC_HTP_SIGN_DIR}/${_name}")
        endforeach()
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/libgeniex-proc-htp.inf.in"
            "${_GENIEXPROC_HTP_SIGN_DIR}/libgeniex-proc-htp.inf" @ONLY)
        if(GENIEXPROC_HTP_CERT_THUMBPRINT)
            set(_GENIEXPROC_HTP_SIGN_CMD
                "${_GENIEXPROC_HTP_SIGNTOOL}" sign /fd sha256 /sm /s My
                /sha1 "${GENIEXPROC_HTP_CERT_THUMBPRINT}")
        else()
            set(_GENIEXPROC_HTP_SIGN_CMD
                "${_GENIEXPROC_HTP_SIGNTOOL}" sign /fd sha256 /f "${GENIEXPROC_HTP_CERT}")
            if(GENIEXPROC_HTP_CERT_PASS)
                list(APPEND _GENIEXPROC_HTP_SIGN_CMD /p "${GENIEXPROC_HTP_CERT_PASS}")
            endif()
        endif()
        add_custom_target(geniex_proc_htp_sign ALL
            DEPENDS ${_GENIEXPROC_HTP_DSP_TARGETS}
            ${_GENIEXPROC_HTP_COPY_COMMANDS}
            COMMAND "${_GENIEXPROC_HTP_INF2CAT}" /driver:${_GENIEXPROC_HTP_SIGN_DIR} /os:10_GE_ARM64
            COMMAND ${_GENIEXPROC_HTP_SIGN_CMD} "${_GENIEXPROC_HTP_SIGN_DIR}/libgeniex-proc-htp.cat"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_GENIEXPROC_HTP_SIGN_DIR}/libgeniex-proc-htp.cat" "${_GENIEXPROC_HTP_BIN_DIR}/libgeniex-proc-htp.cat"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_GENIEXPROC_HTP_SIGN_DIR}/libgeniex-proc-htp.inf" "${_GENIEXPROC_HTP_BIN_DIR}/libgeniex-proc-htp.inf"
            COMMENT "Signing geniex-proc HTP DSP skels"
            VERBATIM
        )
        add_dependencies(geniex_proc_htp_runtime geniex_proc_htp_sign)
    endif()
endif()

include(GNUInstallDirs)
set(_GENIEXPROC_HTP_INSTALL_COMPONENT geniex-proc-htp)
set(_GENIEXPROC_HTP_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/geniex-proc-htp")

install(CODE
    "file(REMOVE
        \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_GENIEXPROC_HTP_INSTALL_DIR}/libgeniex-proc-htp.cat\"
        \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_GENIEXPROC_HTP_INSTALL_DIR}/libgeniex-proc-htp.inf\"
    )"
    COMPONENT "${_GENIEXPROC_HTP_INSTALL_COMPONENT}"
)
install(FILES
    "${_GENIEXPROC_HTP_BIN_DIR}/libgeniex-proc-htp-v73.so"
    "${_GENIEXPROC_HTP_BIN_DIR}/libgeniex-proc-htp-v81.so"
    DESTINATION "${_GENIEXPROC_HTP_INSTALL_DIR}"
    COMPONENT "${_GENIEXPROC_HTP_INSTALL_COMPONENT}"
)
if(GENIEXPROC_HTP_CDSPRPC_DLL)
    install(FILES "${GENIEXPROC_HTP_CDSPRPC_DLL}"
        DESTINATION "${_GENIEXPROC_HTP_INSTALL_DIR}"
        RENAME libcdsprpc.dll
        COMPONENT "${_GENIEXPROC_HTP_INSTALL_COMPONENT}"
    )
else()
    install(CODE
        "file(REMOVE
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_GENIEXPROC_HTP_INSTALL_DIR}/libcdsprpc.dll\"
        )"
        COMPONENT "${_GENIEXPROC_HTP_INSTALL_COMPONENT}"
    )
endif()
if(GENIEXPROC_HTP_CERT OR GENIEXPROC_HTP_CERT_THUMBPRINT)
    install(FILES
        "${_GENIEXPROC_HTP_BIN_DIR}/libgeniex-proc-htp.cat"
        "${_GENIEXPROC_HTP_BIN_DIR}/libgeniex-proc-htp.inf"
        DESTINATION "${_GENIEXPROC_HTP_INSTALL_DIR}"
        COMPONENT "${_GENIEXPROC_HTP_INSTALL_COMPONENT}"
    )
endif()
