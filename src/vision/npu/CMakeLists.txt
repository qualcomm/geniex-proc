# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

if(WIN32)
    set(_GENIEXPROC_HTP_DEFAULT_ARCHS v73 v81)
    set(_GENIEXPROC_HTP_TARGET_ARCH_AVAILABLE FALSE)
    foreach(_variable CMAKE_GENERATOR_PLATFORM CMAKE_CXX_COMPILER_ARCHITECTURE_ID)
        if(NOT "${${_variable}}" STREQUAL "")
            set(_GENIEXPROC_HTP_TARGET_ARCH_AVAILABLE TRUE)
            string(TOLOWER "${${_variable}}" _value)
            if(_variable STREQUAL "CMAKE_GENERATOR_PLATFORM")
                string(REGEX REPLACE ",.*$" "" _value "${_value}")
            endif()
            if(NOT _value STREQUAL "arm64" AND NOT _value STREQUAL "aarch64")
                message(FATAL_ERROR "GENIEXPROC_ENABLE_HTP on Windows requires an ARM64 target; ${_variable} is '${${_variable}}'")
            endif()
        endif()
    endforeach()
    if(NOT _GENIEXPROC_HTP_TARGET_ARCH_AVAILABLE)
        string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _GENIEXPROC_HTP_PROCESSOR)
        if(NOT _GENIEXPROC_HTP_PROCESSOR STREQUAL "arm64" AND NOT _GENIEXPROC_HTP_PROCESSOR STREQUAL "aarch64")
            message(FATAL_ERROR "GENIEXPROC_ENABLE_HTP on Windows requires an ARM64 target (got '${CMAKE_SYSTEM_PROCESSOR}')")
        endif()
    endif()
elseif(ANDROID)
    set(_GENIEXPROC_HTP_DEFAULT_ARCHS v79 v81)
    if(NOT CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
        message(FATAL_ERROR "GENIEXPROC_ENABLE_HTP on Android requires CMAKE_ANDROID_ARCH_ABI=arm64-v8a (got '${CMAKE_ANDROID_ARCH_ABI}')")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_GENIEXPROC_HTP_DEFAULT_ARCHS v73 v75)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _GENIEXPROC_HTP_PROCESSOR)
    if(NOT _GENIEXPROC_HTP_PROCESSOR STREQUAL "aarch64" AND NOT _GENIEXPROC_HTP_PROCESSOR STREQUAL "arm64")
        message(FATAL_ERROR "GENIEXPROC_ENABLE_HTP on Linux requires an ARM64 target (got '${CMAKE_SYSTEM_PROCESSOR}')")
    endif()
else()
    message(FATAL_ERROR "GENIEXPROC_ENABLE_HTP supports Windows ARM64, Linux ARM64, and Android arm64-v8a only (got ${CMAKE_SYSTEM_NAME})")
endif()

set(_GENIEXPROC_HTP_SUPPORTED_ARCHS v68 v69 v73 v75 v79 v81)
set(GENIEXPROC_HTP_DSP_ARCHS "${_GENIEXPROC_HTP_DEFAULT_ARCHS}" CACHE STRING
    "DSP architectures to build for GENIEXPROC_ENABLE_HTP (supported: ${_GENIEXPROC_HTP_SUPPORTED_ARCHS})")
if(NOT GENIEXPROC_HTP_DSP_ARCHS MATCHES "^v[0-9]+(;v[0-9]+)*$")
    message(FATAL_ERROR "Invalid GENIEXPROC_HTP_DSP_ARCHS='${GENIEXPROC_HTP_DSP_ARCHS}'; use a non-empty semicolon-separated architecture list")
endif()
foreach(_arch IN LISTS GENIEXPROC_HTP_DSP_ARCHS)
    list(FIND _GENIEXPROC_HTP_SUPPORTED_ARCHS "${_arch}" _supported_arch_index)
    if(_supported_arch_index EQUAL -1)
        message(FATAL_ERROR "Unsupported HTP DSP architecture ${_arch}; supported architectures are ${_GENIEXPROC_HTP_SUPPORTED_ARCHS}")
    endif()
endforeach()
set(_GENIEXPROC_HTP_UNIQUE_ARCHS ${GENIEXPROC_HTP_DSP_ARCHS})
list(REMOVE_DUPLICATES _GENIEXPROC_HTP_UNIQUE_ARCHS)
if(NOT "${GENIEXPROC_HTP_DSP_ARCHS}" STREQUAL "${_GENIEXPROC_HTP_UNIQUE_ARCHS}")
    message(FATAL_ERROR "GENIEXPROC_HTP_DSP_ARCHS must not contain duplicate architectures: '${GENIEXPROC_HTP_DSP_ARCHS}'")
endif()
message(STATUS "geniex-proc HTP: ${CMAKE_SYSTEM_NAME} target, DSP architectures ${GENIEXPROC_HTP_DSP_ARCHS}")

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
if(NOT EXISTS "${HEXAGON_SDK_ROOT}/build/cmake/hexagon_toolchain.cmake")
    message(FATAL_ERROR "Hexagon SDK toolchain was not found under HEXAGON_SDK_ROOT: ${HEXAGON_SDK_ROOT}/build/cmake/hexagon_toolchain.cmake")
endif()
set(_GENIEXPROC_HTP_HEXAGON_CLANG "${HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-clang")
if(CMAKE_HOST_WIN32)
    string(APPEND _GENIEXPROC_HTP_HEXAGON_CLANG ".exe")
endif()
if(NOT EXISTS "${_GENIEXPROC_HTP_HEXAGON_CLANG}")
    message(FATAL_ERROR "Hexagon DSP compiler was not found: ${_GENIEXPROC_HTP_HEXAGON_CLANG}")
endif()
foreach(_arch IN LISTS GENIEXPROC_HTP_DSP_ARCHS)
    if(NOT IS_DIRECTORY "${HEXAGON_TOOLS_ROOT}/Tools/target/hexagon/lib/${_arch}")
        message(FATAL_ERROR "Hexagon Tools at '${HEXAGON_TOOLS_ROOT}' do not provide compiler support for ${_arch}")
    endif()
endforeach()

include(ExternalProject)

set(_GENIEXPROC_HTP_IDL "${CMAKE_CURRENT_SOURCE_DIR}/geniex_proc_htp.idl")
set(_GENIEXPROC_HTP_IDL_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/idl")
set(_GENIEXPROC_HTP_HEADER "${_GENIEXPROC_HTP_IDL_OUT_DIR}/geniex_proc_htp.h")
set(_GENIEXPROC_HTP_STUB "${_GENIEXPROC_HTP_IDL_OUT_DIR}/geniex_proc_htp_stub.c")
set(_GENIEXPROC_HTP_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${_GENIEXPROC_HTP_IDL_OUT_DIR}" "${_GENIEXPROC_HTP_BIN_DIR}")

if(CMAKE_HOST_WIN32)
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
set(_GENIEXPROC_HTP_SKEL_FILES)
foreach(_arch IN LISTS GENIEXPROC_HTP_DSP_ARCHS)
    string(REGEX REPLACE "^v" "" _arch_number "${_arch}")
    list(APPEND _GENIEXPROC_HTP_ARCH_NUMBERS "${_arch_number}")
    set(_target "geniex-proc-htp-${_arch}")
    set(_skel "${_GENIEXPROC_HTP_BIN_DIR}/libgeniex-proc-htp-${_arch}.so")
    list(APPEND _GENIEXPROC_HTP_DSP_TARGETS "${_target}")
    list(APPEND _GENIEXPROC_HTP_SKEL_FILES "${_skel}")
    ExternalProject_Add(${_target}
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dsp"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/dsp-${_arch}"
        BUILD_ALWAYS TRUE
        BUILD_BYPRODUCTS "${_skel}"
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
)
add_dependencies(geniex_proc_htp_runtime geniex_proc_htp_idl ${_GENIEXPROC_HTP_DSP_TARGETS})
target_include_directories(geniex_proc_htp_runtime PUBLIC
    "${HEXAGON_SDK_ROOT}/incs"
    "${HEXAGON_SDK_ROOT}/incs/stddef"
    "${_GENIEXPROC_HTP_IDL_OUT_DIR}"
)
target_compile_definitions(geniex_proc_htp_runtime PUBLIC
    GENIEXPROC_HTP_ARCHS=${_GENIEXPROC_HTP_ARCH_DEFINITION}
)
set_property(TARGET geniex_proc_htp_runtime PROPERTY GENIEXPROC_HTP_SKEL_FILES
    "${_GENIEXPROC_HTP_SKEL_FILES}")
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
set(GENIEXPROC_HTP_FASTRPC_LIBRARY "" CACHE FILEPATH
    "Optional real target libcdsprpc.so for Linux ARM64 or Android arm64-v8a")
if(NOT WIN32 AND (GENIEXPROC_HTP_CERT OR GENIEXPROC_HTP_CERT_THUMBPRINT OR GENIEXPROC_HTP_CDSPRPC_DLL))
    message(FATAL_ERROR "GENIEXPROC_HTP_CERT, GENIEXPROC_HTP_CERT_THUMBPRINT, and GENIEXPROC_HTP_CDSPRPC_DLL apply to Windows only")
endif()

if(WIN32)
    if(GENIEXPROC_HTP_FASTRPC_LIBRARY)
        message(FATAL_ERROR "GENIEXPROC_HTP_FASTRPC_LIBRARY is only used on Linux and Android targets")
    endif()
else()
    if(GENIEXPROC_HTP_FASTRPC_LIBRARY)
        set(_GENIEXPROC_HTP_FASTRPC_LIBRARY "${GENIEXPROC_HTP_FASTRPC_LIBRARY}")
    elseif(ANDROID)
        set(_GENIEXPROC_HTP_FASTRPC_LIBRARY "${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so")
    else()
        set(_GENIEXPROC_HTP_FASTRPC_LIBRARY "${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/UbuntuARM_aarch64/libcdsprpc.so")
    endif()
    if(NOT EXISTS "${_GENIEXPROC_HTP_FASTRPC_LIBRARY}" OR IS_DIRECTORY "${_GENIEXPROC_HTP_FASTRPC_LIBRARY}")
        message(FATAL_ERROR "Target FastRPC library not found: ${_GENIEXPROC_HTP_FASTRPC_LIBRARY}; set GENIEXPROC_HTP_FASTRPC_LIBRARY to a real ARM64 libcdsprpc.so")
    endif()
    target_link_libraries(geniex_proc_htp_runtime PUBLIC "${_GENIEXPROC_HTP_FASTRPC_LIBRARY}")
endif()

if(WIN32)
    find_program(_GENIEXPROC_HTP_LIB_EXECUTABLE NAMES lib.exe lib REQUIRED)
    set(_GENIEXPROC_HTP_CDSPRPC_DEF "${CMAKE_CURRENT_SOURCE_DIR}/cdsprpc.def")
    set(_GENIEXPROC_HTP_CDSPRPC_LIB "${_GENIEXPROC_HTP_BIN_DIR}/cdsprpc.lib")
    add_custom_command(
        OUTPUT "${_GENIEXPROC_HTP_CDSPRPC_LIB}"
        COMMAND "${_GENIEXPROC_HTP_LIB_EXECUTABLE}" /DEF:${_GENIEXPROC_HTP_CDSPRPC_DEF}
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
        file(TO_CMAKE_PATH "$ENV{WINDOWS_SDK_BIN}/x64" _GENIEXPROC_HTP_WINSDK_X64)
        file(TO_CMAKE_PATH "$ENV{WINDOWS_SDK_BIN}/x86" _GENIEXPROC_HTP_WINSDK_X86)
        find_program(_GENIEXPROC_HTP_INF2CAT NAMES inf2cat.exe Inf2Cat.exe
            PATHS ${_GENIEXPROC_HTP_WINSDK_X86} ${_GENIEXPROC_HTP_WINSDK_ARM64} REQUIRED)
        string(TOLOWER "${CMAKE_HOST_SYSTEM_PROCESSOR}" _GENIEXPROC_HTP_HOST_PROCESSOR)
        if(_GENIEXPROC_HTP_HOST_PROCESSOR MATCHES "^(amd64|x86_64|x64)$")
            set(_GENIEXPROC_HTP_SIGNTOOL_PATHS ${_GENIEXPROC_HTP_WINSDK_X64} ${_GENIEXPROC_HTP_WINSDK_X86})
        elseif(_GENIEXPROC_HTP_HOST_PROCESSOR MATCHES "^(arm64|aarch64)$")
            set(_GENIEXPROC_HTP_SIGNTOOL_PATHS ${_GENIEXPROC_HTP_WINSDK_ARM64} ${_GENIEXPROC_HTP_WINSDK_X86})
        else()
            set(_GENIEXPROC_HTP_SIGNTOOL_PATHS ${_GENIEXPROC_HTP_WINSDK_X86})
        endif()
        find_program(_GENIEXPROC_HTP_SIGNTOOL NAMES signtool.exe
            PATHS ${_GENIEXPROC_HTP_SIGNTOOL_PATHS} REQUIRED)

        set(_GENIEXPROC_HTP_SIGN_DIR "${CMAKE_CURRENT_BINARY_DIR}/sign")
        file(MAKE_DIRECTORY "${_GENIEXPROC_HTP_SIGN_DIR}")
        set(GENIEXPROC_HTP_SOURCE_FILES "")
        set(GENIEXPROC_HTP_COPY_FILES "")
        set(_GENIEXPROC_HTP_COPY_COMMANDS)
        set(_GENIEXPROC_HTP_VERIFY_COMMANDS)
        foreach(_arch IN LISTS GENIEXPROC_HTP_DSP_ARCHS)
            set(_name "libgeniex-proc-htp-${_arch}.so")
            string(APPEND GENIEXPROC_HTP_SOURCE_FILES "${_name} = 1\n")
            string(APPEND GENIEXPROC_HTP_COPY_FILES "${_name},,,0x10\n")
            list(APPEND _GENIEXPROC_HTP_COPY_COMMANDS COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_GENIEXPROC_HTP_BIN_DIR}/${_name}" "${_GENIEXPROC_HTP_SIGN_DIR}/${_name}")
            list(APPEND _GENIEXPROC_HTP_VERIFY_COMMANDS COMMAND "${_GENIEXPROC_HTP_SIGNTOOL}"
                verify /pa /c "${_GENIEXPROC_HTP_SIGN_DIR}/libgeniex-proc-htp.cat"
                "${_GENIEXPROC_HTP_SIGN_DIR}/${_name}")
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
            COMMAND ${CMAKE_COMMAND} -E remove
                "${_GENIEXPROC_HTP_BIN_DIR}/libgeniex-proc-htp.cat"
                "${_GENIEXPROC_HTP_BIN_DIR}/libgeniex-proc-htp.inf"
            ${_GENIEXPROC_HTP_COPY_COMMANDS}
            COMMAND "${_GENIEXPROC_HTP_INF2CAT}" /driver:${_GENIEXPROC_HTP_SIGN_DIR} /os:10_GE_ARM64
            COMMAND ${_GENIEXPROC_HTP_SIGN_CMD} "${_GENIEXPROC_HTP_SIGN_DIR}/libgeniex-proc-htp.cat"
            ${_GENIEXPROC_HTP_VERIFY_COMMANDS}
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

set(_GENIEXPROC_HTP_INSTALL_COMPONENT geniex-proc-htp)
set(_GENIEXPROC_HTP_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/geniex-proc-htp")

geniexproc_install_htp_cleanup()
if(GENIEXPROC_HTP_CERT OR GENIEXPROC_HTP_CERT_THUMBPRINT)
    set(_GENIEXPROC_HTP_INSTALL_VERIFY_CODE
        "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E compare_files \"${_GENIEXPROC_HTP_BIN_DIR}/libgeniex-proc-htp.inf\" \"${_GENIEXPROC_HTP_SIGN_DIR}/libgeniex-proc-htp.inf\" RESULT_VARIABLE _geniexproc_htp_inf_result)
        if(NOT \"\${_geniexproc_htp_inf_result}\" STREQUAL \"0\")
            message(FATAL_ERROR \"The published geniex-proc HTP INF does not match the selected DSP architecture list\")
        endif()\n")
    foreach(_skel IN LISTS _GENIEXPROC_HTP_SKEL_FILES)
        string(APPEND _GENIEXPROC_HTP_INSTALL_VERIFY_CODE
            "execute_process(COMMAND \"${_GENIEXPROC_HTP_SIGNTOOL}\" verify /pa /c \"${_GENIEXPROC_HTP_BIN_DIR}/libgeniex-proc-htp.cat\" \"${_skel}\" RESULT_VARIABLE _geniexproc_htp_verify_result OUTPUT_VARIABLE _geniexproc_htp_verify_output ERROR_VARIABLE _geniexproc_htp_verify_error)\n"
            "if(NOT \"\${_geniexproc_htp_verify_result}\" STREQUAL \"0\")\n"
            "    message(FATAL_ERROR \"The selected geniex-proc HTP skel is not verified by the published catalog: ${_skel}\\n\${_geniexproc_htp_verify_output}\\n\${_geniexproc_htp_verify_error}\")\n"
            "endif()\n")
    endforeach()
    install(CODE "${_GENIEXPROC_HTP_INSTALL_VERIFY_CODE}"
        COMPONENT "${_GENIEXPROC_HTP_INSTALL_COMPONENT}"
    )
endif()
install(FILES ${_GENIEXPROC_HTP_SKEL_FILES}
    DESTINATION "${_GENIEXPROC_HTP_INSTALL_DIR}"
    COMPONENT "${_GENIEXPROC_HTP_INSTALL_COMPONENT}"
)
if(WIN32 AND GENIEXPROC_HTP_CDSPRPC_DLL)
    install(FILES "${GENIEXPROC_HTP_CDSPRPC_DLL}"
        DESTINATION "${_GENIEXPROC_HTP_INSTALL_DIR}"
        RENAME libcdsprpc.dll
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
