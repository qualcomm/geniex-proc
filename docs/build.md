# Building geniex-proc

This document provides instructions for building the geniex-proc project. For detailed dependency installation, see the specialized documentation referenced throughout.

## Quick Start

```bash
git clone https://github.com/qualcomm/geniex-proc.git --recursive
cd geniex-proc
cmake -B build
cmake --build build
```

This builds the core geniex-proc library with basic functionality.

## Prerequisites

- **CMake 3.10+**
- **C++20 compatible compiler**
- **Git** (for submodules)
- **Rust toolchain** (cargo) - Required for tokenizers-cpp

Platform-specific compilers: VS2019+ (Windows), GCC 9+/Clang 10+ (Linux), Xcode 11+ (macOS)

### Installing Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo --version  # verify installation
```

## Build Options

geniex-proc uses CMake options to control which features and modules to build:

### Core Options
| Option | Default | Description |
|--------|---------|-------------|
| `GENIEXPROC_BUILD_SHARED_LIBS` | `OFF` | Build as shared library instead of static |
| `GENIEXPROC_BUILD_TESTS` | `OFF` | Build and register CTest unit tests (GoogleTest via FetchContent) |
| `GENIEXPROC_INSTALL` | `OFF` | Enable installation targets |

### Modality Options
| Option | Default | Description |
|--------|---------|-------------|
| `GENIEXPROC_ENABLE_VISION` | `OFF` | Build image processing + all VLM processors (stb) |
| `GENIEXPROC_ENABLE_OPENCL` | `OFF` | Build the OpenCL image backend (OpenCL SDK required; GPU required at runtime). |
| `GENIEXPROC_ENABLE_HTP` | `OFF` | Build HTP on Windows/Linux ARM64 or Android arm64-v8a (matching Hexagon SDK/tools and device FastRPC required). |
| `GENIEXPROC_ENABLE_AUDIO` | `OFF` | Build audio processing + all Omni processors (libsndfile, soxr, fftw3) |
| `GENIEXPROC_ENABLE_VIDEO` | `OFF` | Build video processing (decord, ffmpeg) |

The Snapdragon SDK preset needs `-DGENIEXPROC_ENABLE_OPENCL=ON -DGENIEXPROC_ENABLE_HTP=ON` for both proc backends; `GGML_OPENCL=ON` enables llama.cpp OpenCL only.

Qwen2VL defaults to CPU. Before launch, `GENIEX_QAIRT_PREPROCESS_DEVICE=cpu|gpu|npu|auto` selects a backend (`auto`/unset means CPU); `Qwen2VLConfig::preprocess_device` overrides the environment. Invalid choices and unavailable backends fail without fallback. Other processors ignore this setting.

## Platform-Specific Instructions

### Windows
Use vcpkg for dependency management:
```powershell
cmake -B build -DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build --config Release
```
📖 **See [vcpkg-static.md](vcpkg-static.md) for vcpkg setup**

### Linux
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```
Install system dependencies as needed: `build-essential cmake git libfftw3-dev zlib1g-dev libopencv-dev`

### macOS
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```
📖 **See [openmp.md](openmp.md) for OpenMP setup (required for audio processing)**

## Common Build Configurations

### 1. Basic Library Only
```bash
cmake -B build
cmake --build build
```

### 2. With Multi-Modal Processing (Image Only)
```bash
cmake -B build -DGENIEXPROC_BUILD_MMPROCESS=ON
cmake --build build
```

### 3. Full Multi-Modal Support (Image + Audio + Video)
```bash
cmake -B build \
  -DGENIEXPROC_BUILD_MMPROCESS=ON \
  -DGENIEXPROC_BUILD_MMPROCESS_AUDIO=ON \
  -DGENIEXPROC_BUILD_MMPROCESS_VIDEO=ON
cmake --build build
```

### 4. With Computer Vision Features
```bash
cmake -B build \
  -DGENIEXPROC_BUILD_WITH_OPENCV=ON \
  -DGENIEXPROC_BUILD_TRANSFORM=ON \
  -DGENIEXPROC_BUILD_PADDLE_OCR_PROC=ON
cmake --build build
```

### 5. Development Build (All Features)
```bash
cmake -B build \
  -DGENIEXPROC_BUILD_MMPROCESS=ON \
  -DGENIEXPROC_BUILD_MMPROCESS_AUDIO=ON \
  -DGENIEXPROC_BUILD_MMPROCESS_VIDEO=ON \
  -DGENIEXPROC_BUILD_WITH_OPENCV=ON \
  -DGENIEXPROC_BUILD_TRANSFORM=ON \
  -DGENIEXPROC_BUILD_PADDLE_OCR_PROC=ON \
  -DGENIEXPROC_BUILD_C_WRAPPERS=ON \
  -DGENIEXPROC_BUILD_EXAMPLES=ON
cmake --build build
```

## Dependencies

Most dependencies are built-in as git submodules. External dependencies needed:

- **Multi-modal processing**: FFTW3, OpenMP, MP3 libraries, and FFmpeg (install via system package manager)
- **Computer vision**: OpenCV (install via system package manager or build from source)
- **Windows dependencies**: 📖 **See [vcpkg-static.md](vcpkg-static.md)**
- **macOS OpenMP**: 📖 **See [openmp.md](openmp.md)**

## Troubleshooting

### Common Issues
- **OpenMP on macOS**: 📖 **See [openmp.md](openmp.md)**
- **FFTW3/vcpkg on Windows**: 📖 **See [vcpkg-static.md](vcpkg-static.md)**

### Debug Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --verbose  # for detailed output
```

## Installation

To install the built libraries and headers:

```bash
cmake -B build -DGENIEXPROC_INSTALL=ON
cmake --build build
cmake --install build --prefix /path/to/install
```

With `GENIEXPROC_ENABLE_HTP=ON`, build first, then install the HTP runtime payload independently (also works with `GENIEXPROC_INSTALL=OFF`):

```bash
cmake --install build --prefix /path/to/install --component geniex-proc-htp
```

The component installs selected skels to `bin/geniex-proc-htp`; Windows may sign/install the matching `.cat`/`.inf` with a PFX (`GENIEXPROC_HTP_CERT`) or thumbprint (`GENIEXPROC_HTP_CERT_THUMBPRINT`) using SDK INF2CAT and host-native SignTool, and optionally install `GENIEXPROC_HTP_CDSPRPC_DLL`. Linux/Android require a system FastRPC runtime; CMake links the target SDK library but does not install it. Cleanup removes proc-owned files only. Consumers can query `GENIEXPROC_HTP_SKEL_FILES` on `geniex_proc_htp_runtime` and stage runtime files separately. See [HTP deployment notes](../src/vision/npu/README.md).

Default skels are Windows v73/v81, Linux v73/v75, and Android arm64-v8a v79/v81. Override with a unique supported list (`v68`, `v69`, `v73`, `v75`, `v79`, `v81`) using `GENIEXPROC_HTP_DSP_ARCHS`; it controls dispatch and signing. Linux/Android may set `GENIEXPROC_HTP_FASTRPC_LIBRARY`; otherwise CMake selects the target SDK library.

## Testing

The common `device` test checks selection and a real RGB resize on configured hardware (no fallback). Existing processor tests use the selected Qwen2VL backend, for example:

```bash
GENIEX_QAIRT_PREPROCESS_DEVICE=npu ctest --test-dir build -R '^(vision|processor|device)$' --output-on-failure
```

After building, you can run the examples to verify functionality:

```bash
# Run built examples (if GENIEXPROC_BUILD_EXAMPLES=ON)
./build/bin/geniex-proc-test-*
```

## Advanced Options

- **Custom toolchain**: `-DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake`
- **Custom build directory**: `cmake -B my-custom-build`  
- **Parallel builds**: `cmake --build build -j$(nproc)` (Linux/macOS) or `--parallel` (Windows)
- **Cross-compilation**: Use a custom toolchain file for the target platform

## Related Documentation

- 📖 [vcpkg-static.md](vcpkg-static.md) - Windows dependency management  
- 📖 [openmp.md](openmp.md) - OpenMP setup for macOS
