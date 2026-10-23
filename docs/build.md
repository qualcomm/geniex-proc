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
| `GENIEXPROC_ENABLE_OPENCL` | `OFF` | Enable Qwen2VL OpenCL GPU preprocessing; requires an OpenCL SDK at build time and a GPU at runtime. |
| `GENIEXPROC_ENABLE_HTP` | `OFF` | Enable Qwen2VL Hexagon HTP preprocessing; currently Windows-only, requires a Hexagon SDK and a device FastRPC runtime. |
| `GENIEXPROC_ENABLE_AUDIO` | `OFF` | Build audio processing + all Omni processors (libsndfile, soxr, fftw3) |
| `GENIEXPROC_ENABLE_VIDEO` | `OFF` | Build video processing (decord, ffmpeg) |

When building geniex SDK with the Snapdragon CMake preset, also pass `-DGENIEXPROC_ENABLE_OPENCL=ON -DGENIEXPROC_ENABLE_HTP=ON` to enable both Qwen2VL backends. The preset's `GGML_OPENCL=ON` enables llama.cpp OpenCL, **not** geniex-proc OpenCL; the proc options default to OFF.

Qwen2VL remains on CPU by default. For an unmodified geniex/QAIRT application, set `GENIEX_QAIRT_PREPROCESS_DEVICE=cpu|gpu|npu|auto` **before launching** the process; `auto` (and an unset variable) means CPU. Only Qwen2VL reads this setting, at `Qwen2VLProcessor::create`. Callers can set `Qwen2VLConfig::preprocess_device` to `CPU`, `GPU`, or `NPU` to override the environment; the default `Auto` reads it. Unsupported backends and invalid values raise an error rather than falling back to CPU. The decoded image is uploaded to OpenCL once; resize and normalization share device memory, and the GPU writes the final patch-ordered float32 output before returning it to the host. HTP fuses resize, normalization, and patch packing in one FastRPC call. Neither backend provides GPU/NPU zero-copy: `BatchFeatures` and the QNN input still require host data.

`geniex::PreprocessDevice` is a shared type and environment resolution lives in shared vision utilities, but accelerated preprocessing is currently supported only by Qwen2VL. The `qwen2vl_*` vision backend files are intentionally model-specific because their kernels produce Qwen2VL's temporal/merge-aware patch layout, not a generic image output.

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

The component requires both build-produced v73/v81 skels and installs them under `bin/geniex-proc-htp`; when signing is configured, it also installs the matching `.cat` and `.inf`. `GENIEXPROC_HTP_CDSPRPC_DLL` optionally adds `libcdsprpc.dll`. An unsigned component install removes any previously installed proc `.cat`/`.inf`; if HTP is disabled, invoking the same component removes the proc HTP payload directory. This component contains only HTP runtime payload files, not the proc libraries or SDK/plugin dependencies; the consuming SDK/plugin chooses any final runtime or DSP search destination.

## Testing

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
