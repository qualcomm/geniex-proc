# Building geniex-proc

This document provides instructions for building the geniex-proc project. For detailed dependency installation, see the specialized documentation referenced throughout.

## Quick Start

```bash
git clone https://github.com/GenieXAI/geniex-proc.git --recursive
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

📖 **See [examples/hf-tokenizers/README.md](../examples/hf-tokenizers/README.md) for tokenizer usage examples**

## Build Options

geniex-proc uses CMake options to control which features and modules to build:

### Core Options
| Option | Default | Description |
|--------|---------|-------------|
| `GENIEXPROC_BUILD_SHARED_LIB` | `OFF` | Build as shared library instead of static |
| `GENIEXPROC_BUILD_EXAMPLES` | `ON` | Build example programs |
| `GENIEXPROC_INSTALL` | `OFF` | Enable installation targets |

### Module Options
| Option | Default | Description |
|--------|---------|-------------|
| `GENIEXPROC_BUILD_MMPROCESS` | `OFF` | Build multi-modal processing module |
| `GENIEXPROC_BUILD_MMPROCESS_AUDIO` | `OFF` | Add audio support to mm-process |
| `GENIEXPROC_BUILD_MMPROCESS_VIDEO` | `OFF` | Add video support to mm-process |
| `GENIEXPROC_BUILD_TRANSFORM` | `OFF` | Build transform utilities |
| `GENIEXPROC_BUILD_PADDLE_OCR_PROC` | `OFF` | Build PaddleOCR processor |
| `GENIEXPROC_BUILD_WITH_OPENCV` | `OFF` | Enable OpenCV integration |
| `GENIEXPROC_BUILD_WITH_ZLIB` | `OFF` | Build with zlib support |
| `GENIEXPROC_BUILD_C_WRAPPERS` | `OFF` | Build C API wrappers |

> **Note**: Some options have dependencies. For example:
> - `GENIEXPROC_BUILD_PADDLE_OCR_PROC` automatically enables `GENIEXPROC_BUILD_TRANSFORM` and `GENIEXPROC_BUILD_WITH_OPENCV`
> - `GENIEXPROC_BUILD_TRANSFORM` automatically enables `GENIEXPROC_BUILD_WITH_OPENCV`

## Platform-Specific Instructions

### Windows
Use vcpkg for dependency management:
```powershell
cmake -B build -DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build --config Release
```
📖 **See [vcpkg-static.md](vcpkg-static.md) for vcpkg setup and [zlib.md](zlib.md) for Windows dependencies**

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

- **Multi-modal processing**: 📖 **See [mm-process.md](mm-process.md)** for FFTW3, OpenMP, MP3 libraries, and FFmpeg
- **Computer vision**: 📖 **See [build-opencv.md](build-opencv.md)** for OpenCV installation
- **Windows dependencies**: 📖 **See [vcpkg-static.md](vcpkg-static.md)** and [zlib.md](zlib.md)
- **macOS OpenMP**: 📖 **See [openmp.md](openmp.md)**

## Build Scripts

The project includes several build scripts in the `scripts/` directory:

- `build-android.sh`: Android cross-compilation
- `build-linux-arm64.sh`: ARM64 Linux cross-compilation  
- `build-opencv-*.sh/.ps1`: OpenCV-specific builds
- `update-dylib-deps.sh`: macOS library dependency management

## Troubleshooting

### Common Issues
- **OpenMP on macOS**: 📖 **See [openmp.md](openmp.md)**
- **FFTW3/vcpkg on Windows**: 📖 **See [vcpkg-static.md](vcpkg-static.md)** and [mm-process.md](mm-process.md)
- **zlib conflicts**: 📖 **See [zlib.md](zlib.md)**

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
- **Cross-compilation**: Use scripts in `scripts/` directory

## Related Documentation

- 📖 [mm-process.md](mm-process.md) - Multi-modal processing setup
- 📖 [vcpkg-static.md](vcpkg-static.md) - Windows dependency management  
- 📖 [zlib.md](zlib.md) - zlib installation
- 📖 [openmp.md](openmp.md) - OpenMP setup for macOS
- 📖 [build-opencv.md](build-opencv.md) - OpenCV build instructions
