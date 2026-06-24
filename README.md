# geniex-proc

A high-performance C++ library for multimodal data preprocessing, designed to efficiently prepare image, audio, video, and text data for local inference models (GGML, NPU, etc.).

## Supported Modalities

| Modality | Libraries | Features |
|----------|-----------|----------|
| Image    | [STB Image](https://github.com/nothings/stb) | Resizing, normalization, color conversion |
| Audio    | *(planned)* [libsndfile](https://github.com/libsndfile/libsndfile), [xtensor-fftw](https://github.com/xtensor-stack/xtensor-fftw), [soxr](https://github.com/dofuuz/soxr) | Resampling, MFCC extraction, spectrograms |
| Video    | *(planned)* [Decord](https://github.com/dmlc/decord), FFmpeg | Frame extraction, temporal sampling |
| Text     | [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp), [minja](https://github.com/google/minja) | Tokenization (BPE), chat templates |
| Tensors  | [xtensor](https://github.com/xtensor-stack/xtensor), [xtl](https://github.com/xtensor-stack/xtl), [xtensor-io](https://github.com/xtensor-stack/xtensor-io), [xtensor-blas](https://github.com/xtensor-stack/xtensor-blas) | General multi-dimensional tensors, linear algebra, npy/npz I/O |

## Key Features

- **Modular Design**: Each modality has dedicated, optimized processors
- **Lightweight & Full-Featured**: Choose between minimal dependencies (STB, built-in libraries) or full capabilities
- **Memory Efficient**: Optimized for minimal copying and efficient memory usage
- **Integration Ready**: Simplified APIs for direct integration with inference engines
- **Cross-Platform**: Supports Windows, Linux, macOS with comprehensive build system

## Quick Start

```bash
git clone https://github.com/qualcomm/geniex-proc.git --recursive
cd geniex-proc
cmake -B build
cmake --build build
```

📖 **For detailed build instructions, dependencies, and platform-specific setup, see [docs/build.md](docs/build.md)**

## Module Overview

geniex-proc is built with a modular architecture. Enable only the features you need:

- **Core Library** (`geniex-proc`): Tokenizer + sampler + grammar-constrained sampling (always built)
- **Vision Library** (`geniex-proc-vision`): Image processing + VLM processors — enable with `-DGENIEXPROC_ENABLE_VISION=ON`
- **Audio Library** (`geniex-proc-audio`): Audio processing + Omni processors — enable with `-DGENIEXPROC_ENABLE_AUDIO=ON` *(scaffolding only; sources WIP)*
- **Video Support**: Enable with `-DGENIEXPROC_ENABLE_VIDEO=ON` *(planned)*

Additional build options:

- `-DGENIEXPROC_BUILD_SHARED_LIBS=ON` — build shared libraries instead of static
- `-DGENIEXPROC_BUILD_TESTS=ON` — build and register CTest unit tests (uses GoogleTest)
- `-DGENIEXPROC_INSTALL=ON` — generate install rules

## Repository Layout

```
include/
  geniex-proc.h              Umbrella header
  geniex-vision.h            Vision umbrella header
  geniex-audio.h             Audio umbrella header (placeholder)
  geniex-proc/
    export.h                 API export macros
    processor.h              Base processor interface
    qwen2vl.h                Qwen2-VL processor
    sampler.h                Sampler
    tokenizer.h              Tokenizer
    types.h                  Common types

src/
  tokenizer/                 Tokenizer + tokenizer config
  sampler/                   Sampling implementation
  geniex-sampling/           Grammar-constrained sampling
  vision/                    Image loading / preprocessing (stb-based)
  processors/                Model-specific processors (Qwen2-VL, ...)
  internal/                  Internal utilities

tests/                       GoogleTest unit tests
third-party/                 Vendored dependencies (xtensor, tokenizers-cpp, minja, stb, ...)
docs/                        Build and integration documentation
```

## Development Guidelines

### Adding a New Vision/VLM Processor

Follow these steps to add a new model processor (e.g., `qwen3vl`):

1. **Create the interface**:
   - Add `include/geniex-proc/qwen3vl.h` with the processor class declaration
   - Follow the naming pattern of existing processors (e.g., `qwen2vl.h`)

2. **Implement the processor**:
   - Add `src/processors/qwen3vl.cpp` with the implementation
   - Follow existing patterns from `src/processors/qwen2vl.cpp` for consistent API design

3. **Update build configuration** in `CMakeLists.txt`:
   ```cmake
   add_library(geniex-proc-vision ${_GENIEXPROC_LIB_TYPE}
       src/vision/vision.cpp
       src/processors/processor.cpp
       src/processors/qwen2vl.cpp
       src/processors/qwen3vl.cpp   # Add your processor here
   )
   ```

4. **Add a test**:
   - Add a test file under `tests/` (e.g., `tests/qwen3vl.cpp`)
   - Register it in `tests/CMakeLists.txt`

### Adding Audio / Video Processing

The audio and video modules are scaffolded in `CMakeLists.txt` but not yet implemented:

1. **Audio processors**:
   - Create `src/audio/audio.cpp` for feature extraction (FFT, mel spectrograms, ...)
   - Add model-specific processors under `src/processors/` (e.g., `qwen25omni.cpp`, `whisper.cpp`)
   - Wire up `libsndfile`, `soxr`, and `fftw3` in the `geniex-proc-audio` target

2. **Video processors**:
   - Add a `src/video/` directory and a new `geniex-proc-video` target
   - Integrate `decord` / `ffmpeg` for frame extraction

## Roadmap

### ✅ Completed Features
- **Core Processing**: Tokenizer support (BPE via tokenizers-cpp)
- **LLM Integration**: Decoding strategies (sampling, temperature, repetition penalty)
- **Grammar-Constrained Sampling**: Structured output via `geniex-sampling`
- **Chat Templates**: Multi-model chat template support (via minja)
- **Vision Processing**: Qwen2-VL processor with stb-based image preprocessing

### 🚧 In Development
- Audio processing (WhisperFeatureExtractor, Parakeet, ...)
- Additional VLM processors (Qwen3-VL, OmniVLM, ...)
- Multi-modal Omni processors (Qwen2.5 Omni)
- Video processing workflows

### 📋 Planned Features
- CLIP processor with image preprocessing
- C wrappers and additional language bindings
- Expanded examples and sample data

## Getting in Contact

* [Report an Issue on GitHub](../../issues)
* [Open a Discussion on GitHub](../../discussions)
* E-mail the maintainers for general questions:
  * Perry Cheng — [zhic@qti.qualcomm.com](mailto:zhic@qti.qualcomm.com)
  * Alex Chen — [alexcw@qti.qualcomm.com](mailto:alexcw@qti.qualcomm.com)

## Contributing

For details on contributing, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

geniex-proc is licensed under the [BSD 3-Clause License](https://spdx.org/licenses/BSD-3-Clause.html). See [LICENSE.txt](LICENSE.txt) for the full license text.
