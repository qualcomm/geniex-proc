# geniex-proc

A high-performance C++ library for multimodal data preprocessing, designed to efficiently prepare image, audio, video, and text data for local inference models (GGML, NPU, etc.).

## Supported Modalities

| Modality | Libraries | Features |
|----------|-----------|----------|
| Image    | STB Image (lightweight), OpenCV (advanced), Clipper/Clipper2,  | Resizing, normalization, color conversion, augmentation, contour extraction |
| Audio    | [libsndfile](https://github.com/libsndfile/libsndfile), [xtensor-fftw](https://github.com/xtensor-stack/xtensor-fftw), [soxr](https://github.com/dofuuz/soxr) | Resampling, MFCC extraction, spectrograms |
| Video    | OpenCV, FFmpeg, [Decord](https://github.com/dmlc/decord) | Frame extraction, temporal sampling, motion detection |
| Text     | [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) | Tokenization, BPE |
| Others   | xtensor, xtl, xtensor-io, xtensor-blas, zlib, etc. | General multi-dimensional tensors, linear algebra, npy/npz, etc. |

## Key Features

- **Modular Design**: Each modality has dedicated, optimized processors
- **Lightweight & Full-Featured**: Choose between minimal dependencies (STB, built-in libraries) or full capabilities (OpenCV, FFmpeg)
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

- **Core Library**: Basic processing capabilities (always built)
- **Multi-Modal Processing**: Image, audio, video support (`-DGENIEXPROC_BUILD_MMPROCESS=ON`)
- **Computer Vision**: OpenCV-based features (`-DGENIEXPROC_BUILD_WITH_OPENCV=ON`)
- **PaddleOCR Integration**: OCR capabilities (`-DGENIEXPROC_BUILD_PADDLE_OCR_PROC=ON`)

## Documentation

- 📖 [Build Instructions](docs/build.md) - Complete build guide for all platforms
- 📖 [Multi-Modal Processing](docs/mm-process.md) - Audio, video, and image processing setup
- 📖 [Platform-Specific Guides](docs/) - Windows (vcpkg), macOS (OpenMP), zlib setup


## Development Guidelines

### Adding a New Processor

Follow these steps to add a new model processor (e.g., `model-x`):

1. **Create the interface**:
   - Add `include/model-x-proc.h` with the processor class declaration
   - Follow the naming pattern of existing processors (e.g., `convnext-proc.h`)

2. **Implement the processor**:
   - Add `src/model-x-proc.cpp` with the implementation
   - Follow existing patterns for consistent API design

3. **Update build configuration**:
   ```cmake
   set(SOURCES
       src/convnext-proc.cpp
       src/model-x-proc.cpp  # Add your processor here
       # ... other sources
   )
   ```

4. **Add examples**:
   - Create `examples/model-x/` directory
   - Include CMakeLists.txt and example program
   - Demonstrate processor usage and validation

### Adding Multi-Modal Processing Features

For mm-process module enhancements:

1. **Audio/Video processors**:
   - Implement in `src/mm-process/` (e.g., new model audio/video handlers)
   - Add corresponding headers in the same directory
   - Follow patterns from existing processors like `qwen2-5-omni.cpp`, `whisper.cpp`

2. **Vision processors**:
   - Add vision-specific implementations to `mm-process-vision.cpp`
   - Or create dedicated files like `qwen2-vl.cpp`, `qwen3-vl.cpp`

3. **Add comprehensive examples**:
   - Create subdirectory in `examples/geniex-mm-process/`
   - Include individual feature tests (audio, image, video, e2e, chat-template)
   - Provide sample data in `examples/geniex-mm-process/samples/`

## Roadmap

### ✅ Completed Features
- **Core Processing**: Tokenizer support (BPE)
- **LLM Integration**: Decoding strategies (sampling, temperature, repetition penalty)
- **Streaming**: Text streaming capabilities  
- **Chat Templates**: Multi-model chat template support
- **Audio Processing**: WhisperFeatureExtractor, Parakeet support
- **Vision Processing**: Qwen2VL, Qwen3VL, OmniVLM processors
- **Multi-Modal**: Complete Qwen2.5 Omni processor (audio + vision + text)
- **Development Tools**: C wrappers, comprehensive examples, build system

### 🚧 In Development
- CLIP processor with image preprocessing
- Enhanced video processing workflows

### 📋 Planned Features
- More vision-language model processors

## Getting in Contact

* [Report an Issue on GitHub](../../issues)
* [Open a Discussion on GitHub](../../discussions)
* E-mail the maintainers for general questions:
  * Zhi Cheng — [zhic@qti.qualcomm.com](mailto:zhic@qti.qualcomm.com)
  * Alex Chen — [alexcw@qti.qualcomm.com](mailto:alexcw@qti.qualcomm.com)

## Contributing

For details on contributing, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

geniex-proc is licensed under the [BSD 3-Clause License](https://spdx.org/licenses/BSD-3-Clause.html). See [LICENSE.txt](LICENSE.txt) for the full license text.
