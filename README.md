# geniex-proc

A high-performance C++ library for multimodal data preprocessing, designed to efficiently prepare image, audio, video, and text data for local inference models (GGML, NPU, etc.).

## Supported Modalities

| Modality | Libraries | Features |
|----------|-----------|----------|
| Image    | [STB Image](https://github.com/nothings/stb) | Loading (JPEG/PNG/BMP/…), bicubic resizing, normalization, patch/tensor construction for VLMs |
| Text     | [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp), [minja](https://github.com/google/minja) | Tokenization (BPE), chat templates |
| Tensors  | [xtensor](https://github.com/xtensor-stack/xtensor), [xtl](https://github.com/xtensor-stack/xtl), [xtensor-io](https://github.com/xtensor-stack/xtensor-io), [xtensor-blas](https://github.com/xtensor-stack/xtensor-blas) | Multi-dimensional tensors, linear algebra, npy/npz I/O |

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

## Image Preprocessing Pipeline

All image processing is built on top of **[STB Image](https://github.com/nothings/stb)** (single-header C library) and **xtensor**, with no external CV frameworks. The implementation lives in `src/vision/` and `src/processors/`.

The pipeline covers:
- **Loading**: decodes JPEG/PNG/BMP/… into a `uint8` HWC tensor via `stbi_load`.
- **Resizing**: bicubic (Catmull-Rom) resize via `stbir_resize`, matching HuggingFace defaults. Includes a `smart_resize` helper that preserves aspect ratio while keeping dimensions divisible by the patch stride and total pixels within a `[min, max]` budget.
- **Normalization**: cast to `float32`, rescale by `1/255`, then per-channel mean/std subtraction (CLIP defaults) — all vectorized via xtensor views.
- **Patch/tensor construction** (VLM-specific): HWC → CHW, temporal tiling, reshape into a patch grid, and flatten to `[n_patches, C×T×P×P]` — the `pixel_values` tensor fed to the visual encoder, alongside `image_grid_thw` for positional encoding.


## Roadmap

### ✅ Completed Features
- **Core Processing**: Tokenizer support (BPE via tokenizers-cpp)
- **LLM Integration**: Decoding strategies (sampling, temperature, repetition penalty)
- **Grammar-Constrained Sampling**: Structured output via `geniex-sampling`
- **Chat Templates**: Multi-model chat template support (via minja)
- **Vision Processing**: Qwen2-VL processor with stb-based image preprocessing

### 📋 Planned Features
- Audio processing (WhisperFeatureExtractor, Parakeet, ...)
- Additional VLM processors (Qwen3-VL, OmniVLM, ...)
- Multi-modal Omni processors (Qwen2.5 Omni)

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