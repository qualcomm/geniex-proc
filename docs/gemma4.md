# Gemma4 vision preprocessing

`geniex::gemma4::Gemma4Processor` is a C++ port of transformers'
`Gemma4ImageProcessorPil`. It produces the two tensors the exported Gemma4 VEG
(Visual Embedding Generator) graph consumes:

```
pixel_values       [n_images, max_patches, patch_size^2 * 3]  float32
image_position_ids [n_images, max_patches, 2]                 int32   (x, y), -1 = padding
```

## Pipeline

```
image.jpg (any size)
  │  convert to RGB
  │  resize to force_square_size x force_square_size   (BICUBIC-family; default 768)
  ▼
  │  aspect-ratio-preserving resize onto the patch budget   <- a no-op at 768
  │  rescale /255      (do_normalize = false: Gemma4 trains on raw [0,1] pixels)
  │  patchify 16x16 -> 48x48 = 2304 patches x 768
  │  pad to max_patches = max_soft_tokens * pooling^2 = 280 * 9 = 2520, position id -1
  ▼
pixel_values [1,2520,768] + image_position_ids [1,2520,2],  256 soft tokens
```

Soft tokens = patches / pooling² = 2304 / 9 = **256**, which is what the v73 VEG
export emits.

## Why the square pre-resize

The exported graph is traced at a **fixed** soft-token count. A non-square image
would come out of the aspect-preserving step with a different patch count and
therefore a different number of soft tokens, which no longer matches the graph's
output shape. `force_square_size` pins the geometry.

768 specifically: the aspect-preserving resize *targets* 768×768 for any square
input, so at 768 that step is a no-op and the image is resampled exactly once,
from the original pixels. Pre-squaring to 448 (as the original reference script
did) makes the processor upscale straight back to 768 — a second resample that
only costs detail.

Set `force_square_size = 0` to disable and let the true aspect-preserving path
run, for graphs traced with a dynamic soft-token count.

## Patch memory layout

Upstream reshapes `(C,H,W) -> (C,nph,p,npw,p)`, transposes to `(nph,npw,p,p,C)`
and flattens, so **within a patch the channel is the fastest-varying axis**
(`[patch_row][patch_col][channel]`). Getting this wrong yields a plausible-looking
but meaningless embedding.

Position ids come from `meshgrid(..., indexing="xy")` flattened row-major, i.e.
patch index `row * npw + col` carries `(x, y) = (col, row)`.

## Validation

`gemma4_prep_check` (in the plugin, built with `-DGENIEX_BUILD_VLM=ON`) diffs this
implementation against the Python reference:

```powershell
python driver/prep_vision.py images.jpg <ref_dir>
gemma4_prep_check.exe --image images.jpg --ref-dir <ref_dir>
```

Result on `images.jpg` (597×335):

| | |
|---|---|
| `image_position_ids` mismatches | **0 / 5040** (exact) |
| `pixel_values` mean abs diff | 0.000416 (0.11 / 255) |
| `pixel_values` max abs diff | 0.039 (10 / 255) |
| elements differing > 1/255 | 3.1 % |

Position ids are exact. The small pixel differences are the resampler: this port
uses stb's Catmull-Rom, the reference uses PIL's BICUBIC. Both are a = −0.5 cubic,
but they differ in edge handling and coordinate convention.

## Usage

```cpp
#include "geniex-proc/gemma4.h"

geniex::gemma4::Gemma4Config cfg;              // patch 16, pooling 3, 280 soft tokens
auto proc = geniex::gemma4::Gemma4Processor::create(tokenizer_json, tokenizer_config_json, cfg);

// image only — no tokenizer needed for this path
auto feats = proc->process_images({"image.jpg"});
int  n_soft = feats.num_soft_tokens_per_image[0];   // 256

// text + image: each image marker expands to boi + n_soft x image_token + eoi
auto text  = proc->apply_chat_template({{geniex::Role::User, proc->image_marker() + "describe this"}});
auto batch = proc->process(text, {"image.jpg"});     // batch.input_ids has a contiguous image run
```

The image-token run is contiguous by construction, which is what lets the runtime
splice the VEG output into `inputs_embeds` by position.
