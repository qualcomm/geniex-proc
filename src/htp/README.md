# Private HTP backend

`GENIEXPROC_ENABLE_HTP` is OFF by default. The DSP build produces v73 and v81
skels with the Hexagon SDK. Hardware targets are Windows ARM64 on qcs (v73) and
X2Elite (v81). Public headers and processor APIs are unchanged.

## Boundaries

- `src/vision/vision.cpp`: shared image loading and resize-dimension helpers.
- `src/vision/vision_cpu.cpp`: CPU resize, using stb as the reference.
- `src/vision/vision_htp.cpp`: tensor/argument validation and direct invocation of
  the QAIC-generated DSP resize stub.
- `htp_runtime.h/.cpp`: session lifecycle and guarded access to its native handle;
  no image operators are declared here.
- `geniex_proc_htp.idl`: private RPC contract and QAIC-generated bindings.
- `dsp/geniex_proc_htp.c`: DSP RPC argument checks and dispatch.
- `dsp/resize_kernel.h/.c`: independent RGB Catmull-Rom resize, with HVX vertical
  filtering. Horizontal filtering, coefficient generation and saturated output
  conversion remain scalar. It does not depend on stb.

CMake selects `*_cpu.cpp` or `*_htp.cpp` in the vision and processor directories.
The private backend target supplies SDK/generated headers to host adapters;
public headers do not expose SDK types. The host and DSP builds are separate. Both skels are covered by one signed
catalog. Session open queries FastRPC `ARCH_VER` and selects the matching skel;
an architecture not included in the build is rejected before module loading.

## InternVL Normalize

`src/processors/internvl_cpu.cpp` and `internvl_htp.cpp` implement the same private
Normalize + HWC-to-CHW operation. Input is packed RGB uint8; output is planar
float32. It evaluates `(pixel * rescale_factor - mean[c]) * (1 / std[c])` in that
order, without folding operations into FMA. Parameters must be finite, with
nonzero standard deviations and finite reciprocals. Host and DSP check lengths
and dimensions independently. The DSP kernel uses HVX byte lookups to deinterleave
32 pixels, float32 arithmetic, unaligned output stores and bounded tail copies.
Resize, tiling and thumbnail generation are not part of this operation.

## Session lifecycle

```cpp
geniex::htp::Session session;
geniex::htp::open(session);
{
    auto access = geniex::htp::acquire(session);
    // Invoke the generated DSP stub with access.handle.
}
geniex::htp::close(session);
```

Open and close are idempotent. Acquire requires an already-open session and keeps
close blocked until the access guard is released. Explicit close reports errors;
destruction provides non-throwing fallback cleanup. The Vision adapter borrows the
lazily opened runtime-owned default session.

## Correctness tests

- `tests/htp.cpp` tests only session open, close, reopen and guarded access on qcs.
- `tests/vision.cpp` is shared by CPU and HTP builds. Resize coverage is limited
  to identity, constant rectangular input, a fixed nonconstant Catmull-Rom vector,
  antialias downsampling and invalid arguments, alongside the existing helpers.
  With HTP ON, these calls use the actual CDSP.

- `tests/internvl_normalize.cpp` is compiled into the processor tests for both
  backends. It checks channel layout, every uint8 value, vector tails, custom
  parameters and invalid inputs. Formula comparisons allow `2e-6 * max(1, abs(ref))`
  floating-point error; a fixed unscaled RGB vector checks planar channel layout.

There is no separate runtime test file or standalone smoke executable. The fixed
small RGB oracle remains byte-exact. Additional antialias oracle checks allow one
uint8 level of floating-point rounding difference. The 17-case qcs comparison
found 15 byte-exact cases and a maximum one-level difference in two large downsample
cases; this is not a claim of bit-exact output for every image. Performance tuning
is in progress, and no speedup over CPU is claimed. HMX experiments remain outside
the production resize path.
