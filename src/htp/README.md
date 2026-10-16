# Private HTP backend

`GENIEXPROC_ENABLE_HTP` is OFF by default. The current hardware validation target
is Windows ARM64 on qcs, with the Hexagon SDK v73 toolchain. Public headers and
processor APIs are unchanged.

## Boundaries

- `src/vision/vision.cpp`: shared image loading and resize-dimension helpers.
- `src/vision/vision_cpu.cpp`: CPU resize, using stb as the reference.
- `src/vision/vision_htp.cpp`: tensor/argument validation and direct invocation of
  the QAIC-generated DSP resize stub.
- `htp_runtime.h/.cpp`: session lifecycle and guarded access to its native handle;
  no image operators are declared here.
- `geniex_proc_htp.idl`: private RPC contract and QAIC-generated bindings.
- `dsp/geniex_proc_htp.c`: DSP RPC argument checks and dispatch.
- `dsp/resize_kernel.h/.c`: independent RGB Catmull-Rom resize, with scalar
  horizontal filtering and HVX vertical filtering. It does not depend on stb.

CMake selects `*_cpu.cpp` or `*_htp.cpp` in the vision and processor directories.
The private backend target supplies SDK/generated headers to host adapters;
public headers do not expose SDK types. The host and DSP builds are separate.

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

There is no separate runtime test file or standalone smoke executable. The fixed
small RGB oracle remains byte-exact. Additional antialias oracle checks allow one
uint8 level of floating-point rounding difference. The 17-case qcs comparison
found 15 byte-exact cases and a maximum one-level difference in two large downsample
cases; this is not a claim of bit-exact output for every image. Performance
optimization is deferred.
