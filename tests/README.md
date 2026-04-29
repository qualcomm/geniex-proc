# Tests

GoogleTest-based unit tests for `geniex-proc`. Pure CPU — runs in GitHub CI.

## Build and run

```pwsh
cmake -B build `
      -DGENIEXPROC_ENABLE_VISION=ON `
      -DGENIEXPROC_BUILD_TESTS=ON
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

GoogleTest is pulled by CMake `FetchContent`; no system install needed.

## Test binaries

| Binary | Source | Covers |
|---|---|---|
| `sampler`   | [`sampler.cpp`](sampler.cpp)     | `Sampler::sample_greedy`, deterministic seeded sampling, logit bias, EOG detection, reset, init. No tokenizer needed. |
| `tokenizer` | [`tokenizer.cpp`](tokenizer.cpp) | `Tokenizer::from_file`, encode / decode roundtrip (ASCII + UTF-8), `id_to_piece` ⇄ `piece_to_id`, special-token detection. |
| `vision`    | [`vision.cpp`](vision.cpp)       | `round_by_factor`, `ceil_by_factor`, `floor_by_factor`, `smart_resize` invariants. Built only when `GENIEXPROC_ENABLE_VISION=ON`. |

## Tokenizer fixture

`tokenizer` needs a real `tokenizer.json`. CMake downloads one on the
first configure (`Qwen/Qwen2.5-0.5B`, ~7 MB) into `build/tests/fixtures/` and
caches it.

To use a local file instead (e.g. air-gapped build):

```pwsh
cmake -B build -DGENIEXPROC_TEST_TOKENIZER="C:\path\to\tokenizer.json" ...
```

If the download fails and no override is set, `tokenizer` reports the
individual tests as **skipped** (via `GTEST_SKIP`), not failed.
