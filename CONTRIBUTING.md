# Contributing to geniex-proc

Thanks for your interest in contributing! Whether you're fixing a bug, adding a processor, or improving the docs, this guide takes you from `git clone` to a merged PR. The same rules apply to external and internal contributors — this is the single source of truth.

Please also read our [Code of Conduct](CODE-OF-CONDUCT.md) and [license](LICENSE.txt).

## Getting started

Prerequisites:

- **CMake 3.10+**
- **C++20 compatible compiler** — VS 2019+ (Windows), GCC 9+/Clang 10+ (Linux), Xcode 11+ (macOS).
- **Rust toolchain** (`cargo`) — required by the `tokenizers-cpp` submodule.
- **[vcpkg](docs/vcpkg-static.md)** — for dependency management on Windows.

Clone with submodules — the build vendors `third-party/minja` and `third-party/tokenizers-cpp`:

```bash
git clone --recursive https://github.com/qualcomm/geniex-proc.git
cd geniex-proc
cmake -B build
cmake --build build
```

That builds the core library. Feature modules (multimodal, OpenCV, PaddleOCR, audio/video) are opt-in via `-DGENIEXPROC_BUILD_*` options. Full per-platform instructions, the complete option table, and dependency setup live in [docs/build.md](docs/build.md).

## Running tests

GoogleTest-based unit tests (pure CPU, fetched via CMake — no system install). This is exactly what CI runs:

```bash
cmake -B build -DGENIEXPROC_ENABLE_VISION=ON -DGENIEXPROC_BUILD_TESTS=ON
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

The `tokenizer` and `processor` suites need a real `tokenizer.json`; CMake downloads a small fixture on first configure and caches it. See [tests/README.md](tests/README.md) for the binary matrix and how to point at local fixtures for air-gapped builds.

## Project structure

| Directory        | Contents                                                            |
|------------------|---------------------------------------------------------------------|
| [src/](src/)     | Library sources: `sampler/`, `tokenizer/`, `vision/`, `processors/`, `geniex-sampling/`, `internal/`. |
| [include/](include/) | Public headers (`geniex-proc/`) and vendored `nlohmann/`.       |
| [tests/](tests/) | GoogleTest unit tests (`sampler`, `tokenizer`, `vision`, `processor`). |
| [docs/](docs/)   | Build, vcpkg, and OpenMP guides.                                    |
| `third-party/`   | Vendored `minja` and `tokenizers-cpp` submodules.                   |

## Making a change

Develop on a branch off `main`; PRs target `main`.

1. [Fork](https://github.com/qualcomm/geniex-proc/fork) and clone your fork.
2. Create a branch: `git checkout -b <type>/<short-topic> main` (e.g. `fix/cjk-detokenization`).
3. Add an upstream remote to keep your branch current:

   ```bash
   git remote add upstream https://github.com/qualcomm/geniex-proc.git
   git pull --rebase upstream main
   ```

4. Make your change, add tests, keep the change focused — split independent changes into separate PRs.

### Commits — Conventional Commits

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>
```

| Type       | Meaning                                      |
|------------|----------------------------------------------|
| `feat`     | New user-visible feature.                    |
| `fix`      | Bug fix.                                      |
| `perf`     | Performance improvement, no behavior change. |
| `refactor` | Internal restructure, no behavior change.    |
| `docs`     | Documentation only.                          |
| `chore`    | Build, deps, tooling, misc.                  |
| `test`     | Test-only change.                            |
| `ci`       | CI config only.                              |

Subject: imperative mood (`add`, not `added`), ≤ 72 characters, no trailing period. Add a body only when the "why" is non-obvious.

### Sign your commits (DCO)

This project uses the [Developer Certificate of Origin](https://developercertificate.org/). Every commit must carry a `Signed-off-by` line — add it with `-s`:

```bash
git commit -s -m "fix(tokenizer): extend byte-level reverse table for CJK"
```

The DCO check in CI rejects PRs whose commits are not signed off.

## Code style & linting

This project uses **clang-format** (Google base style, 4-space indent, 120-column limit — see [`.clang-format`](.clang-format)). Run it on every file you change before committing:

```bash
clang-format -i <files>
```

## Opening a PR

1. Push your branch to your fork (`git push -u origin <type>/<short-topic>`).
2. [Open a PR](https://github.com/qualcomm/geniex-proc/pulls) against `main`.

- **Title**: follow the Conventional Commits format above.
- **Description**: what changed and why; note any new build options or dependencies.
- **Checks**: the [Build Check](.github/workflows/build-check.yml) builds and runs the test suite on Windows x64. In parallel, [QC Preflight Checks](.github/workflows/qcom-preflight-checks.yml) runs **Semgrep** static analysis plus DCO, copyright/license, and repolinter checks. Resolve anything they flag before merge.
- Reviewers are assigned automatically via [CODEOWNERS](CODEOWNERS). It's a good idea to discuss large features or architecture changes in an issue first — reviews go faster with no surprises.

## Reporting issues

Open an issue at [github.com/qualcomm/geniex-proc/issues](https://github.com/qualcomm/geniex-proc/issues). A good bug report includes a minimal repro, your environment (OS, compiler, CMake options), and relevant logs. Feature requests are welcome too — describe the use case.

For **security vulnerabilities**, do **not** open a public issue — follow [SECURITY.md](SECURITY.md).

## Community & Code of Conduct

Participation is governed by our [Code of Conduct](CODE-OF-CONDUCT.md). Be respectful and constructive.
