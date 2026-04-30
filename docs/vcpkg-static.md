# Use `vcpkg` for static linking

This document describes how to use `vcpkg` to install static libraries, then use them in the project.

## Install `vcpkg`

```powershell
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
```

## Install static libraries

Here we take `fftw3` and `arm64-windows` platform as an example.

```powershell
cd C:/path/to/vcpkg
.\vcpkg.exe install fftw3:arm64-windows-static
```

> [!NOTE]
> The `arm64-windows-static` part after the colon is called *triplet*.
> You can list all available triplets by running:
> ```powershell
> .\vcpkg.exe help triplet
> ```
> Then you may choose triplets for desired platforms and architectures.

## Configure the project

To use the installed static libraries in the project, you need to configure the project with `cmake` to use the corresponding triplet by setting `VCPKG_TARGET_TRIPLET`.

You can either use `-D` as below:

```powershell
cmake -B build `
  -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=arm64-windows-static `
  ...
```

Or set `VCPKG_TARGET_TRIPLET` in the CMakeLists.txt file directly:

```cmake
set(VCPKG_TARGET_TRIPLET arm64-windows-static)
```

## References

- https://devblogs.microsoft.com/cppblog/vcpkg-updates-static-linking-is-now-available/