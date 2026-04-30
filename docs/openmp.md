# OpenMP

## Mac

```bash
brew install libomp
```

```bash
export OpenMP_ROOT=$(brew --prefix)/opt/libomp
export LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$LD_LIBRARY_PATH"
```

Add following definations to your cmake configuration commands:

```bash
  -DCMAKE_C_FLAGS="-I/opt/homebrew/opt/libomp/include" \
  -DCMAKE_CXX_FLAGS="-I/opt/homebrew/opt/libomp/include" \
```