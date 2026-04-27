# Overview
## xtensor Library Documentation

> **Note**: xtensor provides several data types to accommodate different use cases:
>
> - **xt::xarray**: Can be reshaped dynamically to any number of dimensions. Most similar to NumPy arrays.
> - **xt::xtensor**: Has dimensions set at compilation time, enabling optimizations like stack-allocated shapes and strides.
> - **xt::xtensor_fixed**: Has shape fixed at compile time, allowing even more optimizations including stack storage allocation and compile-time stride computation.

This section gives a short guide to the use of the `xtensor` library and how it can be combined with other processing tools.

## Data transfer between the xtensor and the GGML
We show an example below how to use xtensor to optimize the GGML. It is a minimum example, but it is enough for us to understand how to make it communicate with the GGML library:
```cpp
struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    struct ggml_context * ctx;
};


void load_model(simple_model & model,
    const xt::xtensor<float, 2>& matrix_A,
    const xt::xtensor<float, 2>& matrix_B) {
    int rows_A = matrix_A.shape()[0];
    int cols_A = matrix_A.shape()[1];
    int rows_B = matrix_B.shape()[0];
    int cols_B = matrix_B.shape()[1];

    size_t ctx_size = 0;
    {
        ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32) * 100;
        ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F32);
        ctx_size += 2 * ggml_tensor_overhead(),
        ctx_size += ggml_graph_overhead();
        ctx_size += 1024;
    }

    // Rest of init code...
    struct ggml_init_params params {
    /*.mem_size   =*/ ctx_size,
    /*.mem_buffer =*/ NULL,
    /*.no_alloc   =*/ false,
    };

    model.ctx = ggml_init(params);
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_A, rows_A);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_B, rows_B);

    memcpy(model.a->data, matrix_A.begin(), ggml_nbytes(model.a));
    memcpy(model.b->data, matrix_B.begin(), ggml_nbytes(model.b));
}

struct ggml_cgraph * build_graph(const simple_model& model) {
    struct ggml_cgraph  * gf = ggml_new_graph(model.ctx);

    struct ggml_tensor * result = ggml_mul_mat(model.ctx, model.a, model.b);

    ggml_build_forward_expand(gf, result);
    return gf;
}

struct ggml_tensor * compute(const simple_model & model) {
    struct ggml_cgraph * gf = build_graph(model);

    int n_threads = 1;

    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);
    return ggml_graph_node(gf, -1);
}

int main(void) {
    ggml_time_init();

    xt::xtensor<float, 2> matrix_A = {
        {2.0f, 8.0f},
        {5.0f, 1.0f},
        {4.0f, 2.0f},
        {8.0f, 6.0f}
    };

    xt::xtensor<float, 2> matrix_B = {
        {10.0f, 5.0f},
        {9.0f, 9.0f},
        {5.0f, 4.0f}
    };

    simple_model model;
    load_model(model, matrix_A, matrix_B);

    struct ggml_tensor * result = compute(model);

    std::array<std::size_t, 2> shape = {
        static_cast<std::size_t>(result->ne[0]),
        static_cast<std::size_t>(result->ne[1])
    };

    xt::xarray<float> result_tensor = xt::adapt(
        static_cast<float*>(result->data),
        static_cast<std::size_t>(ggml_nelements(result)),
        xt::no_ownership(),
        shape
    );

    std::cout << "result min: " << xt::amin(result_tensor)() << std::endl;
    std::cout << "result max: " << xt::amax(result_tensor)() << std::endl;
    std::cout << "result mean: " << xt::mean(result_tensor)() << std::endl;
    std::cout << "result top 10 values:\n" << xt::view(result_tensor, xt::range(0, 10), xt::all()) << std::endl;
    ggml_free(model.ctx);
    return 0;
}
```

Now, we note how to add the library, in your main project, `CMakeLists.txt`, we can add
```cmake
add_subdirectory(third_party/xtl)
add_subdirectory(third_party/xtensor)
include_directories(${CMAKE_SOURCE_DIR}/third_party/xtensor/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/xtl/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(xtensor_DIR ${CMAKE_BINARY_DIR}/third_party/xtensor)
set(xtl_DIR ${CMAKE_BINARY_DIR}/third_party/xtl)
```

And inside the example, or the object folder's CMakeLists.txt, we can add
```cmake
find_package(xtensor REQUIRED)
```
After this, we can begin to use the `xtensor` library.

## Tricky parts of `xtensor`

- Operations like `xt::stack(xtuple(a, b, c), 1)`, `xt::hstack(xtuple(a, b, c))`, `xt::vstack(xtuple(a, b, c))`, `xt::concatenate(xtuple(a, b, c), 1)` require the input to be grouped by `xt::xtuple`, which is essentially `std::make_tuple`. Use `xt::xtuple` may zero out all elements, so use `std::make_tuple` instead.

```cpp
// use xt::xtuple will zero out all elements
// x = xt::concatenate(xt::xtuple(x, y), 0);
x = xt::concatenate(std::make_tuple(x, y), 0);
```

- A combination of mathematical function (e.g., `xt::maximum`) and reducer (e.g., `xt::amax`) will be super slow and even stucks there forever. So use `xt::eval` to evaluate the result first.

```cpp
// use xt::eval to evaluate the result first
// x = xt::maximum(x, xt::amax(y));
float y_max_float = xt::eval(xt::amax(y)).data()[0];
x = xt::maximum(x, y_max_float);
```

# Guidelines for Adding New Processors

Each processor is designed for a specific model family to ensure it can be used as conveniently as possible. This document outlines the steps to add a new processor to our system.

## Basic Structure

1. **Define the interface in the include folder**
   - Create a header file in the `include/` directory with the processor class declaration
   - Follow naming convention: `model-family-proc.h` (e.g., `convnext-proc.h`, `omnivlm-proc.h`)
   - Header files inside include can include each other as needed

2. **Implement in the src folder**
   - Create a source file in the `src/` directory with the processor implementation
   - Match the name of the header file: `model-family-proc.cpp`
   - Include the corresponding header file from the include directory

3. **Test the processor in the examples folder**
   - Create a new directory in the `examples/` folder with the model family name
   - Create a CMakeLists.txt file for the example
   - Add a simple example program that demonstrates the processor's usage

## Implementation Patterns

### Header File (`include/model-family-proc.h`)
- Use a namespace specific to your model family
- Declare a class with a clear processor name (e.g., `ModelFamilyPreprocessor`)
- Define public methods for processing input data
- Define private members for configuration parameters

### Source File (`src/model-family-proc.cpp`)
- Implement the methods declared in the header file
- Use xtensor for data manipulation
- Follow existing code style and conventions

### Example Usage (`examples/model-family/main.cpp`)
- Demonstrate basic usage of the processor
- Include error handling and validation
- Show expected output and verification

## Adding to the Build System

1. Add the new processor to the project's CMakeLists.txt:
   ```cmake
   set(SOURCES
       src/convnext-proc.cpp
       src/model-family-proc.cpp
       # Add your new processor here
   )
   ```

2. Update the examples/CMakeLists.txt to include your new example folder.

## Integration with xtensor

- Use the `xt::xarray` and other xtensor data types for tensor operations
- Be aware of xtensor's performance considerations as mentioned in the documentation
- Use `xt::eval` when combining mathematical functions and reducers to avoid performance issues
- Use `std::make_tuple` instead of `xt::xtuple` for operations like `xt::concatenate`

## Integration with GGML

- When transferring data between xtensor and GGML, follow the pattern shown in the documentation
- Use the `xt::adapt` function to create views of GGML tensor data without copying
- Pay attention to tensor shapes and memory layout when transferring data

## Testing

- Test your processor with various inputs to ensure robustness
- Verify outputs match expected values from reference implementations
- Include validation logic to handle edge cases and invalid inputs
