cmake_minimum_required(VERSION 3.15)

project(cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCES cpp/bindings.cpp)

find_package(OpenMP REQUIRED)

# cblas.h lives next to openblas_config.h. UCRT64 puts that under
# <toolchain>/include/openblas. Debian and Ubuntu use the multiarch directory.
get_filename_component(_cxx_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(_toolchain "${_cxx_bin}" DIRECTORY)
find_path(OPENBLAS_INCLUDE_DIR cblas.h
    HINTS
        "${_toolchain}/include/openblas"
        /usr/include/openblas
        /usr/include/x86_64-linux-gnu/openblas
        /usr/local/include/openblas
)
find_library(OPENBLAS_LIBRARY NAMES openblas
    HINTS
        "${_toolchain}/lib"
        /usr/lib/x86_64-linux-gnu
        /usr/lib
        /usr/local/lib
)
if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIBRARY)
    message(FATAL_ERROR
        "OpenBLAS was not found. Install libopenblas-dev, or set OPENBLAS_INCLUDE_DIR and OPENBLAS_LIBRARY.")
endif()

find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
if(Python3_FOUND AND pybind11_FOUND)
pybind11_add_module(cpp
    ${SOURCES}
    cpp/Utils/src/Tensor.cpp
    cpp/Utils/src/Kernel.cpp
    cpp/Utils/src/Serialization.cpp
    cpp/Layers/src/Network.cpp
    cpp/Layers/src/Dense.cpp
    cpp/Layers/src/Conv2D.cpp
    cpp/Layers/src/ReLU.cpp
    cpp/Layers/src/SoftMax.cpp
    cpp/Layers/src/Flatten.cpp
    cpp/Layers/src/MaxPool2D.cpp
    cpp/Layers/src/Dropout.cpp
    cpp/Layers/src/BatchNorm2D.cpp
    cpp/Optimizers/src/SGD.cpp
    cpp/Optimizers/src/AdamW.cpp
    cpp/Loss/src/CrossEntropy.cpp
)

# Headers are included as "Layers/include/...", "Utils/include/...", etc.
target_include_directories(cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp ${OPENBLAS_INCLUDE_DIR})
target_link_libraries(cpp PRIVATE OpenMP::OpenMP_CXX ${OPENBLAS_LIBRARY})
else()
    message(STATUS "pybind11 was not found. The Python module is skipped. The CUDA benches do not need it.")
endif()

# The CUDA benches launch the kernels. Skip them when nvcc is not installed
# so the Python module still configures on a CPU-only machine.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_executable(bench_conv
        cpp_bench/bench_conv.cpp
        cuda/src/TensorOps.cu
        cuda/src/Convolution.cu
        cpp/Utils/src/Tensor.cpp
        cpp/Utils/src/Kernel.cpp
        cpp/Layers/src/Conv2D.cpp
    )
    set_source_files_properties(cpp_bench/bench_conv.cpp PROPERTIES LANGUAGE CUDA)
    target_include_directories(bench_conv PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp
        ${OPENBLAS_INCLUDE_DIR}
    )
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(bench_conv PRIVATE OpenMP::OpenMP_CXX ${OPENBLAS_LIBRARY} CUDA::cublas)

    add_executable(bench_eval
        cpp_bench/bench_eval.cpp
        cuda/src/TensorOps.cu
        cuda/src/Convolution.cu
        cuda/src/ReLU.cu
        cuda/src/MaxPool2D.cu
        cpp/Utils/src/Tensor.cpp
        cpp/Utils/src/Kernel.cpp
        cpp/Layers/src/Conv2D.cpp
        cpp/Layers/src/ReLU.cpp
        cpp/Layers/src/MaxPool2D.cpp
        cpp/Layers/src/Flatten.cpp
        cpp/Layers/src/Dense.cpp
        cpp/Layers/src/Dropout.cpp
    )
    set_source_files_properties(cpp_bench/bench_eval.cpp PROPERTIES LANGUAGE CUDA)
    target_include_directories(bench_eval PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp
        ${OPENBLAS_INCLUDE_DIR}
    )
    target_link_libraries(bench_eval PRIVATE OpenMP::OpenMP_CXX ${OPENBLAS_LIBRARY})
endif()