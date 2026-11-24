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
    message(STATUS "pybind11 was not found. The Python module is skipped. The CUDA bench does not need it.")
endif()

# The CUDA bench launches the kernels. Skip it when nvcc is not installed
# so the Python module still configures on a CPU-only machine.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_executable(bench_eval
        cpp_bench/bench_eval.cpp
        cpp_bench/train_images.cpp
        cuda/Utils/src/TensorOps.cu
        cuda/Layers/src/Convolution.cu
        cuda/Layers/src/ReLU.cu
        cuda/Layers/src/MaxPool2D.cu
        cuda/Layers/src/Dropout.cu
        cuda/Optimizers/src/SGD.cu
        cuda/Optimizers/src/SGDDevice.cu
        cuda/Launch/src/Conv2D.cu
        cuda/Launch/src/ReLUDevice.cu
        cuda/Launch/src/MaxPoolDevice.cu
        cuda/Launch/src/DenseDevice.cu
        cuda/Launch/src/DropoutDevice.cu
        cuda/Layers/src/BatchNorm2D.cu
        cuda/Launch/src/BatchNormDevice.cu
        cpp/Utils/src/Tensor.cpp
        cpp/Utils/src/Kernel.cpp
        cpp/Layers/src/Network.cpp
        cpp/Layers/src/Conv2D.cpp
        cpp/Layers/src/ReLU.cpp
        cpp/Layers/src/MaxPool2D.cpp
        cpp/Layers/src/Flatten.cpp
        cpp/Layers/src/Dense.cpp
        cpp/Layers/src/Dropout.cpp
        cpp/Layers/src/BatchNorm2D.cpp
        cpp/Layers/src/SoftMax.cpp
        cpp/Loss/src/CrossEntropy.cpp
        cpp/Optimizers/src/SGD.cpp
    )
    set_source_files_properties(cpp_bench/bench_eval.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(bench_eval PRIVATE GTSRB_CUDA)
    target_include_directories(bench_eval PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp
        ${OPENBLAS_INCLUDE_DIR}
    )
    target_link_libraries(bench_eval PRIVATE OpenMP::OpenMP_CXX ${OPENBLAS_LIBRARY} CUDA::cublas)

    # The Python module stays CPU-only unless this configure found nvcc.
    if(TARGET cpp)
        target_sources(cpp PRIVATE
            cuda/Utils/src/TensorOps.cu
            cuda/Layers/src/Convolution.cu
            cuda/Layers/src/ReLU.cu
            cuda/Layers/src/MaxPool2D.cu
            cuda/Layers/src/Dropout.cu
            cuda/Launch/src/Conv2D.cu
            cuda/Launch/src/ReLUDevice.cu
            cuda/Launch/src/MaxPoolDevice.cu
            cuda/Launch/src/DenseDevice.cu
            cuda/Launch/src/DropoutDevice.cu
            cuda/Layers/src/BatchNorm2D.cu
            cuda/Launch/src/BatchNormDevice.cu
        )
        target_compile_definitions(cpp PRIVATE GTSRB_CUDA)
        target_include_directories(cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(cpp PRIVATE CUDA::cudart)
    endif()
endif()