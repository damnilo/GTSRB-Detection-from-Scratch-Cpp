cmake_minimum_required(VERSION 3.15)

project(cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCES cpp/bindings.cpp)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

# cblas.h lives in <toolchain>/include/openblas and includes openblas_config.h
# from that same directory. The UCRT64 package is the OpenMP build, so it
# shares OMP_NUM_THREADS with the rest of the library.
get_filename_component(_cxx_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(_toolchain "${_cxx_bin}" DIRECTORY)
find_path(OPENBLAS_INCLUDE_DIR cblas.h HINTS "${_toolchain}/include/openblas" REQUIRED)
find_library(OPENBLAS_LIBRARY NAMES openblas HINTS "${_toolchain}/lib" REQUIRED)
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