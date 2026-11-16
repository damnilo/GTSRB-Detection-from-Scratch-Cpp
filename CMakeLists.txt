cmake_minimum_required(VERSION 3.15)

project(cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCES cpp/bindings.cpp)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)
pybind11_add_module(cpp
    ${SOURCES}
    cpp/Utils/src/Tensor.cpp
    cpp/Utils/src/Serialization.cpp
    cpp/Layers/src/Network.cpp
    cpp/Layers/src/Dense.cpp
    cpp/Layers/src/Conv2D.cpp
    cpp/Layers/src/ReLU.cpp
    cpp/Layers/src/SoftMax.cpp
    cpp/Layers/src/Flatten.cpp
    cpp/Layers/src/MaxPool2D.cpp
    cpp/Layers/src/Dropout.cpp
    cpp/Optimizers/src/SGD.cpp
    cpp/Loss/src/CrossEntropy.cpp
)

# Headers are included as "Layers/include/...", "Utils/include/...", etc.
target_include_directories(cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(cpp PRIVATE OpenMP::OpenMP_CXX)