// pybind11 bindings exposing the C++ library as the `cpp` Python extension module.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Utils/include/Tensor.h"
#include "Utils/include/Serialization.h"
#include "Layers/include/Layer.h"
#include "Layers/include/Dense.h"
#include "Layers/include/ReLU.h"
#include "Layers/include/Conv2D.h"
#include "Layers/include/Flatten.h"
#include "Layers/include/SoftMax.h"
#include "Layers/include/Network.h"
#include "Layers/include/MaxPool2D.h"
#include "Layers/include/Dropout.h"
#include "Layers/include/BatchNorm2D.h"
#include "Loss/include/CrossEntropy.h"
#include "Loss/include/Loss.h"
#include "Optimizers/include/SGD.h"
#include "Optimizers/include/AdamW.h"

#ifdef GTSRB_CUDA
#include <cuda_runtime.h>
#endif

#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

#ifdef GTSRB_CUDA
struct DeviceInput {
    float* data = nullptr;
    size_t capacity = 0;
};

DeviceInput& device_input() {
    static DeviceInput slot;
    return slot;
}

void must_cuda(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
}
#endif

}  // namespace

PYBIND11_MODULE(cpp, m) {
    m.doc() = "Neural Network Library Bindings";
    // One seed for weight init and one for dropout masks. Both generators are
    // reseeded, but they stay independent streams.
    m.def("seed", [](unsigned int seed) {
        Tensor::seed(seed);
        Dropout::seed(seed);
    }, py::arg("seed"));
    m.def("createNetwork", []() { return new Network(); }, py::return_value_policy::take_ownership);
    m.def("cuda_available", []() {
#ifdef GTSRB_CUDA
        return true;
#else
        return false;
#endif
    });
    m.def("forward_device", [](Network& network, py::array_t<float, py::array::c_style | py::array::forcecast> batch) -> py::tuple {
#ifndef GTSRB_CUDA
        (void)network;
        (void)batch;
        throw std::runtime_error("CUDA was not built");
#else
        py::buffer_info info = batch.request();
        if (info.ndim != 4 || info.shape[1] != 3 || info.shape[2] != 32 || info.shape[3] != 32) {
            throw std::runtime_error("forward_device expects an array of shape (N, 3, 32, 32)");
        }
        const auto count = static_cast<size_t>(info.shape[0]);
        if (count < 1 || count > 64) {
            throw std::runtime_error("forward_device batch must be from 1 to 64");
        }

        DeviceInput& slot = device_input();
        if (count > slot.capacity) {
            if (slot.data != nullptr) {
                must_cuda(cudaFree(slot.data), "cudaFree");
                slot.data = nullptr;
            }
            must_cuda(cudaMalloc(&slot.data, count * 3 * 32 * 32 * sizeof(float)), "cudaMalloc");
            slot.capacity = count;
        }

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        must_cuda(cudaEventCreate(&start), "cudaEventCreate");
        must_cuda(cudaEventCreate(&stop), "cudaEventCreate");
        must_cuda(cudaEventRecord(start), "cudaEventRecord");
        must_cuda(cudaMemcpy(
            slot.data, info.ptr, count * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice
        ), "cudaMemcpy HtoD");
        float* logits = network.forwardDevice(slot.data, count, 3, 32, 32);
        py::array_t<float> out({static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(43)});
        must_cuda(cudaMemcpy(
            out.mutable_data(), logits, count * 43 * sizeof(float), cudaMemcpyDeviceToHost
        ), "cudaMemcpy DtoH");
        must_cuda(cudaEventRecord(stop), "cudaEventRecord");
        must_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
        float milliseconds = 0.0f;
        must_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return py::make_tuple(out, milliseconds);
#endif
    }, py::arg("network"), py::arg("batch"));

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init<std::initializer_list<size_t>>())
        .def("getShape", &Tensor::getShape)
        .def("size", &Tensor::size)
        .def("dimensions", &Tensor::dimensions)
        .def("at", (float& (Tensor::*)(const std::vector<size_t>&)) &Tensor::at)
        .def("at", (const float& (Tensor::*)(const std::vector<size_t>&) const) &Tensor::at)
        .def("__getitem__", [](const Tensor& t, const std::vector<size_t>& indices) { return t.at(indices); })
        .def("__setitem__", [](Tensor& t, const std::vector<size_t>& indices, float value) { t.at(indices) = value; })
        // Zero-copy numpy view into the Tensor's own buffer (not a copy like to_numpy()).
        // `self` is passed as the array's base object so pybind11 keeps the Tensor alive
        // for as long as the returned array exists — critical for safety here.
        // Used by clip_gradients() in Python for fast in-place vectorized scaling.
        .def("raw", [](py::object self) {
            Tensor& t = self.cast<Tensor&>();
            return py::array_t<float>({t.size()}, {sizeof(float)}, t.raw().data(), self);
        }, py::return_value_policy::reference)
        .def("reshape", &Tensor::reshape)
        .def("fill", &Tensor::fill)
        .def("randomize", &Tensor::randomize)
        .def_static("zeros", &Tensor::zeros)
        // Constructs a Tensor directly from a numpy array, copying element-by-element
        // in row-major order to match Tensor's internal layout.
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> array) {
            auto buf = array.request();
            std::vector<size_t> shape(buf.shape.begin(), buf.shape.end());
            Tensor tensor(shape);
            float* src = static_cast<float*>(buf.ptr);
            std::vector<size_t> idx(shape.size(), 0);
            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor.at(idx) = src[i];
                for(int j = (int)shape.size() - 1; j >= 0; --j) {
                    if (++idx[j] < shape[j]) break;
                    idx[j] = 0;
                }
            }
            return tensor;
        }))
        // Copies the Tensor's contents into a brand-new numpy array (unlike raw(),
        // this allocates fresh memory rather than viewing the Tensor's own buffer).
        .def("to_numpy", [](const Tensor& t) {
            auto shape = t.getShape();
            py::array_t<float> out(shape);
            float* dst = static_cast<float*>(out.request().ptr);
            std::vector<size_t> idx(shape.size(), 0);
            for (size_t i = 0; i < t.size(); ++i) {
                dst[i] = t.at(idx);
                for(int j = (int)shape.size() - 1; j >= 0; --j) {
                    if (++idx[j] < shape[j]) break;
                    idx[j] = 0;
                }
            }
            return out;
        });

    py::classh<Layer>(m, "Layer");

    // NOTE: parameters()/gradients() bindings below use py::return_value_policy::reference
    // because they return raw pointers into each layer's own member Tensors (e.g. &weights).
    // Without this, pybind11 defaults to take_ownership and will attempt to delete these
    // sub-object pointers when the Python wrapper is garbage-collected — undefined behavior.
    py::classh<Dense, Layer>(m, "Dense")
        .def(py::init<size_t, size_t>())
        .def("forward", &Dense::forward)
        .def("backward", &Dense::backward)
        .def("parameters", &Dense::parameters, py::return_value_policy::reference)
        .def("gradients", &Dense::gradients, py::return_value_policy::reference);

    py::classh<ReLU, Layer>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward)
        .def("backward", &ReLU::backward);

    py::classh<Conv2D, Layer>(m, "Conv2D")
        .def(py::init<size_t, size_t, size_t, size_t, size_t>())
        .def("forward", &Conv2D::forward)
        .def("backward", &Conv2D::backward)
        .def("parameters", &Conv2D::parameters, py::return_value_policy::reference)
        .def("gradients", &Conv2D::gradients, py::return_value_policy::reference);

    py::classh<Flatten, Layer>(m, "Flatten")
        .def(py::init<>())
        .def("forward", &Flatten::forward)
        .def("backward", &Flatten::backward);

    py::classh<SoftMax, Layer>(m, "SoftMax")
        .def(py::init<>())
        .def("forward", &SoftMax::forward)
        .def("backward", &SoftMax::backward);

    py::class_<Loss>(m, "Loss");

    py::class_<CrossEntropy, Loss>(m, "CrossEntropy")
        .def(py::init<>())
        .def("forward", &CrossEntropy::forward)
        .def("backward", &CrossEntropy::backward);

    py::class_<Network>(m, "Network")
        .def(py::init<>())
        .def("addLayer", [](Network& net, std::shared_ptr<Layer> layer) { net.addLayer(layer); })
        .def("forward", &Network::forward)
        .def("backward", &Network::backward)
        .def("numLayers", &Network::numLayers)
        .def("getLayer", &Network::getLayer, py::return_value_policy::reference)
        .def("parameters", &Network::parameters, py::return_value_policy::reference)
        .def("gradients", &Network::gradients, py::return_value_policy::reference);

    py::class_<SGD>(m, "SGD")
        .def(py::init<float, float>(), py::arg("learningRate"), py::arg("momentum") = 0.0f)
        .def("update", &SGD::update)
        .def("setLearningRate", &SGD::setLearningRate)
        .def("getLearningRate", &SGD::getLearningRate);

    py::class_<AdamW>(m, "AdamW")
        .def(py::init<float, float, float, float, float>(),
             py::arg("learningRate") = 1e-3f,
             py::arg("weightDecay") = 1e-2f,
             py::arg("beta1") = 0.9f,
             py::arg("beta2") = 0.999f,
             py::arg("epsilon") = 1e-8f)
        .def("update", &AdamW::update)
        .def("setLearningRate", &AdamW::setLearningRate)
        .def("getLearningRate", &AdamW::getLearningRate);

    py::class_<Serialization>(m, "Serialization")
        .def_static("saveNetwork", &Serialization::saveNetwork)
        .def_static("loadNetwork", &Serialization::loadNetwork);

    py::classh<MaxPool2D, Layer>(m, "MaxPool2D")
        .def(py::init<size_t, size_t>())
        .def("forward", &MaxPool2D::forward)
        .def("backward", &MaxPool2D::backward);

    py::classh<Dropout, Layer>(m, "Dropout")
        .def(py::init<float>())
        .def("forward", &Dropout::forward)
        .def("backward", &Dropout::backward)
        .def("setTraining", &Dropout::setTraining)   // Call with False before evaluation/inference
        .def("isTraining", &Dropout::isTraining);

    py::classh<BatchNorm2D, Layer>(m, "BatchNorm2D")
        .def(py::init<size_t>())
        .def("forward", &BatchNorm2D::forward)
        .def("backward", &BatchNorm2D::backward)
        .def("parameters", &BatchNorm2D::parameters, py::return_value_policy::reference)
        .def("gradients", &BatchNorm2D::gradients, py::return_value_policy::reference)
        .def("setTraining", &BatchNorm2D::setTraining)
        .def("isTraining", &BatchNorm2D::isTraining)
        .def("runningMean", &BatchNorm2D::runningMeanBuffer, py::return_value_policy::reference)
        .def("runningVar", &BatchNorm2D::runningVarBuffer, py::return_value_policy::reference);
}