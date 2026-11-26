# CPU image. CUDA is not installed. The GTSRB files are not copied in.
# docker build -t gtsrb-cnn .
# docker run --rm gtsrb-cnn
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/src/build \
    OMP_NUM_THREADS=4 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        libopenblas-dev \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip \
    && python -m pip install numpy pytest opencv-python-headless pybind11 pandas \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /src
COPY . .
RUN cmake -S . -B build -G Ninja \
        -DPython3_EXECUTABLE=/opt/venv/bin/python \
        -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build build

CMD ["python", "-m", "pytest", "tests/test_reference.py"]
