FROM nvcr.io/nvidia/isaac-sim:2022.1.0

# There is an issue with the Isaac Sim docker where it can't install from Apt
RUN echo "deb http://archive.ubuntu.com/ubuntu/ bionic main restricted" > /etc/apt/sources.list \
  && echo "deb http://archive.ubuntu.com/ubuntu/ bionic-updates main restricted" >> /etc/apt/sources.list \
  && echo "deb http://archive.ubuntu.com/ubuntu/ bionic universe" >> /etc/apt/sources.list \
  && echo "deb http://archive.ubuntu.com/ubuntu/ bionic-updates universe" >> /etc/apt/sources.list \
  && echo "deb http://archive.ubuntu.com/ubuntu/ bionic multiverse" >> /etc/apt/sources.list \
  && echo "deb http://archive.ubuntu.com/ubuntu/ bionic-updates multiverse" >> /etc/apt/sources.list \
  && echo "deb http://archive.ubuntu.com/ubuntu/ bionic-backports main restricted universe multiverse" >> /etc/apt/sources.list \
  && echo "deb http://security.ubuntu.com/ubuntu bionic-security main restricted" >> /etc/apt/sources.list \
  && echo "deb http://security.ubuntu.com/ubuntu bionic-security universe" >> /etc/apt/sources.list \
  && echo "deb http://security.ubuntu.com/ubuntu bionic-security multiverse" >> /etc/apt/sources.list \
  && rm ~/.pip/pip.conf

# # Install apt dependencies necessary for OMPL
RUN apt update \
  && apt install -y --no-install-recommends \
    g++ \
    cmake \
    pkg-config \
    libboost-serialization-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libboost-test-dev \
    libeigen3-dev \
    libode-dev \
    wget \
    libyaml-cpp-dev \
    python3.7-dev \
    libboost-python-dev \
    libboost-numpy-dev \
    git \
  && apt-get clean \
  && rm -rf /var/lib/apt

# Clone repos necessary to build OMPL and build it against the Isaac Sim Python
RUN /isaac-sim/kit/python/bin/python3 -m pip install -vU https://github.com/CastXML/pygccxml/archive/develop.zip pyplusplus numpy \
 && wget -q -O- https://data.kitware.com/api/v1/file/5e8b740d2660cbefba944189/download | tar zxf - -C ${HOME} \
 && wget -O - https://github.com/ompl/ompl/archive/1.5.2.tar.gz | tar zxf - \
 && export CXX=g++ \
 && export MAKEFLAGS="-j `nproc`" \
 && export PATH=${HOME}/castxml/bin:${PATH} \
 && mkdir -p ompl-1.5.2/build/Release \
 && cd ompl-1.5.2/build/Release \
 && cmake ../.. -DPYTHON_EXEC=/isaac-sim/kit/python/bin/python3 \
 && make update_bindings \
 && make \
 && make install

# Set the PYTHONPATH so that Python has access to Lula (and Geometric Fabrics)
ENV PYTHONPATH=/isaac-sim/exts/omni.isaac.lula/pip_prebundle/
ENV PATH="/isaac-sim/kit/python/bin:${PATH}"

# Install cuda 11.3 (this should be upgraded if Pytorch is also upgraded)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
  && dpkg -i cuda-keyring_1.0-1_all.deb \
  && apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-3=11.3.109-1 \
    cuda-compat-11-3 \
    cuda-libraries-11-3=11.3.1-1 \
    libnpp-11-3=11.3.3.95-1 \
    cuda-nvtx-11-3=11.3.109-1 \
    libcusparse-11-3=11.6.0.109-1 \
    libcublas-11-3=11.5.1.109-1 \
    libnccl2=2.9.9-1+cuda11.3 \
    cuda-cudart-dev-11-3=11.3.109-1 \
    cuda-command-line-tools-11-3=11.3.1-1 \
    cuda-minimal-build-11-3=11.3.1-1 \
    cuda-libraries-dev-11-3=11.3.1-1 \
    cuda-nvml-dev-11-3=11.3.58-1 \
    cuda-nvprof-11-3=11.3.111-1 \
    libnpp-dev-11-3=11.3.3.95-1 \
    libcusparse-dev-11-3=11.6.0.109-1 \
    libcublas-dev-11-3=11.5.1.109-1 \
    libnccl-dev=2.9.9-1+cuda11.3 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold libcublas-11-3 libnccl2 libcublas-dev-11-3 libnccl-dev


# # Alias the system Python to the Kit Python and install dependencies
# For some reason, the ikfast dependency do not get properly installed if
# ikfast and robofin are in the same pip install statement
RUN python3 -m pip install --upgrade pip setuptools \
  && python3 -m pip install \
    torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113 \
    pytorch_lightning \
    h5py \
    GitPython \
    trimesh \
    wandb \
    ipython \
    pybullet \
    pyquaternion \
    geometrout \
    ikfast-pybind  \
    tqdm \
    urchin \
    yourdfpy \
    shapely \
    rtree \
    triangle \
    git+https://github.com/fishbotics/pointnet2_ops.git@main \
  && python3 -m pip install git+https://github.com/fishbotics/robofin.git@v0.0.1 git+https://github.com/fishbotics/atob@v0.0.1

WORKDIR /root
ENTRYPOINT []