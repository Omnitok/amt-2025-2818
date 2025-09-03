FROM  tensorflow/tensorflow:2.10.0-gpu


RUN apt-get update && apt-get -qq install -y \
        wget \
        unzip \
        libtbb2 \
        libtbb-dev \
        git \
        pkg-config \
        libjpeg8-dev \
        libtiff-dev \
        libpng-dev \
        libgtk2.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libatlas-base-dev \
        gfortran \
        libhdf5-dev \
        python3-pip \
        libgl1-mesa-dev \
        libusb-1.0 \
        libglib2.0-0 \
        libgl1-mesa-glx

RUN pip3 install \
        git+https://github.com/tensorflow/examples.git \
        tensorflow_datasets \
        tensorflow_addons \
        ipython \
        matplotlib \
        protobuf==3.20.* \
        scikit-image \
        pyrealsense2 \
        opencv-python==4.4.0.46 \
        tf2onnx \
        onnxruntime-gpu \
        onnx \
        GitPython


COPY update_resnet_init_filters.patch /
RUN git clone https://github.com/qubvel/classification_models.git && \
        cd classification_models && git apply /update_resnet_init_filters.patch && \
        pip3 install .

RUN pip3 install segmentation_models
