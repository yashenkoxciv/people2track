FROM nvidia/cuda:11.0-base-ubuntu20.04


# set tzdata
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install face recognition dependencies
RUN apt update -y; apt install -y \
wget \
git \
cmake \
libsm6 \
libxext6 \
libxrender-dev \
python3 \
python3-pip \
libopenblas-dev \
liblapack-dev \
ffmpeg \
libsm6 \
libxext6

RUN pip3 install --upgrade pip
RUN pip3 install scikit-build

# Install compilers
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt update -y; apt install -y gcc-7 g++-7

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 50
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 50

#Install dlib
RUN git clone -b 'v19.21' --single-branch https://github.com/davisking/dlib.git
RUN mkdir -p /dlib/build

RUN cmake -H/dlib -B/dlib/build -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
RUN cmake --build /dlib/build

RUN cd /dlib; python3 /dlib/setup.py install

# Install the face recognition package
RUN pip3 install face_recognition

# Install OpenCV
RUN pip3 install opencv-python
RUN pip3 install ipdb
RUN pip3 install tf-nightly-gpu
RUN pip3 install matplotlib
RUN pip3 install pyyaml 
RUN apt-get -y install python3-tk
RUN apt-get -y install cuda-cudart-11-1 libcudnn8

# Set WORKDIR and others
WORKDIR "/app"
#
