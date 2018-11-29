FROM nvidia/cudagl:10.0-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# Install basic packages:
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git \        
    ssh \        
    rsync \        
    python \       
    csh \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install X libraries and X authentication
RUN apt-get update && apt-get install -y --no-install-recommends \        
    libx11-6 \
    libxi6 \
    xauth && \
    rm -rf /var/lib/apt/lists/*

# Install additional x libraries and VMD dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \        
    x11proto-gl-dev \        
    libdrm-dev \        
    libxdamage-dev \        
    libx11-xcb-dev \        
    libxcb-glx0-dev \       
    libxcb-dri2-0-dev \ 
    libxinerama-dev && \
    rm -rf /var/lib/apt/lists/* 

# Add all neccessary driver capabilities
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video,compat32,display

RUN apt-get -y update && apt-get install -y python3-dev python3-pip \
    libboost-all-dev build-essential python-dev libboost-python-dev \
    libboost-thread-dev hdf5-tools git python3-tk freeglut3-dev mesa-utils \
    binutils libfreetype6-dev libpng-dev pkg-config \
    libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev && \
    rm -rf /var/lib/apt/lists/* 

COPY instagraal /app/instagraal
COPY scripts /app/scripts
COPY example /apps/example

COPY *.* /app/

WORKDIR /app

RUN pip3 install -Ur requirements.txt

RUN git clone git://github.com/inducer/pycuda /root/pycuda && \
    cd /root/pycuda && git submodule update --init --recursive && \
    python3 configure.py --cuda-enable-gl --no-use-shipped-boost && \
    pip3 install . && \
    rm /root/pycuda* -rf

RUN cd /app & \
    pip3 install .

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH

# ENV PATH /usr/local/cuda/lib64/stubs/:$PATH
# ENV PATH ${LIBRARY_PATH}:${PATH}

#RUN ldconfig



ENTRYPOINT ["instagraal"]