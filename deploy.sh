#!/bin/bash

set -o pipefail
set -e

if [ "$EUID" -ne 0 ]; then
  echo "Please run this script as root."
  exit 1
fi

command -v nvcc -V >/dev/null 2>&1 || {
  echo "Warning: nvcc not found. Please make sure you have installed the NVIDIA CUDA toolkit."
  echo "It is available at https://developer.nvidia.com/cuda-downloads"
  exit 1
}

# Install required libraries:
#   * OpenGL dependencies
#   * HDF5
#   * Boost libraries for Python
sudo apt-get install libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev libhdf5-100 libboost-python1.65

# Install required Python libraries other than pycuda
sudo pip3 install --upgrade -r requirements.txt

# Grab latest pycuda to compile it with specific flags
git clone git://github.com/inducer/pycuda
cd pycuda || {
  echo "Couldn't access pycuda directory."
  exit 1
}
git submodule update --init --recursive

# Enable OpenGL support and disable custom boost linking because it is known
# to cause bugs
python3 configure.py --cuda-enable-gl --no-use-shipped-boost
sudo python3 setup.py install

echo "You're good to go!"
