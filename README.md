# instaGRAAL
Large genome reassembly based on Hi-C data (continuation and partial rewrite of [GRAAL](https://github.com/koszullab/GRAAL)) and post-scaffolding polishing libraries.

This work is under continuous development/improvement - see [GRAAL](https://github.com/koszullab/GRAAL) for information about the basic principles of GRAAL and on how to deploy it. 

## Requirements

The scaffolder and polishing libraries are written in Python 3.6 and CUDA. The Python 2 version is available at the ```python2``` branch of this repository, but be aware that development will mainly focus on the Python 3 version. The software has been tested for Ubuntu 17.04 and most dependencies can be downloaded with its package manager (or Python's ```pip```).

### External libraries

You will need to download and install the [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux). Manual installation is recommended - installing ```nvidia-cuda-toolkit``` from Ubuntu's package manager has been known to cause glitches.

OpenGL libraries:
* ```libglu1-mesa```
* ```libxi-dev```
* ```libxmu-dev```
* ```libglu1-mesa-dev```

HDF5 serialization library:
* ```libhdf5-100```

Boost libraries:
* ```libboost-all-dev```

### Python libraries

* ```numpy```
* ```scipy```
* ```matplotlib```
* ```codepy```
* ```h5py```
* ```pyopengl```
* ```docopt```
* ```biopython```

You will also need to build  ```pycuda``` with OpenGL support and **disable** its use of custom Boost libraries.

You may run (as root)  ```deploy.sh```, an all-in-one script to handle all the above dependencies on Ubuntu 17+.

## How to use
Until more detailed documentation is added, run the following to see options:

    python main_single_proc.py -h 

Unlike GRAAL, this is meant to be run from the command line.

## Polishing

Lingering artifacts found in output genomes can be corrected by editing the info_frags.txt file, either by hand or with a script. Look at options by running the following:

    python parse_info_frags.py -h 
