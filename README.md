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

These should be handily installed using the supplied requirements file:

    pip install -r requirements.txt

You will also need to build  ```pycuda``` with OpenGL support and **disable** its use of custom Boost libraries.

You may run (as root)  ```deploy.sh```, an all-in-one script to handle all the above dependencies on Ubuntu 17+.

## How to use

Unlike GRAAL, this is meant to be run from the command line.

### Usage

    ./main_single_proc.py <hic_folder> <reference.fa> [<output_folder>]
                        [--level=4] [--cycles=100] [--coverage-std=1]
                        [--neighborhood=5] [--device=0] [--circular]
                        [--bomb] [--quiet] [--debug]

### Options

    -h, --help              Display this help message.
    --version               Display the program's current version.
    -l 4, --level 4         Level (resolution) of the contact map.
                            Increasing level by one means a threefold smaller
                            resolution but also a threefold faster computation
                            time. [default: 4]
    -n 100, --cycles 100    Number of iterations to perform for each bin.
                            (row/column of the contact map). A high number of
                            cycles has diminishing returns but there is a
                            necessary minimum for assembly convergence.
                            [default: 100]
    -c 1, --coverage-std 1  Number of standard deviations below the mean.
                            coverage, below which fragments should be filtered
                            out prior to binning. [default: 1]
    -N 5, --neighborhood 5  Number of neighbors to sample for potential
                            mutations for each bin. [default: 5]
    --device 0              If multiple graphic cards are available, select
                            a specific device (numbered from 0). [default: 0]
    -C, --circular          Indicates genome is circular. [default: False]
    -b, --bomb              Explode the genome prior to scaffolding.
                            [default: False]
    --quiet                 Only display warnings and errors as outputs.
                            [default: False]
    --debug                 Display debug information. For development purposes
                            only. Mutually exclusive with --quiet, and will
                            override it. [default: False]

## Polishing

Lingering artifacts found in output genomes can be corrected by editing the info_frags.txt file, either by hand or with a script. Look at options by running the following:

    python parse_info_frags.py -h

## Troubleshooting

### Loading CUDA libraries

If you encounter the following error, *despite* having installed the NVIDIA CUDA Toolkit:

    ImportError: libcurand.so.9.2: cannot open shared object file: No such file or directory

it probably means the CUDA-related libraries haven't been properly added to your ```$PATH``` for some reason. A quick solution is to simply add this at the end of your ```.bashrc``` or ```.bash_profile``` (replace the paths with wherever you installed the toolkit and change the version number accordingly):

    export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

### Remote running

If you encounter the following error:

    freeglut (main_single_proc.py): failed to open display ''

it most likely means you attempted to run an instaGRAAL instance remotely (e.g. over ssh) but didn't configure a proper ```$DISPLAY``` variable. In order to avoid this, simply run the following beforehand:

    export DISPLAY=:0

Note that this will disable the movie (it will play on the remote machine instead).

However, instaGRAAL is based on OpenGL, which means there has to be an X server of some kind running on your target machine no matter what. While this allows for pretty movies and visualizations, it may prove problematic on an environment you don't have total control over, *e.g.* a server cluster. Currently, your best bet is asking the system administrator of the target machine to set up an X instance if they haven't already.