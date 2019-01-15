# instaGRAAL

![ ](example/example.gif "instaGRAAL demo")

[![PyPI version](https://badge.fury.io/py/instagraal.svg)](https://badge.fury.io/py/instagraal)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/serpentine.svg)
[![Docker Automated build](https://img.shields.io/docker/build/koszullab/instagraal.svg)](https://hub.docker.com/r/koszullab/instagraal/)
[![Read the docs](https://readthedocs.org/projects/instagraal/badge)](https://instagraal.readthedocs.io)
[![License: GPLv3](https://img.shields.io/badge/License-GPL%203-0298c3.svg)](https://opensource.org/licenses/GPL-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Large genome reassembly based on Hi-C data (continuation and partial rewrite of [GRAAL](https://github.com/koszullab/GRAAL)) and post-scaffolding polishing libraries.

This work is under continuous development/improvement - see [GRAAL](https://github.com/koszullab/GRAAL) for information about the basic principles.

## Installation

Install from PyPI:

```sh
    sudo pip3 install -U instagraal
```

or, if you want to get the very latest version:

```sh
   sudo pip3 install -e git+https://github.com/koszullab/instagraal.git@master#egg=instagraal
```

This should automatically handle most dependencies.

### Requirements

The scaffolder and polishing libraries are written in Python 3 and CUDA. The Python 2 version is available at the ```python2``` branch of this repository, but be aware that development will mainly focus on the Python 3 version. The software has been tested for Ubuntu 17.04 and most dependencies can be downloaded with its package manager (or Python's ```pip```).

#### External libraries

You will need to download and install the [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux). Manual installation is recommended - installing ```nvidia-cuda-toolkit``` from Ubuntu's package manager has been known to cause glitches.

Because some Python dependencies (such as ```pyopengl``` or ```h5py```) require to be built against specific header files, it is recommended that you install the following packages if you encounter errors.

##### OpenGL libraries

* ```libglu1-mesa```
* ```libxi-dev```
* ```libxmu-dev```
* ```libglu1-mesa-dev```

##### HDF5 serialization library

* ```hdf5-tools```

##### Boost libraries

* ```libboost-all-dev```

#### Python dependencies

Python package requirements should be handled automatically by ```pip```, but should you
wish to install them manually, these are:

* ```numpy```
* ```scipy```
* ```matplotlib```
* ```codepy```
* ```h5py```
* ```pyopengl```
* ```docopt```
* ```biopython```

They can also be handily installed using the supplied requirements file in the repo:

    pip3 install -Ur requirements.txt

You will also need to build  ```pycuda``` with OpenGL support and **disable** its use of custom Boost libraries. Installing it directly from PyPI will cause errors at runtime. Here is how to do it manually with Git on Ubuntu:

```sh
    git clone --recurse-submodules https://github.com/inducer/pycuda.git
    cd pycuda
    python3 configure.py --cuda-enable-gl --no-use-shipped-boost
    sudo python3 setup.py install
```

You may run (as root)  ```instagraal-setup```, an all-in-one script to handle all the above dependencies on Ubuntu 17+.

### Container

There is experimental Docker support for instaGRAAL. You may fetch the corresponding image by running the following:

```sh
    docker pull koszullab/instagraal
```

## How to use

Unlike GRAAL, this is meant to be run from the command line.

### Usage

    instagraal <hic_folder> <reference.fa> [<output_folder>]
               [--level=4] [--cycles=100] [--coverage-std=1]
               [--neighborhood=5] [--device=0] [--circular] [--bomb]
               [--save-matrix] [--pyramid-only] [--save-pickle] [--simple]
               [--quiet] [--debug]

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
    --pyramid-only          Only build multi-resolution contact maps (pyramids)
                            and don't do any scaffolding. [default: False]
    --save-pickle           Dump all info from the instaGRAAL run into a
                            pickle. Primarily for development purposes, but
                            also for advanced post hoc introspection.
                            [default: False]
    --save-matrix           Saves a preview of the contact map after each
                            cycle, in csv format. [default: False]
    --simple                Only perform operations at the edge of the contigs.
                            [default: False]
    --quiet                 Only display warnings and errors as outputs.
                            [default: False]
    --debug                 Display debug information. For development purposes
                            only. Mutually exclusive with --quiet, and will
                            override it. [default: False]

### Input datasets

#### Format specification

The above ```<hic_folder>``` passed as an argument to instaGRAAL needs three files:

* A file named ```abs_fragments_contacts_weighted.txt```, containing the (sparse) Hi-C map itself. The first line must be ```id_frag_a    id_frag_b    n_contact```. All subsequent lines must represent the map's contacts in coordinate format (```id_frag_a``` being the row indices, ```id_frag_b``` being the column indices, ```n_contact``` being the number of contacts between each locus or index pair, _e.g._ if 5 contacts are found between fragments #2 and #3, there should be a line reading ```2 3 5``` in the file). ```n_contact``` must be an integer. The list should be sorted according to ```id_frag_a``` first, then ```id_frag_b```. Fragment ids start at 0.
* A file named ```fragments_list.txt``` containing information related to each fragment of the genome. The first line must be ```id    chrom    start_pos    end_pos    size    gc_content```, and subsequent lines (representing the fragments themselves) should follow that template. The fields should be self-explanatory; notably, ```chrom``` can be any string representing the chromosome's name to which the fragment at a given line belongs, and fragment ids should start over at 1 when the chromosome name changes. Aside from the ```chrom``` field and the ```gc``` field which is currently unused in this version and can be filled with any value, all fields should be integers. Note that ```start_pos``` starts at 0.
* A file named ```info_contigs.txt``` containing information related to each contig/scaffold/chromosome in the genome. The first line must be ```contig    length_kb    n_frags    cumul_length```. Field names should be again self-explanatory; naturally the contig field must contain names that are consistent with those found in ```fragments_list.txt```. Also ```length_kb``` should be an integer (rounded up or down if need be), and ```n_frags``` and ```cumul_length``` are supposed to be consistent with each other in that the cumulated length (in fragments) of contig N should be equal to the sum of the fields found in ```n_frags``` for the N-1 preceding lines. Note that ```cumul_length``` starts at 0.

All fields (including those in the files' headers) must be separated by tabs.

Minimal working templates are provided in the ```example``` folder.

#### Matrix generation

If you want to generate instaGRAAL-compatible matrices from scratch (i.e. from reads and a reference genome, as opposed to existing Hi-C data in one of the numerous existing formats), you may do so with [hicstuff](https://github.com/koszullab/hicstuff), which acts as both a Python library and a pipeline. A [graphical interface](https://github.com/koszullab/HiC-Box) is also available. Instructions, parameters and optional arguments are detailed in the repo's readme.

## Output

After the scaffolder is done running, whatever path you specified as output will contain a ```test_mcmc_X``` directory, where X is the level (resolution) at which scaffolding was performed. This directory, in turn, will contain the following:

* ```genome.fasta```: the scaffolded genome. Scaffolds will be ordered by increasing size *in fragments*, which roughly (but not always) translates into increasing size in bp.
* ```info_frags.txt```: a file that contains, for each newly formed scaffold, the original coordinates of every single bin in that scaffold, in the format ```chromosome, id, orientation, start, end```. Each bin has a unique ID that provides a convenient way of tracking consecutive stretches. Orientations are relative to one another, and when "-1" is supplied, it is understood that the reverse complement should be taken.

Other files are mostly for developmental purposes and keep track of the evolution of various metrics and model parameters.

## Polishing

Lingering artifacts found in output genomes can be corrected by editing the ```info_frags.txt``` file, either by hand or with a script. Look at options by running the following:

    instagraal-polish -h

The most common use case is to run all polishing procedures at once:

    instagraal-polish -m polishing -i info_frags.txt -f reference.fasta -o polished_assembly.fa

## Troubleshooting

### Loading CUDA libraries

If you encounter the following error, *despite* having installed the NVIDIA CUDA Toolkit:

    ImportError: libcurand.so.9.2: cannot open shared object file: No such file or directory

it probably means the CUDA-related libraries haven't been properly added to your ```$PATH``` for some reason. A quick solution is to simply add this at the end of your ```.bashrc``` or ```.bash_profile``` (replace the paths with wherever you installed the toolkit and change the version number accordingly):

    export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

### Remote running

If you encounter the following error:

    freeglut (instagraal.py): failed to open display ''

it most likely means you attempted to run an instaGRAAL instance remotely (e.g. over ssh) but didn't configure a proper ```$DISPLAY``` variable. In order to avoid this, simply run the following beforehand:

    export DISPLAY=:0

Note that this will disable the movie (it will play on the remote machine instead).

However, instaGRAAL is based on OpenGL, which means there has to be an X server of some kind running on your target machine no matter what. While this allows for pretty movies and visualizations, it may prove problematic on an environment you don't have total control over, *e.g.* a server cluster. Currently, your best bet is asking the system administrator of the target machine to set up an X instance if they haven't already.

## Documentation

As a Python package, instaGRAAL provides both a scaffolding and polishing library, as well as a convenient Hi-C matrix handling framework, and we've tried to expose much of the API behind these on [readthedocs](https://instagraal.readthedocs.io). If you wish to know more about how the scaffolder works, see the [references](#References), especially the [supplementary method](https://github.com/koszullab/GRAAL/blob/master/GRAALprinciple.pdf) delving deeper into the details of the model.

## References

### Principle

* [High-quality genome assembly using chromosomal contact data](https://www.ncbi.nlm.nih.gov/pubmed/25517223), Hervé Marie-Nelly, Martial Marbouty, Axel Cournac, Jean-François Flot, Gianni Liti, Dante Poggi Parodi, Sylvie Syan, Nancy Guillén, Antoine Margeot, Christophe Zimmer and Romain Koszul, Nature Communications, 2014
* [A probabilistic approach for genome assembly from high-throughput chromosome conformation capture data](https://www.theses.fr/2013PA066714), Hervé Marie-Nelly, 2013, PhD thesis

### Use cases

* [Proximity ligation scaffolding and comparison of two Trichoderma reesei strains genomes](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5469131/), Etienne Jourdier, Lyam Baudry, Dante Poggi-Parodi, Yoan Vicq, Romain Koszul, Antoine Margeot, Martial Marbouty, and Frédérique Bidard, Biotechnology for Biofuels, 2017
* [Scaffolding bacterial genomes and probing host-virus interactions in gut microbiome by proximity ligation (chromosome capture) assay](https://www.ncbi.nlm.nih.gov/pubmed/28232956), Martial Marbouty, Lyam Baudry, Axel Cournac, and Romain Koszul, Science Advances, 2017

## Contact

* lyam.baudry@pasteur.fr
* romain.koszul@pasteur.fr