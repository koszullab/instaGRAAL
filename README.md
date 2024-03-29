# instaGRAAL

![ ](example/example.gif "instaGRAAL demo")

[![PyPI version](https://badge.fury.io/py/instagraal.svg)](https://badge.fury.io/py/instagraal)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/serpentine.svg)
[![Docker Cloud Automated build](https://img.shields.io/docker/cloud/automated/koszullab/instagraal)](https://hub.docker.com/r/koszullab/instagraal/)
[![Read the docs](https://readthedocs.org/projects/instagraal/badge)](https://instagraal.readthedocs.io)
[![DOI](https://zenodo.org/badge/131966807.svg)](https://zenodo.org/badge/latestdoi/131966807)
[![License: GPLv3](https://img.shields.io/badge/License-GPL%203-0298c3.svg)](https://opensource.org/licenses/GPL-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Large genome reassembly based on Hi-C data (continuation and partial rewrite of [GRAAL](https://github.com/koszullab/GRAAL); Marie-Nelly et al., [2013](http://www.theses.fr/2013PA066714); [2014](https://www.nature.com/articles/ncomms6695)) and post-scaffolding polishing libraries.

This work is under continuous development/improvement - see [GRAAL](https://github.com/koszullab/GRAAL) for information about the basic principles.

You can now easily install instaGRAAL using a docker container available below or you can try it on [Galaxy Europe](https://usegalaxy.eu/).

## Table of contents
* [Installation](#Installation)
* [Usage](#Usage)
* [Output](#Output)
* [Curation](#Curation)
* [Troubleshooting](#Troubleshooting)
* [Documentation](#Documentation)
* [References](#References)

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

The scaffolder and polishing libraries are written in Python 3 and CUDA. As such, an NVIDIA graphics card is required for the scaffolder to run. The Python 2 version is available at the ```python2``` branch of this repository, but be aware that development will mainly focus on the Python 3 version. The software has been tested for Ubuntu 17.04 and later, and most dependencies can be downloaded with its package manager (or Python's ```pip```).

#### External libraries

You will need to download and install the [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-downloads). Manual installation is recommended - installing ```nvidia-cuda-toolkit``` from Ubuntu's package manager has been known to cause glitches. It is fairly straightforward on OS X thanks to the installation wizard. Here is how to quickly do it on Ubuntu 18.04:

```sh
    wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
    chmod +x cuda_10.0.130_410.48_linux
    sudo ./cuda_10.0.130_410.48_linux
```

**Note to Ubuntu users**: Be aware that the installation script will fail if it isn't run as root, or if a graphical instance (e.g. X) is running as well. You may need to temporarily shut it down, for instance by switching to tty1 and running the following (prior to the installation script):

```sh
    sudo service lightdm stop
```

(Replace ```lightdm``` with ```mdm```, ```gdm``` or whichever login manager is present on your machine if that fails; if all else fails as well, you may have to run something like ```sudo pkill Xorg``` instead.)

**Note to OS X users**: There is currently [no CUDA support](https://devtalk.nvidia.com/default/topic/1042279/cuda-10-and-macos-10-14/) on Mojave (10.14) and it is unclear when it is going to be added, if it is to be added at all. This means instaGRAAL (or indeed any CUDA-based application) will *not* work on Mojave. If you wish to run it on OS X, the only solution for now is to downgrade to High Sierra (10.13).

#### Recommended libraries

Because some Python dependencies (such as ```pyopengl``` or ```h5py```) require to be built against specific files, it is recommended that you install the following packages if you encounter errors.

##### OpenGL libraries

* ```libglu1-mesa```
* ```libxi-dev```
* ```libxmu-dev```
* ```libglu1-mesa-dev```
* ```freeglut3-dev```

##### HDF5 serialization library

* ```hdf5-tools``` (```hdf5``` for OS X in brew)

##### Boost libraries

* ```libboost-all-dev``` (```boost``` and ```boost-python``` for OS X in brew)

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

You will also need to build  ```pycuda``` with OpenGL support and **disable** its use of custom Boost libraries. Installing it directly from PyPI will cause errors at runtime. Here is how to do it manually with Git on Ubuntu or OS X:

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
And run it with
```sh
docker run --gpus all koszullab/instagraal
```

> Note: Running the container requires the dependency nvidia-docker2 \[[installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)\]

## Usage

Unlike GRAAL, this is meant to be run from the command line.

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

If you want to generate instaGRAAL-compatible matrices from scratch (i.e. from reads and a reference genome, as opposed to existing Hi-C data in one of the numerous existing formats), you may do so with [hicstuff](https://github.com/koszullab/hicstuff), which acts as both a Python library and a pipeline. Instructions, parameters and optional arguments are detailed in the repo's readme. We strongly recommend using hicstuff with the parameter -m iterative or -m cutsite to improve mapping.

## Output

After the scaffolder is done running, whatever path you specified as output will contain a ```test_mcmc_X``` directory, where X is the level (resolution) at which scaffolding was performed. This directory, in turn, will contain the following:

* ```genome.fasta```: the scaffolded genome. Scaffolds will be ordered by increasing size *in fragments*, which roughly (but not always) translates into increasing size in bp.
* ```info_frags.txt```: a file that contains, for each newly formed scaffold, the original coordinates of every single bin in that scaffold, in the format ```chromosome, id, orientation, start, end```. Each bin has a unique ID that provides a convenient way of tracking consecutive stretches. Orientations are relative to one another, and when "-1" is supplied, it is understood that the reverse complement should be taken.

Other files are mostly for developmental purposes and keep track of the evolution of various metrics and model parameters.

## Curation

__This step is strongly recommended to improve the quality of your scaffolds__, unless your input contigs have many misassemblies. Lingering artifacts found in output genomes can be corrected by editing the ```info_frags.txt``` file, either by hand or with a script. Look at options by running the following:

    instagraal-polish -h

The most common use case is to run all curation procedures at once:

    instagraal-polish -m polishing -i info_frags.txt -f contigs.fasta -o curated_assembly.fa

You can add gaps with the parameter -j (necessary for subsequent gap filling), for instance gaps with 10 Ns in this example:

    instagraal-polish -m polishing -i info_frags.txt -f contigs.fasta -o curated_assembly.fa -j NNNNNNNNNN

## Troubleshooting

### "I am not happy with the scaffolds"

If the output is not as you would expect:
* check the Hi-C mapping rate; if the mapping rate is low, this may be due to:
    * a poor contig completeness (check BUSCO and _k_-mer completeness)  
    * not using the parameter iterative or cutsite when running hicstuff
* make sure that there are few artefactual duplications
* if the assembly was obtained with low-accuracy Nanopore reads, polish the assembly with highly accurate reads prior to scaffolding
* check that there are _trans_ contacts between contigs in the contact map prior to scaffolding, as insufficient contacts will lead to poor Hi-C scaffolding
* try switching aligner from bwa to bowtie2 or vice versa, as we have noticed sometimes different scaffolding outputs depending on the aligner used by hicstuff

### Scaffolding is too slow

By default, the parameter --level is set to 4. For genomes larger than 500 Mb, increasing it to 5 is often more adapted to improve runtime, and 6 for genomes larger than 3 Gb.

### KeyError on contig names

This is due to spaces and special characters in contig names. Check that the contig names match the ones in the outputs from hicstuff, and rename your contigs if necessary.

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

However, instaGRAAL is based on OpenGL, which means there has to be an X server of some kind running on your target machine no matter what. While this allows for pretty movies and visualizations, it may prove problematic on an environment you don't have total control over, *e.g.* a server cluster. Currently, your best bet is asking the system administrator of the target machine to set up an X instance (possibly virtual, such as Xvfb or ```xserver-xorg-video-dummy```) if they haven't already.

### PyOpenGL/GLUT error

If you encounter the following:

    NullFunctionError: Attempt to call an undefined function glutInit, check for bool(glutInit) before calling

check whether you have installed ```freeglut3-dev```. It seems that the ```pyopengl``` library [does not include a GLUT implementation](https://stackoverflow.com/questions/26700719/pyopengl-glutinit-nullfunctionerror) when installed from PyPI. Alternatively, just installing ```pyopengl``` with your package manager (*e.g.* ```python3-pyopengl``` on Ubuntu) seems to work as well.

### Codepy toolchain

If you encounter an error like the following :

      File "/usr/local/lib/python3.6/dist-packages/codepy/toolchain.py", line 382, in _guess_toolchain_kwargs_from_python_config
    object_suffix = '.' + make_vars['MODOBJS'].split()[0].split('.')[1]
    IndexError: list index out of range

You may need to upgrade to a more recent version of ```codepy```.

```sh
    sudo pip3 install --upgrade --no-cache-dir -e git+https://github.com/inducer/codepy.git@master#egg=codepy
```

No such error has been found as of commit [10a014f](https://github.com/inducer/codepy/tree/10a014f), so if you encounter regressions after this, you should stick to that version.

Depending on your system, you may also need to upgrade to gcc/g++ 8:

```sh
    sudo apt install gcc-8 g++-8
```

If for some reason your system does not automatically switch to gcc/g++-8, you should manually configure your system to do so, *e.g.* on Ubuntu:

```sh
    sudo update-alternatives --remove-all gcc 
    sudo update-alternatives --remove-all g++

    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 10
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 10
```

### General tips

* instaGRAAL will attempt to detect already-built pyramids in hdf5 format, but if building was interrupted for some reason, and you re-run the pyramid building step, the hdf5 files will still be there, but corrupted. You will need to manually delete the ```pyramids``` folder and try again.

* If there is a mismatch between the version of CUDA you installed and the one instaGRAAL seems to rely on (*e.g.* you installed CUDA 10 but instaGRAAL complains that it can't find ```libcurand.so.9.1```), try reinstalling ```pycuda``` and ```instagraal``` with the ```--no-cache-dir``` option.

* You may experience issues if you handle dependencies with conda, such as ```pycuda``` failing to build because some header files that would be present when you installed ```libboost-all-dev``` aren't automatically recognized. If you don't want to manually mess with your ```$PATH```, it's probably best to just deactivate conda altogether and install everything with your OS's normal package manger (and ```pip```).

## Documentation

As a Python package, instaGRAAL provides both a scaffolding and polishing library, as well as a convenient Hi-C matrix handling framework, and we've tried to expose much of the API behind these on [readthedocs](https://instagraal.readthedocs.io). If you wish to know more about how the scaffolder works, see the [references](#References), especially the [supplementary method](https://github.com/koszullab/GRAAL/blob/master/GRAALprinciple.pdf) delving deeper into the details of the model.

## References

### Principle
* [instaGRAAL: chromosome-level quality scaffolding of genomes using a proximity ligation-based scaffolder](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02041-z) Lyam Baudry, Nadège Guiglielmoni, Hervé Marie-Nelly, Alexandre Cormier, Martial Marbouty, Komlan Avia, Yann Loe Mie, Olivier Godfroy, Lieven Sterck, J. Mark Cock, Christophe Zimmer, Susana M. Coelho & Romain Koszul 
* [High-quality genome assembly using chromosomal contact data](https://www.ncbi.nlm.nih.gov/pubmed/25517223), Hervé Marie-Nelly, Martial Marbouty, Axel Cournac, Jean-François Flot, Gianni Liti, Dante Poggi Parodi, Sylvie Syan, Nancy Guillén, Antoine Margeot, Christophe Zimmer and Romain Koszul, Nature Communications, 2014
* [A probabilistic approach for genome assembly from high-throughput chromosome conformation capture data](https://www.theses.fr/2013PA066714), Hervé Marie-Nelly, 2013, PhD thesis

### Use cases

* [Proximity ligation scaffolding and comparison of two Trichoderma reesei strains genomes](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5469131/), Etienne Jourdier, Lyam Baudry, Dante Poggi-Parodi, Yoan Vicq, Romain Koszul, Antoine Margeot, Martial Marbouty, and Frédérique Bidard, Biotechnology for Biofuels, 2017
* [Scaffolding bacterial genomes and probing host-virus interactions in gut microbiome by proximity ligation (chromosome capture) assay](https://www.ncbi.nlm.nih.gov/pubmed/28232956), Martial Marbouty, Lyam Baudry, Axel Cournac, and Romain Koszul, Science Advances, 2017

## Contact

* romain.koszul@pasteur.fr
