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

### Input datasets

The above ```<hic_folder>``` passed as an argument to instaGRAAL needs three files:

* A file named ```abs_fragments_contacts_weighted.txt```, containing the (sparse) Hi-C map itself. The first line must be ```id_frag_a    id_frag_b    n_contact```. All subsequent lines must represent the map's contacts in coordinate format (```id_frag_a``` being the row indices, ```id_frag_b``` being the column indices, ```n_contact``` being the number of contacts between each locus or index pair, _e.g._ if 5 contacts are found between fragments #2 and #3, there should be a line reading ```2 3 5``` in the file). ```n_contact``` must be an integer. The list should be sorted according to ```id_frag_a``` first, then ```id_frag_b```. Fragment ids start at 0.
* A file named ```fragments_list.txt``` containing information related to each fragment of the genome. The first line must be ```id    chrom    start_pos    end_pos    size    gc_content```, and subsequent lines (representing the fragments themselves) should follow that template. The fields should be self-explanatory; notably, ```chrom``` can be any string representing the chromosome's name to which the fragment at a given line belongs, and fragment ids should start over at 1 when the chromosome name changes. Aside from the ```chrom``` field and the ```gc``` field which is currently unused in this version and can be filled with any value, all fields should be integers. Note that ```start_pos``` starts at 0.
* A file named ```info_contigs.txt``` containing information related to each contig/scaffold/chromosome in the genome. The first line must be ```contig    length_kb    n_frags    cumul_length```. Field names should be again self-explanatory; naturally the contig field must contain names that are consistent with those found in ```fragments_list.txt```. Also ```length_kb``` should be an integer (rounded up or down if need be), and ```n_frags``` and ```cumul_length``` are supposed to be consistent with each other in that the cumulated length (in fragments) of contig N should be equal to the sum of the fields found in ```n_frags``` for the N-1 preceding lines. Note that ```cumul_length``` starts at 0.

All fields (including those in the files' headers) must be separated by tabs.

Minimal working templates are provided in the ```example``` folder.

## Polishing

Lingering artifacts found in output genomes can be corrected by editing the ```info_frags.txt``` file, either by hand or with a script. Look at options by running the following:

    python parse_info_frags.py -h

The most common use case is to run all polishing procedures at once:

    python parse_info_frags.py -m polishing -i info_frags.txt -f reference.fasta -o polished_assembly.fa

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