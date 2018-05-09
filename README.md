# instaGRAAL
Large genome reassembly based on Hi-C data (continuation and partial rewrite of [GRAAL](https://github.com/koszullab/GRAAL))

This work is under continuous development/improvement - see [GRAAL](https://github.com/koszullab/GRAAL) for information about the basic principles of GRAAL and on how to deploy it. 

## How to use
Until more detailed documentation is added, run the following to see options:

    python main_single_proc.py -h 

Unlike GRAAL, this is meant to be run from the command line.

## Polishing

Lingering artifacts found in output genomes can be corrected by editing the info_frags.txt file, either by hand or with a script. Look at options by running the following:

    python parse_info_frags.py -h 
