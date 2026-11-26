# Citing instaGRAAL

Please cite:

> Lyam Baudry, Nadège Guiglielmoni, Hervé Marie-Nelly, Alexandre Cormier,
> Martial Marbouty, Komlan Avia, Yann Loe Mie, Olivier Godfroy, Lieven
> Sterck, J. Mark Cock, Christophe Zimmer, Susana M. Coelho and Romain
> Koszul. 2020. instaGRAAL: chromosome-level quality scaffolding of genomes
> using a proximity ligation-based scaffolder. *Genome Biology* 21: 148.
> <https://doi.org/10.1186/s13059-020-02041-z>

```bibtex
@article{baudry2020instagraal,
  author  = {Baudry, Lyam and Guiglielmoni, Nad{\`e}ge and Marie-Nelly, Herv{\'e} and
             Cormier, Alexandre and Marbouty, Martial and Avia, Komlan and
             Mie, Yann Loe and Godfroy, Olivier and Sterck, Lieven and Cock, J. Mark and
             Zimmer, Christophe and Coelho, Susana M. and Koszul, Romain},
  title   = {{instaGRAAL}: chromosome-level quality scaffolding of genomes using a
             proximity ligation-based scaffolder},
  journal = {Genome Biology},
  volume  = {21},
  pages   = {148},
  year    = {2020},
  doi     = {10.1186/s13059-020-02041-z}
}
```

## GRAAL

instaGRAAL continues GRAAL, whose model and MCMC it rewrites:

- [High-quality genome (re)assembly using chromosomal contact data](https://doi.org/10.1038/ncomms6695),
  Hervé Marie-Nelly, Martial Marbouty, Axel Cournac, Jean-François Flot,
  Gianni Liti, Dante Poggi Parodi, Sylvie Syan, Nancy Guillén, Antoine
  Margeot, Christophe Zimmer and Romain Koszul. *Nature Communications*, 2014.
- [A probabilistic approach for genome assembly from high-throughput chromosome conformation capture data](https://www.theses.fr/2013PA066714),
  Hervé Marie-Nelly. PhD thesis, Université Pierre et Marie Curie, 2013.
- [The principle of GRAAL](https://github.com/koszullab/GRAAL/blob/master/GRAALprinciple.pdf),
  the details of the model.

## Studies that use it

- [Proximity ligation scaffolding and comparison of two *Trichoderma reesei* strains genomes](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5469131/),
  Etienne Jourdier, Lyam Baudry, Dante Poggi-Parodi, Yoan Vicq, Romain Koszul,
  Antoine Margeot, Martial Marbouty and Frédérique Bidard. *Biotechnology for
  Biofuels*, 2017.
- [Scaffolding bacterial genomes and probing host-virus interactions in gut microbiome by proximity ligation (chromosome capture) assay](https://www.ncbi.nlm.nih.gov/pubmed/28232956),
  Martial Marbouty, Lyam Baudry, Axel Cournac and Romain Koszul.
  *Science Advances*, 2017.

## The tools instaGRAAL runs

Please also cite the tools a run relies on:

- The MCMC on the GPU: [PyCUDA](https://doi.org/10.1016/j.parco.2011.09.001)
  (Klöckner *et al.*, *Parallel Computing* 2012).
- Contact maps: [cooler](https://doi.org/10.1093/bioinformatics/btz540)
  (Abdennur and Mirny, *Bioinformatics* 2020).
- Restriction sites and sequences: [Biopython](https://doi.org/10.1093/bioinformatics/btp163)
  (Cock *et al.*, *Bioinformatics* 2009).
- The pairs, from the reads: [hicstuff](https://github.com/koszullab/hicstuff),
  with [bowtie2](https://github.com/BenLangmead/bowtie2) or
  [bwa](https://github.com/lh3/bwa).

The test data comes from the Hi-C library
[SRR22130071](https://www.ebi.ac.uk/ena/browser/view/SRR22130071) of
*Saccharomyces cerevisiae*, on the S288c reference genome, and is archived on
Zenodo ([10.5281/zenodo.19813387](https://doi.org/10.5281/zenodo.19813387)).
