---
title: instaGRAAL
hide:
  - navigation
  - toc
  - footer
---

<div class="cx-landing" markdown>

<h1 class="cx-masthead"><img src="assets/mark.svg" alt="" width="64" height="64"><span>instaGRAAL</span></h1>

<p class="cx-tagline">Chromosome-level scaffolding from Hi-C contacts.</p>

[instaGRAAL](https://github.com/koszullab/instaGRAAL) orders and orients the
contigs of a genome assembly into chromosomes from the contacts of a Hi-C
library. Loci of one chromosome touch each other far more often than they
touch other chromosomes, and the closer they lie along it, the more often:
instaGRAAL finds the layout of the contigs that best explains those contacts,
with an MCMC on a GPU, re-joins whole contigs where the evidence is clear, and
writes the contact maps of the new assembly.
{ .cx-lede }

<div class="cx-install" markdown>

```bash
pip install instagraal
```

</div>

<div class="cx-actions" markdown>

[Why instaGRAAL?](why.md){ .cx-button }
[Quickstart :lucide-arrow-right:](quickstart.md){ .cx-button .cx-button--filled }

</div>

<div class="cx-badges" markdown>

[![PyPI version](https://img.shields.io/pypi/v/instagraal.svg)](https://pypi.org/project/instagraal/)
[![Python versions](https://img.shields.io/pypi/pyversions/instagraal.svg)](https://pypi.org/project/instagraal/)
[![Tests](https://github.com/koszullab/instaGRAAL/actions/workflows/test-cpu.yml/badge.svg)](https://github.com/koszullab/instaGRAAL/actions/workflows/test-cpu.yml)
[![DOI](https://img.shields.io/badge/DOI-10.1186%2Fs13059--020--02041--z-blue)](https://doi.org/10.1186/s13059-020-02041-z)

</div>

<p class="cx-kicker">Start here</p>

<div class="cx-tiles" markdown>

[<span class="cx-glyph">:lucide-rocket:</span> <strong>Quickstart</strong> <span class="cx-tile-text">From contigs and Hi-C pairs to chromosomes, on the test data, in a minute, without a GPU.</span>](quickstart.md){ .cx-tile }
[<span class="cx-glyph">:lucide-lightbulb:</span> <strong>Concepts</strong> <span class="cx-tile-text">The contact model, the MCMC, levels, refine and polish.</span>](concepts.md){ .cx-tile }
[<span class="cx-glyph">:lucide-square-terminal:</span> <strong>CLI reference</strong> <span class="cx-tile-text">Every command and option.</span>](cli.md){ .cx-tile }
[<span class="cx-glyph">:lucide-braces:</span> <strong>Python API</strong> <span class="cx-tile-text">The functions behind the commands.</span>](api/pre.md){ .cx-tile }

</div>

</div>
