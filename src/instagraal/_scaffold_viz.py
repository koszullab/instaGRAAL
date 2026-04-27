#!/usr/bin/env python
# coding: utf-8

"""Visualisation helpers for scaffold composition plots.

This module is internal; import from ``instagraal.parse_info_frags`` instead.
"""

import numpy as np
from matplotlib import pyplot as plt

from ._scaffold_io import format_info_frags, parse_info_frags


def plot_info_frags(scaffolds):
    """A crude way to visualize new scaffolds according to their origin on the
    initial scaffolding. Each scaffold spawns a new plot. Orientations are
    represented by different colors.
    """

    scaffolds = format_info_frags(scaffolds)
    for name, scaffold in scaffolds.items():
        plt.figure()
        xs = range(len(scaffold))
        color = []
        names = {}
        ys = []
        for my_bin in scaffold:
            current_color = "r" if my_bin[4] > 0 else "g"
            color += [current_color]
            name = my_bin[0]
            if name in names:
                ys.append(names[name])
            else:
                names[name] = len(names)
                ys.append(names[name])
        plt.scatter(xs, ys, c=color)
    plt.show()


def plot_contig_composition(new_info_frags_path, output_path=None):
    """Plot a stacked barplot of new-contig composition by source contig.

    For each new contig on the X-axis the bar height (Y-axis, bp) is split
    into coloured segments representing the contribution (in bp) of each
    original contig.  New contigs are sorted from longest to shortest.

    Parameters
    ----------
    new_info_frags_path:
        Path to ``new_info_frags.txt`` produced by ``instagraal-polish``.
    output_path:
        Path where the figure is saved.  When ``None`` the figure is shown
        interactively.
    """
    scaffolds = parse_info_frags(str(new_info_frags_path))

    # Compute per-source-contig length contribution for each new contig
    new_names = list(scaffolds.keys())
    all_src: list[str] = []
    contribs: dict[str, dict[str, int]] = {}
    for new_name, frags in scaffolds.items():
        contrib: dict[str, int] = {}
        for frag in frags:
            src = frag[0]
            length = frag[3] - frag[2]  # end - start
            contrib[src] = contrib.get(src, 0) + length
            if src not in all_src:
                all_src.append(src)
        contribs[new_name] = contrib

    all_src = sorted(all_src)

    # Sort new contigs by total length (descending)
    total_len = {n: sum(contribs[n].values()) for n in new_names}
    new_names_sorted = sorted(new_names, key=lambda n: total_len[n], reverse=True)

    # Build per-source height matrix (n_src x n_new_contigs)
    n_new = len(new_names_sorted)
    heights = np.array(
        [[contribs[n].get(src, 0) for n in new_names_sorted] for src in all_src],
        dtype=float,
    )

    # Assign colours
    cmap = plt.cm.get_cmap("tab20", len(all_src))
    colors = [cmap(i) for i in range(len(all_src))]

    fig, ax = plt.subplots(figsize=(max(8, n_new * 0.25), 5))
    x = np.arange(n_new)
    bottoms = np.zeros(n_new)
    for i, src in enumerate(all_src):
        ax.bar(x, heights[i], bottom=bottoms, color=colors[i], label=src, width=0.8)
        bottoms += heights[i]

    # X-axis: strip the common prefix (e.g. "3C-assembly|") for readability
    short_labels = [name.split("|")[-1] if "|" in name else name for name in new_names_sorted]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=90, fontsize=max(4, min(8, 120 // n_new)))
    ax.set_xlabel("New contigs")
    ax.set_ylabel("Contig length (bp)")
    ax.set_title("Source-contig composition of new assembly contigs")
    ax.legend(
        title="Original contigs",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=6,
        ncol=max(1, len(all_src) // 30),
    )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
