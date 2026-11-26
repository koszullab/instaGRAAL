# Polishing

`instagraal.parse_info_frags` reads and writes layouts (`info_frags` files)
and the FASTA files of their scaffolds, and holds the steps of
`instagraal-polish`, which take and return a layout as a dictionary: for each
scaffold, its pieces as `[contig, id, start, end, orientation]`. See
[Polishing the scaffolds](../guides/polishing.md) and
[info_frags files](../formats.md#info_frags-files).

::: instagraal.parse_info_frags
