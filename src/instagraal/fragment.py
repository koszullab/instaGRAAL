#!/usr/bin/env python3


class basic_fragment:
    id_init: int
    init_contig: str
    init_name: str
    start_pos: int
    end_pos: int
    length_kb: float
    gc_content: float
    np_id_abs: int
    curr_id: int
    curr_name: str
    pos_kb: float
    contig_id: int
    orientation: str
    init_frag_start: int
    init_frag_end: int
    sub_frag_start: int
    sub_frag_end: int
    super_index: int
    n_accu_frags: int

    def __init__(self) -> None:
        "standard init fragment"

    @classmethod
    def initiate(
        cls,
        np_id_abs: int,
        id_init: int,
        init_contig: str,
        curr_id: int,
        start_pos: int,
        end_pos: int,
        length_kb: float,
        gc_content: float,
        init_frag_start: int,
        init_frag_end: int,
        sub_frag_start: int,
        sub_frag_end: int,
        super_index: int,
        id_contig: int,
        n_accu_frags: int,
    ) -> "basic_fragment":
        obj = cls()
        obj.id_init = id_init
        obj.init_contig = init_contig
        obj.init_name = str(id_init) + "-" + init_contig
        obj.start_pos = start_pos
        obj.end_pos = end_pos
        obj.length_kb = length_kb
        obj.gc_content = gc_content
        obj.np_id_abs = np_id_abs
        obj.curr_id = curr_id
        obj.curr_name = ""
        obj.pos_kb = 0
        obj.contig_id = id_contig
        obj.orientation = "w"
        obj.init_frag_start = init_frag_start
        obj.init_frag_end = init_frag_end
        obj.sub_frag_start = sub_frag_start
        obj.sub_frag_end = sub_frag_end
        obj.super_index = super_index
        obj.n_accu_frags = n_accu_frags
        return obj
