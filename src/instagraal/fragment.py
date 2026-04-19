#!/usr/bin/env python3


class basic_fragment:
    def __init__(self):
        "standard init fragment"

    @classmethod
    def initiate(
        cls,
        np_id_abs,
        id_init,
        init_contig,
        curr_id,
        start_pos,
        end_pos,
        length_kb,
        gc_content,
        init_frag_start,
        init_frag_end,
        sub_frag_start,
        sub_frag_end,
        super_index,
        id_contig,
        n_accu_frags,
    ):
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
