#!/usr/bin/env python3


class fragment:
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
        obj.contig_id = 0
        obj.orientation = "w"
        return obj

    def update_name(self, contig_id):
        self.curr_name = (
            str(self.curr_id) + "-" + str(contig_id) + "-" + str(self.pos_kb)
        )
        self.contig_id = contig_id

    @classmethod
    def copy(cls, frag):
        obj = cls()
        obj.id_init = frag.id_init
        obj.init_contig = frag.init_contig
        obj.init_name = frag.init_name
        obj.start_pos = frag.start_pos
        obj.end_pos = frag.end_pos
        obj.length_kb = frag.length_kb
        obj.gc_content = frag.gc_content
        obj.np_id_abs = frag.np_id_abs
        obj.cur_id = 0
        obj.curr_name = ""
        obj.pos_kb = 0
        obj.orientation = "w"
        return obj


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
