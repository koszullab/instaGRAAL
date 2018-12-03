"""instaGRAAL testing

Basic testing for the instaGRAAL scaffolder.
"""

import pathlib
import pickle
import pytest
from instagraal import instagraal

TEST_DATASETS = ("demo_name", ["trichoderma"])


@pytest.mark.parametrize(*TEST_DATASETS)
def test_single_run(demo_name):
    """Test run on Trichoderma dataset.
    """
    dataset_path = pathlib.Path("/media/rsg/DATA/instaGRAAL/demos") / demo_name
    p2 = instagraal.window(
        name=demo_name,
        folder_path=dataset_path,
        fasta=dataset_path / ("{}.fa".format(demo_name)),
        device=0,
        level=3,
        n_iterations_em=100,
        n_iterations_mcmc=30,
        is_simu=False,
        scrambled=False,
        perform_em=False,
        use_rippe=True,
        gl_size_im=1000,
        sample_param=True,
        thresh_factor=0,
        output_folder=dataset_path,
    )

    p2.full_em(
        n_cycles=1,
        n_neighbours=3,
        bomb=True,
        id_start_sample_param=4,
        save_matrix=True,
    )

    with open("graal.pkl", "wb") as pickle_handle:
        pickle.dump(p2, pickle_handle)

    p2.ctx_gl.pop()
