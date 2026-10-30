import random

import numpy as np

from reviewlens.config import load_config, set_seed


def test_set_seed_is_deterministic():
    set_seed(7)
    a = (random.random(), np.random.rand())
    set_seed(7)
    assert a == (random.random(), np.random.rand())


def test_load_data_config():
    cfg = load_config("data")
    assert cfg["seed"] == 42
    assert cfg["labels"]["negative"] == [1, 2]
