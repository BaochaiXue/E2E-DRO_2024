import pytest

from e2edro import DataLoad


def test_synthetic_exp_shapes():
    X, Y = DataLoad.synthetic_exp(n_x=2, n_y=3, n_tot=20, n_obs=4)
    assert X.data.shape[1] == 2
    assert Y.data.shape[1] == 3
