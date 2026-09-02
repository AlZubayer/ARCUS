import numpy as np

from arcus.module_a.representation import estimate_cross_formulation_subspace, project_out, subspace_overlap


def test_subspace_and_project_out():
    x = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    subspace = estimate_cross_formulation_subspace(x, rank=1)
    y = project_out(x, subspace.basis)
    assert np.allclose(y[:, 0], 0.0, atol=1e-8)


def test_identical_subspace_overlap():
    basis = np.eye(3)[:, :2]
    assert np.isclose(subspace_overlap(basis, basis), 1.0)
