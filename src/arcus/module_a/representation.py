from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubspaceEstimate:
    mean: np.ndarray
    basis: np.ndarray  # [hidden_dim, rank], orthonormal columns
    explained_variance: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])


def estimate_cross_formulation_subspace(activations: np.ndarray, rank: int) -> SubspaceEstimate:
    """Estimate a low-rank cross-formulation subspace with centered SVD.

    This is only a descriptive candidate representation. It must pass held-out causal projection and
    patch tests before being described as a causal fact-representation locus.
    """
    if activations.ndim != 2:
        raise ValueError("activations must have shape [n_examples, hidden_dim]")
    if rank < 1 or rank > min(activations.shape):
        raise ValueError("rank must be between 1 and min(n_examples, hidden_dim)")

    mean = activations.mean(axis=0, keepdims=True)
    centered = activations - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:rank].T
    variance = singular_values**2
    explained = variance[:rank] / max(float(variance.sum()), 1e-12)
    return SubspaceEstimate(mean=mean.squeeze(0), basis=basis, explained_variance=explained)


def project_out(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the component of x lying in span(basis)."""
    return x - (x @ basis) @ basis.T


def projection_energy(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coeff = x @ basis
    return np.sum(coeff**2, axis=-1)


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared canonical overlap for two orthonormal bases; 1 means identical subspaces."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("bases must be 2D")
    r = min(a.shape[1], b.shape[1])
    if r == 0:
        return float("nan")
    singular_values = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(np.mean(singular_values[:r] ** 2))
