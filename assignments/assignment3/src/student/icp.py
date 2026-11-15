"""Rigid registration: point-set registration, coarse initialisation and ICP."""

from dataclasses import dataclass, field
from data_types.nominal_types import vct3, Rot, Frame
from typing import Dict, List, Optional

import numpy as np

from data_types.nominal_types import vct3, Frame
from ..mesh import Mesh, PatchBox


@dataclass
class IcpResult:
    """Output of icp()."""
    F_reg: Frame                  # reference body -> CT
    residuals: np.ndarray         # (K,) ||F_reg d_k - closest_k|| for every sample
    inlier_mask: np.ndarray       # (K,) bool, False for rejected samples
    closest: np.ndarray           # (K,3) closest mesh points of the last iteration
    normals: np.ndarray           # (K,3) mesh normals at those points
    n_iter: int
    converged: bool
    history: List[float] = field(default_factory=list)   # rms inlier residual per iteration


def register_points(a: List[vct3], b: List[vct3]) -> Frame:
    """
    Compute the least-squares rigid transform between paired point sets.

    inputs
    ----------
    a: List[vct3]
        Source points; at least three non-collinear points are required.
    b: List[vct3]
        Target points corresponding to a in the same order and count.

    return
    -------
    Frame
        Rigid transform F mapping source coordinates to target coordinates,
        minimizing sum_i ||F * a_i - b_i||^2.
    """
    source = np.stack([point.vec.ravel() for point in a])
    target = np.stack([point.vec.ravel() for point in b])

    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Point coordinates must be finite")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)

    source_centered = source - source_center
    target_centered = target - target_center

    cross_covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(cross_covariance)

    rank_tolerance = (
        max(cross_covariance.shape)
        * np.finfo(float).eps
        * singular_values[0]
    )
    if np.count_nonzero(singular_values > rank_tolerance) < 2:
        raise ValueError("Correspondences do not constrain rotation")
    correction = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[2, 2] = -1.0
    rotation = vt.T @ correction @ u.T
    translation = target_center - rotation @ source_center

    return Frame(Rot(matrix=rotation), vct3(translation))

def coarse_registration(d_by_patch: Dict[int, np.ndarray], boxes: List[PatchBox]) -> Frame:
    """
    input: {patch id: (n_i,3) tip points in reference coordinates}, the patch boxes
    output: a rough F_reg (reference -> CT) from patch centroids <-> box centres
    """
    raise NotImplementedError("TODO: implement coarse_registration")


def icp(d: np.ndarray, mesh: Mesh, F_init: Frame, max_iter: int = 100, tol: float = 1e-4,
        reject: Optional[str] = "mad") -> IcpResult:
    """
    TODO: implement.
    input: d (K,3) tip points in reference coordinates, mesh, initial F_reg,
           iteration cap, convergence tolerance, outlier-rejection method
    output: IcpResult
    """
    raise NotImplementedError("TODO: implement icp")
