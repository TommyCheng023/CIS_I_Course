"""Rigid registration: point-set registration, coarse initialisation and ICP."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame
from ..mesh import Mesh, PatchBox
from .closest_point import closest_points_on_mesh


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
        Rigid transform F mapping source coordinates to target coordinates, minimizing sum_i ||F * a_i - b_i||^2.
    """
    source = np.stack([point.vec.ravel() for point in a])
    target = np.stack([point.vec.ravel() for point in b])
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
    Estimate an initial reference-to-CT transform using patch identities.
    inputs
    ----------
    d_by_patch : Dict[int, np.ndarray]
        Patch IDs mapped to tip samples of shape (N_i, 3) in reference coordinates. Each patch must contain at least one sample.
    boxes : List[PatchBox]
        Patch boxes providing corresponding centers in CT coordinates.

    return
    -------
    Frame
        Coarse reference-to-CT transform.
    """
    box_by_id = {box.id: box for box in boxes}
    source = []
    target = []

    for patch_id, samples in sorted(d_by_patch.items()):
        if patch_id not in box_by_id:
            raise ValueError(f"Unknown patch ID: {patch_id}")

        samples = np.asarray(samples, dtype=float)
        if samples.size == 0:
            # skip useless patches
            continue

        representative = np.median(samples, axis=0)
        source.append(vct3(representative))
        target.append(vct3(box_by_id[patch_id].p))

    if len(source) < 3:
        raise ValueError(
            "Too few usable patches to determine a rigid transform."
        )

    return register_points(source, target)


def icp(d: np.ndarray, mesh: Mesh, F_init: Frame, max_iter: int = 100, tol: float = 1e-4,
        reject: Optional[str] = "mad") -> IcpResult:
    """
    Register reference-space samples to a CT mesh using robust ICP.

    inputs
    ----------
    d : np.ndarray, shape (K, 3)
        Measured tip positions in reference coordinates.
    mesh : Mesh
        Target surface in CT coordinates.
    F_init : Frame
        Initial reference-to-CT transform.
    max_iter : int
        Maximum number of transformation updates; must be positive.
    tol : float
        Convergence tolerance in millimeters. Stop when maximum inlier movement is below `tol`, or RMS changes remain below tol for three updates.
    reject : Optional[str]
        "mad" for adaptive distance rejection, or None for no rejection.

    return
    -------
    IcpResult
        Final transform, residuals, inlier mask, closest points, normals, iteration count, convergence status, and RMS history.
    """
    d = np.asarray(d, dtype=float)
    F_reg = F_init
    history = []
    conv_flag = False

    mad_scale = 1.4826
    rej_factor = 3.0
    minimum_margin = 1e-6
    for i in range(max_iter):
        transformed = (
            d @ F_reg.R.matrix.T + F_reg.p.vec.ravel()
        )
        closest, triangle_indices, residuals = closest_points_on_mesh(transformed, mesh)
        if reject == "mad":
            median = np.median(residuals)
            mad = np.median(np.abs(residuals - median))
            margin = max(rej_factor * mad_scale * mad, minimum_margin)
            inlier_mask = residuals <= margin + median
        else:
            inlier_mask = np.ones(len(d), dtype=bool)
        if inlier_mask.sum() < 3:
            raise ValueError(
                "Too few inlier matches to determine a rigid transform."
            )
        source = [vct3(p) for p in d[inlier_mask]]
        target = [vct3(p) for p in closest[inlier_mask]]
        F_update = register_points(source, target)
        updated = (
            d @ F_update.R.matrix.T + F_update.p.vec.ravel()
        )
        max_movement = np.linalg.norm(updated[inlier_mask] - transformed[inlier_mask], axis=1).max()
        fit_err = updated[inlier_mask] - closest[inlier_mask]
        rms = np.sqrt(np.mean(np.sum(fit_err ** 2, axis=1)))
        history.append(rms)
        F_reg = F_update
        rms_stable = (
            len(history) >= 4
            and np.all(np.abs(np.diff(history[-4:])) <= tol)
        )

        if max_movement <= tol or rms_stable:
            conv_flag = True
            break

    # recompute for once
    transformed = d @ F_reg.R.matrix.T + F_reg.p.vec.ravel()
    closest, triangle_indices, residuals = closest_points_on_mesh(
        transformed, mesh
    )
    if reject == "mad":
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        margin = max(rej_factor * mad_scale * mad, minimum_margin)
        inlier_mask = residuals <= margin + median
    else:
        inlier_mask = np.ones(len(d), dtype=bool)
    if inlier_mask.sum() < 3:
        raise ValueError(
            "Too few inlier matches to determine a rigid transform."
        )

    return IcpResult(
        F_reg=F_reg,
        residuals=residuals,
        inlier_mask=inlier_mask,
        closest=closest,
        normals=mesh.face_normals[triangle_indices],
        n_iter=len(history),
        converged=conv_flag,
        history=history,
    )
