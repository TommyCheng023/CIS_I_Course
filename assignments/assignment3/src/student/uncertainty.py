"""Statistical estimate of the registration error and the predicted TRE."""

from typing import Tuple

import numpy as np

from data_types.nominal_types import Frame
from ..mesh import Mesh
from .icp import IcpResult
from uncertainty_networks.se3 import log_se3


def registration_covariance(result: IcpResult, F0_reg: Frame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate registration correction and local surface-fit covariance.
    inputs
    ----------
    result: IcpResult
        Final ICP result with CT-space closest points, normals, residual distances, and inlier mask.
    F0_reg: Frame
        Initial reference-to-CT registration guess.

    returns
    -------
    mu: np.ndarray, shape (6,)
        Estimated correction relative to F0_reg, ordered as rotation followed by translation, using a left CT perturbation.
    C_reg: np.ndarray, shape (6, 6)
        Approximate covariance around the final registration, using the same perturbation convention.
    """
    mask = result.inlier_mask
    points = result.closest[mask]
    normals = result.normals[mask]
    residuals = result.residuals[mask]
    n_inliers = len(points)
    if n_inliers <= 6:
        raise ValueError(f"Too few inliers ({n_inliers}) to estimate registration covariance.")
    J = np.column_stack((
        np.cross(points, normals), normals
    ))
    u, sigma, vt = np.linalg.svd(J, full_matrices=False)
    rank_tolerance = (
        max(J.shape) * np.finfo(float).eps * sigma[0]
    )
    variance = np.sum(residuals**2) / (n_inliers - 6)
    C_reg = (variance * (1.0 / sigma ** 2)) @ vt
    C_reg = 0.5 * (C_reg + C_reg.T)
    correction = result.F_reg * F0_reg.inv()
    correction_matrix = np.eye(4)
    correction_matrix[:3, :3] = correction.R.matrix
    correction_matrix[:3, 3] = correction.p.vec.ravel()
    mu = log_se3(correction_matrix)
    return mu, C_reg


def predicted_tre(mesh: Mesh, F_reg: Frame, C_reg: np.ndarray) -> dict:
    """
    Summarize predicted pointwise RMS TRE over CT mesh vertices.

    inputs
    ----------
    mesh: Mesh
        Target mesh with vertices in CT coordinates.
    F_reg: Frame
        Estimated reference-to-CT registration. Retained for the required interface. CT vertices need no further transformation.
    C_reg: np.ndarray, shape (6, 6)
        Registration covariance for a left CT perturbation, ordered as rotation followed by translation.

    return
    -------
    dict
        "rms": root mean predicted squared TRE across vertices.
        "mean", "max", "p90": statistics of pointwise RMS TRE, in mm.
        These are spatial summaries, not probability quantiles.
    """
    vertices = mesh.vertices
    n_vertices = len(vertices)
    jacobian = np.zeros((n_vertices, 3, 6))
    x, y, z = vertices.T

    jacobian[:, 0, 1] = z
    jacobian[:, 0, 2] = -y
    jacobian[:, 1, 0] = -z
    jacobian[:, 1, 2] = x
    jacobian[:, 2, 0] = y
    jacobian[:, 2, 1] = -x
    jacobian[:, :, 3:] = np.eye(3)

    expected_squared_error = np.einsum(
        "nij,jk,nik->n",
        jacobian,
        C_reg,
        jacobian,
    )

    expected_squared_error = np.maximum(expected_squared_error, 0.0)
    pointwise_rms = np.sqrt(expected_squared_error)

    return {
        "rms": float(np.sqrt(expected_squared_error.mean())),
        "mean": float(pointwise_rms.mean()),
        "max": float(pointwise_rms.max()),
        "p90": float(np.percentile(pointwise_rms, 90)),
    }