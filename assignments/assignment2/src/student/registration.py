"""
Rigid registration helpers: point-set registration, frame averaging and
marker-body refinement.
"""

from typing import List

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame

from assignments.assignment1.src.tracker import Tracker
from .. import simulation
from ..simulation import PlacedBody


def register_points(a: List[vct3], b: List[vct3]) -> Frame:
    """
    TODO: implement (your own -- do not call the simulator's solver).

    Rigid point-set registration: find the least-squares Frame F such that
    F * a[i] ~= b[i] for every corresponding pair. Think about what happens
    with fewer than 3 points, with collinear points, and with a reflection.

    input:
        a - points in the source frame (e.g. a marker body's local layout)
        b - the same points observed in the target frame, same order
    output: Frame mapping a -> b
    """
    A = np.array([point.vec.ravel() for point in a], dtype=float)
    B = np.array([point.vec.ravel() for point in b], dtype=float)

    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
        raise ValueError("Point coordinates must be finite")

    center_a = np.mean(A, axis=0)
    center_b = np.mean(B, axis=0)
    A_centered = A - center_a
    B_centered = B - center_b

    if (
        np.linalg.matrix_rank(A_centered) < 2
        or np.linalg.matrix_rank(B_centered) < 2
    ):
        raise ValueError("Points must not be collinear")

    H = A_centered.T @ B_centered
    U, _, Vt = np.linalg.svd(H)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ correction @ U.T
    translation = center_b - R @ center_a

    return Frame(Rot(matrix=R), vct3(translation))


def average_frames(frames: List[Frame]) -> Frame:
    """
    TODO: implement.

    Average of several noisy estimates of the same Frame. Rotations cannot
    simply be averaged element-wise -- the result must be a proper
    rotation matrix.

    input:
        frames - List[Frame], all estimates of the same transform
    output: the averaged Frame
    """
    p_mean = np.mean([F.p.vec.ravel() for F in frames], axis=0)
    R_mean = np.mean([F.R.matrix for F in frames], axis=0)

    U, _, Vt = np.linalg.svd(R_mean)

    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(U @ Vt)
    R_avg = U @ correction @ Vt

    return Frame(Rot(matrix=R_avg), vct3(p_mean))


def refine_marker_body(trk: Tracker, placed: PlacedBody, n_readings: int = 30) -> List[vct3]:
    """
    TODO: implement.

    "It is possible to improve your estimate of a_m,k through a calibration
    process involving multiple observations of the marker body using the
    tracking camera and then averaging them."

    Take n_readings SampleMarkerBodies readings, use
    most_recent_observed_markers(...) plus your register_points to bring
    each reading's b_m,k back into the body's own frame, average, and push
    the result into the body with update_marker_body(...). Note that each
    reading costs delta_t_samp of simulated time, and that the tracker
    jiggles between readings.

    input:
        trk - Tracker the body is tracked by
        placed - PlacedBody whose body's nominal marker positions should be refined
        n_readings - number of readings to average
    output: the new List[vct3] of marker positions (also stored on the body)
    """
    body = placed.body
    nominal = list(body.nominal_marker_positions)
    local_readings = []

    for _ in range(n_readings):
        simulation.sample_marker_bodies(trk)
        observed = simulation.most_recent_observed_markers(body)
        if observed is None:
            raise ValueError(f"Marker body {body.name} is not visible")
        F_body_to_tracker = register_points(nominal, observed)
        F_tracker_to_body = F_body_to_tracker.inv()

        points_local = [
            (F_tracker_to_body * point).vec.ravel()
            for point in observed
        ]
        local_readings.append(points_local)

    mean_positions = np.mean(local_readings, axis=0)
    refined = [vct3(point) for point in mean_positions]
    simulation.update_marker_body(body, refined)
    return refined
