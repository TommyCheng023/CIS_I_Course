"""The end-to-end registration workflow: design, patch plan, data collection, registration."""

from dataclasses import dataclass, field
from data_types.nominal_types import vct3, Rot, Frame
from typing import List

import numpy as np
import time

from .icp import coarse_registration, icp
from .uncertainty import registration_covariance, predicted_tre
from ..mesh import Mesh, PatchBox
from ..session import Session, SceneDesign, PointerPlacement
from .report import RegistrationReport, residual_statistics


@dataclass
class SampleSet:
    """Tip positions collected through the tracker."""
    handles: List[PointerPlacement] = field(default_factory=list)
    patch_ids: List[int] = field(default_factory=list)
    d_ref: List[np.ndarray] = field(default_factory=list)   # (3,) tip in reference-body coordinates
    t_collect: float = 0.0
    skipped: List[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.d_ref)

    def as_array(self) -> np.ndarray:
        return np.array(self.d_ref).reshape(-1, 3)


def design_scene() -> SceneDesign:
    """
    TODO: implement -- return the same values the notebook's Step 1 cell uses
    (grading calls this instead of the notebook).
    Define the tracker pose, reference and pointer marker layouts, and pointer tip position.

    return
    -------
    SceneDesign
        Tracker pose in workspace coordinates, reference and pointer markers in their respective body coordinates, and pointer tip in pointer coordinates. All positions are in millimeters.
    """
    tracker_distance = 1600.0 #mm
    pointer_length = 120.0    #mm

    tracker_pose = Frame(
        Rot.eye(),
        vct3(0.0, 0.0, -tracker_distance),
    )
    reference_markers = [
        vct3(-40.0, -30.0, 0.0),
        vct3(40.0, -30.0, 0.0),
        vct3(0.0, 40.0, 0.0),
        vct3(0.0, 0.0, 30.0),
    ]
    pointer_markers = [
        vct3(-35.0, -25.0, 0.0),
        vct3(35.0, -25.0, 0.0),
        vct3(0.0, 35.0, 0.0),
        vct3(0.0, 0.0, 30.0),
    ]

    pointer_tip = vct3(0.0, 0.0, -pointer_length)

    return SceneDesign(
        F_tracker_approx=tracker_pose,
        ref_marker_positions=reference_markers,
        ptr_marker_positions=pointer_markers,
        p_tip=pointer_tip,
    )


def plan_patches(boxes: List[PatchBox], n_patches: int, samples_per_patch: int) -> List[int]:
    """
    Select spatially and directionally diverse patches for sampling.

    inputs
    ----------
    boxes: List[PatchBox]
        Available patch boxes in CT coordinates.
    n_patches: int
        Number of distinct patches to select, between 1 and len(boxes).
    samples_per_patch: int
        Positive number of pointer placements per selected patch.

    return
    -------
    List[int]
        Ordered patch IDs, repeating each selected patch
        samples_per_patch times.
    """
    centers = np.array([box.p for box in boxes])
    normals = np.array([box.normal for box in boxes])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    extent = np.linalg.norm(np.ptp(centers, axis=0))
    if extent == 0.0:
        extent = 1.0

    offsets = centers[:, None, :] - centers[None, :, :]
    spatial_distance = np.sum(offsets ** 2, axis=2) / extent ** 2
    normal_similarity = np.clip(
        np.abs(normals @ normals.T), 0.0, 1.0
    )
    normal_distance = 1.0 - normal_similarity

    diversity = spatial_distance + normal_distance
    center_distances = np.linalg.norm(
        centers - centers.mean(axis=0), axis=1
    )
    selected = [int(np.argmax(center_distances))]

    while len(selected) < n_patches:
        scores = diversity[:, selected].min(axis=1)
        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))

    return [
        boxes[index].id
        for index in selected
        for _ in range(samples_per_patch)
    ]


def collect_samples(session: Session, plan: List[int]) -> SampleSet:
    """
    Collect pointer-tip measurements in reference-body coordinates.

    inputs
    ----------
    session: Session
        Simulation interface providing pointer placement and simultaneous observations of the reference and pointer bodies.
    plan: List[int]
        Ordered patch IDs, with one pointer placement per entry.

    return
    -------
    SampleSet
        Valid tip positions with placement handles and patch IDs, skipped-observation records, and simulated collection duration.
    """
    samples = SampleSet()
    start_time = session.elapsed()
    for patch_id in plan:
        placement = session.place_pointer(patch_id)
        observations = session.sample_marker_bodies()
        reference = observations["ref"]
        pointer = observations["ptr"]
        if reference is None or pointer is None:
            missing = []
            if reference is None:
                missing.append("reference")
            if pointer is None:
                missing.append("pointer")
            samples.skipped.append({
                "sample_index": placement.sample_index,
                "patch_id": placement.patch_id,
                "t_sim": placement.t_sim,
                "reason": "invisible: " + ", ".join(missing),
            })
            continue

        tip_reference = reference.frame.inv() * pointer.frame * session.design.p_tip
        samples.handles.append(placement)
        samples.patch_ids.append(placement.patch_id)
        samples.d_ref.append(tip_reference.vec.ravel().copy())

    samples.t_collect = session.elapsed() - start_time
    return samples


def run_registration(session: Session, mesh: Mesh, boxes: List[PatchBox], **kwargs) -> RegistrationReport:
    """
    Collect surface samples, register them, and assemble a report.

    inputs
    ----------
    session: Session
        Simulation interface with an initialized scene design.
    mesh: Mesh
        Femur surface in CT coordinates.
    boxes: List[PatchBox]
        Available sampling-patch geometry in CT coordinates.
    **kwargs
        n_patches: int, default 12
            Maximum number of visible patches to select.
        samples_per_patch: int, default 6
            Pointer placements per selected patch.
        max_iter: int, default 100
            ICP iteration limit.
        tol: float, default 1e-4
            ICP convergence tolerance in mm.
        reject: str or None, default "mad"
            ICP outlier-rejection method.

    return
    -------
    RegistrationReport
        Registration, sample decisions, residual statistics, uncertainty, predicted TRE, and timing information.

    """
    start_wall = time.perf_counter()
    F0_reg = session.initial_registration_guess()

    n_patches = kwargs.get("n_patches", 12)
    samples_per_patch = kwargs.get("samples_per_patch", 6)
    reference_pose = session.nominal_reference_pose()
    reference_markers = [
        reference_pose * marker
        for marker in session.design.ref_marker_positions
    ]
    if not session.is_visible(reference_markers):
        raise RuntimeError("Reference markers are not nominally visible")
    visible_boxes = []
    for box in boxes:
        pointer_pose = session.nominal_pointer_pose(box.id)
        pointer_markers = [
            pointer_pose * marker
            for marker in session.design.ptr_marker_positions
        ]
        if session.is_visible(pointer_markers):
            visible_boxes.append(box)

    if len(visible_boxes) < 3:
        raise RuntimeError("Fewer than three patches are nominally visible")
    plan = plan_patches(
        visible_boxes,
        n_patches=min(n_patches, len(visible_boxes)),
        samples_per_patch=samples_per_patch,
    )
    samples = collect_samples(session, plan)
    if len(samples) == 0:
        raise RuntimeError("No valid tracker measurements were collected")

    by_patch = {}
    for patch_id, point in zip(samples.patch_ids, samples.d_ref):
        by_patch.setdefault(patch_id, []).append(point)
    F_coarse = coarse_registration(by_patch, boxes)

    result = icp(
        samples.as_array(),
        mesh,
        F_coarse,
        max_iter=kwargs.get("max_iter", 100),
        tol=kwargs.get("tol", 1e-4),
        reject=kwargs.get("reject", "mad"),
    )

    mu, covariance = registration_covariance(result, F0_reg)
    tre_prediction = predicted_tre(mesh, result.F_reg, covariance)
    sample_records = []
    for index, placement in enumerate(samples.handles):
        used = bool(result.inlier_mask[index])
        sample_records.append({
            "sample_index": placement.sample_index,
            "patch_id": placement.patch_id,
            "t_sim": placement.t_sim,
            "d_ref": samples.d_ref[index].tolist(),
            "residual_mm": float(result.residuals[index]),
            "used": used,
            "reason": "inlier" if used else "icp_outlier",
        })

    for skipped in samples.skipped:
        sample_records.append({
            **skipped,
            "d_ref": None,
            "residual_mm": None,
            "used": False,
        })

    sample_records.sort(key=lambda record: record["sample_index"])

    patch_records = []
    for patch_id in dict.fromkeys(plan):
        records = [
            record
            for record in sample_records
            if record["patch_id"] == patch_id
        ]
        n_used = sum(record["used"] for record in records)
        patch_records.append({
            "patch_id": patch_id,
            "n_samples": len(records),
            "n_used": n_used,
            "n_rejected": len(records) - n_used,
        })

    simulated_time = session.elapsed()

    return RegistrationReport(
        F_reg=result.F_reg,
        patches=patch_records,
        samples=sample_records,
        residual_stats=residual_statistics(
            result.residuals[result.inlier_mask]
        ),
        t_simulated=simulated_time,
        t_wall=time.perf_counter() - start_wall,
        n_icp_iterations=result.n_iter,
        converged=result.converged,
        eta_reg={
            "mu": mu.tolist(),
            "C_reg": covariance.tolist(),
            "convention": "left_ct",
        },
        predicted_tre=tre_prediction,
    )