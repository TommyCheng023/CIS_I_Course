"""
Unit tests for YOUR components (src/student). The complete tests below fail
with NotImplementedError until you implement what they call; the ones
marked TODO are yours to write.

Run from assignments/assignment3:  python -m pytest tests -q
"""

import numpy as np
import pytest

from pathlib import Path

from src import vct3, Rot, Frame, load_sur, load_patch_boxes

MESH_FILE = Path(__file__).resolve().parents[1] / "data" / "ProximalFemurMeshFile.sur"
from src.student import closest_point, icp, uncertainty, report


def random_frame(rng, angle=1.0, dist=100.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return Frame(Rot(vct3(axis), angle=rng.uniform(-angle, angle)), vct3(rng.uniform(-dist, dist, 3)))


def apply(F, P):
    return P @ F.R.matrix.T + F.p.vec.ravel()


@pytest.fixture(scope="module")
def mesh():
    return load_sur(MESH_FILE)


@pytest.fixture(scope="module")
def surface_points(mesh):
    rng = np.random.default_rng(0)
    tri = rng.choice(mesh.n_triangles, size=150, p=mesh.areas / mesh.areas.sum())
    r1, r2 = rng.uniform(size=(2, 150))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    bary = np.stack([1 - r1 - r2, r1, r2], axis=1)
    return np.einsum("kj,kjd->kd", bary, mesh.tri_vertices[tri])


# ---------------------------------------------------------------- closest point

def test_closest_point_on_triangle_regions():
    a, b, c = vct3(0, 0, 0), vct3(10, 0, 0), vct3(0, 10, 0)
    assert np.allclose(closest_point.closest_point_on_triangle(vct3(2, 3, 5), a, b, c).vec.ravel(), [2, 3, 0])   # face
    assert np.allclose(closest_point.closest_point_on_triangle(vct3(5, -4, 1), a, b, c).vec.ravel(), [5, 0, 0])  # edge ab
    assert np.allclose(closest_point.closest_point_on_triangle(vct3(-3, -3, 2), a, b, c).vec.ravel(), [0, 0, 0]) # vertex a
    assert np.allclose(closest_point.closest_point_on_triangle(vct3(8, 8, 0), a, b, c).vec.ravel(), [5, 5, 0])   # edge bc


def test_closest_points_on_mesh_matches_brute_force(mesh):
    rng = np.random.default_rng(1)
    q = rng.uniform(-70, 70, size=(40, 3))
    closest, tri, dist = closest_point.closest_points_on_mesh(q, mesh)
    assert closest.shape == (40, 3) and tri.shape == (40,) and dist.shape == (40,)
    for k in range(40):
        best = min((closest_point.closest_point_on_triangle(vct3(q[k]), *[vct3(v) for v in mesh.tri_vertices[t]]) - vct3(q[k])).norm()
                   for t in range(mesh.n_triangles))
        assert np.isclose(dist[k], best, atol=1e-6)
        assert np.isclose(np.linalg.norm(closest[k] - q[k]), dist[k])


def test_closest_points_on_surface_points_have_zero_distance(mesh, surface_points):
    _, _, dist = closest_point.closest_points_on_mesh(surface_points, mesh)
    assert dist.max() < 1e-6


# ---------------------------------------------------------------- registration

def test_register_points_recovers_known_transform():
    rng = np.random.default_rng(2)
    F = random_frame(rng)
    a = [vct3(rng.uniform(-50, 50, 3)) for _ in range(6)]
    b = [F * p for p in a]
    F_est = icp.register_points(a, b)
    assert np.allclose(F_est.R.matrix, F.R.matrix, atol=1e-9) and np.allclose(F_est.p.vec, F.p.vec, atol=1e-9)


def test_icp_recovers_transform_from_surface_samples(mesh, surface_points):
    rng = np.random.default_rng(3)
    F_true = random_frame(rng, angle=0.5, dist=20.0)             # reference -> CT
    d = apply(F_true.inv(), surface_points)                        # samples in reference coordinates
    F_init = Frame(Rot(vct3(0, 0, 1), angle=0.15), vct3(5, -5, 3)) * F_true
    res = icp.icp(d, mesh, F_init, max_iter=100)
    assert res.converged
    err = np.linalg.norm(apply(res.F_reg, d) - surface_points, axis=1)
    assert err.max() < 0.05


def test_icp_rejects_injected_outliers(mesh, surface_points):
    rng = np.random.default_rng(4)
    F_true = random_frame(rng, angle=0.5, dist=20.0)
    d = apply(F_true.inv(), surface_points)
    bad = rng.choice(len(d), size=15, replace=False)
    d[bad] += rng.normal(size=(15, 3)) * 20.0
    res = icp.icp(d, mesh, F_true, max_iter=100, reject="mad")
    assert not res.inlier_mask[bad].any()
    assert res.inlier_mask.sum() >= len(d) - 15 - 5
    err = np.linalg.norm(apply(res.F_reg, d[res.inlier_mask]) - surface_points[res.inlier_mask], axis=1)
    assert err.max() < 0.05


# ---------------------------------------------------------------- report (provided)

def test_report_round_trip(tmp_path):
    F = Frame(Rot('z', 0.3), vct3(1, 2, 3))
    r = report.RegistrationReport(F_reg=F, patches=[{"patch_id": 0, "n_samples": 2, "n_used": 1, "n_rejected": 1}],
                                  samples=[{"sample_index": 0, "patch_id": 0, "t_sim": 1.0, "d_ref": [0, 0, 0],
                                            "residual_mm": 0.1, "used": True, "reason": "ok"}],
                                  residual_stats=report.residual_statistics(np.array([0.1])),
                                  t_simulated=1.03, t_wall=0.5, n_icp_iterations=3, converged=True)
    r.to_json(tmp_path / "r.json")
    back = report.RegistrationReport.from_json(tmp_path / "r.json")
    assert np.allclose(back.F_reg.R.matrix, F.R.matrix) and np.allclose(back.F_reg.p.vec, F.p.vec)
    assert "Valid samples (1)" in back.to_text()


# ---------------------------------------------------------------- TODO: yours

def test_registration_covariance_is_consistent():
    """Check covariance coverage under the assumed linear Gaussian model."""
    from scipy.stats import chi2
    from uncertainty_networks.se3 import exp_se3
    rng = np.random.default_rng(21)
    n_points = 100
    n_trials = 300
    noise_sigma = 0.1
    points = rng.normal(size=(n_points, 3)) * 30.0
    normals = rng.normal(size=(n_points, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    jacobian = np.column_stack((
        np.cross(points, normals),
        normals,
    ))
    mahalanobis_squared = []

    for _ in range(n_trials):
        noise = rng.normal(0.0, noise_sigma, size=n_points)
        estimate = np.linalg.lstsq(
            jacobian, noise, rcond=None
        )[0]
        fit_residuals = noise - jacobian @ estimate
        result = icp.IcpResult(
            F_reg=Frame(exp_se3(estimate)),
            residuals=np.abs(fit_residuals),
            inlier_mask=np.ones(n_points, dtype=bool),
            closest=points,
            normals=normals,
            n_iter=1,
            converged=True,
        )
        mu, covariance = uncertainty.registration_covariance(
            result, Frame.eye()
        )
        squared_distance = mu @ np.linalg.solve(covariance, mu)
        mahalanobis_squared.append(squared_distance)

    mahalanobis_squared = np.asarray(mahalanobis_squared)
    coverage = np.mean(
        mahalanobis_squared <= chi2.ppf(0.95, df=6)
    )

    assert 0.90 <= coverage <= 0.99
    assert 4.0 <= mahalanobis_squared.mean() <= 8.0

def test_patch_plan_keeps_bodies_visible():
    """Check nominal visibility in a synthetic CT-aligned workspace."""
    from collections import Counter
    from src.student import workflow
    from assignments.assignment1.src.tracker import Tracker

    data_dir = MESH_FILE.parent
    boxes = load_patch_boxes(
        data_dir / "box-centroids-rotations-world.json"
    )
    design = workflow.design_scene()
    tracker = Tracker("visibility_test", design.F_tracker_approx)

    plan = workflow.plan_patches(
        boxes,
        n_patches=12,
        samples_per_patch=6,
    )

    counts = Counter(plan)
    assert len(plan) == 72
    assert len(counts) == 12
    assert all(count == 6 for count in counts.values())
    reference_pose = Frame.eye()
    reference_points = [
        reference_pose * marker
        for marker in design.ref_marker_positions
    ]
    assert tracker.is_visible(reference_points)

    box_by_id = {box.id: box for box in boxes}

    for patch_id in counts:
        box = box_by_id[patch_id]
        pointer_rotation = box.F_box.R
        pointer_translation = (
            vct3(box.p) - pointer_rotation * design.p_tip
        )
        pointer_pose = Frame(
            pointer_rotation, pointer_translation
        )

        pointer_points = [
            pointer_pose * marker
            for marker in design.ptr_marker_positions
        ]
        assert tracker.is_visible(pointer_points), (
            f"Pointer not visible at patch {patch_id}"
        )

def test_registration_covariance_matches_linear_model():
    """Check covariance against a directly solved full-rank normal model."""
    rng = np.random.default_rng(20)
    n = 80
    points = rng.normal(size=(n, 3)) * 30.0
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    residuals = rng.uniform(0.05, 0.15, size=n)
    result = icp.IcpResult(
        F_reg=Frame.eye(),
        residuals=residuals,
        inlier_mask=np.ones(n, dtype=bool),
        closest=points,
        normals=normals,
        n_iter=1,
        converged=True,
    )
    mu, covariance = uncertainty.registration_covariance(result, Frame.eye())
    jacobian = np.column_stack((
        np.cross(points, normals),
        normals,
    ))
    variance = np.sum(residuals**2) / (n - 6)
    expected = variance * np.linalg.solve(jacobian.T @ jacobian, np.eye(6))

    assert mu.shape == (6,)
    assert np.allclose(mu, 0.0)
    assert covariance.shape == (6, 6)
    assert np.allclose(covariance, expected)
    assert np.allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() > 0.0


def test_predicted_tre_translation_only(mesh):
    """Translation-only isotropic uncertainty gives equal RMS at all vertices."""
    sigma = 0.2
    covariance = np.zeros((6, 6))
    covariance[3:, 3:] = sigma**2 * np.eye(3)
    prediction = uncertainty.predicted_tre(mesh, Frame.eye(), covariance)
    expected = np.sqrt(3.0) * sigma
    for key in ("rms", "mean", "max", "p90"):
        assert np.isclose(prediction[key], expected)


def test_coarse_registration_recovers_known_transform():
    """Recover a known transform from symmetric per-patch samples."""
    from src.student import workflow

    boxes = load_patch_boxes(
        MESH_FILE.parent / "box-centroids-rotations-world.json"
    )
    plan = workflow.plan_patches(
        boxes, n_patches=12, samples_per_patch=1
    )
    box_by_id = {box.id: box for box in boxes}

    rng = np.random.default_rng(22)
    F_true = random_frame(rng)

    offsets = np.vstack((
        np.zeros((1, 3)),
        np.eye(3),
        -np.eye(3),
    ))

    by_patch = {}
    for patch_id in plan:
        center_ref = (
            F_true.inv() * vct3(box_by_id[patch_id].p)
        ).vec.ravel()
        by_patch[patch_id] = center_ref + offsets

    F_est = icp.coarse_registration(by_patch, boxes)

    assert np.allclose(F_est.R.matrix, F_true.R.matrix, atol=1e-9)
    assert np.allclose(F_est.p.vec, F_true.p.vec, atol=1e-9)


def test_collect_samples_relative_transform_and_invisibility():
    """Check relative tip coordinates, skipped records, and acquisition time."""
    from types import SimpleNamespace
    from src.student import workflow
    from src.session import Observation, PointerPlacement

    design = workflow.design_scene()
    shared = Frame(Rot("x", 0.3), vct3(50, -30, 20))
    reference_pose = Frame(Rot("z", 0.2), vct3(10, 20, 30))
    pointer_pose = Frame(Rot("y", -0.4), vct3(40, -10, 60))

    reference = Observation(shared * reference_pose, [])
    pointer = Observation(shared * pointer_pose, [])

    class FakeSession:
        """Minimal deterministic session for testing collection."""

        def __init__(self):
            """Initialize the clock and synthetic observations."""
            self.design = design
            self.clock = 0.0
            self.index = -1
            self.readings = [
                {"ref": reference, "ptr": pointer},
                {"ref": reference, "ptr": None},
                {"ref": None, "ptr": pointer},
            ]

        def elapsed(self):
            """Return simulated time in seconds."""
            return self.clock

        def place_pointer(self, patch_id):
            """Place a pointer and return its bookkeeping handle."""
            self.index += 1
            self.clock += 1.0
            return PointerPlacement(
                patch_id, self.index, self.clock
            )

        def sample_marker_bodies(self):
            """Return paired observations and advance the sample clock."""
            self.clock += 0.03
            return self.readings[self.index]

    session = FakeSession()
    samples = workflow.collect_samples(session, [2, 4, 6])

    expected = (
        reference_pose.inv() * pointer_pose * design.p_tip
    ).vec.ravel()

    assert len(samples) == 1
    assert samples.as_array().shape == (1, 3)
    assert np.allclose(samples.d_ref[0], expected)
    assert samples.patch_ids == [2]
    assert samples.handles[0].sample_index == 0

    assert len(samples.skipped) == 2
    assert samples.skipped[0]["sample_index"] == 1
    assert samples.skipped[1]["sample_index"] == 2
    assert samples.skipped[0]["reason"] == "invisible: pointer"
    assert samples.skipped[1]["reason"] == "invisible: reference"
    assert np.isclose(samples.t_collect, 3 * 1.03)