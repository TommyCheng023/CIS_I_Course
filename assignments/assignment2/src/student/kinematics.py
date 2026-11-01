"""
The calibrated kinematic model: ImprovedFwdKins / ImprovedInvKins.
"""

from typing import Dict, Optional
from scipy.optimize import least_squares
import numpy as np

from data_types.nominal_types import vct3, Rot, Frame
from uncertainty_networks.se3 import log_so3, exp_so3, exp_se3

from ..robot import Robot
from .workflow import CalibrationData


class ImprovedKinematics(Robot):
    """
    TODO: implement -- the core deliverable.

    A Robot whose fwd_kins / inv_kins account for the calibrated error
    model. You choose the parameterization; the handout's error model is
        theta*_k = theta_k + delta_theta_k,sys(theta_k)   (all joints; ||.|| <= delta_sys, slope <= omega_sys)
        d*_k = R(phi_k) d_k                                (k = 2..6, ||phi_k|| <= phi_max; d*_1 = d_1)
        F*_we = F_we dF_ee                                 (||alpha_ee|| <= 0.01, ||epsilon_ee|| <= 0.5)
    plus the constant base frame F_X relating the robot base to the
    tracker (see workflow.estimate_base_frame). HomeRobot() gives you
    delta_theta_sys at home directly (workflow.measure_home_offsets); pass
    it as joint_offsets. Every parameter you include must be identifiable
    from your data -- look at the singular values of the Jacobian.

    A parameter vector that covers this model, for example:
        params[0:6]     xi_X          twist of the base frame about F_X_init, F_X = F_X_init * Exp(xi_X)
        params[6:16]    (t1, t2) x 5  tilt of d_2, d_3, n_4, n_5, n_6, two components each in the
                                      plane perpendicular to the nominal axis
        params[16:22]   xi_we         twist of the tool frame, F_we = F_WE * Exp(xi_we)
        params[22:]     c_k,j         per-joint distortion polynomial coefficients
                                      delta_theta_k,sys = offset_k + sum_j c_k,j (theta_k / scale_k)^j
    with every entry zero meaning "the nominal robot at F_X_init".

    The nominal Robot's attributes (AXES, F_CW, F_WE, limits, timing,
    wrist_solutions) are inherited and may be reused.
    """

    def __init__(self, robot: Robot, F_X_init: Frame, joint_offsets=None, n_poly: int = 2):
        super().__init__(robot.name + "_improved", robot.F_rob, robot.theta_min, robot.theta_max,
                         robot.t_settle, robot.theta_dot_max, robot.theta_ddot_max)
        self.F_X_init = F_X_init  # starting estimate of the base frame (from the nominal fit)
        self.joint_offsets = np.zeros(6) if joint_offsets is None else np.asarray(joint_offsets, dtype=np.float64).reshape(6)
        self.n_poly = n_poly      # e.g. the degree of a per-joint distortion polynomial
        self.params = self.initial_params()

    # ------------------------------------------------------------------ parameterization

    def initial_params(self) -> np.ndarray:
        """
        TODO: implement.

        output: the parameter vector that makes this model identical to the
                nominal kinematics with base frame F_X_init -- with the
                layout in the class docstring that is simply
                np.zeros(6 + 10 + 6 + 6 * self.n_poly)
        """
        return np.zeros(22 + 6 * self.n_poly)

    def base_frame(self, params: Optional[np.ndarray] = None) -> Frame:
        """
        TODO: implement.

        output: F_X (robot base relative to the tracker) for params (default:
                self.params) -- e.g. F_X_init * Exp(params[0:6]), see
                uFrame.perturb / uncertainty_networks.se3.exp_se3
        """
        p = self.params if params is None else np.asarray(params)
        T = exp_se3(np.asarray(p[:6], dtype=float))
        return self.F_X_init * Frame(Rot(matrix=T[:3, :3]), vct3(T[:3, 3]))

    # ------------------------------------------------------------------ kinematics

    def fwd_kins(self, theta, params: Optional[np.ndarray] = None) -> Frame:
        """
        ImprovedFwdKins(theta).

        TODO: implement.

        input: theta - 6 encoder joint values; params - defaults to self.params
        output: F_ee relative to the robot base frame, with the calibrated
                error model (joint offsets + distortion, tilted axes, tool
                frame offset) applied
        """
        p = self.params if params is None else np.asarray(params)
        theta = np.asarray(theta, dtype=float).reshape(6)
        axes = np.array([
            axis.vec.ravel() for axis in self.AXES
        ], dtype=float)

        for k in range(1, 6):
            perpendicular = [j for j in range(3) if j != k % 3]

            tilt = np.zeros(3)
            start = 6 + 2 * (k - 1)
            tilt[perpendicular] = p[start:start + 2]

            axes[k] = exp_so3(tilt) @ axes[k]

        scales = np.maximum(
            np.abs(self.theta_min),
            np.abs(self.theta_max),
        )
        normalized = theta / scales
        coefficients = p[22:].reshape(6, self.n_poly)

        correction = self.joint_offsets.copy()
        for degree in range(1, self.n_poly + 1):
            correction += (
                coefficients[:, degree - 1] * normalized ** degree
            )

        theta_corrected = theta + correction
        position = axes[:3].T @ theta_corrected[:3]
        F_cartesian = Frame(Rot(), vct3(position))
        R_wrist = np.eye(3)
        for k in range(3, 6):
            R_wrist = R_wrist @ exp_so3(
                axes[k] * theta_corrected[k]
            )

        F_wrist = Frame(Rot(matrix=R_wrist), vct3())
        T_tool = exp_se3(np.asarray(p[16:22], dtype=float))
        F_tool = self.F_WE * Frame(
            Rot(matrix=T_tool[:3, :3]),
            vct3(T_tool[:3, 3]),
        )

        return F_cartesian * self.F_CW * F_wrist * F_tool

    def predict_in_tracker(self, theta, params: Optional[np.ndarray] = None) -> Frame:
        """The end-effector pose in the tracker frame: base_frame * fwd_kins."""
        return self.base_frame(params) * self.fwd_kins(theta, params)

    def inv_kins(self, F: Frame, seed=None, tol: float = 1e-10, max_iter: int = 50) -> np.ndarray:
        """
        ImprovedInvKins(F) -- provided.

        There is no closed form once the axes are tilted, so this iterates
        on your fwd_kins: both nominal wrist branches (Robot.inv_kins'
        closed form) are refined with damped Gauss-Newton / Levenberg-
        Marquardt steps on the pose error [alpha; epsilon] of
        fwd_kins(theta)^-1 F, and the in-limit solution nearest `seed`
        (default: the current command) is returned. Works as soon as your
        fwd_kins does.

        input: F - desired F_ee relative to the robot base frame
        output: theta with fwd_kins(theta) ~= F; raises ValueError if no
                branch satisfies the joint limits
        """
        seed = self.theta_cmd if seed is None else np.asarray(seed, dtype=np.float64).reshape(6)
        p_w = F.p - F.R * self.F_WE.p - self.F_CW.p   # nominal wrist center -> seeds for the linear joints
        step_max = np.array([50.0] * 3 + [0.3] * 3)
        h = np.array([1e-3] * 3 + [1e-6] * 3)

        def pose_delta(theta):
            dF = self.fwd_kins(theta).inv() * F
            return np.concatenate([log_so3(dF.R.matrix), dF.p.vec.ravel()])

        candidates = []
        for wrist in self.wrist_solutions(F.R):
            theta = np.concatenate([[p_w.x, p_w.y, p_w.z], wrist])
            lam, delta = 1e-4, pose_delta(theta)
            for _ in range(max_iter):
                cost = float(delta @ delta)
                if np.sqrt(cost) < tol:
                    break
                J = np.zeros((6, 6))
                for k in range(6):
                    th = theta.copy()
                    th[k] += h[k]
                    J[:, k] = (pose_delta(th) - delta) / h[k]
                while True:  # Levenberg-Marquardt: shrink the step until the cost decreases
                    JtJ = J.T @ J
                    step = -np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), J.T @ delta)
                    trial = theta + np.clip(step, -step_max, step_max)
                    delta_trial = pose_delta(trial)
                    if float(delta_trial @ delta_trial) < cost or lam > 1e6:
                        theta, delta, lam = trial, delta_trial, max(lam / 10.0, 1e-12)
                        break
                    lam *= 10.0
            theta[3] = (theta[3] + np.pi) % (2 * np.pi) - np.pi
            theta[5] = (theta[5] + np.pi) % (2 * np.pi) - np.pi
            if self.within_limits(theta):
                candidates.append(theta)
        if not candidates:
            raise ValueError(f"inv_kins: no joint solution within limits for {F!r}")
        return min(candidates, key=lambda th: float(np.linalg.norm(th - seed)))

    # ------------------------------------------------------------------ fitting

    def residuals(self, params: np.ndarray, data: CalibrationData) -> np.ndarray:
        """
        TODO: implement.

        input: params - candidate parameter vector; data - CalibrationData
        output: flat residual vector comparing predict_in_tracker(theta_j,
                params) with data.frames[j] for every sample (choose how to
                weigh rotation vs translation)
        """
        residuals = []
        for theta, observed in zip(data.thetas, data.frames):
            predicted = self.predict_in_tracker(theta, params)
            dF = observed.inv() * predicted
            rotation_error = log_so3(dF.R.matrix)
            position_error = dF.p.vec.ravel()
            residuals.extend(rotation_error / 1e-3)
            residuals.extend(position_error / 0.2)
        return np.asarray(residuals, dtype=float)
        

    def fit(self, data: CalibrationData, **kwargs):
        """
        TODO: implement.

        Non-linear least squares over the parameters (scipy.optimize is
        allowed). Must set self.params. Return whatever diagnostics you want
        to report (final cost, iterations, ...).
        """
        fixed_index = 6

        reference = self.params.copy()
        reference[fixed_index] = 0.0

        active = np.ones(reference.size, dtype=bool)
        active[fixed_index] = False

        def unpack(values):
            params = reference.copy()
            params[active] = values
            return params

        def objective(values):
            return self.residuals(unpack(values), data)

        options = {
            "method": "trf",
            "x_scale": "jac",
            "max_nfev": 200,
        }
        options.update(kwargs)

        result = least_squares(
            objective,
            reference[active],
            **options,
        )

        self.params = unpack(result.x)

        return {
            "success": bool(result.success),
            "message": result.message,
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "n_active_params": int(np.sum(active)),
        }

    def jacobian_rank_report(self, data: CalibrationData, params: Optional[np.ndarray] = None) -> dict:
        """
        TODO: implement.

        Numerically differentiate residuals w.r.t. params and report the
        singular values (and which parameter directions are poorly
        determined). Use it to justify your parameterization and your
        choice of calibration poses in the report.
        """
        p = self.params.copy() if params is None else np.asarray(
            params, dtype=float
        ).copy()

        active = np.array([i for i in range(len(p)) if i != 6])

        residual = self.residuals(p, data)
        J = np.zeros((len(residual), len(active)))

        for column, index in enumerate(active):
            step = 1e-6 * max(1.0, abs(p[index]))

            p_plus = p.copy()
            p_minus = p.copy()
            p_plus[index] += step
            p_minus[index] -= step

            J[:, column] = (
                self.residuals(p_plus, data)
                - self.residuals(p_minus, data)
            ) / (2.0 * step)

        column_norms = np.linalg.norm(J, axis=0)
        scales = np.where(column_norms > 0.0, column_norms, 1.0)
        J_scaled = J / scales

        _, singular_values, Vt = np.linalg.svd(
            J_scaled, full_matrices=True
        )
        tolerance = (
            1e-6 * singular_values[0]
            if singular_values.size else 0.0
        )
        rank = int(np.sum(singular_values > tolerance))
        weak_directions = np.zeros((len(active) - rank, len(p)))
        weak_directions[:, active] = Vt[rank:] / scales

        return {
            "rank": rank,
            "n_params": len(active),
            "n_total_params": len(p),
            "active_indices": active,
            "singular_values": singular_values,
            "tolerance": tolerance,
            "weak_parameter_directions": weak_directions,
        }

    # ------------------------------------------------------------------ reporting

    def parameters_report(self) -> Dict[str, np.ndarray]:
        """
        TODO: implement.

        The handout (method 1 (e)) asks your program to report the updated
        kinematic parameters in physical terms, and your debugging runs to
        compare them with ground truth (rob.ground_truth_parameters()).
        Report at least: the axis tilt vectors / unit axes, the angles
        between the axes, the per-joint offsets and scale factors, and the
        tool-frame offset. Note which of these are only defined up to the
        base-frame gauge (and so are better compared as angles between axes).
        """
        p = self.params
        axes = np.array([
            axis.vec.ravel() for axis in self.AXES
        ], dtype=float)

        tilts = np.zeros((6, 3))

        for k in range(1, 6):
            perpendicular = [j for j in range(3) if j != k % 3]

            start = 6 + 2 * (k - 1)
            tilts[k, perpendicular] = p[start:start + 2]

            axes[k] = exp_so3(tilts[k]) @ axes[k]

        cos_angles = np.clip(axes @ axes.T, -1.0, 1.0)
        axis_angles_deg = np.degrees(np.arccos(cos_angles))
        scales = np.maximum(
            np.abs(self.theta_min),
            np.abs(self.theta_max),
        )
        coefficients = p[22:].reshape(6, self.n_poly)
        joint_scale = np.zeros(6)
        if self.n_poly >= 1:
            joint_scale = coefficients[:, 0] / scales

        return {
            "axes": axes,
            "axis_tilts": tilts,
            "axis_angles_deg": axis_angles_deg,
            "joint_offset": self.joint_offsets.copy(),
            "joint_scale": joint_scale,
            "joint_gain": 1.0 + joint_scale,
            "polynomial_coefficients": coefficients.copy(),
            "joint_normalization_scales": scales,
            "F_we_twist": p[16:22].copy(),
            "base_twist": p[:6].copy(),
        }
