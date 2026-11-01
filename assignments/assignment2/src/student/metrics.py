"""
Pose-error metrics used to report calibration accuracy.
"""

from typing import Dict, Sequence, Tuple

import numpy as np

from data_types.nominal_types import Frame
from uncertainty_networks.se3 import log_so3


def frame_error(F_a: Frame, F_b: Frame) -> Tuple[float, float]:
    """
    TODO: implement.

    The mismatch between two frames is defined as
    dF = F_a^-1 F_b ~= Frame(I + sk(alpha), epsilon); report its size.

    input:
        F_a - reference Frame (e.g. ground truth)
        F_b - Frame to compare (e.g. your prediction)
    output:
        (angle_rad, dist_mm) - ||alpha|| (the rotation angle of dF.R) and
            ||epsilon|| (the norm of dF.p)
    """
    dF = F_a.inv() * F_b
    alpha = float(np.linalg.norm(log_so3(dF.R.matrix)))
    epsilon = dF.p.norm()
    return alpha, epsilon

def summarize_errors(errors: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """
    TODO: implement.

    input:
        errors - list of (angle_rad, dist_mm) pairs from frame_error
    output:
        dict with at least the keys
            n, rms_angle, max_angle, rms_pos, max_pos
    """
    values = np.asarray(errors, dtype=float)
    angles = values[:, 0]
    positions = values[:, 1]
    return {
        "n": len(values),
        "rms_angle": float(np.sqrt(np.mean(angles**2))),
        "max_angle": float(np.max(angles)),
        "rms_pos": float(np.sqrt(np.mean(positions**2))),
        "max_pos": float(np.max(positions))
    }
