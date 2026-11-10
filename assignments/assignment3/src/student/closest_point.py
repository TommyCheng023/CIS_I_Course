"""Closest-point queries against the triangle mesh."""

from typing import Tuple

import numpy as np

from data_types.nominal_types import vct3
from ..mesh import Mesh


def closest_point_on_triangle(p: vct3, a: vct3, b: vct3, c: vct3) -> vct3:
    """
    TODO: implement.
    input: query point p, triangle corners a, b, c
    output: the point of the triangle (face, edge or vertex) closest to p
    """
    ab = b - a
    ac = c - a
    ap = p - a
    bp = p - b
    cp = p - c

    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    d3 = ab.dot(bp)
    d4 = ac.dot(bp)
    d5 = ab.dot(cp)
    d6 = ac.dot(cp)

    if (d1 <= 0 and d2 <= 0):
        return a

    if (d3 >= 0 and d4 <= d3):
        return b

    vc = d1*d4 - d2*d3
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        w = d1/(d1 - d3)
        return a + w*ab

    if (d6 >= 0 and d5 <= d6):
        return c

    vb = d5*d2 - d1*d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2/(d2 - d6)
        return a + w*ac

    va = d3*d6 - d5*d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3)/((d4 - d3) + (d5 - d6))
        return b + w*(c - b)

    denom = (va + vb + vc)
    v = vb / denom
    w = vc / denom

    return a + v*ab + w*ac


def closest_points_on_mesh(points: np.ndarray, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TODO: implement (vectorised; a per-triangle Python loop is far too slow).
    input: points (K,3) in CT coordinates, mesh
    output: closest (K,3), triangle index (K,), distance (K,)
    """
    raise NotImplementedError("TODO: implement closest_points_on_mesh")
