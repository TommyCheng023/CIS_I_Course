"""Closest-point queries against the triangle mesh."""

from typing import Tuple

import numpy as np

from data_types.nominal_types import vct3
from ..mesh import Mesh


def closest_point_on_triangle(p: vct3, a: vct3, b: vct3, c: vct3) -> vct3:
    """
    Find the closest point on a triangle to a query point.
    inputs
    ----------
    p: vct3
        Query point.
    a, b, c: vct3
        Vertices of a nondegenerate triangle, in the same coordinate
        system as p.

    return
    -------
    vct3
        Closest point on the triangle's face, edges, or vertices.
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
    Find the closest mesh surface point for each query point.
    inputs
    ----------
    points: np.ndarray, shape (K, 3)
        Query points in CT coordinates.
    mesh: Mesh
        Nonempty triangular mesh in CT coordinates.

    returns
    -------
    closest: np.ndarray, shape (K, 3)
        Closest surface point for each query.
    triangle_indices: np.ndarray, shape (K,)
        Indices of triangles containing the closest points.
    distances: np.ndarray, shape (K,)
        Euclidean distances from queries to their closest points.
    """
    points = np.asarray(points, dtype=float)

    triangles = mesh.tri_vertices
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]

    ab = b - a
    ac = c - a
    normals = np.cross(ab, ac)
    normal_squared = np.einsum("ij,ij->i", normals, normals)

    ab_squared = np.einsum("ij,ij->i", ab, ab)
    ac_squared = np.einsum("ij,ij->i", ac, ac)
    ab_dot_ac = np.einsum("ij,ij->i", ab, ac)

    relative_tolerance = 64.0 * np.finfo(float).eps
    valid_faces = (
        normal_squared > relative_tolerance * ab_squared * ac_squared
    )

    edge_starts = np.stack((a, b, c), axis=1)
    edge_vectors = np.stack((ab, c - b, a - c), axis=1)
    edge_squared = np.einsum(
        "mij,mij->mi", edge_vectors, edge_vectors
    )

    n_queries = len(points)
    closest = np.empty((n_queries, 3), dtype=float)
    triangle_indices = np.empty(n_queries, dtype=int)
    distances = np.empty(n_queries, dtype=float)
    triangle_rows = np.arange(mesh.n_triangles)

    for query_index, point in enumerate(points):
        offsets = point - edge_starts
        numerators = np.einsum(
            "mij,mij->mi", offsets, edge_vectors
        )
        parameters = np.divide(
            numerators,
            edge_squared,
            out=np.zeros_like(numerators),
            where=edge_squared > 0,
        )
        parameters = np.clip(parameters, 0.0, 1.0)

        edge_points = (
            edge_starts + parameters[:, :, None] * edge_vectors
        )
        edge_errors = edge_points - point
        edge_dist_squared = np.einsum(
            "mij,mij->mi", edge_errors, edge_errors
        )

        best_edges = np.argmin(edge_dist_squared, axis=1)
        candidates = edge_points[triangle_rows, best_edges].copy()

        ap = point - a
        plane_parameters = np.divide(
            np.einsum("ij,ij->i", ap, normals),
            normal_squared,
            out=np.zeros(mesh.n_triangles),
            where=valid_faces,
        )
        projections = point - plane_parameters[:, None] * normals

        projected_offsets = projections - a
        dot_ab = np.einsum("ij,ij->i", projected_offsets, ab)
        dot_ac = np.einsum("ij,ij->i", projected_offsets, ac)

        weight_b = np.divide(
            ac_squared * dot_ab - ab_dot_ac * dot_ac,
            normal_squared,
            out=np.zeros(mesh.n_triangles),
            where=valid_faces,
        )
        weight_c = np.divide(
            ab_squared * dot_ac - ab_dot_ac * dot_ab,
            normal_squared,
            out=np.zeros(mesh.n_triangles),
            where=valid_faces,
        )

        inside = (
            valid_faces
            & (weight_b >= 0.0)
            & (weight_c >= 0.0)
            & (weight_b + weight_c <= 1.0)
        )
        candidates[inside] = projections[inside]

        errors = candidates - point
        squared_distances = np.einsum("ij,ij->i", errors, errors)
        best_triangle = int(np.argmin(squared_distances))

        closest[query_index] = candidates[best_triangle]
        triangle_indices[query_index] = best_triangle
        distances[query_index] = np.sqrt(
            squared_distances[best_triangle]
        )

    return closest, triangle_indices, distances
