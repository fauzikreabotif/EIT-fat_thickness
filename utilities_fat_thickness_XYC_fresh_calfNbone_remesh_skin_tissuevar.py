# -*- coding: utf-8 -*-
"""
Utilities for synthetic multi-frequency calf fat-thickness EIT dataset
with an explicit skin layer.

Input:
    [208 measurement pairs, 2 real/imaginary parts, n_frequencies]

Outputs:
    X : [208 measurement pairs, 2 real/imaginary parts, n_frequencies]
    Y : 16 effective subcutaneous fat-thickness values in millimeters
        (skin thickness is NOT included in the target)
    C : one calf-circumference value used as an auxiliary model input

Geometry:
    calf surface -> skin -> subcutaneous fat -> muscle
                                  + tibia/fibula bone overrides

By default the skin electrical properties use the ``SkinWet`` row in the
dielectric-property table, which matches gel/conductive skin preparation.
"""

import copy
import glob
import os
import time

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from matplotlib.path import Path
from scipy.interpolate import interp1d

try:
    import triangle as tr
except Exception:
    tr = None

try:
    from pyeit.eit.fem import EITForward
    import pyeit.eit.protocol as protocol
except Exception:
    EITForward = None
    protocol = None

EPS0 = 8.8541878128e-12

# =========================================================
# re-order the calf boundary so it start same with the first electrode
# =========================================================

def reorder_boundary_from_point(boundary, first_point):
    """
    Circularly reorder a closed polygon so boundary[0] is the vertex
    nearest to first_point.

    The polygon direction is preserved.
    """
    boundary = np.asarray(boundary, dtype=float)
    first_point = np.asarray(first_point, dtype=float)

    distances = np.linalg.norm(
        boundary - first_point,
        axis=1,
    )

    start_index = int(np.argmin(distances))

    reordered = np.roll(
        boundary,
        shift=-start_index,
        axis=0,
    )

    return reordered, start_index

# =========================================================
# Boundary / fat-layer geometry
# =========================================================
def cumulative_boundary_length(boundary):
    boundary = np.asarray(boundary)
    closed = np.vstack([boundary, boundary[0]])
    d = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))
    s = np.insert(np.cumsum(d), 0, 0.0)
    return closed, s


def polygon_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def cross_2d(a, b):
    """2D scalar cross product."""
    return a[0] * b[1] - a[1] * b[0]


def ray_segment_intersection_distance(
    ray_origin,
    ray_direction,
    segment_start,
    segment_end,
    eps=1e-12,
):
    """
    Find the distance t from a ray origin to its intersection
    with a finite line segment.
    Ray:
        p = ray_origin + t * ray_direction, t >= 0
    Segment:
        q = segment_start + u * (segment_end - segment_start),
        0 <= u <= 1
    Returns
    -------
    float or None
        Distance along the normalized ray direction.
    """
    p = np.asarray(ray_origin, dtype=float)
    r = np.asarray(ray_direction, dtype=float)
    q = np.asarray(segment_start, dtype=float)
    s = np.asarray(segment_end, dtype=float) - q
    denominator = cross_2d(r, s)
    if abs(denominator) < eps:
        return None
    q_minus_p = q - p
    t = cross_2d(q_minus_p, s) / denominator
    u = cross_2d(q_minus_p, r) / denominator
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return float(t)
    return None


def distance_to_polygon_along_ray(
    ray_origin,
    ray_direction,
    polygon,
):
    """
    Return the nearest forward intersection distance between
    a ray and a closed polygon.
    """
    polygon = np.asarray(polygon, dtype=float)
    if len(polygon) < 3:
        return np.inf
    direction = np.asarray(ray_direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.inf
    direction = direction / norm
    distances = []
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        distance = ray_segment_intersection_distance(
            ray_origin=ray_origin,
            ray_direction=direction,
            segment_start=a,
            segment_end=b,
        )
        if distance is not None and distance > 1e-10:
            distances.append(distance)
    if len(distances) == 0:
        return np.inf
    return min(distances)


def maximum_thickness_before_bone(
    outer_point,
    inward_normal,
    tibia_boundary,
    fibula_boundary,
    safety_margin_mesh=0.0,
):
    """
    Maximum inward thickness before reaching either tibia or fibula.
    """
    tibia_distance = distance_to_polygon_along_ray(
        ray_origin=outer_point,
        ray_direction=inward_normal,
        polygon=tibia_boundary,
    )
    fibula_distance = distance_to_polygon_along_ray(
        ray_origin=outer_point,
        ray_direction=inward_normal,
        polygon=fibula_boundary,
    )
    nearest_bone_distance = min(tibia_distance, fibula_distance)
    if not np.isfinite(nearest_bone_distance):
        return np.inf
    return max(0.0, nearest_bone_distance - safety_margin_mesh)

def create_skin_fat_muscle_boundary_polygons(
    calf_boundary,
    electrode_xy,
    skin_thickness_mm,
    fat_thickness_mm,
    circumference_mm,
    tibia_boundary=None,
    fibula_boundary=None,
    bone_margin_mm=0.0,
):
    """
    Create explicit skin-fat and fat-muscle boundaries.

    The requested depth from the calf surface is

        skin-fat boundary   : skin thickness
        fat-muscle boundary : skin thickness + fat thickness

    Bone clipping is applied to the total inward depth.  Skin is preserved
    first, and fat is reduced when the available depth before bone is too small.

    Parameters
    ----------
    calf_boundary : ndarray, shape [N, 2]
        Outer calf surface.
    electrode_xy : ndarray, shape [n_electrodes, 2]
        Electrode coordinates around the calf surface.
    skin_thickness_mm : ndarray, shape [n_electrodes]
        Requested local skin thickness.
    fat_thickness_mm : ndarray, shape [n_electrodes]
        Requested local subcutaneous-fat thickness.  This excludes skin.
    circumference_mm : float
        Physical calf circumference.
    tibia_boundary, fibula_boundary : ndarray or None
        Bone polygons.
    bone_margin_mm : float
        Safety margin between the fat-muscle boundary and bone.

    Returns
    -------
    skin_fat_boundary : ndarray [N, 2]
    fat_muscle_boundary : ndarray [N, 2]
    us_outer_points : ndarray [n_electrodes, 2]
    us_skin_inner_points : ndarray [n_electrodes, 2]
    us_inner_points : ndarray [n_electrodes, 2]
        Fat-muscle boundary points along electrode/US directions.
    effective_skin_thickness_mesh : ndarray [n_electrodes]
    effective_fat_thickness_mesh : ndarray [n_electrodes]
    mm_per_mesh_unit : float
    effective_skin_thickness_mm : ndarray [n_electrodes]
    effective_fat_thickness_mm : ndarray [n_electrodes]
    """
    calf_boundary = np.asarray(calf_boundary, dtype=float)
    electrode_xy = np.asarray(electrode_xy, dtype=float)
    requested_skin_mm = np.asarray(skin_thickness_mm, dtype=float)
    requested_fat_mm = np.asarray(fat_thickness_mm, dtype=float)

    expected_shape = (len(electrode_xy),)
    if requested_skin_mm.shape != expected_shape:
        raise ValueError(
            "skin_thickness_mm must have one value for each electrode. "
            f"Received {requested_skin_mm.shape}, expected {expected_shape}."
        )
    if requested_fat_mm.shape != expected_shape:
        raise ValueError(
            "fat_thickness_mm must have one value for each electrode. "
            f"Received {requested_fat_mm.shape}, expected {expected_shape}."
        )
    if np.any(requested_skin_mm < 0.0) or np.any(requested_fat_mm < 0.0):
        raise ValueError("Skin and fat thickness values must be non-negative.")

    # 1. Convert physical thicknesses to mesh units.
    closed, s = cumulative_boundary_length(calf_boundary)
    mesh_circumference = float(s[-1])
    mm_per_mesh_unit = float(circumference_mm) / mesh_circumference
    requested_skin_mesh = requested_skin_mm / mm_per_mesh_unit
    requested_fat_mesh = requested_fat_mm / mm_per_mesh_unit
    bone_margin_mesh = float(bone_margin_mm) / mm_per_mesh_unit

    # 2. Locate electrodes on the polygon arc length.
    el_s, boundary_idx = [], []
    for point in electrode_xy:
        distance = np.linalg.norm(calf_boundary - point, axis=1)
        index = int(np.argmin(distance))
        el_s.append(s[index])
        boundary_idx.append(index)
    el_s = np.asarray(el_s, dtype=float)
    boundary_idx = np.asarray(boundary_idx, dtype=int)

    # 3. Periodically interpolate both requested profiles around the calf.
    order = np.argsort(el_s)
    el_s_sorted = el_s[order]
    el_s_extended = np.r_[el_s_sorted, el_s_sorted[0] + mesh_circumference]

    skin_sorted = requested_skin_mesh[order]
    fat_sorted = requested_fat_mesh[order]
    skin_extended = np.r_[skin_sorted, skin_sorted[0]]
    fat_extended = np.r_[fat_sorted, fat_sorted[0]]

    skin_interp = interp1d(
        el_s_extended, skin_extended, kind="linear", fill_value="extrapolate"
    )
    fat_interp = interp1d(
        el_s_extended, fat_extended, kind="linear", fill_value="extrapolate"
    )
    requested_skin_local = np.asarray(skin_interp(s[:-1]), dtype=float)
    requested_fat_local = np.asarray(fat_interp(s[:-1]), dtype=float)

    # 4. Calculate inward unit normals.
    number_of_points = len(calf_boundary)
    area = polygon_area(calf_boundary)
    inward_normals = np.zeros_like(calf_boundary, dtype=float)

    for i in range(number_of_points):
        previous_point = calf_boundary[(i - 1) % number_of_points]
        next_point = calf_boundary[(i + 1) % number_of_points]
        tangent = next_point - previous_point
        tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
        normal_left = np.array([-tangent[1], tangent[0]])
        normal_right = np.array([tangent[1], -tangent[0]])
        inward_normals[i] = normal_left if area > 0 else normal_right

    # 5. Clip total depth before bone.  Preserve skin first, then fat.
    effective_skin_local = np.maximum(requested_skin_local.copy(), 0.0)
    effective_fat_local = np.maximum(requested_fat_local.copy(), 0.0)

    use_bone_limit = tibia_boundary is not None or fibula_boundary is not None
    if use_bone_limit:
        tibia_boundary = (
            np.empty((0, 2), dtype=float)
            if tibia_boundary is None else np.asarray(tibia_boundary, dtype=float)
        )
        fibula_boundary = (
            np.empty((0, 2), dtype=float)
            if fibula_boundary is None else np.asarray(fibula_boundary, dtype=float)
        )

        for i in range(number_of_points):
            maximum_allowed = maximum_thickness_before_bone(
                outer_point=calf_boundary[i],
                inward_normal=inward_normals[i],
                tibia_boundary=tibia_boundary,
                fibula_boundary=fibula_boundary,
                safety_margin_mesh=bone_margin_mesh,
            )
            if np.isfinite(maximum_allowed):
                effective_skin_local[i] = min(
                    effective_skin_local[i], maximum_allowed
                )
                available_for_fat = max(
                    0.0, maximum_allowed - effective_skin_local[i]
                )
                effective_fat_local[i] = min(
                    effective_fat_local[i], available_for_fat
                )

    # 6. Create the two nested tissue boundaries.
    skin_fat_boundary = (
        calf_boundary + inward_normals * effective_skin_local[:, None]
    )
    total_effective_local = effective_skin_local + effective_fat_local
    fat_muscle_boundary = (
        calf_boundary + inward_normals * total_effective_local[:, None]
    )

    # 7. Effective values at electrode / ultrasound locations.
    us_outer_points = calf_boundary[boundary_idx]
    effective_skin_thickness_mesh = effective_skin_local[boundary_idx]
    effective_fat_thickness_mesh = effective_fat_local[boundary_idx]

    us_skin_inner_points = (
        us_outer_points
        + inward_normals[boundary_idx]
        * effective_skin_thickness_mesh[:, None]
    )
    us_inner_points = (
        us_outer_points
        + inward_normals[boundary_idx]
        * (effective_skin_thickness_mesh + effective_fat_thickness_mesh)[:, None]
    )

    effective_skin_thickness_mm = (
        effective_skin_thickness_mesh * mm_per_mesh_unit
    )
    effective_fat_thickness_mm = (
        effective_fat_thickness_mesh * mm_per_mesh_unit
    )

    return (
        skin_fat_boundary,
        fat_muscle_boundary,
        us_outer_points,
        us_skin_inner_points,
        us_inner_points,
        effective_skin_thickness_mesh,
        effective_fat_thickness_mesh,
        mm_per_mesh_unit,
        effective_skin_thickness_mm,
        effective_fat_thickness_mm,
    )


# Backward-compatible wrapper for older scripts that do not request skin.
def create_fat_muscle_boundary_polygon(
    calf_boundary,
    electrode_xy,
    fat_thickness_mm,
    circumference_mm,
    tibia_boundary=None,
    fibula_boundary=None,
    bone_margin_mm=0.0,
):
    """
    Legacy wrapper.  Creates a zero-thickness skin layer and returns the
    original six outputs expected by older code.
    """
    zero_skin = np.zeros(len(electrode_xy), dtype=float)
    (
        _skin_fat_boundary,
        fat_muscle_boundary,
        us_outer_points,
        _us_skin_inner_points,
        us_inner_points,
        _skin_mesh,
        fat_mesh,
        mm_per_mesh_unit,
        _skin_mm,
        fat_mm,
    ) = create_skin_fat_muscle_boundary_polygons(
        calf_boundary=calf_boundary,
        electrode_xy=electrode_xy,
        skin_thickness_mm=zero_skin,
        fat_thickness_mm=fat_thickness_mm,
        circumference_mm=circumference_mm,
        tibia_boundary=tibia_boundary,
        fibula_boundary=fibula_boundary,
        bone_margin_mm=bone_margin_mm,
    )
    return (
        fat_muscle_boundary,
        us_outer_points,
        us_inner_points,
        fat_mesh,
        mm_per_mesh_unit,
        fat_mm,
    )


# =========================================================
# Protocol and tissue dielectric table
# =========================================================
def load_tissue_table(excel_path):
    """
    Read table formatted as:
    row 0: Hz, SkinDry, NaN, SkinWet, NaN, fat, NaN, muscle, NaN, ...
    row 1: NaN, cond, eps, cond, eps, ...
    """
    raw = pd.read_excel(excel_path, header=None)
    tissues = {}
    freq = raw.iloc[2:, 0].astype(float).to_numpy()

    for col in range(1, raw.shape[1] - 1, 2):
        tissue_name = raw.iloc[0, col]
        if pd.isna(tissue_name):
            continue
        cond = raw.iloc[2:, col].astype(float).to_numpy()
        eps_r = raw.iloc[2:, col + 1].astype(float).to_numpy()
        tissues[str(tissue_name).strip().lower()] = {
            "freq": freq,
            "cond": cond,
            "eps": eps_r,
        }
    return tissues


def interp_tissue_property(tissue_table, tissue, freq_hz):
    tissue = tissue.lower()
    if tissue not in tissue_table:
        raise KeyError(f"Tissue '{tissue}' not found. Available: {list(tissue_table.keys())}")
    tab = tissue_table[tissue]
    cond = np.interp(freq_hz, tab["freq"], tab["cond"])
    eps_r = np.interp(freq_hz, tab["freq"], tab["eps"])
    return cond, eps_r


def complex_admittivity(cond_s_per_m, eps_r, freq_hz):
    """Complex admittivity: sigma* = sigma + j omega epsilon0 epsilon_r."""
    return cond_s_per_m + 1j * 2.0 * np.pi * freq_hz * EPS0 * eps_r


def load_quasi_protocol_csv(csv_path, n_el=16):
    """
    Create a pyEIT protocol object from csv columns: hc, lc, hp, lp.
    The csv may have 208 rows: each row is one measurement pair for one injection pair.
    """
    if protocol is None:
        raise ImportError("pyeit is required for load_quasi_protocol_csv().")

    df = pd.read_csv(csv_path)
    required = {"hc", "lc", "hp", "lp"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}. Found: {df.columns.tolist()}")

    groups = list(df.groupby(["hc", "lc"], sort=False))
    ex_mat = np.array([list(k) for k, _ in groups], dtype=int)
    meas_list = [g[["hp", "lp"]].to_numpy(dtype=int) for _, g in groups]

    n_meas_each = [m.shape[0] for m in meas_list]
    if len(set(n_meas_each)) != 1:
        raise ValueError(f"Each excitation must have same number of voltage pairs. Got {n_meas_each}")

    meas_mat = np.stack(meas_list, axis=0)  # [n_exc, n_meas_per_exc, 2]
    keep_ba = np.ones(meas_mat.shape[:2], dtype=bool)

    # Try modern pyEIT Protocol dataclass first.
    try:
        return protocol.PyEITProtocol(ex_mat=ex_mat, meas_mat=meas_mat, keep_ba=keep_ba)
    except Exception:
        # Fallback: create default object then replace matrices.
        p = protocol.create(n_el=n_el, dist_exc=1, step_meas=1, parser_meas="std")
        p.ex_mat = ex_mat
        p.meas_mat = meas_mat
        p.keep_ba = keep_ba
        return p


# =========================================================
# Mesh tissue assignment
# =========================================================
def get_nodes_elements(mesh_obj):
    nodes = np.asarray(mesh_obj.node)
    elements = np.asarray(mesh_obj.element)
    return nodes, elements


def element_centers(mesh_obj):
    nodes, elements = get_nodes_elements(mesh_obj)
    return nodes[elements, :2].mean(axis=1)


def points_in_poly(points, poly):
    return Path(np.asarray(poly)).contains_points(points)


def assign_calf_tissue_admittivity(
    mesh_obj,
    calf_boundary,
    skin_fat_boundary,
    fat_muscle_boundary,
    tibia_boundary,
    fibula_boundary,
    tissue_table,
    freq_hz,
    skin_tissue="skinwet",
    fat_tissue="fat",
    muscle_tissue="muscle",
    bone_tissue="bonecortical",
):
    """
    Assign an explicit layered calf model:

        calf surface -> skin -> fat -> muscle

    Tibia/fibula override the soft-tissue labels as bone.
    """
    centers = element_centers(mesh_obj)
    n_elem = centers.shape[0]

    in_calf = points_in_poly(centers, calf_boundary)
    in_skin_inner = points_in_poly(centers, skin_fat_boundary)
    in_muscle = points_in_poly(centers, fat_muscle_boundary)
    in_bone = (
        points_in_poly(centers, tibia_boundary)
        | points_in_poly(centers, fibula_boundary)
    )

    labels = np.full(n_elem, "outside", dtype=object)
    labels[in_calf] = skin_tissue.lower()
    labels[in_skin_inner & in_calf] = fat_tissue.lower()
    labels[in_muscle & in_calf] = muscle_tissue.lower()
    labels[in_bone] = bone_tissue.lower()

    # Meshes are expected to contain only the calf interior.  If a numerical
    # centroid falls outside the polygon, assign skin properties as a safe
    # fallback instead of requesting a nonexistent "outside" tissue row.
    labels[labels == "outside"] = skin_tissue.lower()

    adm = np.zeros(n_elem, dtype=np.complex128)
    for tissue in np.unique(labels):
        cond, eps_r = interp_tissue_property(tissue_table, tissue, freq_hz)
        adm[labels == tissue] = complex_admittivity(cond, eps_r, freq_hz)

    return adm, labels


def set_mesh_perm(mesh_obj, elem_perm):
    mesh_new = copy.deepcopy(mesh_obj)
    mesh_new.perm = np.asarray(elem_perm)
    return mesh_new


def scale_coordinates(points, scale_factor, center):
    """
    Uniformly scale x-y coordinates around a specified center.

    Parameters
    ----------
    points : ndarray, shape [N, 2] or [N, 3]
        Coordinates to scale.
    scale_factor : float
        Uniform scaling factor.
    center : ndarray, shape [2]
        Scaling center in x-y coordinates.

    Returns
    -------
    ndarray
        Copy of ``points`` with scaled x-y coordinates.
        Any additional coordinates, such as z, are unchanged.
    """
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)

    if center.shape != (2,):
        raise ValueError(f"center must have shape (2,), received {center.shape}.")
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError("scale_factor must be a positive finite number.")

    scaled = points.copy()
    scaled[:, :2] = center + scale_factor * (points[:, :2] - center)
    return scaled


def boundary_centroid(boundary):
    """Return a stable x-y center for uniform coordinate scaling."""
    boundary = np.asarray(boundary, dtype=float)
    if boundary.ndim != 2 or boundary.shape[1] != 2 or len(boundary) < 3:
        raise ValueError("boundary must have shape [N, 2] with at least three points.")
    return boundary.mean(axis=0)


def affine_transform_coordinates(points, matrix, center):
    """Apply a 2-D affine linear transform around ``center``."""
    points = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    center = np.asarray(center, dtype=float)

    if matrix.shape != (2, 2):
        raise ValueError("matrix must have shape (2, 2).")
    if center.shape != (2,):
        raise ValueError("center must have shape (2,).")

    transformed = points.copy()
    transformed[:, :2] = center + (points[:, :2] - center) @ matrix.T
    return transformed


def rotate_scale_translate_polygon(poly, scale, angle_deg, shift_xy):
    """Scale and rotate a polygon about its centroid, then translate it."""
    poly = np.asarray(poly, dtype=float)
    center = poly.mean(axis=0)
    angle = np.deg2rad(float(angle_deg))
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle),  np.cos(angle)]], dtype=float)
    transformed = center + float(scale) * (poly - center) @ rotation.T
    return transformed + np.asarray(shift_xy, dtype=float)


def polygon_vertices_inside_polygon(inner_poly, outer_poly):
    """Return True when all inner-polygon vertices lie inside the outer polygon."""
    return bool(np.all(Path(np.asarray(outer_poly)).contains_points(np.asarray(inner_poly))))


def minimum_polygon_distance(poly_a, poly_b):
    """Approximate minimum distance between polygon vertices."""
    poly_a = np.asarray(poly_a, dtype=float)
    poly_b = np.asarray(poly_b, dtype=float)
    distances = np.linalg.norm(poly_a[:, None, :] - poly_b[None, :, :], axis=2)
    return float(np.min(distances))




# =========================================================
# Remeshed calf-template bank
# =========================================================
def interpolate_on_boundary(closed_boundary, cumulative_s, target_s):
    """Interpolate one coordinate at arc-length ``target_s``."""
    total_length = float(cumulative_s[-1])
    target_s = float(target_s) % total_length
    index = np.searchsorted(cumulative_s, target_s, side="right") - 1
    index = int(np.clip(index, 0, len(closed_boundary) - 2))
    s0, s1 = cumulative_s[index], cumulative_s[index + 1]
    p0, p1 = closed_boundary[index], closed_boundary[index + 1]
    if np.isclose(s1, s0):
        return p0.copy()
    weight = (target_s - s0) / (s1 - s0)
    return (1.0 - weight) * p0 + weight * p1


def place_electrodes_by_start_fraction(boundary, start_fraction, n_el=16, mode="counterclockwise"):
    """Place electrodes uniformly along a closed boundary."""
    closed, cumulative_s = cumulative_boundary_length(boundary)
    circumference = float(cumulative_s[-1])
    spacing = circumference / int(n_el)
    start_s = (float(start_fraction) % 1.0) * circumference
    sign = 1.0 if mode == "counterclockwise" else -1.0
    if mode not in ("clockwise", "counterclockwise"):
        raise ValueError("mode must be 'clockwise' or 'counterclockwise'.")
    coordinates = [
        interpolate_on_boundary(closed, cumulative_s, start_s + sign * k * spacing)
        for k in range(int(n_el))
    ]
    return np.asarray(coordinates, dtype=float), circumference


def electrode_xy_to_unique_node_indices(nodes, electrode_xy):
    """Snap electrodes to unique nearest mesh nodes."""
    nodes_xy = np.asarray(nodes, dtype=float)[:, :2]
    electrode_xy = np.asarray(electrode_xy, dtype=float)
    selected = []
    for point in electrode_xy:
        order = np.argsort(np.linalg.norm(nodes_xy - point, axis=1))
        index = next((int(i) for i in order if int(i) not in selected), None)
        if index is None:
            raise RuntimeError("Unable to assign a unique mesh node to every electrode.")
        selected.append(index)
    return np.asarray(selected, dtype=int)


def polygon_segments(poly, start_index=0):
    """Return constrained segments for a closed polygon."""
    count = len(poly)
    return np.asarray(
        [[start_index + i, start_index + (i + 1) % count] for i in range(count)],
        dtype=int,
    )


def remesh_calf_polygon(calf_boundary, max_area=80.0, min_angle=30.0):
    """Create a new constrained triangular mesh of the calf interior."""
    if tr is None:
        raise ImportError("The 'triangle' package is required to generate the mesh bank.")
    calf_boundary = np.asarray(calf_boundary, dtype=float)
    source = {
        "vertices": calf_boundary,
        "segments": polygon_segments(calf_boundary),
    }
    result = tr.triangulate(source, f"pq{float(min_angle):g}a{float(max_area):g}")
    if "vertices" not in result or "triangles" not in result:
        raise RuntimeError("Triangle failed to create a valid mesh.")
    return np.asarray(result["vertices"], dtype=float), np.asarray(result["triangles"], dtype=int)


def smooth_radial_boundary_deformation(
    boundary, rng, aspect_ratio_range=(0.85, 1.15), shear_range=(-0.05, 0.05),
    harmonic_amplitude_range=(0.0, 0.035), harmonics=(2, 3, 4),
):
    """Create one smooth outer-calf shape while preserving polygon ordering."""
    boundary = np.asarray(boundary, dtype=float)
    center = boundary.mean(axis=0)
    aspect = float(rng.uniform(*aspect_ratio_range))
    shear = float(rng.uniform(*shear_range))
    affine = np.array([[np.sqrt(aspect), shear], [0.0, 1.0 / np.sqrt(aspect)]])
    shaped = center + (boundary - center) @ affine.T
    vectors = shaped - center
    radii = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    factors = np.ones(len(shaped), dtype=float)
    amplitudes, phases = [], []
    for harmonic in harmonics:
        amplitude = float(rng.uniform(*harmonic_amplitude_range))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        factors += amplitude * np.sin(float(harmonic) * angles + phase)
        amplitudes.append(amplitude)
        phases.append(phase)
    shaped = center + vectors * factors[:, None]
    return shaped, {
        "aspect_ratio": np.float32(aspect),
        "shear": np.float32(shear),
        "shape_center": center.astype(np.float32),
        "affine_matrix": affine.astype(np.float32),
        "harmonic_amplitudes": np.asarray(amplitudes, dtype=np.float32),
        "harmonic_phases": np.asarray(phases, dtype=np.float32),
        "harmonics": np.asarray(harmonics, dtype=np.int16),
    }


def apply_template_shape_map(points, shape_meta):
    """Apply the same smooth template deformation to internal anatomy."""
    points = np.asarray(points, dtype=float)
    center = np.asarray(shape_meta["shape_center"], dtype=float)
    affine = np.asarray(shape_meta["affine_matrix"], dtype=float)
    transformed = center + (points - center) @ affine.T
    vectors = transformed - center
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    factors = np.ones(len(points), dtype=float)
    for harmonic, amplitude, phase in zip(
        shape_meta["harmonics"],
        shape_meta["harmonic_amplitudes"],
        shape_meta["harmonic_phases"],
    ):
        factors += float(amplitude) * np.sin(float(harmonic) * angles + float(phase))
    return center + vectors * factors[:, None]


def triangle_quality(nodes, elements):
    """Return signed area and a scale-free triangle quality score in [0, 1]."""
    xy = np.asarray(nodes, dtype=float)[:, :2]
    tri_nodes = xy[np.asarray(elements, dtype=int)]
    a = np.linalg.norm(tri_nodes[:, 1] - tri_nodes[:, 0], axis=1)
    b = np.linalg.norm(tri_nodes[:, 2] - tri_nodes[:, 1], axis=1)
    c = np.linalg.norm(tri_nodes[:, 0] - tri_nodes[:, 2], axis=1)
    signed_area = 0.5 * np.cross(tri_nodes[:, 1] - tri_nodes[:, 0], tri_nodes[:, 2] - tri_nodes[:, 0])
    quality = 4.0 * np.sqrt(3.0) * np.abs(signed_area) / (a*a + b*b + c*c + 1e-12)
    return signed_area, quality


def generate_calf_mesh_bank(
    output_dir, calf_boundary, tibia_boundary, fibula_boundary,
    n_templates=100, n_el=16, electrode_start_fraction=0.0,
    electrode_mode="counterclockwise", reference_circumference_mm=360.0,
    max_area=80.0, min_angle=30.0, aspect_ratio_range=(0.85, 1.15),
    shear_range=(-0.05, 0.05), harmonic_amplitude_range=(0.0, 0.035),
    tibia_scale_range=(0.85, 1.15), fibula_scale_range=(0.85, 1.15),
    template_bone_shift_max_mm=5.0, template_bone_rotation_max_deg=10.0,
    minimum_bone_gap_mm=1.0, max_attempts_per_template=100, seed=None,
):
    """Generate and save a bank of independently remeshed calf templates.

    Large outer-shape and baseline bone differences belong here. Later, the
    dataset applies only common sample-level variations to each saved template.
    """
    if n_templates <= 0:
        raise ValueError("n_templates must be positive.")
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    calf_boundary = np.asarray(calf_boundary, dtype=float)
    tibia_boundary = np.asarray(tibia_boundary, dtype=float)
    fibula_boundary = np.asarray(fibula_boundary, dtype=float)
    _, ref_s = cumulative_boundary_length(calf_boundary)
    ref_mm_per_unit = float(reference_circumference_mm) / float(ref_s[-1])
    max_shift_units = float(template_bone_shift_max_mm) / ref_mm_per_unit
    min_gap_units = float(minimum_bone_gap_mm) / ref_mm_per_unit
    saved_paths = []
    for template_index in range(int(n_templates)):
        accepted = False
        for attempt in range(int(max_attempts_per_template)):
            shaped_calf, shape_meta = smooth_radial_boundary_deformation(
                calf_boundary, rng, aspect_ratio_range, shear_range,
                harmonic_amplitude_range, harmonics=(2, 3, 4),
            )
            shaped_tibia = apply_template_shape_map(tibia_boundary, shape_meta)
            shaped_fibula = apply_template_shape_map(fibula_boundary, shape_meta)
            tibia = rotate_scale_translate_polygon(
                shaped_tibia, rng.uniform(*tibia_scale_range),
                rng.uniform(-template_bone_rotation_max_deg, template_bone_rotation_max_deg),
                rng.uniform(-max_shift_units, max_shift_units, size=2),
            )
            fibula = rotate_scale_translate_polygon(
                shaped_fibula, rng.uniform(*fibula_scale_range),
                rng.uniform(-template_bone_rotation_max_deg, template_bone_rotation_max_deg),
                rng.uniform(-max_shift_units, max_shift_units, size=2),
            )
            if not polygon_vertices_inside_polygon(tibia, shaped_calf):
                continue
            if not polygon_vertices_inside_polygon(fibula, shaped_calf):
                continue
            if minimum_polygon_distance(tibia, fibula) < min_gap_units:
                continue
            try:
                nodes, elements = remesh_calf_polygon(shaped_calf, max_area=max_area, min_angle=min_angle)
            except Exception:
                continue
            _, quality = triangle_quality(nodes, elements)
            if len(elements) == 0 or np.min(quality) < 0.05:
                continue
            electrode_xy, mesh_circumference = place_electrodes_by_start_fraction(
                shaped_calf, electrode_start_fraction, n_el=n_el, mode=electrode_mode
            )
            el_pos = electrode_xy_to_unique_node_indices(nodes, electrode_xy)
            path = os.path.join(output_dir, f"calf_mesh_{template_index:03d}.npz")
            np.savez_compressed(
                path, template_id=np.int32(template_index), nodes=nodes, tri=elements,
                calf_boundary=shaped_calf, tibia_boundary=tibia, fibula_boundary=fibula,
                el_pos=el_pos, electrode_xy=electrode_xy,
                reference_circumference_mm=np.float32(reference_circumference_mm),
                mesh_circumference_units=np.float32(mesh_circumference),
                minimum_triangle_quality=np.float32(np.min(quality)),
                generation_attempt=np.int32(attempt + 1), **shape_meta,
            )
            saved_paths.append(path)
            accepted = True
            break
        if not accepted:
            raise RuntimeError(f"Could not generate valid template {template_index} after {max_attempts_per_template} attempts.")
    return saved_paths


def load_calf_mesh_bank(mesh_bank_dir, reference_mesh_obj):
    """Load NPZ templates and reconstruct pyEIT-compatible mesh objects."""
    paths = sorted(glob.glob(os.path.join(mesh_bank_dir, "calf_mesh_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No calf_mesh_*.npz files found in: {mesh_bank_dir}")
    bank = []
    for path in paths:
        data = np.load(path, allow_pickle=False)
        mesh = copy.deepcopy(reference_mesh_obj)
        mesh.node = np.asarray(data["nodes"], dtype=float)
        mesh.element = np.asarray(data["tri"], dtype=int)
        mesh.el_pos = np.asarray(data["el_pos"], dtype=int)
        bank.append({
            "template_id": int(data["template_id"]), "path": path, "mesh": mesh,
            "nodes": mesh.node, "elements": mesh.element, "el_pos": mesh.el_pos,
            "calf_boundary": np.asarray(data["calf_boundary"], dtype=float),
            "tibia_boundary": np.asarray(data["tibia_boundary"], dtype=float),
            "fibula_boundary": np.asarray(data["fibula_boundary"], dtype=float),
            "reference_circumference_mm": float(data["reference_circumference_mm"]),
        })
    return bank


# =========================================================
# Dataset
# =========================================================
class FatSkinThicknessEITDataset(Dataset):
    """
    Synthetic multi-frequency EIT dataset with explicit wet skin,
    random subcutaneous-fat thickness, circumference, outer-calf shape,
    and bone geometry.

    Geometry strategy
    -----------------
    The input mesh is treated as a reusable reference geometry in
    arbitrary image/mesh units. ``reference_circumference_mm`` assigns
    a physical circumference to that reference geometry. For every
    generated sample, a new physical circumference is sampled and all
    x-y coordinates are uniformly scaled using

        scale_factor = sampled_circumference_mm
                       / reference_circumference_mm

    With ``generate_new_mesh_per_sample=True``, node count, element count,
    element connectivity, and electrode node indices may change for every
    sample. The EIT protocol and resulting measurement dimensions remain fixed.

    Input
    -----
    x.shape = [n_measurements, 2, n_frequencies]
    x[:, 0, :] = real voltage
    x[:, 1, :] = imaginary voltage

    Target
    ------
    y.shape = [n_electrodes]
    y is effective subcutaneous-fat thickness in millimeters.
    Skin thickness is simulated but is not included in y.

    Notes
    -----
    Because the node coordinates change between samples, the forward
    solver must be constructed for each sampled circumference. Fixed
    frequency-dependent tissue properties are still cached.
    """

    def __init__(
        self,
        num_samples,
        mesh_obj,
        protocol_obj,
        calf_boundary,
        tibia_boundary,
        fibula_boundary,
        el_pos,
        frequencies,
        tissue_table,
        reference_circumference_mm=360.0,
        circumference_min_mm=None,
        circumference_max_mm=None,
        fat_min_mm=3.0,
        fat_max_mm=10.0,
        skin_min_mm=0.5,
        skin_max_mm=2.5,
        skin_tissue="skinwet",
        bone_margin_mm=0.5,
        fixed_dataset=False,
        noise_std=0.0,  # fraction: 1% -> 0.01
        normalize_circumference=False,
        vary_shape=True,
        aspect_ratio_range=(0.90, 1.10),
        shear_range=(-0.05, 0.05),
        tibia_scale_range=(0.85, 1.15),
        fibula_scale_range=(0.85, 1.15),
        bone_shift_max_mm=5.0,
        bone_rotation_max_deg=10.0,
        minimum_bone_gap_mm=1.0,
        geometry_max_attempts=20,
        mesh_bank_dir=None,
        mesh_bank=None,
        generate_new_mesh_per_sample=False,
        electrode_start_fraction=0.0,
        electrode_mode="counterclockwise",
        harmonic_amplitude_range=(0.0, 0.03),
        shape_harmonics=(2, 3, 4),
        mesh_max_area=80.0,
        mesh_min_angle=30.0,
        minimum_triangle_quality=0.05,
        seed=None,
    ):
        if EITForward is None:
            raise ImportError("pyEIT is required for FatSkinThicknessEITDataset.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if fat_max_mm <= fat_min_mm:
            raise ValueError("fat_max_mm must be greater than fat_min_mm.")
        if skin_max_mm <= skin_min_mm:
            raise ValueError("skin_max_mm must be greater than skin_min_mm.")
        if skin_min_mm < 0.0:
            raise ValueError("skin_min_mm must be non-negative.")
        if reference_circumference_mm <= 0.0:
            raise ValueError("reference_circumference_mm must be positive.")

        # If no augmentation range is supplied, preserve the old behavior:
        # every sample uses the reference circumference.
        if circumference_min_mm is None:
            circumference_min_mm = reference_circumference_mm
        if circumference_max_mm is None:
            circumference_max_mm = reference_circumference_mm

        if circumference_min_mm <= 0.0 or circumference_max_mm <= 0.0:
            raise ValueError("Circumference values must be positive.")
        if circumference_max_mm < circumference_min_mm:
            raise ValueError(
                "circumference_max_mm must be greater than or equal to "
                "circumference_min_mm."
            )

        self.num_samples = int(num_samples)
        self.mesh_obj = mesh_obj
        self.protocol_obj = protocol_obj

        # Reference geometry in arbitrary image/mesh coordinates.
        self.calf_boundary = np.asarray(calf_boundary, dtype=float)
        self.tibia_boundary = np.asarray(tibia_boundary, dtype=float)
        self.fibula_boundary = np.asarray(fibula_boundary, dtype=float)
        self.el_pos = np.asarray(el_pos, dtype=int)
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.tissue_table = tissue_table

        self.reference_circumference_mm = float(reference_circumference_mm)
        self.circumference_min_mm = float(circumference_min_mm)
        self.circumference_max_mm = float(circumference_max_mm)
        self.fat_min_mm = float(fat_min_mm)
        self.fat_max_mm = float(fat_max_mm)
        self.skin_min_mm = float(skin_min_mm)
        self.skin_max_mm = float(skin_max_mm)
        self.skin_tissue = str(skin_tissue).strip().lower()
        self.bone_margin_mm = float(bone_margin_mm)
        self.fixed_dataset = bool(fixed_dataset)
        self.noise_std = float(noise_std)
        self.normalize_circumference = bool(normalize_circumference)

        self.vary_shape = bool(vary_shape)
        self.aspect_ratio_range = tuple(map(float, aspect_ratio_range))
        self.shear_range = tuple(map(float, shear_range))
        self.tibia_scale_range = tuple(map(float, tibia_scale_range))
        self.fibula_scale_range = tuple(map(float, fibula_scale_range))
        self.bone_shift_max_mm = float(bone_shift_max_mm)
        self.bone_rotation_max_deg = float(bone_rotation_max_deg)
        self.minimum_bone_gap_mm = float(minimum_bone_gap_mm)
        self.geometry_max_attempts = int(geometry_max_attempts)

        # Fresh-remeshing configuration. When enabled, every sample receives
        # a newly deformed calf boundary, newly varied bones, new triangle
        # connectivity, and newly assigned electrode node indices.
        self.generate_new_mesh_per_sample = bool(generate_new_mesh_per_sample)
        self.electrode_start_fraction = float(electrode_start_fraction) % 1.0
        self.electrode_mode = str(electrode_mode)
        self.harmonic_amplitude_range = tuple(map(float, harmonic_amplitude_range))
        self.shape_harmonics = tuple(int(v) for v in shape_harmonics)
        self.mesh_max_area = float(mesh_max_area)
        self.mesh_min_angle = float(mesh_min_angle)
        self.minimum_triangle_quality = float(minimum_triangle_quality)

        # Preserve immutable reference polygons for fresh geometry generation.
        self.reference_calf_boundary = self.calf_boundary.copy()
        self.reference_tibia_boundary = self.tibia_boundary.copy()
        self.reference_fibula_boundary = self.fibula_boundary.copy()
        self.reference_mesh_obj = copy.deepcopy(mesh_obj)

        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")
        if self.geometry_max_attempts <= 0:
            raise ValueError("geometry_max_attempts must be positive.")
        if self.bone_shift_max_mm < 0.0 or self.bone_rotation_max_deg < 0.0:
            raise ValueError("Bone shift and rotation limits must be non-negative.")
        if self.minimum_bone_gap_mm < 0.0:
            raise ValueError("minimum_bone_gap_mm must be non-negative.")
        if self.electrode_mode not in ("clockwise", "counterclockwise"):
            raise ValueError("electrode_mode must be 'clockwise' or 'counterclockwise'.")
        if len(self.harmonic_amplitude_range) != 2 or self.harmonic_amplitude_range[1] < self.harmonic_amplitude_range[0]:
            raise ValueError("harmonic_amplitude_range must be an increasing two-value range.")
        if not self.shape_harmonics or any(v <= 0 for v in self.shape_harmonics):
            raise ValueError("shape_harmonics must contain positive integers.")
        if self.mesh_max_area <= 0.0 or self.mesh_min_angle <= 0.0:
            raise ValueError("mesh_max_area and mesh_min_angle must be positive.")
        if not 0.0 <= self.minimum_triangle_quality <= 1.0:
            raise ValueError("minimum_triangle_quality must be in [0, 1].")

        for name, value_range in (
            ("aspect_ratio_range", self.aspect_ratio_range),
            ("shear_range", self.shear_range),
            ("tibia_scale_range", self.tibia_scale_range),
            ("fibula_scale_range", self.fibula_scale_range),
        ):
            if len(value_range) != 2 or value_range[1] < value_range[0]:
                raise ValueError(f"{name} must be a two-value increasing range.")
        if self.aspect_ratio_range[0] <= 0.0:
            raise ValueError("aspect_ratio_range values must be positive.")
        if self.tibia_scale_range[0] <= 0.0 or self.fibula_scale_range[0] <= 0.0:
            raise ValueError("Bone scale range values must be positive.")

        self.rng = np.random.default_rng(seed)

        # A bank entry may have a different node count and connectivity.
        # When no bank is supplied, preserve the original single-mesh behavior.
        if mesh_bank is not None and mesh_bank_dir is not None:
            raise ValueError("Provide either mesh_bank or mesh_bank_dir, not both.")
        if mesh_bank_dir is not None:
            self.mesh_bank = load_calf_mesh_bank(mesh_bank_dir, reference_mesh_obj=mesh_obj)
        elif mesh_bank is not None:
            self.mesh_bank = list(mesh_bank)
        else:
            nodes, elements = get_nodes_elements(mesh_obj)
            self.mesh_bank = [{
                "template_id": 0,
                "path": None,
                "mesh": copy.deepcopy(mesh_obj),
                "nodes": np.asarray(nodes, dtype=float),
                "elements": np.asarray(elements, dtype=int),
                "el_pos": self.el_pos.copy(),
                "calf_boundary": self.calf_boundary.copy(),
                "tibia_boundary": self.tibia_boundary.copy(),
                "fibula_boundary": self.fibula_boundary.copy(),
                "reference_circumference_mm": self.reference_circumference_mm,
            }]
        if not self.mesh_bank:
            raise ValueError("mesh_bank must contain at least one template.")

        electrode_counts = {len(np.asarray(item["el_pos"])) for item in self.mesh_bank}
        if len(electrode_counts) != 1:
            raise ValueError("All mesh templates must use the same number of electrodes.")
        self.n_electrodes = electrode_counts.pop()
        for item in self.mesh_bank:
            nodes = np.asarray(item["nodes"], dtype=float)
            elements = np.asarray(item["elements"], dtype=int)
            el_template = np.asarray(item["el_pos"], dtype=int)
            if np.any(el_template < 0) or np.any(el_template >= len(nodes)):
                raise ValueError(f"Template {item.get('template_id')} has invalid el_pos.")
            item["nodes"] = nodes
            item["elements"] = elements
            item["el_pos"] = el_template
            item["calf_boundary"] = np.asarray(item["calf_boundary"], dtype=float)
            item["tibia_boundary"] = np.asarray(item["tibia_boundary"], dtype=float)
            item["fibula_boundary"] = np.asarray(item["fibula_boundary"], dtype=float)
            _, template_s = cumulative_boundary_length(item["calf_boundary"])
            item["mesh_circumference_units"] = float(template_s[-1])
            item["reference_circumference_mm"] = float(item.get("reference_circumference_mm", self.reference_circumference_mm))
            item["reference_mm_per_mesh_unit"] = item["reference_circumference_mm"] / item["mesh_circumference_units"]
            item["scale_center"] = boundary_centroid(item["calf_boundary"])

        # Backward-compatible aliases use the first template only.
        first = self.mesh_bank[0]
        self.mesh_obj = first["mesh"]
        self.nodes = first["nodes"]
        self.elements = first["elements"]
        self.el_pos = first["el_pos"]
        self.calf_boundary = first["calf_boundary"]
        self.tibia_boundary = first["tibia_boundary"]
        self.fibula_boundary = first["fibula_boundary"]
        self.reference_mesh_circumference_units = first["mesh_circumference_units"]
        self.reference_mm_per_mesh_unit = first["reference_mm_per_mesh_unit"]
        self.scale_center = first["scale_center"]

        # Fixed frequency-dependent tissue properties.
        self.tissue_admittivity = self._precompute_tissue_admittivity()

        # Determine dimensions after all members are initialized.
        x_sample, y_sample, c_sample = self.generate_one_sample()
        self.input_shape = x_sample.shape
        self.input_dim = int(np.prod(self.input_shape))
        self.output_dim = int(y_sample.shape[0])
        self.circumference_shape = c_sample.shape
        self.circumference_dim = int(np.prod(self.circumference_shape))

        # Small fixed datasets are convenient for validation.
        # Avoid fixed_dataset=True for very large offline datasets because
        # all samples are retained in RAM.
        if self.fixed_dataset:
            X_list, Y_list, C_list = [], [], []

            print("Creating fixed fat-thickness dataset...")
            for sample_index in range(self.num_samples):
                x, y, c = self.generate_one_sample()
                X_list.append(x)
                Y_list.append(y)
                C_list.append(c)

                if (sample_index + 1) % 100 == 0:
                    print(f"Generated {sample_index + 1}/{self.num_samples}")

            self.X = torch.from_numpy(np.stack(X_list).astype(np.float32))
            self.Y = torch.from_numpy(np.stack(Y_list).astype(np.float32))
            self.C = torch.from_numpy(np.stack(C_list).astype(np.float32))

            print("Fixed dataset created")
            print("X shape:", self.X.shape)
            print("Y shape:", self.Y.shape)
            print("C shape:", self.C.shape)

    # -----------------------------------------------------
    # Precomputation
    # -----------------------------------------------------
    def _precompute_tissue_admittivity(self):
        """Precompute complex admittivity for each tissue/frequency."""
        tissue_names = (
            self.skin_tissue,
            "fat",
            "muscle",
            "bonecortical",
        )

        missing = [name for name in tissue_names if name not in self.tissue_table]
        if missing:
            raise KeyError(
                f"Missing tissue rows {missing} in tissue table. "
                f"Available: {list(self.tissue_table.keys())}"
            )

        values_by_tissue = {}
        for tissue_name in tissue_names:
            values = []
            for freq_hz in self.frequencies:
                conductivity, eps_r = interp_tissue_property(
                    tissue_table=self.tissue_table,
                    tissue=tissue_name,
                    freq_hz=freq_hz,
                )
                values.append(
                    complex_admittivity(
                        cond_s_per_m=conductivity,
                        eps_r=eps_r,
                        freq_hz=freq_hz,
                    )
                )
            values_by_tissue[tissue_name] = np.asarray(
                values, dtype=np.complex128
            )

        return values_by_tissue

    # -----------------------------------------------------
    # Random physical parameters
    # -----------------------------------------------------
    def sample_circumference(self):
        """Sample one physical calf circumference in millimeters."""
        if np.isclose(self.circumference_min_mm, self.circumference_max_mm):
            return self.circumference_min_mm

        return float(self.rng.uniform(self.circumference_min_mm, self.circumference_max_mm))

    def encode_circumference(self, circumference_mm):
        """Return circumference as a one-element float32 model input."""
        circumference_mm = float(circumference_mm)
        if not self.normalize_circumference:
            value = circumference_mm
        else:
            denominator = self.circumference_max_mm - self.circumference_min_mm
            value = 0.0 if np.isclose(denominator, 0.0) else (circumference_mm - self.circumference_min_mm) / denominator
        return np.asarray([value], dtype=np.float32)

    def sample_fat_thickness(self, circumference_mm):
        """
        Generate a smooth random fat-thickness profile.
        The maximum allowed fat thickness depends on circumference:
            maximum fat thickness = 0.05 × circumference
        Examples
        --------
        300 mm circumference -> maximum fat = 15 mm
        360 mm circumference -> maximum fat = 18 mm
        400 mm circumference -> maximum fat = 20 mm
        """
        k = np.arange(self.n_electrodes, dtype=float)
        
        # Circumference-dependent maximum.
        # circumference_fat_max = 0.05 * circumference_mm
        # # Never mutate the dataset-wide global limit.
        # sample_fat_max_mm = min(self.fat_max_mm, circumference_fat_max)
        
        sample_fat_max_mm = self.fat_max_mm
        if sample_fat_max_mm <= self.fat_min_mm:
            raise ValueError(
                "The circumference-dependent maximum fat thickness "
                "must be greater than fat_min_mm."
            )
        
        amp1 = self.rng.uniform(0.2, 2.0)
        amp2 = self.rng.uniform(0.0, 1.0)
        margin = amp1 + amp2

        base_low = self.fat_min_mm + margin
        base_high = sample_fat_max_mm - margin

        # Handle unusually narrow requested ranges robustly.
        if base_high < base_low:
            available_half_range = 0.5 * (sample_fat_max_mm - self.fat_min_mm)
            amplitude_sum = amp1 + amp2
            if amplitude_sum > 0.0:
                reduction = 0.95 * available_half_range / amplitude_sum
                amp1 *= reduction
                amp2 *= reduction
            margin = amp1 + amp2
            base_low = self.fat_min_mm + margin
            base_high = sample_fat_max_mm - margin

        if np.isclose(base_low, base_high):
            base = 0.5 * (base_low + base_high)
        else:
            base = self.rng.uniform(base_low, base_high)

        phase1 = self.rng.uniform(0.0, 2.0 * np.pi)
        phase2 = self.rng.uniform(0.0, 2.0 * np.pi)
        fat = (base + 
               amp1 * np.sin(2.0 * np.pi * k / self.n_electrodes + phase1) +
               amp2 * np.sin(4.0 * np.pi * k / self.n_electrodes + phase2)
               )
        fat += self.rng.normal(loc=0.0,scale=0.25,size=self.n_electrodes,)

        return np.clip(fat, self.fat_min_mm, sample_fat_max_mm).astype(np.float32)

    def sample_skin_thickness(self):
        """
        Generate a smooth 16-point skin-thickness profile.

        Skin varies much less than subcutaneous fat, so the same harmonic idea
        is used with smaller amplitudes and smaller local random variation.
        """
        k = np.arange(self.n_electrodes, dtype=float)
        span = self.skin_max_mm - self.skin_min_mm

        # Keep the oscillation safely inside the requested range.
        amp1_max = min(0.30, 0.25 * span)
        amp2_max = min(0.15, 0.125 * span)
        amp1_min = min(0.05, amp1_max)
        amp1 = self.rng.uniform(amp1_min, amp1_max) if amp1_max > 0 else 0.0
        amp2 = self.rng.uniform(0.0, amp2_max) if amp2_max > 0 else 0.0
        margin = amp1 + amp2

        base_low = self.skin_min_mm + margin
        base_high = self.skin_max_mm - margin
        if base_high < base_low:
            available_half_range = 0.5 * span
            amplitude_sum = amp1 + amp2
            if amplitude_sum > 0.0:
                reduction = 0.95 * available_half_range / amplitude_sum
                amp1 *= reduction
                amp2 *= reduction
            margin = amp1 + amp2
            base_low = self.skin_min_mm + margin
            base_high = self.skin_max_mm - margin

        base = (
            0.5 * (base_low + base_high)
            if np.isclose(base_low, base_high)
            else self.rng.uniform(base_low, base_high)
        )
        phase1 = self.rng.uniform(0.0, 2.0 * np.pi)
        phase2 = self.rng.uniform(0.0, 2.0 * np.pi)

        skin = (
            base
            + amp1 * np.sin(2.0 * np.pi * k / self.n_electrodes + phase1)
            + amp2 * np.sin(4.0 * np.pi * k / self.n_electrodes + phase2)
        )

        # Small local variation while keeping the profile smooth overall.
        local_noise_std = min(0.05, 0.05 * span)
        if local_noise_std > 0.0:
            skin += self.rng.normal(
                loc=0.0,
                scale=local_noise_std,
                size=self.n_electrodes,
            )

        return np.clip(
            skin, self.skin_min_mm, self.skin_max_mm
        ).astype(np.float32)

    # -----------------------------------------------------
    # Sample-specific geometry
    # -----------------------------------------------------
    def _sample_shape_matrix(self):
        """Sample a mild global calf-shape deformation matrix."""
        if not self.vary_shape:
            return np.eye(2), 1.0, 0.0

        aspect_ratio = self.rng.uniform(*self.aspect_ratio_range)
        shear = self.rng.uniform(*self.shear_range)

        # Reciprocal y scaling changes aspect ratio while avoiding a large
        # unconstrained area change. Circumference is corrected afterward.
        matrix = np.array([[aspect_ratio, shear],
                           [0.0, 1.0 / aspect_ratio]], dtype=float)
        return matrix, float(aspect_ratio), float(shear)

    def _augment_bones(self, tibia_boundary, fibula_boundary, calf_boundary, mm_per_mesh_unit):
        """Randomly vary bone size, rotation, and position within the calf."""
        mm_per_mesh_unit = float(mm_per_mesh_unit)
        max_shift_mesh = self.bone_shift_max_mm / mm_per_mesh_unit
        minimum_gap_mesh = self.minimum_bone_gap_mm / mm_per_mesh_unit

        original_tibia = np.asarray(tibia_boundary, dtype=float)
        original_fibula = np.asarray(fibula_boundary, dtype=float)

        for _ in range(self.geometry_max_attempts):
            tibia_scale = self.rng.uniform(*self.tibia_scale_range)
            fibula_scale = self.rng.uniform(*self.fibula_scale_range)
            tibia_angle = self.rng.uniform(-self.bone_rotation_max_deg, self.bone_rotation_max_deg)
            fibula_angle = self.rng.uniform(-self.bone_rotation_max_deg, self.bone_rotation_max_deg)
            tibia_shift = self.rng.uniform(-max_shift_mesh, max_shift_mesh, size=2)
            fibula_shift = self.rng.uniform(-max_shift_mesh, max_shift_mesh, size=2)

            tibia = rotate_scale_translate_polygon(original_tibia, tibia_scale, tibia_angle, tibia_shift)
            fibula = rotate_scale_translate_polygon(original_fibula, fibula_scale, fibula_angle, fibula_shift)

            bones_inside = (
                polygon_vertices_inside_polygon(tibia, calf_boundary)
                and polygon_vertices_inside_polygon(fibula, calf_boundary)
            )
            bones_separated = minimum_polygon_distance(tibia, fibula) >= minimum_gap_mesh

            if bones_inside and bones_separated:
                return tibia, fibula, {
                    "tibia_scale": np.float32(tibia_scale),
                    "fibula_scale": np.float32(fibula_scale),
                    "tibia_rotation_deg": np.float32(tibia_angle),
                    "fibula_rotation_deg": np.float32(fibula_angle),
                    "tibia_shift_mm": (tibia_shift * mm_per_mesh_unit).astype(np.float32),
                    "fibula_shift_mm": (fibula_shift * mm_per_mesh_unit).astype(np.float32),
                    "bone_augmentation_fallback": False,
                }

        return original_tibia.copy(), original_fibula.copy(), {
            "tibia_scale": np.float32(1.0),
            "fibula_scale": np.float32(1.0),
            "tibia_rotation_deg": np.float32(0.0),
            "fibula_rotation_deg": np.float32(0.0),
            "tibia_shift_mm": np.zeros(2, dtype=np.float32),
            "fibula_shift_mm": np.zeros(2, dtype=np.float32),
            "bone_augmentation_fallback": True,
        }

    def _generate_fresh_reference_geometry(self):
        """Generate and remesh one new anatomy in reference mesh units.

        The reference calf boundary is assumed to start at electrode 1.
        Increasing boundary arc length must follow the desired E1 -> E2 order
        when ``electrode_mode='counterclockwise'``.
        """
        start_time = time.perf_counter()

        _, ref_s = cumulative_boundary_length(self.reference_calf_boundary)
        reference_mm_per_unit = self.reference_circumference_mm / float(ref_s[-1])
        max_shift_units = self.bone_shift_max_mm / reference_mm_per_unit
        minimum_gap_units = self.minimum_bone_gap_mm / reference_mm_per_unit

        for attempt in range(self.geometry_max_attempts):
            shaped_calf, shape_meta = smooth_radial_boundary_deformation(
                boundary=self.reference_calf_boundary,
                rng=self.rng,
                aspect_ratio_range=self.aspect_ratio_range,
                shear_range=self.shear_range,
                harmonic_amplitude_range=self.harmonic_amplitude_range,
                harmonics=self.shape_harmonics,
            )

            # Internal anatomy follows the same large deformation as the calf.
            shaped_tibia = apply_template_shape_map(
                self.reference_tibia_boundary, shape_meta
            )
            shaped_fibula = apply_template_shape_map(
                self.reference_fibula_boundary, shape_meta
            )

            tibia_scale = float(self.rng.uniform(*self.tibia_scale_range))
            fibula_scale = float(self.rng.uniform(*self.fibula_scale_range))
            tibia_rotation = float(self.rng.uniform(
                -self.bone_rotation_max_deg, self.bone_rotation_max_deg
            ))
            fibula_rotation = float(self.rng.uniform(
                -self.bone_rotation_max_deg, self.bone_rotation_max_deg
            ))
            tibia_shift = self.rng.uniform(
                -max_shift_units, max_shift_units, size=2
            )
            fibula_shift = self.rng.uniform(
                -max_shift_units, max_shift_units, size=2
            )

            tibia = rotate_scale_translate_polygon(
                shaped_tibia, tibia_scale, tibia_rotation, tibia_shift
            )
            fibula = rotate_scale_translate_polygon(
                shaped_fibula, fibula_scale, fibula_rotation, fibula_shift
            )

            if not polygon_vertices_inside_polygon(tibia, shaped_calf):
                continue
            if not polygon_vertices_inside_polygon(fibula, shaped_calf):
                continue
            if minimum_polygon_distance(tibia, fibula) < minimum_gap_units:
                continue

            mesh_start = time.perf_counter()
            try:
                nodes, elements = remesh_calf_polygon(
                    shaped_calf,
                    max_area=self.mesh_max_area,
                    min_angle=self.mesh_min_angle,
                )
            except Exception:
                continue
            mesh_time = time.perf_counter() - mesh_start

            if len(nodes) == 0 or len(elements) == 0:
                continue
            signed_area, quality = triangle_quality(nodes, elements)
            if np.any(np.abs(signed_area) <= 1e-12):
                continue
            minimum_quality = float(np.min(quality))
            if minimum_quality < self.minimum_triangle_quality:
                continue

            electrode_xy, mesh_circumference = place_electrodes_by_start_fraction(
                boundary=shaped_calf,
                start_fraction=self.electrode_start_fraction,
                n_el=self.n_electrodes,
                mode=self.electrode_mode,
            )
            try:
                el_pos = electrode_xy_to_unique_node_indices(nodes, electrode_xy)
            except RuntimeError:
                continue
            if len(np.unique(el_pos)) != self.n_electrodes:
                continue

            mesh = copy.deepcopy(self.reference_mesh_obj)
            mesh.node = np.asarray(nodes, dtype=float)
            mesh.element = np.asarray(elements, dtype=int)
            mesh.el_pos = np.asarray(el_pos, dtype=int)

            total_time = time.perf_counter() - start_time
            return {
                "template_id": -1,
                "path": None,
                "mesh": mesh,
                "nodes": np.asarray(nodes, dtype=float),
                "elements": np.asarray(elements, dtype=int),
                "el_pos": np.asarray(el_pos, dtype=int),
                "calf_boundary": np.asarray(shaped_calf, dtype=float),
                "tibia_boundary": np.asarray(tibia, dtype=float),
                "fibula_boundary": np.asarray(fibula, dtype=float),
                "electrode_xy_ideal": np.asarray(electrode_xy, dtype=float),
                "reference_circumference_mm": self.reference_circumference_mm,
                "mesh_circumference_units": float(mesh_circumference),
                "reference_mm_per_mesh_unit": (
                    self.reference_circumference_mm / float(mesh_circumference)
                ),
                "scale_center": boundary_centroid(shaped_calf),
                "shape_meta": shape_meta,
                "tibia_scale": np.float32(tibia_scale),
                "fibula_scale": np.float32(fibula_scale),
                "tibia_rotation_deg": np.float32(tibia_rotation),
                "fibula_rotation_deg": np.float32(fibula_rotation),
                "tibia_shift_mm": (tibia_shift * reference_mm_per_unit).astype(np.float32),
                "fibula_shift_mm": (fibula_shift * reference_mm_per_unit).astype(np.float32),
                "generation_attempt": np.int32(attempt + 1),
                "mesh_time_seconds": np.float32(mesh_time),
                "geometry_time_seconds": np.float32(total_time),
                "minimum_triangle_quality": np.float32(minimum_quality),
            }

        raise RuntimeError(
            "Could not generate a valid fresh calf geometry after "
            f"{self.geometry_max_attempts} attempts. Reduce the variation "
            "ranges or minimum_triangle_quality."
        )

    def _create_scaled_geometry(self, circumference_mm):
        """Create fresh geometry or select a stored template, then scale it."""
        fresh_remesh = self.generate_new_mesh_per_sample
        if fresh_remesh:
            template = self._generate_fresh_reference_geometry()
            bank_index = -1
        else:
            bank_index = int(self.rng.integers(0, len(self.mesh_bank)))
            template = self.mesh_bank[bank_index]
        base_nodes = template["nodes"]
        base_elements = template["elements"]
        base_calf = template["calf_boundary"]
        base_tibia = template["tibia_boundary"]
        base_fibula = template["fibula_boundary"]
        base_el_pos = template["el_pos"]
        center = template["scale_center"]
        template_reference_mm = template["reference_circumference_mm"]
        template_mm_per_unit = template["reference_mm_per_mesh_unit"]

        # Fresh geometries already include their full calf deformation. Stored
        # templates can optionally receive the older mild within-template affine
        # augmentation controlled by ``vary_shape``.
        if fresh_remesh:
            shape_matrix = np.eye(2, dtype=float)
            shape_meta = template.get("shape_meta", {})
            aspect_ratio = float(shape_meta.get("aspect_ratio", 1.0))
            shear = float(shape_meta.get("shear", 0.0))
            shaped_nodes = np.asarray(base_nodes, dtype=float).copy()
            shaped_calf = np.asarray(base_calf, dtype=float).copy()
            shaped_tibia = np.asarray(base_tibia, dtype=float).copy()
            shaped_fibula = np.asarray(base_fibula, dtype=float).copy()
        else:
            shape_matrix, aspect_ratio, shear = self._sample_shape_matrix()
            shaped_nodes = affine_transform_coordinates(base_nodes, shape_matrix, center)
            shaped_calf = affine_transform_coordinates(base_calf, shape_matrix, center)
            shaped_tibia = affine_transform_coordinates(base_tibia, shape_matrix, center)
            shaped_fibula = affine_transform_coordinates(base_fibula, shape_matrix, center)

        # Match the requested physical circumference exactly.
        _, shaped_s = cumulative_boundary_length(shaped_calf)
        target_mesh_circumference = float(circumference_mm) / template_mm_per_unit
        uniform_scale = target_mesh_circumference / float(shaped_s[-1])
        nodes = scale_coordinates(shaped_nodes, uniform_scale, center)
        calf_boundary = scale_coordinates(shaped_calf, uniform_scale, center)
        tibia_boundary = scale_coordinates(shaped_tibia, uniform_scale, center)
        fibula_boundary = scale_coordinates(shaped_fibula, uniform_scale, center)

        # After circumference scaling, mm per mesh unit changes consistently.
        _, current_s = cumulative_boundary_length(calf_boundary)
        current_mm_per_unit = float(circumference_mm) / float(current_s[-1])
        if fresh_remesh:
            # Bone variability was already applied before remeshing. Do not apply
            # a second independent augmentation to the same sample.
            bone_parameters = {
                "tibia_scale": template["tibia_scale"],
                "fibula_scale": template["fibula_scale"],
                "tibia_rotation_deg": template["tibia_rotation_deg"],
                "fibula_rotation_deg": template["fibula_rotation_deg"],
                "tibia_shift_mm": template["tibia_shift_mm"],
                "fibula_shift_mm": template["fibula_shift_mm"],
                "bone_augmentation_fallback": False,
            }
        else:
            tibia_boundary, fibula_boundary, bone_parameters = self._augment_bones(
                tibia_boundary, fibula_boundary, calf_boundary, current_mm_per_unit
            )

        sample_mesh = copy.deepcopy(template["mesh"])
        sample_mesh.node = nodes
        sample_mesh.element = base_elements.copy()
        sample_mesh.el_pos = base_el_pos.copy()
        if fresh_remesh and "electrode_xy_ideal" in template:
            electrode_xy = scale_coordinates(
                template["electrode_xy_ideal"], uniform_scale, center
            )[:, :2]
        else:
            electrode_xy = nodes[base_el_pos, :2]
        electrode_node_xy = nodes[base_el_pos, :2]
        element_centers = nodes[base_elements, :2].mean(axis=1)
        calculated_circumference_mm = float(current_s[-1]) * current_mm_per_unit

        return {
            "mesh": sample_mesh, "nodes": nodes, "elements": base_elements,
            "el_pos": base_el_pos, "calf_boundary": calf_boundary,
            "tibia_boundary": tibia_boundary, "fibula_boundary": fibula_boundary,
            "electrode_xy": electrode_xy, "electrode_node_xy": electrode_node_xy,
            "element_centers": element_centers,
            "scale_factor": np.float32(uniform_scale), "scale_center": center.copy(),
            "fresh_remesh": bool(fresh_remesh),
            "generation_attempt": np.int32(template.get("generation_attempt", 0)),
            "mesh_time_seconds": np.float32(template.get("mesh_time_seconds", 0.0)),
            "geometry_time_seconds": np.float32(template.get("geometry_time_seconds", 0.0)),
            "minimum_triangle_quality": np.float32(template.get("minimum_triangle_quality", np.nan)),
            "calculated_circumference_mm": calculated_circumference_mm,
            "shape_aspect_ratio": np.float32(aspect_ratio),
            "shape_shear": np.float32(shear),
            "shape_matrix": shape_matrix.astype(np.float32),
            "template_bank_index": np.int32(bank_index),
            "template_id": np.int32(template.get("template_id", bank_index)),
            "template_path": template.get("path"),
            "template_reference_circumference_mm": np.float32(template_reference_mm),
            "template_node_count": np.int32(len(base_nodes)),
            "template_element_count": np.int32(len(base_elements)),
            **bone_parameters,
        }

    # -----------------------------------------------------
    # Sample-dependent tissue masks
    # -----------------------------------------------------
    def _build_sample_tissue_masks(
        self,
        element_centers,
        calf_boundary,
        skin_fat_boundary,
        fat_muscle_boundary,
        tibia_boundary,
        fibula_boundary,
    ):
        """Build explicit skin/fat/muscle/bone masks for one sample."""
        in_calf = points_in_poly(element_centers, calf_boundary)
        in_skin_inner = points_in_poly(element_centers, skin_fat_boundary)
        in_muscle_polygon = points_in_poly(
            element_centers, fat_muscle_boundary
        )
        in_tibia = points_in_poly(element_centers, tibia_boundary)
        in_fibula = points_in_poly(element_centers, fibula_boundary)
        in_bone = in_tibia | in_fibula
        in_outside = ~in_calf

        # Ring between outer calf and skin-fat boundary.
        in_skin = in_calf & ~in_skin_inner & ~in_bone

        # Ring between skin-fat and fat-muscle boundaries.
        in_fat = (
            in_skin_inner
            & in_calf
            & ~in_muscle_polygon
            & ~in_bone
        )

        # Interior of the fat-muscle boundary.
        in_muscle = in_muscle_polygon & in_calf & ~in_bone

        return {
            "outside": in_outside,
            "skin": in_skin,
            "fat": in_fat,
            "muscle": in_muscle,
            "bone": in_bone,
        }

    def _build_element_admittivity(
        self,
        tissue_masks,
        frequency_index,
    ):
        """Build element-wise admittivity for one frequency."""
        n_elements = len(tissue_masks["outside"])
        elem_adm = np.empty(n_elements, dtype=np.complex128)

        skin_value = self.tissue_admittivity[self.skin_tissue][frequency_index]

        # Fresh meshes should contain only calf-interior triangles; the
        # outside assignment is retained as a numerical fallback.
        elem_adm[tissue_masks["outside"]] = skin_value
        elem_adm[tissue_masks["skin"]] = skin_value
        elem_adm[tissue_masks["fat"]] = self.tissue_admittivity["fat"][frequency_index]
        elem_adm[tissue_masks["muscle"]] = self.tissue_admittivity["muscle"][frequency_index]
        elem_adm[tissue_masks["bone"]] = self.tissue_admittivity["bonecortical"][frequency_index]
        return elem_adm

    def _create_numeric_tissue_labels(self, tissue_masks):
        """
        Numeric labels:
            0 outside
            1 skin
            2 fat
            3 muscle
            4 bone
        """
        labels = np.zeros(len(tissue_masks["outside"]), dtype=np.uint8)
        labels[tissue_masks["skin"]] = 1
        labels[tissue_masks["fat"]] = 2
        labels[tissue_masks["muscle"]] = 3
        labels[tissue_masks["bone"]] = 4
        return labels

    # -----------------------------------------------------
    # Forward solver
    # -----------------------------------------------------
    def _solve_with_admittivity(
        self,
        forward_solver,
        sample_mesh,
        elem_adm,
    ):
        """Solve one frequency on the sample-specific geometry."""
        try:
            # print("Admittivity dtype :", elem_adm.dtype)
            # print("Is complex        :", np.iscomplexobj(elem_adm))
            # print("Max |Imag|        :", np.max(np.abs(elem_adm.imag)))
            # print("First 5 values    :", elem_adm[:5])
            voltage = forward_solver.solve_eit(perm=elem_adm)
            # print("=" * 60)
            # print("Voltage dtype :", voltage.dtype)
            # print("Is complex    :", np.iscomplexobj(voltage))
            # print("Max |Imag|    :", np.max(np.abs(np.imag(voltage))))
            # print("First 5 values:")
            # print(voltage[:5])
            # print("=" * 60)
        except TypeError as exc:
            if "perm" not in str(exc):
                raise

            mesh_frequency = set_mesh_perm(sample_mesh, elem_adm)
            fallback_fwd = EITForward(mesh_frequency, self.protocol_obj)
            voltage = fallback_fwd.solve_eit()

        return np.asarray(voltage, dtype=np.complex128)

    # -----------------------------------------------------
    # Shared sample generator
    # -----------------------------------------------------
    def _generate_sample(self, return_details=False):
        sampled_circumference_mm = self.sample_circumference()

        requested_skin_mm = self.sample_skin_thickness()
        requested_fat_mm = self.sample_fat_thickness(
            circumference_mm=sampled_circumference_mm
        )

        geometry = self._create_scaled_geometry(
            circumference_mm=sampled_circumference_mm,
        )

        (
            skin_fat_boundary,
            fat_muscle_boundary,
            us_outer_points,
            us_skin_inner_points,
            us_inner_points,
            skin_thickness_mesh,
            fat_thickness_mesh,
            mm_per_mesh_unit,
            effective_skin_mm,
            effective_fat_mm,
        ) = create_skin_fat_muscle_boundary_polygons(
            calf_boundary=geometry["calf_boundary"],
            electrode_xy=geometry["electrode_xy"],
            skin_thickness_mm=requested_skin_mm,
            fat_thickness_mm=requested_fat_mm,
            circumference_mm=sampled_circumference_mm,
            tibia_boundary=geometry["tibia_boundary"],
            fibula_boundary=geometry["fibula_boundary"],
            bone_margin_mm=self.bone_margin_mm,
        )

        tissue_masks = self._build_sample_tissue_masks(
            element_centers=geometry["element_centers"],
            calf_boundary=geometry["calf_boundary"],
            skin_fat_boundary=skin_fat_boundary,
            fat_muscle_boundary=fat_muscle_boundary,
            tibia_boundary=geometry["tibia_boundary"],
            fibula_boundary=geometry["fibula_boundary"],
        )

        # The solver geometry is sample-dependent, so construct one
        # solver per sample and reuse it across all frequencies.
        sample_fwd = EITForward(geometry["mesh"], self.protocol_obj)

        voltage_list = []
        if return_details:
            admittivity_list = []

        for frequency_index, _ in enumerate(self.frequencies):
            elem_adm = self._build_element_admittivity(
                tissue_masks=tissue_masks,
                frequency_index=frequency_index,
            )

            voltage = self._solve_with_admittivity(
                forward_solver=sample_fwd,
                sample_mesh=geometry["mesh"],
                elem_adm=elem_adm,
            )

            if self.noise_std > 0.0:
                noise_scale = self.noise_std * np.abs(voltage)
                real_noise = self.rng.normal(
                    loc=0.0,
                    scale=noise_scale,
                    size=voltage.shape,
                )
                imaginary_noise = self.rng.normal(
                    loc=0.0,
                    scale=noise_scale,
                    size=voltage.shape,
                )
                voltage = voltage + real_noise + 1j * imaginary_noise

            voltage_list.append(voltage)
            if return_details:
                admittivity_list.append(elem_adm)

        complex_voltage = np.stack(voltage_list, axis=-1)
        x = np.stack(
            [complex_voltage.real, complex_voltage.imag], axis=1
        ).astype(np.float32)

        # IMPORTANT: the target is subcutaneous FAT ONLY.
        # Skin thickness is an independent nuisance/anatomical variable.
        y = effective_fat_mm.astype(np.float32)
        c = self.encode_circumference(sampled_circumference_mm)

        if not return_details:
            return x, y, c

        bone_clipped = (
            effective_fat_mm < requested_fat_mm - 1e-6
        ).astype(np.uint8)
        skin_clipped = (
            effective_skin_mm < requested_skin_mm - 1e-6
        ).astype(np.uint8)

        details = {
            "requested_skin_mm": requested_skin_mm,
            "effective_skin_mm": effective_skin_mm,
            "skin_thickness_mm": effective_skin_mm,
            "skin_clipped": skin_clipped,
            "skin_tissue": self.skin_tissue,
            "requested_fat_mm": requested_fat_mm,
            "effective_fat_mm": effective_fat_mm,
            "fat_thickness_mm": effective_fat_mm,
            "bone_clipped": bone_clipped,
            "circumference_mm": np.float32(sampled_circumference_mm),
            "circumference_input": c.copy(),
            "circumference_is_normalized": self.normalize_circumference,
            "reference_circumference_mm": np.float32(
                self.reference_circumference_mm
            ),
            "scale_factor": np.float32(geometry["scale_factor"]),
            "scale_center": geometry["scale_center"],
            "calculated_circumference_mm": np.float32(
                geometry["calculated_circumference_mm"]
            ),
            "template_bank_index": geometry["template_bank_index"],
            "template_id": geometry["template_id"],
            "template_path": geometry["template_path"],
            "template_node_count": geometry["template_node_count"],
            "template_element_count": geometry["template_element_count"],
            "shape_aspect_ratio": geometry["shape_aspect_ratio"],
            "shape_shear": geometry["shape_shear"],
            "shape_matrix": geometry["shape_matrix"],
            "fresh_remesh": geometry["fresh_remesh"],
            "generation_attempt": geometry["generation_attempt"],
            "mesh_time_seconds": geometry["mesh_time_seconds"],
            "geometry_time_seconds": geometry["geometry_time_seconds"],
            "minimum_triangle_quality": geometry["minimum_triangle_quality"],
            "elements": geometry["elements"],
            "el_pos": geometry["el_pos"],
            "electrode_xy": geometry["electrode_xy"],
            "electrode_node_xy": geometry["electrode_node_xy"],
            "tibia_scale": geometry["tibia_scale"],
            "fibula_scale": geometry["fibula_scale"],
            "tibia_rotation_deg": geometry["tibia_rotation_deg"],
            "fibula_rotation_deg": geometry["fibula_rotation_deg"],
            "tibia_shift_mm": geometry["tibia_shift_mm"],
            "fibula_shift_mm": geometry["fibula_shift_mm"],
            "bone_augmentation_fallback": geometry[
                "bone_augmentation_fallback"
            ],
            "scaled_nodes": geometry["nodes"],
            "scaled_calf_boundary": geometry["calf_boundary"],
            "scaled_tibia_boundary": geometry["tibia_boundary"],
            "scaled_fibula_boundary": geometry["fibula_boundary"],
            "skin_fat_boundary": skin_fat_boundary,
            "fat_muscle_boundary": fat_muscle_boundary,
            "us_outer_points": us_outer_points,
            "us_skin_inner_points": us_skin_inner_points,
            "us_inner_points": us_inner_points,
            "skin_thickness_mesh": skin_thickness_mesh,
            "fat_thickness_mesh": fat_thickness_mesh,
            "mm_per_mesh_unit": np.float32(mm_per_mesh_unit),
            "voltage_complex": complex_voltage,
            "frequencies": self.frequencies.copy(),
            "element_admittivity": np.stack(admittivity_list, axis=0),
            "tissue_labels": self._create_numeric_tissue_labels(tissue_masks),
        }

        return x, y, c, details

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------
    def generate_one_sample(self):
        """Generate one sample as ``(X, Y, C)``."""
        return self._generate_sample(return_details=False)

    def generate_one_sample_with_details(self):
        """Generate one sample with geometry and simulation details."""
        return self._generate_sample(return_details=True)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.fixed_dataset:
            return self.X[idx], self.Y[idx], self.C[idx]

        # x, y, c = self.generate_one_sample()
        # return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(c)
        
        x, y, c, detail = self.generate_one_sample_with_details()
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(c), detail


# =========================================================
# HDF5 dataset
# =========================================================

class FatThicknessH5Dataset(Dataset):
    """
    HDF5 dataset loader for calf fat-thickness EIT deep learning.
    
    Features
    --------
    - Lazy HDF5 loading
    - Frequency selection
    - Optional training-time complex Gaussian noise
    - Optional flattened input for MLP
    - Optional requested-fat and bone-clipping outputs
    
    Expected HDF5 datasets
    ----------------------
    X : [N, 208, 2, n_frequencies]
    Y : [N, 16]
    C : [N, 1]
    
    Optional:
    requested_Y : [N, 16]
    bone_clipped : [N, 16]
    """

    def __init__(
        self,
        h5_path: str,
        flatten_input=False,
        frequency_indices=None,
        # ---------------------------------------------
        # Noise augmentation
        # ---------------------------------------------
        apply_noise=False,
        noise_std=0.0,
        random_noise_std=False,
        # Optional outputs
        return_requested=False,
        return_bone_clipped=False,
        
        seed=None,
    ):
        self.h5_path = h5_path
        self.flatten_input = bool(flatten_input)
        self.frequency_indices = frequency_indices
        self.apply_noise = bool(apply_noise)
        self.noise_std = float(noise_std)
        self.random_noise_std = bool(random_noise_std)
        self.return_requested = bool(return_requested)
        self.return_bone_clipped = bool(return_bone_clipped)
        self.seed = seed
        self.rng = None
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")
        # Base seed used to initialize RNG separately
        # inside each DataLoader worker.
        # ---------------------------------------------
        # HDF5 handles
        # ---------------------------------------------
        # IMPORTANT:
        # Each DataLoader worker must open its own file.
        self.h5_file = None
        self.X = None
        self.Y = None
        self.C = None
        self.requested_Y = None
        self.bone_clipped = None
        # ---------------------------------------------
        # Read only metadata here
        # ---------------------------------------------
        with h5py.File(self.h5_path, "r") as h5_file:
            for key in ("X", "Y", "C"):
                if key not in h5_file:
                    raise KeyError(f"'{key}' was not found in {self.h5_path}")

            self.num_samples = h5_file["X"].shape[0]
            if h5_file["Y"].shape[0] != self.num_samples or h5_file["C"].shape[0] != self.num_samples:
                raise ValueError("X, Y, and C must contain the same number of samples.")

            self.input_shape = h5_file["X"].shape[1:]
            self.output_shape = h5_file["Y"].shape[1:]
            self.circumference_shape = h5_file["C"].shape[1:]

            if self.return_requested and "requested_Y" not in h5_file:
                raise KeyError(f"'requested_Y' was requested but not found in {self.h5_path}")
            if self.return_bone_clipped and "bone_clipped" not in h5_file:
                raise KeyError(f"'bone_clipped' was requested but not found in {self.h5_path}")
            
            # ---------------------------------------------
            # Original X shape should be [208, 2, F]
            # ---------------------------------------------
            original_input_shape = tuple(h5_file["X"].shape[1:])
            if len(original_input_shape) != 3:
                raise ValueError("Expected X shape [N, 208, 2, F], got {h5_file['X'].shape}")
            self.original_input_shape = original_input_shape
            # ---------------------------------------------
            # Validate selected frequencies
            # ---------------------------------------------
            if self.frequency_indices is None:
                self.input_shape = original_input_shape
            else:
                frequency_indices = np.asarray(self.frequency_indices, dtype=np.int64,)
                if frequency_indices.ndim != 1:
                    raise ValueError("frequency_indices must be a 1-D sequence.")
                if len(frequency_indices) == 0:
                    raise ValueError("frequency_indices cannot be empty.")
                if frequency_indices.min() < 0:
                    raise ValueError("Frequency indices cannot be negative.")
    
                n_available_frequencies = original_input_shape[2]
                if frequency_indices.max() >= n_available_frequencies:
                    raise ValueError(
                        f"Frequency index "
                        f"{frequency_indices.max()} is outside "
                        f"available range "
                        f"0-{n_available_frequencies - 1}."
                    )
    
                # Store a simple Python list.
                self.frequency_indices = frequency_indices.tolist()
                self.input_shape = (original_input_shape[0], original_input_shape[1], len(self.frequency_indices),)
    
            self.output_shape = tuple(h5_file["Y"].shape[1:])
            self.circumference_shape = tuple(h5_file["C"].shape[1:])
    
            if (self.return_requested and "requested_Y" not in h5_file):
                raise KeyError("'requested_Y' was requested but not found in {self.h5_path}")
    
            if (self.return_bone_clipped and "bone_clipped" not in h5_file):
                raise KeyError("'bone_clipped' was requested but not found in {self.h5_path}")
    # =====================================================
    # Open HDF5
    # =====================================================
    def _open_file(self):
        """Open the HDF5 file on first access in each worker."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, mode="r")
            self.X = self.h5_file["X"]
            self.Y = self.h5_file["Y"]
            self.C = self.h5_file["C"]

            if self.return_requested:
                self.requested_Y = self.h5_file["requested_Y"]
            if self.return_bone_clipped:
                self.bone_clipped = self.h5_file["bone_clipped"]
    # =====================================================
    # Random number generator
    # =====================================================
    def _get_rng(self):
       """
       Create a random-number generator separately
       for each DataLoader worker.

       This avoids different workers accidentally
       producing identical noise sequences.
       """
       if self.rng is None:
           worker_info = torch.utils.data.get_worker_info()
           if worker_info is None:
               # num_workers = 0
               worker_id = 0
           else:
               worker_id = worker_info.id
           if self.seed is None:
               # Allow NumPy to choose entropy automatically.
               self.rng = np.random.default_rng()
           else:
               worker_seed = ( int(self.seed) + int(worker_id) * 100_000)
               self.rng = np.random.default_rng(worker_seed)
       return self.rng

   # =====================================================
   # Noise augmentation
   # =====================================================
    def _add_complex_gaussian_noise(self, x):
       """
       Add relative Gaussian noise to complex EIT voltage.

       Parameters
       ----------
       x : ndarray
           Shape:
               [n_measurements, 2, n_frequencies]
           x[:, 0, :] = real voltage
           x[:, 1, :] = imaginary voltage
       Returns
       -------
       ndarray
           Noisy EIT data with the same shape.
       Notes
       -----
       The noise standard deviation is:
           sigma = noise_fraction * |V|
       where
           |V| = sqrt(real^2 + imag^2)
       Real and imaginary noise are independently sampled.
       """
       if not self.apply_noise:
           return x
       if self.noise_std <= 0.0:
           return x
       rng = self._get_rng()
       # ---------------------------------------------
       # Reconstruct complex voltage
       # ---------------------------------------------
       voltage = (x[:, 0, :].astype(np.float64) + 1j * x[:, 1, :].astype(np.float64))
       # ---------------------------------------------
       # Select noise level
       # ---------------------------------------------
       if self.random_noise_std:
           # Example:
           # noise_std = 0.10
           # randomly choose:
           # 0.00 <= noise_fraction <= 0.10
           noise_fraction = rng.uniform(0.0, self.noise_std,)
       else:
           # Fixed noise
           # Example:
           # noise_std = 0.05
           # -> always 5%
           noise_fraction = self.noise_std
       # ---------------------------------------------
       # Relative noise scale
       # ---------------------------------------------
       noise_scale = (noise_fraction * np.abs(voltage))
       # ---------------------------------------------
       # Independent real noise
       # ---------------------------------------------
       real_noise = rng.normal( loc=0.0, scale=noise_scale, size=voltage.shape,)
       # ---------------------------------------------
       # Independent imaginary noise
       # ---------------------------------------------
       imaginary_noise = rng.normal(loc=0.0, scale=noise_scale, size=voltage.shape,)
       # ---------------------------------------------
       # Add noise
       # ---------------------------------------------
       voltage_noisy = ( voltage + real_noise + 1j * imaginary_noise)
       # ---------------------------------------------
       # Convert back to original X representation
       # ---------------------------------------------
       x_noisy = np.stack([voltage_noisy.real, voltage_noisy.imag,], axis=1,).astype(np.float32)
       return x_noisy.astype(np.float32)

   # =====================================================
   # Dataset
   # =====================================================
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        self._open_file()
        x = np.asarray(self.X[index], dtype=np.float32)
        if self.frequency_indices is not None:
            x = x[:, :, self.frequency_indices]
        # ---------------------------------------------
        # Noise augmentation AFTER frequency selection
        # ---------------------------------------------
        x = self._add_complex_gaussian_noise(x)
        
        # ---------------------------------------------
        # Y and circumference
        # ---------------------------------------------
        y = np.asarray(self.Y[index], dtype=np.float32)
        c = np.asarray(self.C[index], dtype=np.float32).reshape(-1)
        
        if self.flatten_input:
            # ---------------------------------------------
            # Flatten for MLP if requested
            # ---------------------------------------------
            x = x.reshape(-1)

        outputs = [torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(c)]

        if self.return_requested:
            requested_y = np.asarray(self.requested_Y[index], dtype=np.float32)
            outputs.append(torch.from_numpy(requested_y))
        if self.return_bone_clipped:
            clipped = np.asarray(self.bone_clipped[index], dtype=np.uint8)
            outputs.append(torch.from_numpy(clipped))

        return tuple(outputs)

    def close(self):
        """Explicitly close the HDF5 file."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
            self.X = None
            self.Y = None
            self.C = None
            self.requested_Y = None
            self.bone_clipped = None

    def __del__(self):
        self.close()
