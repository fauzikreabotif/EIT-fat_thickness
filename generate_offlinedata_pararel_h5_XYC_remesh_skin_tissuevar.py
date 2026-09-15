# -*- coding: utf-8 -*-
"""
Parallel offline HDF5 generator for the fresh-remeshing calf EIT dataset.

Each generated sample receives:
    1. a newly varied calf boundary,
    2. newly varied tibia and fibula,
    3. a fresh triangular mesh,
    4. newly assigned 16-electrode node indices,
    5. random circumference, skin-thickness, and fat-thickness profiles,
    6. multi-frequency FEM voltages with optional complex Gaussian noise.

The neural-network arrays remain fixed-size:
    X : [N, 208, 2, n_frequency]
    Y : [N, 16]
    C : [N, 1]

Node and triangle arrays vary in length between samples and therefore are not
stored directly in the main HDF5 arrays. Compact geometry metadata, electrode
coordinates, and mesh statistics are retained for quality control.
"""

# Prevent NumPy/SciPy/BLAS from creating many threads inside each worker.
# These environment variables must be set before importing NumPy/SciPy.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing as mp
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import pyeit.mesh as mesh

# Change only this import if your local utility has a different filename.
from utilities_fat_thickness_XYC_fresh_calfNbone_remesh_skin_tissuevar import (
    FatThicknessEITDataset,
    load_quasi_protocol_csv,
    load_tissue_table,
    reorder_boundary_from_point,
)


# =========================================================
# Paths
# =========================================================
ROOT_SAVEDATA = Path( r"H:\temporary\output_fat_thickness\data_noise00_skin04_25_fat04_25_circum3045_fresh_remesh_propvar_10\val")

CALF_NPZ_PATH = Path(
    r"C:\Users\takeiken-c135\OneDrive - 国立大学法人千葉大学\My research\Fat thickness\DL_fatThickness"
    r"\Right_calf\mesh_right_calf.npz")

PROTOCOL_CSV_PATH = Path(
    r"C:\Users\takeiken-c135\OneDrive - 国立大学法人千葉大学\My research\Fat thickness\DL_fatThickness"
    r"\Right_calf\calf_quasi_16_0-15_stim_.csv")

TISSUE_EXCEL_PATH = Path(
    r"C:\Users\takeiken-c135\OneDrive - 国立大学法人千葉大学\My research\Fat thickness\DL_fatThickness"
    r"\Right_calf\permitivity_conductivity.xlsx")


# =========================================================
# EIT and physical settings
# =========================================================
N_EL = 16
N_FREQUENCIES = 10

REFERENCE_CIRCUMFERENCE_MM = 360.0
CIRCUMFERENCE_MIN_MM = 300.0
CIRCUMFERENCE_MAX_MM = 450.0
NORMALIZE_CIRCUMFERENCE = False

FAT_MIN_MM = 0.4
FAT_MAX_MM = 25.0

# Explicit wet-skin layer. The experiment uses gel electrodes and conductive
# skin preparation, so SkinWet is used for the electrical properties.
SKIN_MIN_MM = 0.4
SKIN_MAX_MM = 2.5
SKIN_TISSUE = "skinwet"

# Sample-level tissue-property domain randomization.
# 0.10 means each tissue receives independent conductivity and permittivity
# scale factors sampled uniformly from 0.90 to 1.10. The same factor is used
# across all frequencies for that tissue in one generated sample.
TISSUE_PROPERTY_VARIATION = 0.10
TISSUE_PROPERTY_ORDER = (SKIN_TISSUE, "fat", "muscle", "bonecortical")

BONE_MARGIN_MM = 0.5

# Fraction of voltage magnitude:
# 0.0 = 0%, 0.005 = 0.5%, 0.01 = 1%, 0.02 = 2%.
NOISE_STD = 0.0


# =========================================================
# Fresh geometry and remeshing settings
# =========================================================
GENERATE_NEW_MESH_PER_SAMPLE = True

# The reference calf boundary must start from E1, and increasing boundary
# order must follow E1 -> E2 -> ... -> E16.
ELECTRODE_START_FRACTION = 0.0
ELECTRODE_MODE = "counterclockwise"

VARY_SHAPE = True
ASPECT_RATIO_RANGE = (0.95, 1.05)
SHEAR_RANGE = (-0.04, 0.04)
HARMONIC_AMPLITUDE_RANGE = (0.0, 0.03)
SHAPE_HARMONICS = (2, 3, 4)

TIBIA_SCALE_RANGE = (0.95, 1.05)
FIBULA_SCALE_RANGE = (0.95, 1.05)
BONE_SHIFT_MAX_MM = 3.0
BONE_ROTATION_MAX_DEG = 3.0
MINIMUM_BONE_GAP_MM = 1.0

MESH_MAX_AREA = 80.0
MESH_MIN_ANGLE = 30.0
MINIMUM_TRIANGLE_QUALITY = 0.05
GEOMETRY_MAX_ATTEMPTS = 100


# =========================================================
# Parallel generation settings
# =========================================================
NUMBER_OF_WORKERS = 20
SAMPLES_PER_WORKER = 50
FLUSH_INTERVAL = 100
BASE_SEED = 12345

# False protects completed shards from accidental replacement.
OVERWRITE_EXISTING = True

# Save compact geometry metadata for all dataset splits.
SAVE_GEOMETRY_METADATA = True

# Enable only for the TEST dataset. When True, all variable-length mesh and
# boundary arrays required for plotting are saved in packed HDF5 datasets.
# Keep False for training and validation to reduce storage and I/O overhead.
SAVE_FULL_GEOMETRY = True


# =========================================================
# Validation helpers
# =========================================================
def validate_input_files():
    """Fail early when an input path is incorrect."""
    required = {
        "CALF_NPZ_PATH": CALF_NPZ_PATH,
        "PROTOCOL_CSV_PATH": PROTOCOL_CSV_PATH,
        "TISSUE_EXCEL_PATH": TISSUE_EXCEL_PATH,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following required files were not found:\n  " + "\n  ".join(missing)
        )


def validate_reference_boundary(calf_boundary, nodes, el_pos):
    """
    Warn when calf_boundary[0] is not close to the reference E1 node.

    Fresh-remeshing mode assumes boundary fraction zero corresponds to E1.
    This is a warning rather than an error because the ideal annotated E1
    coordinate can differ slightly from the nearest FEM node.
    """
    calf_boundary = np.asarray(calf_boundary, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    el_pos = np.asarray(el_pos, dtype=int)

    e1_node = nodes[el_pos[0], :2]
    boundary_spacing = np.linalg.norm(
        np.roll(calf_boundary, -1, axis=0) - calf_boundary,
        axis=1,
    )
    typical_spacing = float(np.median(boundary_spacing))
    start_error = float(np.linalg.norm(calf_boundary[0] - e1_node))

    if start_error > max(3.0 * typical_spacing, 1e-8):
        print(
            "WARNING: calf_boundary[0] may not correspond to E1.\n"
            f"  boundary[0] = {calf_boundary[0]}\n"
            f"  reference E1 node = {e1_node}\n"
            f"  distance = {start_error:.4f} mesh units",
            flush=True,
        )


# =========================================================
# Build one independent generator inside each worker
# =========================================================
def build_dataset(num_samples, seed):
    """Construct one independent online dataset generator."""
    data = np.load(CALF_NPZ_PATH, allow_pickle=True)

    nodes = np.asarray(data["nodes"], dtype=float)
    tri = np.asarray(data["tri"], dtype=int)
    el_pos = np.asarray(data["el_pos"], dtype=int)
    calf_boundary = np.asarray(data["calf_boundary"], dtype=float)
    tibia_boundary = np.asarray(data["tibia_boundary"], dtype=float)
    fibula_boundary = np.asarray(data["fibula_boundary"], dtype=float)
    electrode_xy = nodes[el_pos, :2]
    # IMPORTANT !!!!!!!!!
    first_electrode_xy = electrode_xy[0]
    calf_boundary, start_index = reorder_boundary_from_point(
        calf_boundary,
        first_electrode_xy,
    )

    if len(el_pos) != N_EL:
        raise ValueError(f"Expected {N_EL} electrodes, found {len(el_pos)}.")
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError(f"tri must have shape [n_elements, 3], received {tri.shape}.")
    if tri.min() < 0 or tri.max() >= len(nodes):
        raise ValueError("Reference triangle connectivity contains invalid node indices.")

    validate_reference_boundary(calf_boundary, nodes, el_pos)

    # Build a pyEIT mesh object carrying the reference topology. In fresh
    # remeshing mode, the utility uses this object as the template type and
    # constructs a new sample-specific mesh internally.
    mesh_obj = mesh.create(n_el=N_EL, h0=0.1)
    mesh_obj.node = nodes
    mesh_obj.element = tri
    mesh_obj.el_pos = el_pos

    protocol_obj = load_quasi_protocol_csv(str(PROTOCOL_CSV_PATH), n_el=N_EL)
    tissue_table = load_tissue_table(str(TISSUE_EXCEL_PATH))

    available_frequencies = np.asarray(
        tissue_table["fat"]["freq"],
        dtype=np.float64,
    )
    if len(available_frequencies) < N_FREQUENCIES:
        raise ValueError(
            f"Requested {N_FREQUENCIES} frequencies, but the tissue table "
            f"contains only {len(available_frequencies)}."
        )
    frequencies = available_frequencies[:N_FREQUENCIES]

    dataset = FatThicknessEITDataset(
        num_samples=num_samples,
        mesh_obj=mesh_obj,
        protocol_obj=protocol_obj,
        calf_boundary=calf_boundary,
        tibia_boundary=tibia_boundary,
        fibula_boundary=fibula_boundary,
        el_pos=el_pos,
        frequencies=frequencies,
        tissue_table=tissue_table,
        reference_circumference_mm=REFERENCE_CIRCUMFERENCE_MM,
        circumference_min_mm=CIRCUMFERENCE_MIN_MM,
        circumference_max_mm=CIRCUMFERENCE_MAX_MM,
        fat_min_mm=FAT_MIN_MM,
        fat_max_mm=FAT_MAX_MM,
        skin_min_mm=SKIN_MIN_MM,
        skin_max_mm=SKIN_MAX_MM,
        skin_tissue=SKIN_TISSUE,
        tissue_property_variation=TISSUE_PROPERTY_VARIATION,
        bone_margin_mm=BONE_MARGIN_MM,
        fixed_dataset=False,
        noise_std=NOISE_STD,
        normalize_circumference=NORMALIZE_CIRCUMFERENCE,

        # Fresh geometry for every generated sample.
        generate_new_mesh_per_sample=GENERATE_NEW_MESH_PER_SAMPLE,
        electrode_start_fraction=ELECTRODE_START_FRACTION,
        electrode_mode=ELECTRODE_MODE,

        # Calf-shape variability.
        vary_shape=VARY_SHAPE,
        aspect_ratio_range=ASPECT_RATIO_RANGE,
        shear_range=SHEAR_RANGE,
        harmonic_amplitude_range=HARMONIC_AMPLITUDE_RANGE,
        shape_harmonics=SHAPE_HARMONICS,

        # Bone variability before remeshing.
        tibia_scale_range=TIBIA_SCALE_RANGE,
        fibula_scale_range=FIBULA_SCALE_RANGE,
        bone_shift_max_mm=BONE_SHIFT_MAX_MM,
        bone_rotation_max_deg=BONE_ROTATION_MAX_DEG,
        minimum_bone_gap_mm=MINIMUM_BONE_GAP_MM,

        # Triangulation quality controls.
        mesh_max_area=MESH_MAX_AREA,
        mesh_min_angle=MESH_MIN_ANGLE,
        minimum_triangle_quality=MINIMUM_TRIANGLE_QUALITY,
        geometry_max_attempts=GEOMETRY_MAX_ATTEMPTS,
        seed=seed,
    )

    return dataset, frequencies


# =========================================================
# HDF5 creation helpers
# =========================================================
def create_main_datasets(h5_file, num_samples, dataset):
    """Create the fixed-size model arrays."""
    n_measurements, n_components, n_frequencies = dataset.input_shape
    n_outputs = dataset.output_dim
    c_dim = dataset.circumference_dim

    datasets = {}
    datasets["X"] = h5_file.create_dataset(
        "X",
        shape=(num_samples, n_measurements, n_components, n_frequencies),
        dtype=np.float32,
        chunks=(1, n_measurements, n_components, n_frequencies),
        compression="lzf",
    )
    datasets["Y"] = h5_file.create_dataset(
        "Y",
        shape=(num_samples, n_outputs),
        dtype=np.float32,
        chunks=(min(256, num_samples), n_outputs),
        compression="lzf",
    )
    datasets["C"] = h5_file.create_dataset(
        "C",
        shape=(num_samples, c_dim),
        dtype=np.float32,
        chunks=(min(1024, num_samples), c_dim),
        compression="lzf",
    )
    datasets["circumference_mm"] = h5_file.create_dataset(
        "circumference_mm",
        shape=(num_samples, 1),
        dtype=np.float32,
        chunks=(min(1024, num_samples), 1),
        compression="lzf",
    )
    datasets["requested_Y"] = h5_file.create_dataset(
        "requested_Y",
        shape=(num_samples, n_outputs),
        dtype=np.float32,
        chunks=(min(256, num_samples), n_outputs),
        compression="lzf",
    )
    datasets["requested_skin_mm"] = h5_file.create_dataset(
        "requested_skin_mm",
        shape=(num_samples, n_outputs),
        dtype=np.float32,
        chunks=(min(256, num_samples), n_outputs),
        compression="lzf",
    )
    datasets["effective_skin_mm"] = h5_file.create_dataset(
        "effective_skin_mm",
        shape=(num_samples, n_outputs),
        dtype=np.float32,
        chunks=(min(256, num_samples), n_outputs),
        compression="lzf",
    )
    datasets["skin_clipped"] = h5_file.create_dataset(
        "skin_clipped",
        shape=(num_samples, n_outputs),
        dtype=np.uint8,
        chunks=(min(256, num_samples), n_outputs),
        compression="lzf",
    )
    datasets["bone_clipped"] = h5_file.create_dataset(
        "bone_clipped",
        shape=(num_samples, n_outputs),
        dtype=np.uint8,
        chunks=(min(256, num_samples), n_outputs),
        compression="lzf",
    )
    datasets["tissue_conductivity_factor"] = h5_file.create_dataset(
        "tissue_conductivity_factor",
        shape=(num_samples, len(TISSUE_PROPERTY_ORDER)),
        dtype=np.float32,
        chunks=(min(1024, num_samples), len(TISSUE_PROPERTY_ORDER)),
        compression="lzf",
    )
    datasets["tissue_permittivity_factor"] = h5_file.create_dataset(
        "tissue_permittivity_factor",
        shape=(num_samples, len(TISSUE_PROPERTY_ORDER)),
        dtype=np.float32,
        chunks=(min(1024, num_samples), len(TISSUE_PROPERTY_ORDER)),
        compression="lzf",
    )
    return datasets


def create_geometry_metadata_datasets(h5_file, num_samples, n_electrodes):
    """Create compact fixed-size geometry quality-control arrays."""
    if not SAVE_GEOMETRY_METADATA:
        return {}

    chunk_scalar = (min(1024, num_samples),)
    chunk_electrode = (min(128, num_samples), n_electrodes, 2)

    specifications = {
        "fresh_remesh": ((num_samples,), np.uint8, chunk_scalar),
        "generation_attempt": ((num_samples,), np.int16, chunk_scalar),
        "node_count": ((num_samples,), np.int32, chunk_scalar),
        "element_count": ((num_samples,), np.int32, chunk_scalar),
        "mesh_time_seconds": ((num_samples,), np.float32, chunk_scalar),
        "geometry_time_seconds": ((num_samples,), np.float32, chunk_scalar),
        "minimum_triangle_quality": ((num_samples,), np.float32, chunk_scalar),
        "shape_aspect_ratio": ((num_samples,), np.float32, chunk_scalar),
        "shape_shear": ((num_samples,), np.float32, chunk_scalar),
        "scale_factor": ((num_samples,), np.float32, chunk_scalar),
        "mm_per_mesh_unit": ((num_samples,), np.float32, chunk_scalar),
        "tibia_scale": ((num_samples,), np.float32, chunk_scalar),
        "fibula_scale": ((num_samples,), np.float32, chunk_scalar),
        "tibia_rotation_deg": ((num_samples,), np.float32, chunk_scalar),
        "fibula_rotation_deg": ((num_samples,), np.float32, chunk_scalar),
        "bone_augmentation_fallback": ((num_samples,), np.uint8, chunk_scalar),
        "el_pos": (
            (num_samples, n_electrodes),
            np.int32,
            (min(256, num_samples), n_electrodes),
        ),
        "electrode_xy": (
            (num_samples, n_electrodes, 2),
            np.float32,
            chunk_electrode,
        ),
        "electrode_node_xy": (
            (num_samples, n_electrodes, 2),
            np.float32,
            chunk_electrode,
        ),
        "tibia_shift_mm": (
            (num_samples, 2),
            np.float32,
            (min(512, num_samples), 2),
        ),
        "fibula_shift_mm": (
            (num_samples, 2),
            np.float32,
            (min(512, num_samples), 2),
        ),
    }

    return {
        name: h5_file.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compression="lzf",
        )
        for name, (shape, dtype, chunks) in specifications.items()
    }


# =========================================================
# Optional full variable-length geometry storage
# =========================================================
def create_full_geometry_datasets(h5_file, num_samples, n_electrodes):
    """Create packed variable-length arrays needed to reproduce plots.

    Each variable-size array is concatenated across samples. Offset arrays
    define the slice belonging to each sample. Triangle indices remain local
    to the corresponding sample nodes.
    """
    if not SAVE_FULL_GEOMETRY:
        return {}, {}

    datasets = {}

    def create_packed_points(name, dtype=np.float32, width=2):
        datasets[name] = h5_file.create_dataset(
            name, shape=(0, width), maxshape=(None, width), dtype=dtype,
            chunks=(4096, width), compression="lzf"
        )
        datasets[f"{name}_offsets"] = h5_file.create_dataset(
            f"{name}_offsets", shape=(num_samples + 1,), dtype=np.int64
        )
        datasets[f"{name}_offsets"][0] = 0

    create_packed_points("mesh_nodes", np.float32, 2)
    create_packed_points("mesh_elements", np.int32, 3)
    create_packed_points("calf_boundary_points", np.float32, 2)
    create_packed_points("skin_fat_boundary_points", np.float32, 2)
    create_packed_points("fat_boundary_points", np.float32, 2)
    create_packed_points("tibia_boundary_points", np.float32, 2)
    create_packed_points("fibula_boundary_points", np.float32, 2)

    # Fixed-size plotting arrays.
    datasets["us_outer_points"] = h5_file.create_dataset(
        "us_outer_points", shape=(num_samples, n_electrodes, 2),
        dtype=np.float32, chunks=(min(128, num_samples), n_electrodes, 2),
        compression="lzf"
    )
    datasets["us_skin_inner_points"] = h5_file.create_dataset(
        "us_skin_inner_points", shape=(num_samples, n_electrodes, 2),
        dtype=np.float32, chunks=(min(128, num_samples), n_electrodes, 2),
        compression="lzf"
    )
    datasets["us_inner_points"] = h5_file.create_dataset(
        "us_inner_points", shape=(num_samples, n_electrodes, 2),
        dtype=np.float32, chunks=(min(128, num_samples), n_electrodes, 2),
        compression="lzf"
    )
    datasets["effective_skin_mm"] = h5_file.create_dataset(
        "effective_skin_mm_full", shape=(num_samples, n_electrodes),
        dtype=np.float32, chunks=(min(256, num_samples), n_electrodes),
        compression="lzf"
    )
    datasets["effective_fat_mm"] = h5_file.create_dataset(
        "effective_fat_mm", shape=(num_samples, n_electrodes),
        dtype=np.float32, chunks=(min(256, num_samples), n_electrodes),
        compression="lzf"
    )

    counters = {
        "mesh_nodes": 0,
        "mesh_elements": 0,
        "calf_boundary_points": 0,
        "skin_fat_boundary_points": 0,
        "fat_boundary_points": 0,
        "tibia_boundary_points": 0,
        "fibula_boundary_points": 0,
    }
    return datasets, counters


def _append_packed_array(datasets, counters, name, sample_index, array, dtype):
    """Append one variable-length array and update its sample offset."""
    arr = np.asarray(array, dtype=dtype)
    if arr.ndim != 2 or arr.shape[1] != datasets[name].shape[1]:
        raise ValueError(
            f"{name} has invalid shape {arr.shape}; expected [N, {datasets[name].shape[1]}]."
        )

    start = counters[name]
    end = start + len(arr)
    datasets[name].resize((end, arr.shape[1]))
    datasets[name][start:end] = arr
    datasets[f"{name}_offsets"][sample_index + 1] = end
    counters[name] = end


def write_full_geometry(full_datasets, counters, sample_index, details):
    """Save all mesh and boundary arrays required for later plotting."""
    if not full_datasets:
        return

    nodes = np.asarray(details["scaled_nodes"], dtype=np.float32)[:, :2]
    elements = np.asarray(details["elements"], dtype=np.int32)
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError(f"Invalid element shape: {elements.shape}")
    if elements.size and (elements.min() < 0 or elements.max() >= len(nodes)):
        raise ValueError(
            f"Sample {sample_index}: triangle indices do not match sample nodes."
        )

    _append_packed_array(full_datasets, counters, "mesh_nodes", sample_index, nodes, np.float32)
    _append_packed_array(full_datasets, counters, "mesh_elements", sample_index, elements, np.int32)
    _append_packed_array(full_datasets, counters, "calf_boundary_points", sample_index, details["scaled_calf_boundary"], np.float32)
    _append_packed_array(full_datasets, counters, "skin_fat_boundary_points", sample_index, details["skin_fat_boundary"], np.float32)
    _append_packed_array(full_datasets, counters, "fat_boundary_points", sample_index, details["fat_muscle_boundary"], np.float32)
    _append_packed_array(full_datasets, counters, "tibia_boundary_points", sample_index, details["scaled_tibia_boundary"], np.float32)
    _append_packed_array(full_datasets, counters, "fibula_boundary_points", sample_index, details["scaled_fibula_boundary"], np.float32)

    full_datasets["us_outer_points"][sample_index] = np.asarray(details["us_outer_points"], dtype=np.float32)
    full_datasets["us_skin_inner_points"][sample_index] = np.asarray(details["us_skin_inner_points"], dtype=np.float32)
    full_datasets["us_inner_points"][sample_index] = np.asarray(details["us_inner_points"], dtype=np.float32)
    full_datasets["effective_skin_mm"][sample_index] = np.asarray(details["effective_skin_mm"], dtype=np.float32)
    full_datasets["effective_fat_mm"][sample_index] = np.asarray(details["effective_fat_mm"], dtype=np.float32)


def load_full_geometry(h5_file, sample_index):
    """Load one sample's mesh and plotting boundaries from an HDF5 file."""
    def read_packed(name):
        start = int(h5_file[f"{name}_offsets"][sample_index])
        end = int(h5_file[f"{name}_offsets"][sample_index + 1])
        return np.asarray(h5_file[name][start:end])

    return {
        "nodes": read_packed("mesh_nodes"),
        "elements": read_packed("mesh_elements").astype(np.int32),
        "calf_boundary": read_packed("calf_boundary_points"),
        "skin_fat_boundary": read_packed("skin_fat_boundary_points"),
        "fat_boundary": read_packed("fat_boundary_points"),
        "tibia_boundary": read_packed("tibia_boundary_points"),
        "fibula_boundary": read_packed("fibula_boundary_points"),
        "us_outer_points": np.asarray(h5_file["us_outer_points"][sample_index]),
        "us_skin_inner_points": np.asarray(h5_file["us_skin_inner_points"][sample_index]),
        "us_inner_points": np.asarray(h5_file["us_inner_points"][sample_index]),
        "effective_skin_mm": np.asarray(h5_file["effective_skin_mm_full"][sample_index]),
        "effective_fat_mm": np.asarray(h5_file["effective_fat_mm"][sample_index]),
    }


def write_file_attributes(h5_file, worker_id, seed, num_samples):
    """Record configuration and reproducibility information."""
    attrs = h5_file.attrs
    attrs["worker_id"] = worker_id
    attrs["seed"] = seed
    attrs["num_samples"] = num_samples
    attrs["reference_circumference_mm"] = REFERENCE_CIRCUMFERENCE_MM
    attrs["circumference_min_mm"] = CIRCUMFERENCE_MIN_MM
    attrs["circumference_max_mm"] = CIRCUMFERENCE_MAX_MM
    attrs["circumference_is_normalized"] = NORMALIZE_CIRCUMFERENCE
    attrs["fat_min_mm"] = FAT_MIN_MM
    attrs["fat_max_global_mm"] = FAT_MAX_MM
    attrs["fat_max_rule"] = "independent of circumference"
    attrs["skin_min_mm"] = SKIN_MIN_MM
    attrs["skin_max_mm"] = SKIN_MAX_MM
    attrs["skin_tissue"] = SKIN_TISSUE
    attrs["tissue_property_variation_fraction"] = TISSUE_PROPERTY_VARIATION
    attrs["tissue_property_variation_percent"] = 100.0 * TISSUE_PROPERTY_VARIATION
    attrs["tissue_property_factor_distribution"] = "uniform multiplicative scale"
    attrs["tissue_property_factor_frequency_rule"] = "one factor per tissue per sample shared across all frequencies"
    attrs["tissue_property_order"] = np.asarray(TISSUE_PROPERTY_ORDER, dtype="S32")
    attrs["target_definition"] = "effective subcutaneous fat thickness only; skin excluded"
    attrs["bone_margin_mm"] = BONE_MARGIN_MM
    attrs["noise_std_fraction"] = NOISE_STD
    attrs["noise_std_percent"] = 100.0 * NOISE_STD

    attrs["generate_new_mesh_per_sample"] = GENERATE_NEW_MESH_PER_SAMPLE
    attrs["electrode_start_fraction"] = ELECTRODE_START_FRACTION
    attrs["electrode_mode"] = ELECTRODE_MODE
    attrs["vary_shape"] = VARY_SHAPE
    attrs["aspect_ratio_range"] = ASPECT_RATIO_RANGE
    attrs["shear_range"] = SHEAR_RANGE
    attrs["harmonic_amplitude_range"] = HARMONIC_AMPLITUDE_RANGE
    attrs["shape_harmonics"] = SHAPE_HARMONICS
    attrs["tibia_scale_range"] = TIBIA_SCALE_RANGE
    attrs["fibula_scale_range"] = FIBULA_SCALE_RANGE
    attrs["bone_shift_max_mm"] = BONE_SHIFT_MAX_MM
    attrs["bone_rotation_max_deg"] = BONE_ROTATION_MAX_DEG
    attrs["minimum_bone_gap_mm"] = MINIMUM_BONE_GAP_MM
    attrs["mesh_max_area"] = MESH_MAX_AREA
    attrs["mesh_min_angle"] = MESH_MIN_ANGLE
    attrs["minimum_triangle_quality"] = MINIMUM_TRIANGLE_QUALITY
    attrs["geometry_max_attempts"] = GEOMETRY_MAX_ATTEMPTS

    attrs["input_layout"] = "sample, measurement, real_imaginary, frequency"
    attrs["target_unit"] = "millimeter"
    attrs["circumference_input_layout"] = "sample, scalar"
    attrs["circumference_raw_unit"] = "millimeter"
    attrs["full_variable_geometry_saved"] = SAVE_FULL_GEOMETRY
    attrs["geometry_storage"] = (
        "packed variable-length arrays with per-sample offsets"
        if SAVE_FULL_GEOMETRY else
        "not saved; compact metadata only"
    )
    attrs["triangle_indexing"] = "local to each sample mesh"


def write_geometry_metadata(metadata_datasets, sample_index, details):
    """Write compact metadata for one newly remeshed sample."""
    if not metadata_datasets:
        return

    values = {
        "fresh_remesh": np.uint8(bool(details["fresh_remesh"])),
        "generation_attempt": np.int16(details["generation_attempt"]),
        "node_count": np.int32(details["template_node_count"]),
        "element_count": np.int32(details["template_element_count"]),
        "mesh_time_seconds": np.float32(details["mesh_time_seconds"]),
        "geometry_time_seconds": np.float32(details["geometry_time_seconds"]),
        "minimum_triangle_quality": np.float32(details["minimum_triangle_quality"]),
        "shape_aspect_ratio": np.float32(details["shape_aspect_ratio"]),
        "shape_shear": np.float32(details["shape_shear"]),
        "scale_factor": np.float32(details["scale_factor"]),
        "mm_per_mesh_unit": np.float32(details["mm_per_mesh_unit"]),
        "tibia_scale": np.float32(details["tibia_scale"]),
        "fibula_scale": np.float32(details["fibula_scale"]),
        "tibia_rotation_deg": np.float32(details["tibia_rotation_deg"]),
        "fibula_rotation_deg": np.float32(details["fibula_rotation_deg"]),
        "bone_augmentation_fallback": np.uint8(
            bool(details["bone_augmentation_fallback"])
        ),
        "el_pos": np.asarray(details["el_pos"], dtype=np.int32),
        "electrode_xy": np.asarray(details["electrode_xy"], dtype=np.float32),
        "electrode_node_xy": np.asarray(
            details["electrode_node_xy"], dtype=np.float32
        ),
        "tibia_shift_mm": np.asarray(details["tibia_shift_mm"], dtype=np.float32),
        "fibula_shift_mm": np.asarray(details["fibula_shift_mm"], dtype=np.float32),
    }

    for name, value in values.items():
        metadata_datasets[name][sample_index] = value


# =========================================================
# Worker
# =========================================================
def generate_worker(worker_id, num_samples, seed):
    """Generate one independent HDF5 shard."""
    worker_start = time.perf_counter()
    output_path = ROOT_SAVEDATA / f"fat_dataset_part_{worker_id:02d}.h5"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if output_path.exists() and not OVERWRITE_EXISTING:
        print(
            f"[Worker {worker_id}] Skipping existing shard: {output_path}",
            flush=True,
        )
        return str(output_path)

    if temporary_path.exists():
        temporary_path.unlink()

    print(
        f"[Worker {worker_id}] Starting {num_samples} samples with seed {seed}",
        flush=True,
    )

    try:
        dataset, frequencies = build_dataset(num_samples, seed)
        main_datasets = None

        with h5py.File(temporary_path, mode="w") as h5_file:
            main_datasets = create_main_datasets(h5_file, num_samples, dataset)
            metadata_datasets = create_geometry_metadata_datasets(
                h5_file,
                num_samples,
                dataset.output_dim,
            )
            full_geometry_datasets, geometry_counters = create_full_geometry_datasets(
                h5_file, num_samples, dataset.output_dim
            )
            h5_file.create_dataset("frequencies", data=frequencies)
            write_file_attributes(h5_file, worker_id, seed, num_samples)

            for sample_index in range(num_samples):
                x, y, c, details = dataset.generate_one_sample_with_details()

                main_datasets["X"][sample_index] = np.asarray(x, dtype=np.float32)
                main_datasets["Y"][sample_index] = np.asarray(y, dtype=np.float32)
                main_datasets["C"][sample_index] = np.asarray(c, dtype=np.float32)
                main_datasets["circumference_mm"][sample_index] = np.asarray(
                    [details["circumference_mm"]],
                    dtype=np.float32,
                )
                main_datasets["requested_Y"][sample_index] = np.asarray(
                    details["requested_fat_mm"],
                    dtype=np.float32,
                )
                main_datasets["requested_skin_mm"][sample_index] = np.asarray(
                    details["requested_skin_mm"], dtype=np.float32
                )
                main_datasets["effective_skin_mm"][sample_index] = np.asarray(
                    details["effective_skin_mm"], dtype=np.float32
                )
                main_datasets["skin_clipped"][sample_index] = np.asarray(
                    details["skin_clipped"], dtype=np.uint8
                )
                main_datasets["bone_clipped"][sample_index] = np.asarray(
                    details["bone_clipped"],
                    dtype=np.uint8,
                )

                factors = details["tissue_property_factors"]
                main_datasets["tissue_conductivity_factor"][sample_index] = np.asarray(
                    [factors[name]["conductivity"] for name in TISSUE_PROPERTY_ORDER],
                    dtype=np.float32,
                )
                main_datasets["tissue_permittivity_factor"][sample_index] = np.asarray(
                    [factors[name]["permittivity"] for name in TISSUE_PROPERTY_ORDER],
                    dtype=np.float32,
                )

                write_geometry_metadata(metadata_datasets, sample_index, details)
                write_full_geometry(
                    full_geometry_datasets, geometry_counters, sample_index, details
                )

                completed = sample_index + 1
                if completed % FLUSH_INTERVAL == 0 or completed == num_samples:
                    h5_file.flush()
                    elapsed = time.perf_counter() - worker_start
                    average_time = elapsed / completed
                    remaining_samples = num_samples - completed
                    remaining_minutes = average_time * remaining_samples / 60.0

                    print(
                        f"[Worker {worker_id}] {completed}/{num_samples} | "
                        f"{average_time:.3f} s/sample | "
                        f"mesh={float(details['mesh_time_seconds']):.4f} s | "
                        f"remaining≈{remaining_minutes:.1f} min",
                        flush=True,
                    )

            h5_file.attrs["generation_complete"] = True
            h5_file.flush()

        os.replace(temporary_path, output_path)
        total_minutes = (time.perf_counter() - worker_start) / 60.0
        print(
            f"[Worker {worker_id}] Finished: {output_path} | "
            f"{total_minutes:.2f} minutes",
            flush=True,
        )
        return str(output_path)

    except Exception:
        print(f"[Worker {worker_id}] FAILED", flush=True)
        traceback.print_exc()
        if temporary_path.exists():
            temporary_path.unlink()
        raise


# =========================================================
# Main
# =========================================================
def main():
    validate_input_files()
    ROOT_SAVEDATA.mkdir(parents=True, exist_ok=True)

    total_samples = NUMBER_OF_WORKERS * SAMPLES_PER_WORKER
    print("=" * 72)
    print("Fresh-remeshing parallel EIT dataset generation")
    print(f"Workers               : {NUMBER_OF_WORKERS}")
    print(f"Samples per worker    : {SAMPLES_PER_WORKER}")
    print(f"Total samples         : {total_samples}")
    print(f"Noise level           : {100.0 * NOISE_STD:.3f}%")
    print(f"Skin tissue           : {SKIN_TISSUE}")
    print(f"Skin range            : {SKIN_MIN_MM:.1f}–{SKIN_MAX_MM:.1f} mm")
    print(f"Tissue property var.  : ±{100.0 * TISSUE_PROPERTY_VARIATION:.1f}%")
    print(f"Fat range             : {FAT_MIN_MM:.1f}–{FAT_MAX_MM:.1f} mm")
    print(
        f"Circumference range   : {CIRCUMFERENCE_MIN_MM:.1f}–"
        f"{CIRCUMFERENCE_MAX_MM:.1f} mm"
    )
    print(f"Fresh mesh per sample : {GENERATE_NEW_MESH_PER_SAMPLE}")
    print(f"Save full geometry    : {SAVE_FULL_GEOMETRY}")
    print(f"Output directory      : {ROOT_SAVEDATA}")
    print("=" * 72)

    context = mp.get_context("spawn")
    jobs = []

    for worker_id in range(NUMBER_OF_WORKERS):
        # Large spacing gives every worker a clearly separated random stream.
        seed = BASE_SEED + worker_id * 100_000
        process = context.Process(
            target=generate_worker,
            args=(worker_id, SAMPLES_PER_WORKER, seed),
            name=f"eit-worker-{worker_id:02d}",
        )
        process.start()
        jobs.append((worker_id, process))

    failed_workers = []
    for worker_id, process in jobs:
        process.join()
        if process.exitcode != 0:
            failed_workers.append((worker_id, process.exitcode))

    if failed_workers:
        raise RuntimeError(f"Some workers failed: {failed_workers}")

    print("All dataset workers completed successfully.")


if __name__ == "__main__":
    mp.freeze_support()
    main()