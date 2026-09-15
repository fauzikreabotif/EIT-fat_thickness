# -*- coding: utf-8 -*-
"""
Combine multiple HDF5 skin-enabled fat-thickness dataset shards.

This version supports two cases:

1. Training / validation shards
   - fixed-size datasets only
   - SAVE_FULL_GEOMETRY=False during generation

2. Testing shards
   - fixed-size datasets
   - optional packed variable-length geometry
   - SAVE_FULL_GEOMETRY=True during generation

Packed geometry format
----------------------
mesh_nodes                    : [total_nodes, 2]
mesh_nodes_offsets            : [N + 1]

mesh_elements                 : [total_elements, 3]
mesh_elements_offsets         : [N + 1]

calf_boundary_points          : [total_points, 2]
calf_boundary_points_offsets  : [N + 1]

skin_fat_boundary_points      : [total_points, 2]
skin_fat_boundary_points_offsets : [N + 1]

fat_boundary_points           : [total_points, 2]
fat_boundary_points_offsets   : [N + 1]

tibia_boundary_points         : [total_points, 2]
tibia_boundary_points_offsets : [N + 1]

fibula_boundary_points        : [total_points, 2]
fibula_boundary_points_offsets: [N + 1]

The element indices remain local to each sample. They are not shifted when
combining files.
"""

import os
from typing import Dict, List, Tuple

import h5py
import numpy as np


# User settings
ROOT = (r"H:\temporary\output_fat_thickness\data_noise00_skin04_25_fat04_25_circum3045_fresh_remesh_propvar_10\testing")

OUTPUT_FILENAME = "fat_dataset_testing.h5"

# Choose the source shard range.
# Examples:
#   FILE_START = 0,  FILE_END = 10   -> parts 0 to 9
#   FILE_START = 10, FILE_END = None -> parts 10 onward
FILE_START = 0
FILE_END = None

# Refuse to overwrite an existing merged file unless enabled.
OVERWRITE_OUTPUT = False


# Dataset definitions
REQUIRED_FIXED_DATASETS = {
    "X",
    "Y",
    "C",
    "circumference_mm",
    "requested_Y",
    "bone_clipped",
    "frequencies",
}

SKIN_FIXED_DATASETS = {
    "requested_skin_mm",
    "effective_skin_mm",
    "skin_clipped",
}

# These datasets are copied automatically when present.
# Their first dimension must equal the number of samples.
OPTIONAL_FIXED_DATASETS = {
    "fresh_remesh",
    "generation_attempt",
    "node_count",
    "element_count",
    "mesh_time_seconds",
    "geometry_time_seconds",
    "minimum_triangle_quality",
    "shape_aspect_ratio",
    "shape_shear",
    "scale_factor",
    "mm_per_mesh_unit",
    "tibia_scale",
    "fibula_scale",
    "tibia_rotation_deg",
    "fibula_rotation_deg",
    "tibia_shift_mm",
    "fibula_shift_mm",
    "el_pos",
    "electrode_xy",
    "electrode_node_xy",
    "us_outer_points",
    "us_skin_inner_points",
    "us_inner_points",
    "requested_skin_mm",
    "effective_skin_mm",
    "skin_clipped",
    "effective_fat_mm",
    "tissue_conductivity_factor",
    "tissue_permittivity_factor",
}

# Packed geometry data and their corresponding offsets.
PACKED_GEOMETRY = {
    "mesh_nodes": "mesh_nodes_offsets",
    "mesh_elements": "mesh_elements_offsets",
    "calf_boundary_points": "calf_boundary_points_offsets",
    "skin_fat_boundary_points": "skin_fat_boundary_points_offsets",
    "fat_boundary_points": "fat_boundary_points_offsets",
    "tibia_boundary_points": "tibia_boundary_points_offsets",
    "fibula_boundary_points": "fibula_boundary_points_offsets",
}


# Helpers
def find_source_files(root: str) -> List[str]:
    """Find and slice all part HDF5 files."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Folder does not exist: {root}")

    files = sorted(
        os.path.join(root, name)
        for name in os.listdir(root)
        if name.endswith(".h5") and "part_" in name
    )

    files = files[FILE_START:FILE_END]

    if not files:
        raise FileNotFoundError(
            f"No source HDF5 part files were found in: {root}"
        )

    return files


def require_datasets(h5_file: h5py.File, file_path: str) -> None:
    """Check the mandatory fixed-size datasets."""
    missing = REQUIRED_FIXED_DATASETS.difference(h5_file.keys())
    if missing:
        raise KeyError(
            f"{os.path.basename(file_path)} is missing datasets: "
            f"{sorted(missing)}"
        )


def detect_full_geometry(files: List[str]) -> bool:
    """
    Return True when packed full geometry is present.

    All source files must be consistent: either every file contains all packed
    geometry datasets, or none of them does.
    """
    geometry_presence = []

    for file_path in files:
        with h5py.File(file_path, "r") as h5_file:
            present_pairs = []

            for data_name, offset_name in PACKED_GEOMETRY.items():
                data_exists = data_name in h5_file
                offset_exists = offset_name in h5_file

                if data_exists != offset_exists:
                    raise KeyError(
                        f"{os.path.basename(file_path)} contains only one of "
                        f"{data_name!r} and {offset_name!r}."
                    )

                present_pairs.append(data_exists and offset_exists)

            if any(present_pairs) and not all(present_pairs):
                missing_pairs = [
                    (data_name, offset_name)
                    for (data_name, offset_name), present
                    in zip(PACKED_GEOMETRY.items(), present_pairs)
                    if not present
                ]
                raise KeyError(
                    f"{os.path.basename(file_path)} has incomplete packed "
                    f"geometry. Missing pairs: {missing_pairs}"
                )

            geometry_presence.append(all(present_pairs))

    if any(geometry_presence) and not all(geometry_presence):
        raise ValueError(
            "Some input shards contain full geometry and others do not. "
            "Combine training/validation shards separately from testing shards."
        )

    return all(geometry_presence)


def collect_shapes_and_counts(
    files: List[str],
    save_full_geometry: bool,
) -> Tuple[int, Dict[str, tuple], np.ndarray, Dict[str, int], List[str]]:
    """
    Validate all files and gather:
      - total sample count
      - per-sample dataset shapes
      - frequencies
      - packed geometry row totals
      - optional fixed-size dataset names
    """
    total_samples = 0
    packed_totals = {name: 0 for name in PACKED_GEOMETRY}

    with h5py.File(files[0], "r") as first_file:
        require_datasets(first_file, files[0])

        frequencies = first_file["frequencies"][:]

        fixed_shapes = {
            name: first_file[name].shape[1:]
            for name in REQUIRED_FIXED_DATASETS
            if name != "frequencies"
        }

        optional_names = sorted(
            name
            for name in OPTIONAL_FIXED_DATASETS
            if name in first_file
        )

        present_skin = SKIN_FIXED_DATASETS.intersection(first_file.keys())
        if present_skin and present_skin != SKIN_FIXED_DATASETS:
            missing_skin = sorted(SKIN_FIXED_DATASETS - present_skin)
            raise KeyError(
                "The first shard contains only part of the skin metadata. "
                f"Missing skin datasets: {missing_skin}"
            )

        optional_shapes = {
            name: first_file[name].shape[1:]
            for name in optional_names
        }

    all_shapes = {**fixed_shapes, **optional_shapes}

    for file_path in files:
        with h5py.File(file_path, "r") as h5_file:
            require_datasets(h5_file, file_path)

            n_samples = int(h5_file["X"].shape[0])

            if not np.array_equal(h5_file["frequencies"][:], frequencies):
                raise ValueError(
                    f"Frequency values differ in "
                    f"{os.path.basename(file_path)}"
                )

            for name, expected_shape in all_shapes.items():
                if name not in h5_file:
                    raise KeyError(
                        f"{os.path.basename(file_path)} is missing optional "
                        f"dataset {name!r}, but it exists in the first shard."
                    )

                if h5_file[name].shape[0] != n_samples:
                    raise ValueError(
                        f"{name} in {os.path.basename(file_path)} has "
                        f"{h5_file[name].shape[0]} samples, expected {n_samples}."
                    )

                if h5_file[name].shape[1:] != expected_shape:
                    raise ValueError(
                        f"Incompatible {name} shape in "
                        f"{os.path.basename(file_path)}: "
                        f"{h5_file[name].shape[1:]} vs {expected_shape}"
                    )

            # Reject additional optional datasets that are not shared by all
            # shards, because silently dropping them would be misleading.
            current_optional = {
                name
                for name in OPTIONAL_FIXED_DATASETS
                if name in h5_file
            }
            if current_optional != set(optional_names):
                raise ValueError(
                    f"Optional dataset mismatch in "
                    f"{os.path.basename(file_path)}.\n"
                    f"Expected: {optional_names}\n"
                    f"Found: {sorted(current_optional)}"
                )

            if save_full_geometry:
                for data_name, offset_name in PACKED_GEOMETRY.items():
                    offsets = np.asarray(h5_file[offset_name][:], dtype=np.int64)

                    if offsets.shape != (n_samples + 1,):
                        raise ValueError(
                            f"{offset_name} in {os.path.basename(file_path)} "
                            f"has shape {offsets.shape}; expected "
                            f"({n_samples + 1},)."
                        )

                    if offsets[0] != 0:
                        raise ValueError(
                            f"{offset_name} in {os.path.basename(file_path)} "
                            "must start at zero."
                        )

                    if np.any(np.diff(offsets) < 0):
                        raise ValueError(
                            f"{offset_name} in {os.path.basename(file_path)} "
                            "is not monotonically increasing."
                        )

                    if offsets[-1] != h5_file[data_name].shape[0]:
                        raise ValueError(
                            f"{offset_name}[-1] does not match "
                            f"{data_name}.shape[0] in "
                            f"{os.path.basename(file_path)}."
                        )

                    packed_totals[data_name] += int(
                        h5_file[data_name].shape[0]
                    )

            total_samples += n_samples

    return (
        total_samples,
        all_shapes,
        frequencies,
        packed_totals,
        optional_names,
    )


def choose_chunks(shape: tuple, dataset_name: str) -> tuple:
    """Choose practical chunk shapes for merged fixed-size datasets."""
    total_samples = shape[0]
    trailing_shape = shape[1:]

    if dataset_name == "X":
        return (1, *trailing_shape)

    return (min(256, total_samples), *trailing_shape)


def create_fixed_dataset(
    output_h5: h5py.File,
    name: str,
    total_samples: int,
    sample_shape: tuple,
    dtype,
) -> h5py.Dataset:
    """Create one merged fixed-size dataset."""
    shape = (total_samples, *sample_shape)

    return output_h5.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=choose_chunks(shape, name),
        compression="lzf",
    )


def create_packed_geometry_datasets(
    output_h5: h5py.File,
    packed_totals: Dict[str, int],
    total_samples: int,
    first_file: h5py.File,
) -> Tuple[Dict[str, h5py.Dataset], Dict[str, h5py.Dataset]]:
    """Create packed geometry arrays and their merged offsets."""
    packed_data = {}
    packed_offsets = {}

    for data_name, offset_name in PACKED_GEOMETRY.items():
        source_dataset = first_file[data_name]
        trailing_shape = source_dataset.shape[1:]
        total_rows = packed_totals[data_name]

        if total_rows == 0:
            chunks = None
            compression = None
        else:
            chunks = (min(4096, total_rows), *trailing_shape)
            compression = "lzf"

        packed_data[data_name] = output_h5.create_dataset(
            data_name,
            shape=(total_rows, *trailing_shape),
            dtype=source_dataset.dtype,
            chunks=chunks,
            compression=compression,
        )

        packed_offsets[offset_name] = output_h5.create_dataset(
            offset_name,
            shape=(total_samples + 1,),
            dtype=np.int64,
        )
        packed_offsets[offset_name][0] = 0

    return packed_data, packed_offsets


# Main combination
def main() -> None:
    files = find_source_files(ROOT)
    output_file = os.path.join(ROOT, OUTPUT_FILENAME)

    if os.path.exists(output_file):
        if not OVERWRITE_OUTPUT:
            raise FileExistsError(
                f"Output already exists: {output_file}\n"
                "Set OVERWRITE_OUTPUT=True to replace it."
            )
        os.remove(output_file)

    print("Found files:")
    for file_path in files:
        print(" ", os.path.basename(file_path))

    save_full_geometry = detect_full_geometry(files)

    (
        total_samples,
        all_shapes,
        frequencies,
        packed_totals,
        optional_names,
    ) = collect_shapes_and_counts(
        files=files,
        save_full_geometry=save_full_geometry,
    )

    print(f"\nTotal samples: {total_samples}")
    print(f"Full geometry detected: {save_full_geometry}")
    print(
        "Skin metadata detected:",
        all(name in optional_names for name in SKIN_FIXED_DATASETS),
    )

    if save_full_geometry:
        print("Packed geometry totals:")
        for name, count in packed_totals.items():
            print(f"  {name}: {count}")

    with h5py.File(files[0], "r") as first_file:
        with h5py.File(output_file, "w") as output_h5:

            # Create fixed-size datasets
            output_datasets = {}

            fixed_names = [
                "X",
                "Y",
                "C",
                "circumference_mm",
                "requested_Y",
                "bone_clipped",
                *optional_names,
            ]

            for name in fixed_names:
                output_datasets[name] = create_fixed_dataset(
                    output_h5=output_h5,
                    name=name,
                    total_samples=total_samples,
                    sample_shape=all_shapes[name],
                    dtype=first_file[name].dtype,
                )

            output_h5.create_dataset(
                "frequencies",
                data=frequencies,
            )

            # Create packed geometry datasets
            packed_data = {}
            packed_offsets = {}

            if save_full_geometry:
                packed_data, packed_offsets = (
                    create_packed_geometry_datasets(
                        output_h5=output_h5,
                        packed_totals=packed_totals,
                        total_samples=total_samples,
                        first_file=first_file,
                    )
                )

            # Copy metadata
            for key, value in first_file.attrs.items():
                output_h5.attrs[key] = value

            output_h5.attrs["num_samples"] = total_samples
            output_h5.attrs["combined_file_count"] = len(files)
            output_h5.attrs["full_variable_geometry_saved"] = (
                save_full_geometry
            )
            output_h5.attrs["skin_metadata_saved"] = bool(
                all(name in optional_names for name in SKIN_FIXED_DATASETS)
            )

            if save_full_geometry:
                output_h5.attrs["geometry_storage"] = (
                    "packed variable-length arrays with merged per-sample offsets"
                )
                output_h5.attrs["mesh_elements_indexing"] = (
                    "local to each sample"
                )

            # Copy data
            sample_start = 0

            # Current destination row in each packed array.
            packed_row_start = {
                name: 0 for name in PACKED_GEOMETRY
            }

            for file_path in files:
                print("Copy:", os.path.basename(file_path))

                with h5py.File(file_path, "r") as input_h5:
                    n_samples = int(input_h5["X"].shape[0])
                    sample_end = sample_start + n_samples

                    # Copy all fixed-size arrays.
                    for name, destination in output_datasets.items():
                        destination[sample_start:sample_end] = input_h5[name][:]

                    # Copy packed geometry and rebuild global offsets.
                    if save_full_geometry:
                        for data_name, offset_name in PACKED_GEOMETRY.items():
                            source_data = input_h5[data_name]
                            source_offsets = np.asarray(
                                input_h5[offset_name][:],
                                dtype=np.int64,
                            )

                            source_row_count = int(source_data.shape[0])
                            destination_row_start = packed_row_start[data_name]
                            destination_row_end = (
                                destination_row_start + source_row_count
                            )

                            packed_data[data_name][
                                destination_row_start:destination_row_end
                            ] = source_data[:]

                            # Exclude the first zero because the destination
                            # offset at sample_start already exists.
                            shifted_offsets = (
                                source_offsets[1:]
                                + destination_row_start
                            )

                            packed_offsets[offset_name][
                                sample_start + 1:sample_end + 1
                            ] = shifted_offsets

                            packed_row_start[data_name] = destination_row_end

                    sample_start = sample_end

            # Final validation.
            if sample_start != total_samples:
                raise RuntimeError(
                    f"Copied {sample_start} samples but expected "
                    f"{total_samples}."
                )

            if save_full_geometry:
                for data_name, offset_name in PACKED_GEOMETRY.items():
                    final_row = packed_row_start[data_name]

                    if final_row != packed_totals[data_name]:
                        raise RuntimeError(
                            f"Copied {final_row} rows for {data_name}, "
                            f"expected {packed_totals[data_name]}."
                        )

                    final_offset = int(
                        packed_offsets[offset_name][-1]
                    )

                    if final_offset != packed_totals[data_name]:
                        raise RuntimeError(
                            f"Final {offset_name} value is {final_offset}, "
                            f"expected {packed_totals[data_name]}."
                        )

            output_h5.flush()

    print("\nFinished!")
    print("Output:", output_file)


# Loading helper for one combined test sample
def load_packed_sample(
    h5_file: h5py.File,
    data_name: str,
    offset_name: str,
    sample_index: int,
) -> np.ndarray:
    """Load one variable-size packed array."""
    start = int(h5_file[offset_name][sample_index])
    end = int(h5_file[offset_name][sample_index + 1])
    return h5_file[data_name][start:end]


def load_full_geometry(
    h5_file: h5py.File,
    sample_index: int,
) -> Dict[str, np.ndarray]:
    """Load all plotting geometry for one combined test sample."""
    if not bool(
        h5_file.attrs.get("full_variable_geometry_saved", False)
    ):
        raise ValueError(
            "This merged file does not contain full variable geometry."
        )

    geometry = {
        "nodes": load_packed_sample(
            h5_file,
            "mesh_nodes",
            "mesh_nodes_offsets",
            sample_index,
        ),
        "elements": load_packed_sample(
            h5_file,
            "mesh_elements",
            "mesh_elements_offsets",
            sample_index,
        ).astype(np.int32),
        "calf_boundary": load_packed_sample(
            h5_file,
            "calf_boundary_points",
            "calf_boundary_points_offsets",
            sample_index,
        ),
        "skin_fat_boundary": load_packed_sample(
            h5_file,
            "skin_fat_boundary_points",
            "skin_fat_boundary_points_offsets",
            sample_index,
        ),
        "fat_boundary": load_packed_sample(
            h5_file,
            "fat_boundary_points",
            "fat_boundary_points_offsets",
            sample_index,
        ),
        "tibia_boundary": load_packed_sample(
            h5_file,
            "tibia_boundary_points",
            "tibia_boundary_points_offsets",
            sample_index,
        ),
        "fibula_boundary": load_packed_sample(
            h5_file,
            "fibula_boundary_points",
            "fibula_boundary_points_offsets",
            sample_index,
        ),
    }

    for optional_name in (
        "us_outer_points",
        "us_skin_inner_points",
        "us_inner_points",
        "requested_skin_mm",
        "effective_skin_mm",
        "skin_clipped",
        "effective_fat_mm",
        "electrode_xy",
        "electrode_node_xy",
        "el_pos",
    ):
        if optional_name in h5_file:
            geometry[optional_name] = h5_file[optional_name][sample_index]

    if geometry["elements"].size:
        if geometry["elements"].max() >= len(geometry["nodes"]):
            raise ValueError(
                "Loaded triangle connectivity does not match loaded nodes."
            )

    return geometry


if __name__ == "__main__":
    main()
