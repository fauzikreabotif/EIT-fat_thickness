# -*- coding: utf-8 -*-
"""
Train calf fat-thickness models on limited real multi-frequency EIT data.

Supported modes:
    transfer: simulation-pretrained initialization
    scratch: random initialization

Real-data organization
----------------------
EIT directory:
    2D_<Subject>_RC1_quasi_f=500.0-1000000.0Hz.xlsx
    2D_<Subject>_RC2_quasi_f=500.0-1000000.0Hz.xlsx
    2D_<Subject>_RC3_quasi_f=500.0-1000000.0Hz.xlsx

Each EIT workbook:
    sheet "R": [208 measurement pairs x 10 frequencies]
    sheet "X": [208 measurement pairs x 10 frequencies]

Ultrasound annotation workbook:
    one right-calf sheet per subject, e.g. "Anto_RC"
    column "Fat_Thickness_mm": 48 rows ordered as
        layer 1: E1...E16
        layer 2: E1...E16
        layer 3: E1...E16

Metadata workbook:
    one row per subject and circumference columns for RC layers 1-3.

Important:
    Splitting is performed by SUBJECT, never by measurement, so all three
    layers from one subject remain in the same fold.
"""

from __future__ import annotations

import copy
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from models_fat_thickness_XYC_modulated_transformer import (
    FatThicknessMLP,
    FatThicknessMLP_basic,
    ProtocolAwareFatThicknessTransformer,
    load_protocol_indices_csv,
)


# ============================================================
# User settings
# ============================================================
REAL_EIT_DIR = Path(r"H:\indonesian_student_visit\Taly_FatThickness_Jul-2026\Pakai untuk DL_2\Quasi Calf EIT Mesurement")
US_ANNOTATION_XLSX = Path(r"H:\indonesian_student_visit\Taly_FatThickness_Jul-2026\Pakai untuk DL_2\AnotasiFatThickness_US.xlsx")
METADATA_XLSX = Path(r"H:\indonesian_student_visit\Taly_FatThickness_Jul-2026\Pakai untuk DL_2\Metadata_Taly_FatLayer.xlsx")
# =========================================================
# Frequency selection
# =========================================================
ORIGINAL_N_FREQUENCIES = 10
FREQUENCY_INDICES = [ 2, 3, 4, 5, 6, 7, 8, 9] # you can put "None"
N_SELECTED_FREQUENCIES = len(FREQUENCY_INDICES) if FREQUENCY_INDICES != None else ORIGINAL_N_FREQUENCIES
# sfreq = '_8freqs_' if FREQUENCY_INDICES != None else '_'
print("Selected frequency indices:", FREQUENCY_INDICES)
print("Number of selected frequencies:", N_SELECTED_FREQUENCIES)

# The supplied metadata columns are named *_mm, but their values are around
# 30-40, which indicates centimeters. "auto" converts values < 100 to mm.
CIRCUMFERENCE_UNIT = "auto"  # "auto", "cm", or "mm"

MODEL_TYPE = "transformer"  # "mlp_basic", "mlp_film", or "transformer"
# Choose one experiment mode.
EXPERIMENT_MODE = "transfer"  # "transfer" or "scratch"
TRANSFER_STRATEGY = "head_then_full" # head_then_full, direct_full

IMPEDANCE_COMPONENT = "real"  # "complex", "real", "imag"
n_components = 2 if IMPEDANCE_COMPONENT=="complex" else 1

TRANSFER_STRATEGY = "_" if EXPERIMENT_MODE == "scratch" else TRANSFER_STRATEGY
# Best checkpoint obtained from simulation training.
# PRETRAINED_CHECKPOINT = Path(
#    r"H:\temporary\output_fat_thickness\data_noise00_skin04_25_fat04_25_circum3045_fresh_remesh_propvar_10"
#    r"\Right-Left\results_XYC_remesh_8freqs_transformer_noise_20\best_fat_thickness_mlp_XYC_remesh.pth"
#)
PRETRAINED_CHECKPOINT = Path(r"H:\temporary\output_fat_thickness\data_noise00_skin04_25_fat04_25_circum3045_fresh_remesh_propvar_10\Right-Left\real_results_XYC_remesh_8freqs_transformer_noise_5_skin_fat\best_fat_thickness_mlp_XYC_remesh.pth")

# Needed only for transformer.
PROTOCOL_CSV_PATH = Path(r"new_bilateral\calf_quasi_16_0-15_stim_.csv")

# Run one configuration at a time, e.g. A0-A4; results never overwrite other trials.
# A0: mean/std only; A1: mean/std + complex gain;
# A2: mean/std + gain/offset; A3: no alignment + gain;
# A4: no alignment + gain/offset; N0: no alignment, no calibration.
ADAPTATION_VARIANT = "N0"  # "N0", "A0", "A1", "A2", "A3", "A4"
#{Statistical alignment, Learnable calibration}
ADAPTATION_CONFIGS = {
    "N0": (False, "none"),
    "A0": (True,  "none"),
    "A1": (True,  "gain"),
    "A2": (True,  "gain_offset"),
    "A3": (False, "gain"),
    "A4": (False, "gain_offset"),
}
if ADAPTATION_VARIANT not in ADAPTATION_CONFIGS:
    raise ValueError(f"Unknown ADAPTATION_VARIANT: {ADAPTATION_VARIANT}")
USE_DOMAIN_ALIGNMENT, CALIBRATION_MODE = ADAPTATION_CONFIGS[ADAPTATION_VARIANT]

# Complex calibration learning settings (regularization relative to identity).
CALIBRATION_LR_HEAD = 3e-4
CALIBRATION_LR_FULL = 1e-4
CALIBRATION_REG_WEIGHT = 1e-3
# For comparable scratch/transfer ablations, alignment settings are independent
# of EXPERIMENT_MODE. Scratch with alignment requires SIMULATION_TRAIN_H5.
OUTPUT_DIR = Path(r"H:\temporary\wps\Right-Left") / (
    f"noise_5_{EXPERIMENT_MODE}_{TRANSFER_STRATEGY}_{MODEL_TYPE}_{ADAPTATION_VARIANT}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_MEASUREMENTS = 208
N_COMPONENTS = n_components.copy()
N_FREQUENCIES = N_SELECTED_FREQUENCIES
N_ELECTRODES = 16
N_LAYERS = 3

SEED = 12345
N_SPLITS = 5
BATCH_SIZE = 8
NUM_WORKERS = 0

# Two-stage fine-tuning:
HEAD_EPOCHS = 80
FULL_EPOCHS = 160
HEAD_LR = 3e-4
BACKBONE_LR = 1e-5
FULL_HEAD_LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
EARLY_STOPPING_PATIENCE = 50

# Set to a simulation training H5 to map real data distribution toward the
# simulation domain. The mapping is calculated per component and frequency:
# [real R/X, 10 frequencies], pooled over samples and 208 channels.
#
# This is safer than per-channel matching with only 45 real measurements.
SIMULATION_TRAIN_H5: Optional[Path] = Path(
    r"H:\temporary\output_fat_thickness\data_noise00_skin04_25_fat04_25_circum3045_fresh_remesh_propvar_10\Right-Left\training\fat_dataset_training_bilateral.h5")

# Final model is trained on all 45 measurements after grouped cross-validation.
TRAIN_FINAL_MODEL = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================================================
# for plotting
#=============================================================
PLOT_CV_GT_VS_PRED_ON_CALF = True
RIGHT_CALF_NPZ_PATH = Path(
    r"C:\Users\takeiken-c135\OneDrive - 国立大学法人千葉大学\My research"
    r"\Fat thickness\DL_fatThickness\new_bilateral\Right_calf\mesh_right_calf.npz"
)
LEFT_CALF_NPZ_PATH = Path(
    r"C:\Users\takeiken-c135\OneDrive - 国立大学法人千葉大学\My research"
    r"\Fat thickness\DL_fatThickness\new_bilateral\Left_calf\mesh_left_calf.npz"
)
PLOT_MESH_TRIANGLES = True
PLOT_BONES = True
PLOT_DPI = 220

def close_plot_polygon(poly):
    poly = np.asarray(poly, dtype=np.float64)[:, :2]

    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    return poly


def plot_polygon_length(poly):
    poly = close_plot_polygon(poly)
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())


def scale_plot_xy(points, factor, center):
    points = np.asarray(points, dtype=np.float64)
    scaled = points.copy()
    scaled[:, :2] = center + factor * (points[:, :2] - center)

    return scaled


def load_plot_calf_geometry(side: str):
    npz_path = RIGHT_CALF_NPZ_PATH if side == "Right" else LEFT_CALF_NPZ_PATH
    data = np.load(npz_path, allow_pickle=True)

    nodes = np.asarray(data["nodes"], dtype=np.float64)
    tri = np.asarray(data["tri"], dtype=np.int64)
    el_pos = np.asarray(data["el_pos"], dtype=np.int64).reshape(-1)
    calf = np.asarray(data["calf_boundary"], dtype=np.float64)[:, :2]
    tibia = np.asarray(data["tibia_boundary"], dtype=np.float64)[:, :2]
    fibula = np.asarray(data["fibula_boundary"], dtype=np.float64)[:, :2]

    if "electrode_xy" in data.files:
        electrode_xy = np.asarray(data["electrode_xy"], dtype=np.float64)[:, :2]
    else:
        electrode_xy = nodes[el_pos, :2]

    return nodes, tri, calf, tibia, fibula, electrode_xy


def prepare_plot_calf_geometry(circumference_mm, side):
    nodes, tri, calf, tibia, fibula, electrode_xy = load_plot_calf_geometry(side)

    center = calf.mean(axis=0)
    scale_factor = float(circumference_mm) / plot_polygon_length(calf)

    nodes = scale_plot_xy(nodes, scale_factor, center)
    calf = scale_plot_xy(calf, scale_factor, center)
    tibia = scale_plot_xy(tibia, scale_factor, center)
    fibula = scale_plot_xy(fibula, scale_factor, center)
    electrode_xy = scale_plot_xy(electrode_xy, scale_factor, center)

    center = calf.mean(axis=0)

    boundary_indices = np.asarray(
        [np.argmin(np.linalg.norm(calf - point, axis=1)) for point in electrode_xy]
    )

    outer_points = calf[boundary_indices]
    normals = []

    for boundary_index in boundary_indices:
        tangent = calf[(boundary_index + 1) % len(calf)] - calf[(boundary_index - 1) % len(calf)]
        tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
        normal = np.array([-tangent[1], tangent[0]])

        if np.dot(normal, center - calf[boundary_index]) < 0:
            normal = -normal

        normals.append(normal)

    normals = np.asarray(normals)

    return nodes, tri, calf, tibia, fibula, electrode_xy, outer_points, normals


def plot_gt_prediction_on_calf(prediction_df, fold):
    plot_dir = OUTPUT_DIR / "cv_GT_vs_prediction_on_calf"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for (subject, side), subject_df in prediction_df.groupby(["subject", "side"], sort=True):
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))

        for layer in range(1, N_LAYERS + 1):
            ax = axes[layer - 1]

            layer_df = subject_df[subject_df["layer"] == layer].sort_values("electrode").reset_index(drop=True)

            circumference_mm = float(layer_df["circumference_mm"].iloc[0])
            gt_mm = layer_df["ground_truth_mm"].to_numpy(dtype=np.float64)
            pred_mm = layer_df["prediction_mm"].to_numpy(dtype=np.float64)

            nodes, tri, calf, tibia, fibula, electrode_xy, outer_points, normals = prepare_plot_calf_geometry(circumference_mm, side)

            gt_inner_points = outer_points + normals * gt_mm[:, None]
            pred_inner_points = outer_points + normals * pred_mm[:, None]

            if PLOT_MESH_TRIANGLES:
                ax.triplot(nodes[:, 0], nodes[:, 1], tri, linewidth=0.25, alpha=0.20)

            calf_closed = close_plot_polygon(calf)
            ax.plot(calf_closed[:, 0], calf_closed[:, 1], "k-", linewidth=2.2, label="Calf surface")

            if PLOT_BONES:
                tibia_closed = close_plot_polygon(tibia)
                fibula_closed = close_plot_polygon(fibula)

                ax.plot(tibia_closed[:, 0], tibia_closed[:, 1], linewidth=1.5, label="Tibia")
                ax.plot(fibula_closed[:, 0], fibula_closed[:, 1], linewidth=1.5, label="Fibula")

            for electrode_index in range(N_ELECTRODES):
                ax.plot(
                    [outer_points[electrode_index, 0], gt_inner_points[electrode_index, 0]],
                    [outer_points[electrode_index, 1], gt_inner_points[electrode_index, 1]],
                    linewidth=2.5,
                    label="US GT" if electrode_index == 0 else None,
                )

                ax.plot(
                    [outer_points[electrode_index, 0], pred_inner_points[electrode_index, 0]],
                    [outer_points[electrode_index, 1], pred_inner_points[electrode_index, 1]],
                    "--",
                    linewidth=2.0,
                    label="Prediction" if electrode_index == 0 else None,
                )

            gt_contour = np.vstack([gt_inner_points, gt_inner_points[0]])
            pred_contour = np.vstack([pred_inner_points, pred_inner_points[0]])

            ax.plot(gt_contour[:, 0], gt_contour[:, 1], linewidth=1.7, alpha=0.75)
            ax.plot(pred_contour[:, 0], pred_contour[:, 1], "--", linewidth=1.7, alpha=0.75)

            ax.scatter(
                electrode_xy[:, 0],
                electrode_xy[:, 1],
                s=55,
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
                label="Electrodes",
            )

            diameter = max(np.ptp(calf[:, 0]), np.ptp(calf[:, 1]))
            label_offset = 0.075 * diameter

            for electrode_index in range(N_ELECTRODES):
                label_xy = outer_points[electrode_index] - normals[electrode_index] * label_offset
                error_mm = pred_mm[electrode_index] - gt_mm[electrode_index]

                ax.text(
                    label_xy[0],
                    label_xy[1],
                    f"E{electrode_index + 1}\n"
                    f"GT {gt_mm[electrode_index]:.1f}\n"
                    f"P {pred_mm[electrode_index]:.1f}\n"
                    f"Δ {error_mm:+.1f}",
                    fontsize=7.5,
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.16",
                        "facecolor": "white",
                        "alpha": 0.82,
                        "edgecolor": "none",
                    },
                )

            error = pred_mm - gt_mm
            mae_mm = float(np.mean(np.abs(error)))
            rmse_mm = float(np.sqrt(np.mean(error ** 2)))
            bias_mm = float(np.mean(error))

            ax.set_title(
                f"{'RC' if side == 'Right' else 'LC'}{layer} | C = {circumference_mm:.1f} mm\n"
                f"MAE = {mae_mm:.2f} | RMSE = {rmse_mm:.2f} | "
                f"Bias = {bias_mm:+.2f} mm",
                fontsize=11,
            )

            ax.set_aspect("equal")
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        unique_legend = dict(zip(labels, handles))

        fig.legend(
            unique_legend.values(),
            unique_legend.keys(),
            loc="lower center",
            ncol=len(unique_legend),
            frameon=False,
        )

        fig.suptitle(
            f"Fold {fold} — {subject} — {side}: Ultrasound GT vs Predicted Fat Thickness\n"
            f"{EXPERIMENT_MODE} | {MODEL_TYPE} | Held-out subject",
            fontsize=14,
        )

        fig.tight_layout(rect=(0, 0.07, 1, 0.93))

        safe_subject = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(subject))

        figure_path = plot_dir / (
            f"fold_{fold:02d}_{safe_subject}_{side.lower()}_L1_L2_L3_GT_vs_pred.png"
        )

        fig.savefig(figure_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

        print("Saved GT-vs-pred calf plot:", figure_path)




# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# Data structures
# ============================================================
@dataclass
class RealSample:
    subject: str
    side: str
    layer: int
    eit_path: Path
    x: np.ndarray       # [208, 2, 10]
    y: np.ndarray       # [16]
    circumference: float


class RealFatThicknessDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[RealSample],
        domain_aligner: Optional["DomainAligner"] = None,
    ):
        self.samples = list(samples)
        self.domain_aligner = domain_aligner

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        x = sample.x.astype(np.float32, copy=True)
        if self.domain_aligner is not None:
            x = self.domain_aligner.transform(x)

        y = sample.y.astype(np.float32, copy=False)
        c = np.asarray([sample.circumference], dtype=np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(c),
            sample.subject,
            sample.side,
            int(sample.layer),
        )


# ============================================================
# Excel readers
# ============================================================
def normalize_subject_name(value: object) -> str:
    """Normalize subject identifiers for matching files, sheets, and metadata."""
    text = str(value).strip()
    text = re.sub(r"^2D_", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_(RC|LC)\d*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_(RC|LC)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def read_numeric_eit_sheet(path: Path, sheet_name: str) -> np.ndarray:
    """
    Read one R or X sheet.

    Expected raw layout:
        first row = frequencies
        first column = measurement index
        remaining block = [208, 10]
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    # Remove completely empty rows/columns.
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # The observed files have one header row and one index column.
    block = raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce")

    # Fallback for files that pandas can read directly with headers.
    if block.shape != (N_MEASUREMENTS, ORIGINAL_N_FREQUENCIES):
        direct = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
        direct = direct.apply(pd.to_numeric, errors="coerce")
        block = direct

    if block.shape != (N_MEASUREMENTS, ORIGINAL_N_FREQUENCIES):
        raise ValueError(
            f"{path.name}, sheet {sheet_name}: expected "
            f"({N_MEASUREMENTS}, {ORIGINAL_N_FREQUENCIES}), got {block.shape}."
        )
        
        
    values = block.to_numpy(dtype=np.float32)
    if FREQUENCY_INDICES!=None:
        values = values[ :, FREQUENCY_INDICES,]
    
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))
        raise ValueError(
            f"{path.name}, sheet {sheet_name}: contains NaN/Inf at "
            f"{bad[:10].tolist()}."
        )
    return values


def read_eit_workbook(path: Path) -> np.ndarray:
    """
    Return X with the same layout as the simulation dataset:
        X[:, 0, :] = real component R
        X[:, 1, :] = imaginary component X
    """
    real = read_numeric_eit_sheet(path, "R")
    imag = read_numeric_eit_sheet(path, "X")
    x = np.stack([real, imag], axis=1)  # [208, 2, 10]
    return x.astype(np.float32)


def parse_subject_side_layer_from_filename(path: Path) -> Tuple[str, str, int]:
    """Parse subject, side, and layer from RC/LC EIT filenames."""
    match = re.search(
        r"^2D_(.+?)_(RC|LC)([123])(?:_|$)",
        path.stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Cannot parse subject/side/layer from filename: {path.name}. "
            "Expected 2D_<Subject>_RC1_... or 2D_<Subject>_LC1_..."
        )

    subject = match.group(1).strip()
    side = "Right" if match.group(2).upper() == "RC" else "Left"
    layer = int(match.group(3))
    return subject, side, layer


def load_ultrasound_targets(path: Path) -> Dict[Tuple[str, str], np.ndarray]:
    """Return (normalized_subject, side) -> [3, 16] fat thickness in mm."""
    if not path.is_file():
        raise FileNotFoundError(f"Ultrasound annotation file not found: {path}")

    excel = pd.ExcelFile(path)
    targets: Dict[Tuple[str, str], np.ndarray] = {}

    for sheet in excel.sheet_names:
        match = re.search(r"_(RC|LC)$", sheet, flags=re.IGNORECASE)
        if match is None:
            continue

        side = "Right" if match.group(1).upper() == "RC" else "Left"
        df = pd.read_excel(path, sheet_name=sheet)
        matching_columns = [
            c for c in df.columns
            if str(c).strip().lower() == "fat_thickness_mm"
        ]

        if not matching_columns:
            raise KeyError(
                f"Sheet {sheet}: column 'Fat_Thickness_mm' was not found."
            )

        values = pd.to_numeric(
            df[matching_columns[0]], errors="coerce"
        ).dropna().to_numpy(dtype=np.float32)

        expected = N_LAYERS * N_ELECTRODES
        if len(values) != expected:
            raise ValueError(
                f"Sheet {sheet}: expected {expected} valid values, found {len(values)}."
            )

        subject_key = normalize_subject_name(sheet)
        targets[(subject_key, side)] = values.reshape(N_LAYERS, N_ELECTRODES)

    if not targets:
        raise RuntimeError("No right/left calf (*_RC or *_LC) annotation sheets were found.")

    return targets


def find_subject_column(df: pd.DataFrame) -> str:
    preferred = [
        "Subject", "Subject_ID", "Name", "Nama", "Participant",
        "Participant_ID", "ID",
    ]
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in preferred:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]

    # Use first text-like column as fallback.
    for column in df.columns:
        series = df[column].dropna()
        if len(series) and series.map(lambda v: isinstance(v, str)).mean() > 0.5:
            return column

    raise KeyError(
        "Could not identify the subject column in the metadata workbook."
    )


def find_circumference_column(df: pd.DataFrame, side: str, layer: int) -> str:
    prefix = "RC" if side == "Right" else "LC"
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(c).lower()): c
        for c in df.columns
    }

    candidates = [
        f"Circum_{prefix}_L{layer}_mm",
        f"Circum{prefix}_L{layer}_mm",
        f"{prefix}_L{layer}_mm",
    ]

    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]

    raise KeyError(
        f"Could not find {side} circumference column for layer {layer}. "
        f"Available columns: {list(df.columns)}"
    )


def load_circumference_metadata(path: Path) -> Dict[Tuple[str, str, int], float]:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_excel(path)
    subject_column = find_subject_column(df)
    layer_columns = {
        (side, layer): find_circumference_column(df, side, layer)
        for side in ("Right", "Left")
        for layer in range(1, N_LAYERS + 1)
    }

    result: Dict[Tuple[str, str, int], float] = {}

    for _, row in df.iterrows():
        if pd.isna(row[subject_column]):
            continue

        subject_key = normalize_subject_name(row[subject_column])

        for (side, layer), column in layer_columns.items():
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue

            value = float(value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Invalid circumference for {row[subject_column]}, {side}, layer {layer}: {value}"
                )

            if CIRCUMFERENCE_UNIT == "cm":
                value_mm = value * 10.0
            elif CIRCUMFERENCE_UNIT == "mm":
                value_mm = value
            elif CIRCUMFERENCE_UNIT == "auto":
                value_mm = value * 10.0 if value < 100.0 else value
            else:
                raise ValueError("CIRCUMFERENCE_UNIT must be 'auto', 'cm', or 'mm'.")

            if not 200.0 <= value_mm <= 600.0:
                raise ValueError(
                    f"Implausible circumference for {row[subject_column]}, "
                    f"{side}, layer {layer}: {value_mm:.1f} mm"
                )

            result[(subject_key, side, layer)] = value_mm

    return result


def discover_eit_files(directory: Path) -> List[Path]:
    patterns = [
        "**/2D_*_RC1_quasi*.xlsx",
        "**/2D_*_RC2_quasi*.xlsx",
        "**/2D_*_RC3_quasi*.xlsx",
        "**/2D_*_LC1_quasi*.xlsx",
        "**/2D_*_LC2_quasi*.xlsx",
        "**/2D_*_LC3_quasi*.xlsx",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))
    return sorted(set(files))


def assemble_real_samples(
    eit_dir: Path,
    us_path: Path,
    metadata_path: Path,
) -> List[RealSample]:
    ultrasound = load_ultrasound_targets(us_path)
    circumferences = load_circumference_metadata(metadata_path)
    eit_files = discover_eit_files(eit_dir)

    if not eit_files:
        raise FileNotFoundError(
            f"No EIT workbooks found recursively under {eit_dir}."
        )

    samples: List[RealSample] = []
    seen = set()

    for path in eit_files:
        subject, side, layer = parse_subject_side_layer_from_filename(path)
        subject_key = normalize_subject_name(subject)
        sample_key = (subject_key, side, layer)
        ultrasound_key = (subject_key, side)

        if sample_key in seen:
            raise RuntimeError(
                f"Duplicate EIT measurement for {subject}, layer {layer}: {path}"
            )
        seen.add(sample_key)

        if ultrasound_key not in ultrasound:
            raise KeyError(
                f"No ultrasound sheet matched {subject} {side}."
            )
        if sample_key not in circumferences:
            raise KeyError(
                f"No circumference matched subject '{subject}', layer {layer}."
            )

        x = read_eit_workbook(path)
        y = ultrasound[ultrasound_key][layer - 1]

        samples.append(
            RealSample(
                subject=subject,
                side=side,
                layer=layer,
                eit_path=path,
                x=x,
                y=y.copy(),
                circumference=circumferences[sample_key],
            )
        )

    layer_map: Dict[Tuple[str, str], set] = {}
    for sample in samples:
        key = (normalize_subject_name(sample.subject), sample.side)
        layer_map.setdefault(key, set()).add(sample.layer)

    incomplete = {
        key: sorted(layers)
        for key, layers in layer_map.items()
        if layers != {1, 2, 3}
    }

    if incomplete:
        raise RuntimeError(
            "Each available subject-side must have layers 1, 2, and 3. "
            f"Incomplete subject-sides: {incomplete}"
        )

    n_subjects = len({normalize_subject_name(s.subject) for s in samples})
    print(f"Loaded {len(samples)} measurements from {n_subjects} subjects.")
    print(f"Right measurements: {sum(s.side == 'Right' for s in samples)}")
    print(f"Left measurements : {sum(s.side == 'Left' for s in samples)}")
    return samples


# ============================================================
# Domain alignment
# ============================================================
class DomainAligner:
    """
    Map real EIT to the broad component/frequency distribution of the
    simulation training data.

    Statistics have shape [1, 2, 10], pooled over samples and channels.
    """

    def __init__(
        self,
        real_mean: np.ndarray,
        real_std: np.ndarray,
        sim_mean: np.ndarray,
        sim_std: np.ndarray,
        epsilon: float = 1e-6,
    ):
        self.real_mean = real_mean.astype(np.float32)
        self.real_std = np.maximum(real_std, epsilon).astype(np.float32)
        self.sim_mean = sim_mean.astype(np.float32)
        self.sim_std = np.maximum(sim_std, epsilon).astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (
            (x - self.real_mean) / self.real_std
            * self.sim_std + self.sim_mean
        ).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "real_mean": self.real_mean.tolist(),
            "real_std": self.real_std.tolist(),
            "sim_mean": self.sim_mean.tolist(),
            "sim_std": self.sim_std.tolist(),
        }


def calculate_real_domain_stats(
    samples: Sequence[RealSample],
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.stack([sample.x for sample in samples], axis=0)  # [N,208,2,10]
    mean = x.mean(axis=(0, 1), keepdims=False)[None, :, :]  # [1,2,10]
    std = x.std(axis=(0, 1), keepdims=False)[None, :, :]
    return mean, std


def calculate_simulation_domain_stats(
    h5_path: Path,
    chunk_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Streaming mean/std over simulation samples and measurement channels.
    Expected H5 X shape: [N, 208, 2, 10].
    """
    if not h5_path.is_file():
        raise FileNotFoundError(f"Simulation H5 not found: {h5_path}")

    count = 0
    sum_x = np.zeros((N_COMPONENTS, N_FREQUENCIES), dtype=np.float64)
    sum_x2 = np.zeros_like(sum_x)

    with h5py.File(h5_path, "r") as h5:
        x_ds = h5["X"]
        if tuple(x_ds.shape[1:]) != (
            N_MEASUREMENTS, N_COMPONENTS, ORIGINAL_N_FREQUENCIES
        ):
            raise ValueError(f"Unexpected simulation X shape: {x_ds.shape}")

        for start in range(0, len(x_ds), chunk_size):
            block = np.asarray(x_ds[start:start + chunk_size], dtype=np.float64)
            
            if FREQUENCY_INDICES!=None:
                block = block[:, :, :, FREQUENCY_INDICES,]
            
            sum_x += block.sum(axis=(0, 1))
            sum_x2 += np.square(block).sum(axis=(0, 1))
            count += block.shape[0] * block.shape[1]

    mean = sum_x / count
    variance = np.maximum(sum_x2 / count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    return mean[None, :, :], std[None, :, :]


def make_domain_aligner(
    train_samples: Sequence[RealSample],
    sim_mean: Optional[np.ndarray],
    sim_std: Optional[np.ndarray],
) -> Optional[DomainAligner]:
    if not USE_DOMAIN_ALIGNMENT:
        return None
    if sim_mean is None or sim_std is None:
        raise RuntimeError(
            "Domain alignment is enabled but simulation statistics are absent."
        )
    real_mean, real_std = calculate_real_domain_stats(train_samples)
    return DomainAligner(real_mean, real_std, sim_mean, sim_std)


# ============================================================
# Learnable complex residual calibration for [B,208,2,F]
# ============================================================
class ComplexCalibrationLayer(nn.Module):
    """One complex gain (and optional complex offset) per frequency."""
    def __init__(self, n_frequencies: int, mode: str):
        super().__init__()
        if mode not in {"gain", "gain_offset"}:
            raise ValueError(f"Unsupported calibration mode: {mode}")
        self.mode = mode
        self.gain_real = nn.Parameter(torch.ones(n_frequencies))
        self.gain_imag = nn.Parameter(torch.zeros(n_frequencies))
        if mode == "gain_offset":
            self.offset_real = nn.Parameter(torch.zeros(n_frequencies))
            self.offset_imag = nn.Parameter(torch.zeros(n_frequencies))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = x[:, :, 0, :], x[:, :, 1, :]
        r = self.gain_real * real - self.gain_imag * imag
        i = self.gain_imag * real + self.gain_real * imag
        if self.mode == "gain_offset":
            r = r + self.offset_real
            i = i + self.offset_imag
        return torch.stack((r, i), dim=2)

    def regularization_loss(self) -> torch.Tensor:
        reg = ((self.gain_real - 1).square() + self.gain_imag.square()).mean()
        if self.mode == "gain_offset":
            reg = reg + (self.offset_real.square() + self.offset_imag.square()).mean()
        return reg


class CalibratedFatThicknessModel(nn.Module):
    """Wrapper retains original pretrained architecture and its forward signature."""
    def __init__(self, backbone: nn.Module, n_frequencies: int, mode: str):
        super().__init__()
        self.calibration = ComplexCalibrationLayer(n_frequencies, mode)
        self.backbone = backbone

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.calibration(x), c)

    def regularization_loss(self) -> torch.Tensor:
        return self.calibration.regularization_loss()


def calibration_details(model: nn.Module) -> Optional[dict]:
    if not isinstance(model, CalibratedFatThicknessModel):
        return None
    layer = model.calibration
    result = {"mode": layer.mode,
              "gain_real": layer.gain_real.detach().cpu().tolist(),
              "gain_imag": layer.gain_imag.detach().cpu().tolist()}
    if layer.mode == "gain_offset":
        result["offset_real"] = layer.offset_real.detach().cpu().tolist()
        result["offset_imag"] = layer.offset_imag.detach().cpu().tolist()
    return result


# ============================================================
# Model and checkpoint
# ============================================================
def build_model(checkpoint: dict) -> nn.Module:
    input_dim = int(checkpoint.get(
        "input_dim",
        N_MEASUREMENTS * N_COMPONENTS * N_FREQUENCIES,
    ))
    circumference_dim = int(checkpoint.get("circumference_dim", 1))
    output_dim = int(checkpoint.get("output_dim", N_ELECTRODES))
    dropout = float(checkpoint.get("dropout", DROPOUT))

    if MODEL_TYPE == "mlp_basic":
        model = FatThicknessMLP_basic(
            input_dim=input_dim,
            circumference_dim=circumference_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
    elif MODEL_TYPE == "mlp_film":
        model = FatThicknessMLP(
            input_dim=input_dim,
            circumference_dim=circumference_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
    elif MODEL_TYPE == "transformer":
        protocol_indices = load_protocol_indices_csv(
            str(PROTOCOL_CSV_PATH),
            expected_measurements=N_MEASUREMENTS,
        )
        model = ProtocolAwareFatThicknessTransformer(
            protocol_indices=protocol_indices,
            n_frequencies=N_FREQUENCIES,
            n_components=N_COMPONENTS,
            n_electrodes=N_ELECTRODES,
            output_dim=output_dim,
            circumference_dim=circumference_dim,
            d_model=128,
            n_heads=8,
            n_encoder_layers=4,
            n_decoder_layers=2,
            feedforward_dim=512,
            dropout=0.1,
        )
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    return model


def load_checkpoint_safely(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def set_fold_circumference_statistics(
    checkpoint: dict,
    train_samples: Sequence[RealSample],
) -> None:
    values = np.asarray(
        [sample.circumference for sample in train_samples],
        dtype=np.float64,
    )
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-12:
        std = 1.0
    checkpoint["standardize_circumference"] = True
    checkpoint["circumference_mean"] = mean
    checkpoint["circumference_std"] = std


def initialize_model(
    train_samples: Sequence[RealSample],
) -> Tuple[nn.Module, dict]:
    """Create the same architecture with pretrained or random weights."""
    if EXPERIMENT_MODE not in {"transfer", "scratch"}:
        raise ValueError("EXPERIMENT_MODE must be 'transfer' or 'scratch'.")

    reference = {}
    if PRETRAINED_CHECKPOINT.is_file():
        reference = load_checkpoint_safely(PRETRAINED_CHECKPOINT)
    elif EXPERIMENT_MODE == "transfer":
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {PRETRAINED_CHECKPOINT}"
        )

    checkpoint = {}
    for key in (
        "input_dim", "circumference_dim", "output_dim", "dropout",
        "input_shape", "standardize_circumference",
        "circumference_mean", "circumference_std",
        "frequency_indices", "n_frequencies",
    ):
        if key in reference:
            checkpoint[key] = reference[key]

    checkpoint.setdefault(
        "input_dim", N_MEASUREMENTS * N_COMPONENTS * N_FREQUENCIES
    )
    checkpoint.setdefault("circumference_dim", 1)
    checkpoint.setdefault("output_dim", N_ELECTRODES)
    checkpoint.setdefault("dropout", DROPOUT)
    checkpoint.setdefault(
        "input_shape", (N_MEASUREMENTS, N_COMPONENTS, N_FREQUENCIES)
    )
    checkpoint.setdefault("standardize_circumference", True)
    checkpoint.setdefault("circumference_mean", 0.0)
    checkpoint.setdefault("circumference_std", 1.0)

    model = build_model(checkpoint)

    if EXPERIMENT_MODE == "transfer":
        state = reference.get("model_state_dict", reference)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint/model mismatch.\n"
                f"Missing keys: {missing}\n"
                f"Unexpected keys: {unexpected}\n"
                "Check MODEL_TYPE and architecture settings."
            )
    else:
        set_fold_circumference_statistics(checkpoint, train_samples)

    if CALIBRATION_MODE != "none":
        model = CalibratedFatThicknessModel(
            model, N_FREQUENCIES, CALIBRATION_MODE
        )
    return model.to(DEVICE), checkpoint


def find_head_parameter_names(model: nn.Module) -> List[str]:
    """
    Identify prediction-head parameters by common module-name patterns.
    Falls back to the last four parameter tensors.
    """
    patterns = (
        "head", "output", "regressor", "prediction", "decoder",
        "fc_out", "final",
    )
    names = [
        name for name, _ in model.named_parameters()
        if not name.startswith("calibration.")
        and any(pattern in name.lower() for pattern in patterns)
    ]
    if not names:
        all_names = [name for name, _ in model.named_parameters()]
        names = all_names[-min(4, len(all_names)):]
    return names


def configure_head_only(model: nn.Module) -> List[str]:
    head_names = set(find_head_parameter_names(model))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (name in head_names or name.startswith("calibration."))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Head-only trainable parameters:", trainable)
    return trainable


def build_head_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("calibration.")]
    calibration_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and n.startswith("calibration.")]
    groups = [{"params": head_params, "lr": HEAD_LR}]
    if calibration_params:
        groups.append({"params": calibration_params,
                       "lr": CALIBRATION_LR_HEAD, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def configure_full_finetuning(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def build_full_optimizer(model: nn.Module, head_names: Sequence[str]):
    head_names = set(head_names)
    backbone_parameters, head_parameters, calibration_parameters = [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith("calibration."):
            calibration_parameters.append(parameter)
        elif name in head_names:
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    groups = [
        {"params": backbone_parameters, "lr": BACKBONE_LR},
        {"params": head_parameters, "lr": FULL_HEAD_LR},
    ]
    if calibration_parameters:
        groups.append({"params": calibration_parameters,
                       "lr": CALIBRATION_LR_FULL, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def prepare_circumference(
    c: torch.Tensor,
    checkpoint: dict,
) -> torch.Tensor:
    c = c.to(DEVICE, non_blocking=True)
    if bool(checkpoint.get("standardize_circumference", True)):
        mean = float(checkpoint["circumference_mean"])
        std = max(float(checkpoint["circumference_std"]), 1e-12)
        c = (c - mean) / std
    return c


def freeze_batchnorm_running_statistics(model: nn.Module) -> None:
    """
    Keep BatchNorm layers in evaluation mode during transfer learning.

    This is recommended for the basic MLP because the real dataset is very
    small and a final mini-batch may contain only one sample. BatchNorm cannot
    estimate batch variance from a batch of size one. Evaluation mode uses the
    running mean and variance learned from the simulation dataset instead.

    The affine BatchNorm parameters (weight and bias) remain trainable when
    their requires_grad flags are True.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


# ============================================================
# Training and evaluation
# ============================================================
def make_loader(
    samples: Sequence[RealSample],
    aligner: Optional[DomainAligner],
    shuffle: bool,
) -> DataLoader:
    dataset = RealFatThicknessDataset(samples, aligner)
    return DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    checkpoint: dict,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    is_training = optimizer is not None
    model.train(is_training)

    # FatThicknessMLP_basic contains BatchNorm1d layers. With only 45 real
    # measurements, a mini-batch can occasionally contain one sample.
    # Preserve the simulation-trained BatchNorm running statistics and avoid
    # the "Expected more than 1 value per channel" error.
    if is_training:
        freeze_batchnorm_running_statistics(model)

    loss_sum = 0.0
    abs_sum = 0.0
    sq_sum = 0.0
    n_samples = 0
    n_values = 0

    for x, y, c, _, _, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        c = prepare_circumference(c, checkpoint)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            prediction = model(x, c)
            loss = criterion(prediction, y)
            if is_training and isinstance(model, CalibratedFatThicknessModel):
                loss = loss + CALIBRATION_REG_WEIGHT * model.regularization_loss()

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = x.shape[0]
        error = prediction - y
        loss_sum += loss.item() * batch_size
        abs_sum += error.abs().sum().item()
        sq_sum += error.square().sum().item()
        n_samples += batch_size
        n_values += y.numel()

    return {
        "loss": loss_sum / max(n_samples, 1),
        "mae_mm": abs_sum / max(n_values, 1),
        "rmse_mm": np.sqrt(sq_sum / max(n_values, 1)),
    }


def fit_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint: dict,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    stage_name: str,
) -> Tuple[dict, List[dict]]:
    # Smooth L1 is more robust for only 45 measurements and occasional
    # ultrasound-label outliers.
    criterion = nn.SmoothL1Loss(beta=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val = np.inf
    patience = 0
    history: List[dict] = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, checkpoint, criterion, optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, checkpoint, criterion
        )
        scheduler.step(val_metrics["loss"])

        row = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae_mm": train_metrics["mae_mm"],
            "train_rmse_mm": train_metrics["rmse_mm"],
            "val_loss": val_metrics["loss"],
            "val_mae_mm": val_metrics["mae_mm"],
            "val_rmse_mm": val_metrics["rmse_mm"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{stage_name} {epoch:03d}/{epochs} | "
                f"train MAE {train_metrics['mae_mm']:.3f} | "
                f"val MAE {val_metrics['mae_mm']:.3f} mm"
            )

        if val_metrics["loss"] < best_val - 1e-6:
            best_val = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if patience >= EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict(best_state)
    return best_state, history


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    checkpoint: dict,
) -> pd.DataFrame:
    model.eval()
    rows = []

    with torch.no_grad():
        for x, y, c, subjects, sides, layers in loader:
            x = x.to(DEVICE)
            c_device = prepare_circumference(c, checkpoint)
            prediction = model(x, c_device).cpu().numpy()
            y_np = y.numpy()
            c_np = c.numpy().reshape(-1)

            for b in range(len(subjects)):
                for electrode in range(N_ELECTRODES):
                    rows.append({
                        "subject": subjects[b],
                        "side": sides[b],
                        "layer": int(layers[b]),
                        "electrode": electrode + 1,
                        "circumference_mm": float(c_np[b]),
                        "ground_truth_mm": float(y_np[b, electrode]),
                        "prediction_mm": float(prediction[b, electrode]),
                        "absolute_error_mm": float(
                            abs(prediction[b, electrode] - y_np[b, electrode])
                        ),
                    })

    return pd.DataFrame(rows)


def evaluate_prediction_frame(df: pd.DataFrame) -> dict:
    error = df["prediction_mm"].to_numpy() - df["ground_truth_mm"].to_numpy()
    return {
        "mae_mm": float(np.mean(np.abs(error))),
        "rmse_mm": float(np.sqrt(np.mean(np.square(error)))),
        "bias_mm": float(np.mean(error)),
    }


def train_one_fold(
    fold: int,
    train_samples: Sequence[RealSample],
    val_samples: Sequence[RealSample],
    sim_mean: Optional[np.ndarray],
    sim_std: Optional[np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    print(
        f"\nFold {fold}: {len(train_samples)} train measurements, "
        f"{len(val_samples)} validation measurements"
    )

    aligner = make_domain_aligner(train_samples, sim_mean, sim_std)
    train_loader = make_loader(train_samples, aligner, shuffle=True)
    val_loader = make_loader(val_samples, aligner, shuffle=False)

    model, checkpoint = initialize_model(train_samples)

    if EXPERIMENT_MODE == "transfer":
        if TRANSFER_STRATEGY == "head_then_full":
            head_names = configure_head_only(model)
            head_optimizer = build_head_optimizer(model)
            _, history_head = fit_stage(
                model, train_loader, val_loader, checkpoint,
                head_optimizer, HEAD_EPOCHS, "head"
            )
    
            configure_full_finetuning(model)
            full_optimizer = build_full_optimizer(model, head_names)
            
            best_state, history_full = fit_stage(
                model, train_loader, val_loader, checkpoint,
                full_optimizer, FULL_EPOCHS, "full"
            )
        elif TRANSFER_STRATEGY == "direct_full":
            configure_full_finetuning(model)
            head_names = find_head_parameter_names(model)
            full_optimizer = build_full_optimizer(model, head_names,)
            
            best_state, history_full = fit_stage(
                model, train_loader, val_loader, checkpoint,
                full_optimizer, HEAD_EPOCHS + FULL_EPOCHS, "direct_full_transfer"
            )
            history_head = []
    else:
        configure_full_finetuning(model)
        scratch_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=HEAD_LR,
            weight_decay=WEIGHT_DECAY,
        )
        best_state, history_full = fit_stage(
            model, train_loader, val_loader, checkpoint,
            scratch_optimizer, HEAD_EPOCHS + FULL_EPOCHS, "scratch"
        )
        history_head = []

    prediction_df = predict_loader(model, val_loader, checkpoint)
    prediction_df.insert(0, "fold", fold)
    if PLOT_CV_GT_VS_PRED_ON_CALF:
        plot_gt_prediction_on_calf(prediction_df, fold)
    history_df = pd.DataFrame(history_head + history_full)
    history_df.insert(0, "fold", fold)
    metrics = evaluate_prediction_frame(prediction_df)

    fold_path = OUTPUT_DIR / f"{EXPERIMENT_MODE}_fold_{fold:02d}_best.pth"
    torch.save(
        {
            "model_state_dict": best_state,
            "experiment_mode": EXPERIMENT_MODE,
            "source_checkpoint": (str(PRETRAINED_CHECKPOINT) if EXPERIMENT_MODE == "transfer" else None),
            "model_type": MODEL_TYPE,
            "fold": fold,
            "metrics": metrics,
            "domain_alignment": (
                aligner.to_dict() if aligner is not None else None
            ),
            "adaptation_variant": ADAPTATION_VARIANT,
            "calibration_mode": CALIBRATION_MODE,
            "calibration_details": calibration_details(model),
            "calibration_regularization_weight": CALIBRATION_REG_WEIGHT,
            "circumference_mean": checkpoint.get("circumference_mean"),
            "circumference_std": checkpoint.get("circumference_std"),
            "standardize_circumference": checkpoint.get(
                "standardize_circumference", True
            ),
        },
        fold_path,
    )

    print(
        f"Fold {fold} result: MAE={metrics['mae_mm']:.3f} mm, "
        f"RMSE={metrics['rmse_mm']:.3f} mm"
    )
    return prediction_df, history_df, metrics


def grouped_cross_validation(
    samples: Sequence[RealSample],
    sim_mean: Optional[np.ndarray],
    sim_std: Optional[np.ndarray],
) -> None:
    groups = np.asarray([
        normalize_subject_name(sample.subject) for sample in samples
    ])
    unique_subjects = np.unique(groups)

    if len(unique_subjects) < N_SPLITS:
        raise ValueError(
            f"N_SPLITS={N_SPLITS}, but only {len(unique_subjects)} subjects."
        )

    splitter = GroupKFold(n_splits=N_SPLITS)
    all_predictions = []
    all_histories = []
    fold_metrics = []

    dummy_x = np.zeros(len(samples))
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(dummy_x, groups=groups), start=1
    ):
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        # Explicit leakage check.
        train_subjects = {
            normalize_subject_name(s.subject) for s in train_samples
        }
        val_subjects = {
            normalize_subject_name(s.subject) for s in val_samples
        }
        assert train_subjects.isdisjoint(val_subjects)

        prediction_df, history_df, metrics = train_one_fold(
            fold, train_samples, val_samples, sim_mean, sim_std
        )
        all_predictions.append(prediction_df)
        all_histories.append(history_df)
        fold_metrics.append({"fold": fold, **metrics})

    predictions = pd.concat(all_predictions, ignore_index=True)
    histories = pd.concat(all_histories, ignore_index=True)
    metrics_df = pd.DataFrame(fold_metrics)

    overall = evaluate_prediction_frame(predictions)

    for side in ("Right", "Left"):
        side_df = predictions[predictions["side"] == side]
        if len(side_df):
            fold_metrics.append({"fold": f"overall_{side.lower()}", **evaluate_prediction_frame(side_df)})

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df = pd.concat(
        [
            metrics_df,
            pd.DataFrame([{"fold": "overall", **overall}]),
        ],
        ignore_index=True,
    )

    predictions.to_excel(
        OUTPUT_DIR / f"{EXPERIMENT_MODE}_grouped_cv_predictions.xlsx", index=False
    )
    histories.to_excel(
        OUTPUT_DIR / f"{EXPERIMENT_MODE}_grouped_cv_training_history.xlsx", index=False
    )
    metrics_df.to_excel(
        OUTPUT_DIR / f"{EXPERIMENT_MODE}_grouped_cv_metrics.xlsx", index=False
    )

    subject_metrics = (
        predictions.groupby("subject", as_index=False)
        .apply(
            lambda d: pd.Series(evaluate_prediction_frame(d)),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    subject_metrics.to_excel(
        OUTPUT_DIR / f"{EXPERIMENT_MODE}_grouped_cv_subject_metrics.xlsx", index=False
    )

    print("\nGrouped cross-validation result")
    print(metrics_df.to_string(index=False))


def train_final_model(
    samples: Sequence[RealSample],
    sim_mean: Optional[np.ndarray],
    sim_std: Optional[np.ndarray],
) -> None:
    """
    Fine-tune on all real measurements.

    Since no held-out subjects remain, use a fixed 90/10 measurement split only
    for early stopping. This final model is for deployment, not unbiased
    performance reporting. Report grouped-CV metrics as the study result.
    """
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    n_val = max(3, int(round(0.1 * len(samples))))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    aligner = make_domain_aligner(samples, sim_mean, sim_std)
    train_loader = make_loader(train_samples, aligner, shuffle=True)
    val_loader = make_loader(val_samples, aligner, shuffle=False)

    model, checkpoint = initialize_model(train_samples)
    if EXPERIMENT_MODE == "transfer":
        head_names = configure_head_only(model)
        head_optimizer = build_head_optimizer(model)
        _, history_head = fit_stage(
            model, train_loader, val_loader, checkpoint,
            head_optimizer, HEAD_EPOCHS, "head"
        )
        configure_full_finetuning(model)
        full_optimizer = build_full_optimizer(model, head_names)
        best_state, history_full = fit_stage(
            model, train_loader, val_loader, checkpoint,
            full_optimizer, FULL_EPOCHS, "full"
        )
    else:
        configure_full_finetuning(model)
        scratch_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=HEAD_LR,
            weight_decay=WEIGHT_DECAY,
        )
        best_state, history_full = fit_stage(
            model, train_loader, val_loader, checkpoint,
            scratch_optimizer, HEAD_EPOCHS + FULL_EPOCHS, "scratch"
        )
        history_head = []

    torch.save(
        {
            "model_state_dict": best_state,
            "experiment_mode": EXPERIMENT_MODE,
            "source_checkpoint": (str(PRETRAINED_CHECKPOINT) if EXPERIMENT_MODE == "transfer" else None),
            "model_type": MODEL_TYPE,
            "domain_alignment": (
                aligner.to_dict() if aligner is not None else None
            ),
            "adaptation_variant": ADAPTATION_VARIANT,
            "calibration_mode": CALIBRATION_MODE,
            "calibration_details": calibration_details(model),
            "calibration_regularization_weight": CALIBRATION_REG_WEIGHT,
            "circumference_mean": checkpoint.get("circumference_mean"),
            "circumference_std": checkpoint.get("circumference_std"),
            "standardize_circumference": checkpoint.get(
                "standardize_circumference", True
            ),
            "input_shape": checkpoint.get(
                "input_shape",
                (N_MEASUREMENTS, N_COMPONENTS, N_FREQUENCIES),
            ),
            "note": (
                "Fine-tuned using all available real measurements. "
                "Use grouped CV metrics for unbiased reporting."
            ),
        },
        OUTPUT_DIR / f"final_{EXPERIMENT_MODE}_model_all_real_data.pth",
    )

    pd.DataFrame(history_head + history_full).to_excel(
        OUTPUT_DIR / f"final_{EXPERIMENT_MODE}_model_training_history.xlsx", index=False
    )
    print(f"Saved final {EXPERIMENT_MODE} model.")


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("Device:", DEVICE)
    print("Model type:", MODEL_TYPE)
    print("Experiment mode:", EXPERIMENT_MODE)
    print("Domain alignment:", USE_DOMAIN_ALIGNMENT)
    print("Adaptation variant:", ADAPTATION_VARIANT)
    print("Calibration mode:", CALIBRATION_MODE)
    with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as file:
        json.dump({"experiment_mode": EXPERIMENT_MODE,
                   "transfer_strategy": TRANSFER_STRATEGY,
                   "model_type": MODEL_TYPE,
                   "adaptation_variant": ADAPTATION_VARIANT,
                   "domain_alignment": USE_DOMAIN_ALIGNMENT,
                   "calibration_mode": CALIBRATION_MODE,
                   "calibration_reg_weight": CALIBRATION_REG_WEIGHT,
                   "seed": SEED, "frequency_indices": FREQUENCY_INDICES},
                  file, indent=2)

    samples = assemble_real_samples(
        REAL_EIT_DIR,
        US_ANNOTATION_XLSX,
        METADATA_XLSX,
    )

    # Save a manifest so matching can be checked before training.
    manifest = pd.DataFrame([
        {
            "subject": s.subject,
            "side": s.side,
            "layer": s.layer,
            "eit_file": str(s.eit_path),
            "circumference_mm": s.circumference,
            "mean_fat_thickness_mm": float(s.y.mean()),
            "min_fat_thickness_mm": float(s.y.min()),
            "max_fat_thickness_mm": float(s.y.max()),
        }
        for s in samples
    ])
    manifest.to_excel(OUTPUT_DIR / "real_data_manifest.xlsx", index=False)

    sim_mean = sim_std = None
    if USE_DOMAIN_ALIGNMENT:
        if SIMULATION_TRAIN_H5 is None:
            raise ValueError(
                "Set SIMULATION_TRAIN_H5 or disable USE_DOMAIN_ALIGNMENT."
            )
        sim_mean, sim_std = calculate_simulation_domain_stats(
            SIMULATION_TRAIN_H5
        )
        np.savez(
            OUTPUT_DIR / "simulation_domain_statistics.npz",
            mean=sim_mean,
            std=sim_std,
        )

    grouped_cross_validation(samples, sim_mean, sim_std)

    if TRAIN_FINAL_MODEL:
        train_final_model(samples, sim_mean, sim_std)


if __name__ == "__main__":
    main()
