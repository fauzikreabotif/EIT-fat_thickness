# -*- coding: utf-8 -*-
"""
Train the circumference-aware MLP using remeshed HDF5 datasets.

Training and validation require only:
    X : [N, 208, 2, n_frequency]
    Y : [N, 16]
    C : [N, 1]

The testing HDF5 file may additionally contain packed variable-size geometry:
    mesh_nodes / mesh_nodes_offsets
    mesh_elements / mesh_elements_offsets
    calf_boundary_points / calf_boundary_points_offsets
    fat_boundary_points / fat_boundary_points_offsets
    tibia_boundary_points / tibia_boundary_points_offsets
    fibula_boundary_points / fibula_boundary_points_offsets

This geometry is used only for plotting predictions on the actual remeshed calf
associated with each test sample.
"""

import os
import time
from typing import Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from models_fat_thickness_XYC_modulated_transformer import FatThicknessMLP_basic, FatThicknessMLP, ProtocolAwareFatThicknessTransformer, load_protocol_indices_csv
from utilities_fat_thickness_XYC_fresh_calfNbone_remesh_skin_tissuevar import (
    create_fat_muscle_boundary_polygon, create_skin_fat_muscle_boundary_polygons,
    FatThicknessH5Dataset, )


# =========================================================
# Settings
# =========================================================
ROOT_DATA = ("/mnt/Ubuntu01/lymphedema/Rizki/fat_thickness/data_simulation/data_noise00_skin04_25_fat04_25_circum3045_fresh_remesh_propvar_10")

TRAIN_H5_PATH = os.path.join(ROOT_DATA, "training", "fat_dataset_training.h5")
VAL_H5_PATH = os.path.join(ROOT_DATA, "val", "fat_dataset_val.h5")
TEST_H5_PATH = os.path.join(ROOT_DATA, "testing", "fat_dataset_testing.h5")
OUTPUT_DIR = os.path.join(ROOT_DATA, "results_XYC_remesh_8freqs_transformer_noise_0")
os.makedirs(OUTPUT_DIR, exist_ok=True)

protocol_csv_path ='Right_calf/calf_quasi_16_0-15_stim_.csv'

#================  Model ==============
flatten_input=False
MODEL_TYPE = "transformer" # mlp_basic, mlp_film, transformer

flatten_input = MODEL_TYPE == "mlp_basic"

# =========================================================
# Frequency selection
# =========================================================
FREQUENCY_INDICES = [2, 3, 4, 5, 6, 7, 8, 9] # you can put "None"
N_SELECTED_FREQUENCIES = len(FREQUENCY_INDICES) if FREQUENCY_INDICES != None else 10
print("Selected frequency indices:", FREQUENCY_INDICES)
print("Number of selected frequencies:", N_SELECTED_FREQUENCIES)

# =========================================================
# Training-time noise augmentation
# =========================================================
TRAIN_APPLY_NOISE = True
TRAIN_MAX_NOISE = 0.0       # 0.10 = 10%
TRAIN_RANDOM_NOISE_STD = True #False-> every sample receives exactly 10% noise
TRAIN_RANDOM_NOISE = True

# Keep validation/test clean
VAL_APPLY_NOISE = False;
TEST_APPLY_NOISE = False


# =========================================================
# Training parameters
# =========================================================
BATCH_SIZE = 256
EPOCHS = 500
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.2
NUM_WORKERS = 0
EARLY_STOPPING_PATIENCE = 100
SCHEDULER_PATIENCE = 50
SAMPLING_SEED = 12345

# True: use every training sample once per epoch.
# False: use SAMPLES_PER_EPOCH random non-repeating samples per epoch.
USE_FULL_TRAINING_SET_EACH_EPOCH = True
SAMPLES_PER_EPOCH = None

# C was generated as raw circumference in millimeters.
STANDARDIZE_CIRCUMFERENCE = True

# Number of test geometries to visualize after evaluation.
NUMBER_OF_TEST_PLOTS = 10
TEST_PLOT_START_INDEX = 0


# =========================================================
# Packed remesh geometry loader
# =========================================================
# def load_packed_array(
#     h5_file: h5py.File,
#     data_name: str,
#     offset_name: str,
#     sample_index: int,
# ) -> np.ndarray:
#     start = int(h5_file[offset_name][sample_index])
#     end = int(h5_file[offset_name][sample_index + 1])
#     return np.asarray(h5_file[data_name][start:end])

def load_packed_geometry(f, data_key, offset_key, sample_index):
    start = int(f[offset_key][sample_index])
    end = int(f[offset_key][sample_index + 1])
    data = f[data_key][start:end]

    return data

def test_file_has_skin(h5_path):
    f = h5py.File(h5_path, "r")
    required_keys = ["skin_fat_boundary_points", "skin_fat_boundary_points_offsets", "us_skin_inner_points", "effective_skin_mm"]
    has_skin = all(key in f for key in required_keys)
    f.close()

    return has_skin


def load_test_geometry(h5_path, sample_index):
    f = h5py.File(h5_path, "r")

    geometry = {}
    geometry["nodes"] = np.asarray(load_packed_geometry(f, "mesh_nodes", "mesh_nodes_offsets", sample_index))
    geometry["elements"] = np.asarray(load_packed_geometry(f, "mesh_elements", "mesh_elements_offsets", sample_index), dtype=np.int64)
    geometry["calf_boundary"] = np.asarray(load_packed_geometry(f, "calf_boundary_points", "calf_boundary_points_offsets", sample_index))
    geometry["gt_fat_boundary"] = np.asarray(load_packed_geometry(f, "fat_boundary_points", "fat_boundary_points_offsets", sample_index))
    geometry["tibia_boundary"] = np.asarray(load_packed_geometry(f, "tibia_boundary_points", "tibia_boundary_points_offsets", sample_index))
    geometry["fibula_boundary"] = np.asarray(load_packed_geometry(f, "fibula_boundary_points", "fibula_boundary_points_offsets", sample_index))
    geometry["electrode_xy"] = np.asarray(f["electrode_xy"][sample_index])
    geometry["gt_outer_points"] = np.asarray(f["us_outer_points"][sample_index])
    geometry["gt_inner_points"] = np.asarray(f["us_inner_points"][sample_index])
    geometry["ground_truth_mm"] = np.asarray(f["Y"][sample_index])
    geometry["circumference_mm"] = float(np.asarray(f["circumference_mm"][sample_index]).reshape(-1)[0])
    geometry["has_skin"] = False

    skin_keys = ["skin_fat_boundary_points", "skin_fat_boundary_points_offsets", "us_skin_inner_points", "effective_skin_mm"]

    if all(key in f for key in skin_keys):
        geometry["skin_fat_boundary"] = np.asarray(load_packed_geometry(f, "skin_fat_boundary_points", "skin_fat_boundary_points_offsets", sample_index))
        geometry["gt_skin_inner_points"] = np.asarray(f["us_skin_inner_points"][sample_index])
        geometry["effective_skin_mm"] = np.asarray(f["effective_skin_mm"][sample_index])
        geometry["has_skin"] = True

    f.close()

    return geometry


# =========================================================
# Circumference statistics
# =========================================================
def calculate_circumference_statistics(h5_path: str):
    with h5py.File(h5_path, "r") as h5_file:
        c_values = np.asarray(h5_file["C"], dtype=np.float64).reshape(-1)

    mean = float(c_values.mean())
    std = float(c_values.std())
    if std < 1e-12:
        std = 1.0
    return mean, std


C_MEAN, C_STD = calculate_circumference_statistics(TRAIN_H5_PATH)
print(f"Training circumference mean: {C_MEAN:.3f} mm")
print(f"Training circumference std : {C_STD:.3f} mm")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# =========================================================
# Datasets and loaders
# =========================================================
train_dataset = FatThicknessH5Dataset(
                TRAIN_H5_PATH,
                flatten_input=flatten_input,
                frequency_indices=FREQUENCY_INDICES,
                apply_noise=TRAIN_APPLY_NOISE,
                noise_std=TRAIN_MAX_NOISE,
                random_noise_std=TRAIN_RANDOM_NOISE_STD,
                seed=SAMPLING_SEED,
                )
val_dataset = FatThicknessH5Dataset(
                VAL_H5_PATH,
                flatten_input=flatten_input,
                frequency_indices=FREQUENCY_INDICES,
                # Validation remains CLEAN
                apply_noise=False,
                )
test_dataset = FatThicknessH5Dataset(
                TEST_H5_PATH,
                flatten_input=flatten_input,
                frequency_indices=FREQUENCY_INDICES,
                # Test remains CLEAN
                apply_noise=False,
                )

print("Train samples:", len(train_dataset))

x, y, c = train_dataset[0]
print("X:", x.shape, x.dtype)
print("Y:", y.shape, y.dtype)
print("C:", c.shape, c.dtype)
x, y, c = val_dataset[0]
print("Val X:", x.shape)
x, y, c = test_dataset[0]
print("Test X:", x.shape)
print("HDF5 loaders OK.")

print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))
print("Original input shape:", train_dataset.input_shape)
print("Output shape:", train_dataset.output_shape)
print("Circumference shape:", train_dataset.circumference_shape)

if train_dataset.input_shape != val_dataset.input_shape:
    raise ValueError("Train and validation X shapes differ.")
if train_dataset.input_shape != test_dataset.input_shape:
    raise ValueError("Train and test X shapes differ.")
if train_dataset.output_shape != val_dataset.output_shape:
    raise ValueError("Train and validation Y shapes differ.")
if train_dataset.output_shape != test_dataset.output_shape:
    raise ValueError("Train and test Y shapes differ.")

INPUT_DIM = int(np.prod(train_dataset.input_shape))
OUTPUT_DIM = int(train_dataset.output_shape[0])
CIRCUMFERENCE_DIM = int(np.prod(train_dataset.circumference_shape))

print("Flattened input dimension:", INPUT_DIM)
print("Output dimension:", OUTPUT_DIM)
print("Circumference dimension:", CIRCUMFERENCE_DIM)


def make_loader(dataset, shuffle=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )


val_loader = make_loader(val_dataset, shuffle=False)
test_loader = make_loader(test_dataset, shuffle=False)


# =========================================================
# Model and optimization
# =========================================================
if MODEL_TYPE == "mlp":
    model = FatThicknessMLP(
        input_dim=INPUT_DIM,
        circumference_dim=CIRCUMFERENCE_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)
elif MODEL_TYPE == "mlp_basic":
    model = FatThicknessMLP_basic(
        input_dim=INPUT_DIM,
        circumference_dim=CIRCUMFERENCE_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)
elif MODEL_TYPE == "transformer":
    protocol_indices = load_protocol_indices_csv(
        protocol_csv_path,
        expected_measurements=208,
    )

    model = ProtocolAwareFatThicknessTransformer(
        protocol_indices=protocol_indices,
        n_frequencies=N_SELECTED_FREQUENCIES,
        n_components=2,
        n_electrodes=16,
        output_dim=16,
        circumference_dim=CIRCUMFERENCE_DIM,
        d_model=128,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=2,
        feedforward_dim=512,
        dropout=0.1,
    ).to(DEVICE)

else:
    raise ValueError(
        f"Unknown MODEL_TYPE: {MODEL_TYPE}"
    )


criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.8,
    patience=SCHEDULER_PATIENCE,
    min_lr=1e-6,
)


def prepare_circumference(c):
    c = c.to(DEVICE, non_blocking=True)
    if STANDARDIZE_CIRCUMFERENCE:
        c = (c - C_MEAN) / C_STD
    return c


def calculate_batch_metrics(prediction, target):
    error = prediction - target
    return (
        torch.abs(error).sum().item(),
        torch.square(error).sum().item(),
        target.numel(),
    )


def run_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_absolute_error = 0.0
    total_squared_error = 0.0
    total_values = 0
    total_samples = 0

    for x, y, c in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        c = prepare_circumference(c)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            prediction = model(x, c)
            loss = criterion(prediction, y)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        abs_sum, sq_sum, n_values = calculate_batch_metrics(prediction, y)
        total_absolute_error += abs_sum
        total_squared_error += sq_sum
        total_values += n_values

    if total_samples == 0:
        raise RuntimeError("The DataLoader returned no samples.")

    return {
        "loss": total_loss / total_samples,
        "mae_mm": total_absolute_error / total_values,
        "rmse_mm": np.sqrt(total_squared_error / total_values),
    }


# =========================================================
# Training
# =========================================================
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_fat_thickness_mlp_XYC_remesh.pth")
LAST_MODEL_PATH = os.path.join(OUTPUT_DIR, "last_fat_thickness_mlp_XYC_remesh.pth")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "mlp_training_history_XYC_remesh.xlsx")

best_val_loss = np.inf
epochs_without_improvement = 0
history = []
training_start = time.perf_counter()

rng = np.random.default_rng(SAMPLING_SEED)
all_train_indices = np.arange(len(train_dataset))
rng.shuffle(all_train_indices)
index_pointer = 0

if not USE_FULL_TRAINING_SET_EACH_EPOCH:
    if SAMPLES_PER_EPOCH is None:
        samples_per_epoch = min(len(train_dataset), 10_000)
    else:
        samples_per_epoch = min(len(train_dataset), int(SAMPLES_PER_EPOCH))
    print("Train samples per epoch:", samples_per_epoch)
else:
    samples_per_epoch = len(train_dataset)
    print("Using the complete training dataset each epoch.")

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.perf_counter()

    if USE_FULL_TRAINING_SET_EACH_EPOCH:
        train_loader = make_loader(train_dataset, shuffle=True)
    else:
        if index_pointer + samples_per_epoch > len(train_dataset):
            rng.shuffle(all_train_indices)
            index_pointer = 0

        epoch_indices = all_train_indices[
            index_pointer:index_pointer + samples_per_epoch
        ]
        index_pointer += samples_per_epoch

        train_loader = make_loader(
            train_dataset,
            sampler=SubsetRandomSampler(epoch_indices.tolist()),
        )

    train_metrics = run_epoch(model, train_loader, criterion, optimizer)
    val_metrics = run_epoch(model, val_loader, criterion)

    scheduler.step(val_metrics["loss"])
    current_lr = optimizer.param_groups[0]["lr"]
    epoch_time = time.perf_counter() - epoch_start

    history.append(
        {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_metrics["loss"],
            "train_mae_mm": train_metrics["mae_mm"],
            "train_rmse_mm": train_metrics["rmse_mm"],
            "val_loss": val_metrics["loss"],
            "val_mae_mm": val_metrics["mae_mm"],
            "val_rmse_mm": val_metrics["rmse_mm"],
            "epoch_time_sec": epoch_time,
        }
    )

    print(
        f"Epoch [{epoch:03d}/{EPOCHS}] | "
        f"Train loss: {train_metrics['loss']:.6f} | "
        f"Train MAE: {train_metrics['mae_mm']:.4f} mm | "
        f"Val loss: {val_metrics['loss']:.6f} | "
        f"Val MAE: {val_metrics['mae_mm']:.4f} mm | "
        f"Val RMSE: {val_metrics['rmse_mm']:.4f} mm | "
        f"LR: {current_lr:.2e} | "
        f"Time: {epoch_time:.1f} s"
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_metrics["loss"],
        "input_dim": INPUT_DIM,
        "circumference_dim": CIRCUMFERENCE_DIM,
        "output_dim": OUTPUT_DIM,
        "dropout": DROPOUT,
        "standardize_circumference": STANDARDIZE_CIRCUMFERENCE,
        "circumference_mean": C_MEAN,
        "circumference_std": C_STD,
        "input_shape": train_dataset.input_shape,
        "frequency_indices": FREQUENCY_INDICES,
        "n_frequencies": N_SELECTED_FREQUENCIES,
    }

    # Save the latest state every epoch.
    torch.save(checkpoint, LAST_MODEL_PATH)

    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        epochs_without_improvement = 0
        checkpoint["best_val_loss"] = best_val_loss
        torch.save(checkpoint, BEST_MODEL_PATH)
        print("  Saved best model.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break

print(
    f"Training completed in "
    f"{(time.perf_counter() - training_start) / 60.0:.2f} minutes."
)

pd.DataFrame(history).to_excel(HISTORY_PATH, index=False)
print("Training history saved:", HISTORY_PATH)


# =========================================================
# Load best model and evaluate test data
# =========================================================
checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Loaded best model from epoch:", checkpoint["epoch"])

# Use normalization statistics saved with the model.
C_MEAN = float(checkpoint["circumference_mean"])
C_STD = float(checkpoint["circumference_std"])
STANDARDIZE_CIRCUMFERENCE = bool(
    checkpoint["standardize_circumference"]
)

test_metrics = run_epoch(model, test_loader, criterion)
print("\nFinal test results")
print(f"Test loss: {test_metrics['loss']:.6f}")
print(f"Test MAE: {test_metrics['mae_mm']:.4f} mm")
print(f"Test RMSE: {test_metrics['rmse_mm']:.4f} mm")

TEST_RESULT_PATH = os.path.join(OUTPUT_DIR, "test_results.txt")

with open(TEST_RESULT_PATH, "w") as f:
    f.write("Final test results\n")
    f.write(f"Test loss: {test_metrics['loss']:.6f}\n")
    f.write(f"Test MAE: {test_metrics['mae_mm']:.4f} mm\n")
    f.write(f"Test RMSE: {test_metrics['rmse_mm']:.4f} mm\n")
# =========================================================
# Predict and plot actual remeshed test geometries
# =========================================================
TEST_HAS_SKIN = test_file_has_skin(TEST_H5_PATH)
print("Test HDF5 contains skin:", TEST_HAS_SKIN)


def predict_one_test_sample(sample_index: int):
    x_test, y_test, c_test = test_dataset[sample_index]
    x_input = x_test.reshape(1, -1).to(DEVICE) if MODEL_TYPE == "mlp_basic" else x_test.unsqueeze(0).to(DEVICE)
    c_input = prepare_circumference(c_test.reshape(1, -1))

    with torch.no_grad():
        prediction = model(x_input, c_input)

    predicted_mm = prediction.squeeze(0).cpu().numpy().astype(np.float64)
    ground_truth_mm = y_test.cpu().numpy().astype(np.float64)

    return predicted_mm, ground_truth_mm


def plot_test_prediction(sample_index: int):
    predicted_fat_mm, ground_truth_fat_mm = predict_one_test_sample(sample_index)
    predicted_fat_for_geometry = np.maximum(predicted_fat_mm, 0.0)
    geometry = load_test_geometry(TEST_H5_PATH, sample_index)

    nodes = geometry["nodes"]
    tri = geometry["elements"]
    calf_boundary = geometry["calf_boundary"]
    gt_fat_boundary = geometry["gt_fat_boundary"]
    tibia_boundary = geometry["tibia_boundary"]
    fibula_boundary = geometry["fibula_boundary"]
    electrode_xy = geometry["electrode_xy"]
    gt_outer_points = geometry["gt_outer_points"]
    gt_inner_points = geometry["gt_inner_points"]
    circumference_mm = geometry["circumference_mm"]
    has_skin = geometry["has_skin"]
    # has_skin = False
    if has_skin:
        gt_skin_fat_boundary = geometry["skin_fat_boundary"]
        gt_skin_inner_points = geometry["gt_skin_inner_points"]
        gt_skin_mm = np.asarray(geometry["effective_skin_mm"], dtype=np.float64)
        result = create_skin_fat_muscle_boundary_polygons(calf_boundary=calf_boundary, electrode_xy=electrode_xy, skin_thickness_mm=gt_skin_mm, fat_thickness_mm=predicted_fat_for_geometry, circumference_mm=circumference_mm, tibia_boundary=tibia_boundary, fibula_boundary=fibula_boundary, bone_margin_mm=0.0)
        pred_fat_boundary = result[1]
        pred_effective_mm = np.asarray(result[-1], dtype=np.float64)
        fat_line_start = gt_skin_inner_points
    else:
        result = create_fat_muscle_boundary_polygon(calf_boundary=calf_boundary, electrode_xy=electrode_xy, fat_thickness_mm=predicted_fat_for_geometry, circumference_mm=circumference_mm, tibia_boundary=tibia_boundary, fibula_boundary=fibula_boundary, bone_margin_mm=0.0)
        pred_fat_boundary = result[0]
        pred_effective_mm = np.asarray(result[-1], dtype=np.float64)
        fat_line_start = gt_outer_points

    gt_effective_mm = np.asarray(ground_truth_fat_mm, dtype=np.float64)
    sample_mae = float(np.mean(np.abs(predicted_fat_mm - gt_effective_mm)))
    sample_rmse = float(np.sqrt(np.mean((predicted_fat_mm - gt_effective_mm) ** 2)))

    print(f"\nSample {sample_index}")
    print("Skin included:", has_skin)
    print("Ground truth fat [mm]:", np.round(gt_effective_mm, 2))
    print("Raw prediction fat [mm]:", np.round(predicted_fat_mm, 2))
    print("Negative predictions:", int(np.sum(predicted_fat_mm < 0)))

    if has_skin:
        print("Ground truth skin [mm]:", np.round(gt_skin_mm, 2))

    print(f"Sample MAE: {sample_mae:.4f} mm")
    print(f"Sample RMSE: {sample_rmse:.4f} mm")

    plt.figure(figsize=(9, 9))
    plt.triplot(nodes[:, 0], nodes[:, 1], tri, linewidth=0.35, alpha=0.25)

    if has_skin:
        plt.fill(calf_boundary[:, 0], calf_boundary[:, 1], color="mistyrose", alpha=0.8, label="Skin")
        plt.fill(gt_skin_fat_boundary[:, 0], gt_skin_fat_boundary[:, 1], color="khaki", alpha=0.8, label="Fat")
        plt.fill(gt_fat_boundary[:, 0], gt_fat_boundary[:, 1], color="lightcoral", alpha=0.35, label="Muscle")
    else:
        plt.fill(calf_boundary[:, 0], calf_boundary[:, 1], color="khaki", alpha=0.55, label="Fat")
        plt.fill(gt_fat_boundary[:, 0], gt_fat_boundary[:, 1], color="lightcoral", alpha=0.35, label="Muscle")

    plt.fill(tibia_boundary[:, 0], tibia_boundary[:, 1], color="lightsteelblue", alpha=0.9)
    plt.fill(fibula_boundary[:, 0], fibula_boundary[:, 1], color="lightsteelblue", alpha=0.9)
    plt.plot(calf_boundary[:, 0], calf_boundary[:, 1], "k-", linewidth=2.5, label="Calf surface")

    if has_skin:
        plt.plot(gt_skin_fat_boundary[:, 0], gt_skin_fat_boundary[:, 1], "m-", linewidth=2.2, label="Skin-fat boundary")

    plt.plot(gt_fat_boundary[:, 0], gt_fat_boundary[:, 1], "r-", linewidth=2.5, label="GT fat-muscle")
    plt.plot(pred_fat_boundary[:, 0], pred_fat_boundary[:, 1], "g--", linewidth=2.5, label="Predicted fat-muscle")
    plt.plot(tibia_boundary[:, 0], tibia_boundary[:, 1], "b-", linewidth=1.8, label="Tibia")
    plt.plot(fibula_boundary[:, 0], fibula_boundary[:, 1], "c-", linewidth=1.8, label="Fibula")

    if has_skin:
        for p0, p1 in zip(gt_outer_points, gt_skin_inner_points):
            plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "m-", linewidth=1.3, alpha=0.8)

    for electrode_index, (p0, p1) in enumerate(zip(fat_line_start, gt_inner_points)):
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "r-", linewidth=1.8)
        error_mm = predicted_fat_mm[electrode_index] - gt_effective_mm[electrode_index]
        plt.text(p1[0], p1[1], f"E{electrode_index + 1}\nGT: {gt_effective_mm[electrode_index]:.1f}\nP: {predicted_fat_mm[electrode_index]:.1f}\nΔ: {error_mm:+.1f}", fontsize=11, ha="center", va="center", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5})

    plt.scatter(electrode_xy[:, 0], electrode_xy[:, 1], edgecolors="black", linewidths=1.2, s=110, zorder=5)

    anatomy_text = "Skin + Fat" if has_skin else "Fat only"
    plt.title(f"{MODEL_TYPE} Fat-Thickness Prediction on Remeshed Geometry\n{anatomy_text} | Sample {sample_index} | C = {circumference_mm:.1f} mm | MAE = {sample_mae:.3f} mm | RMSE = {sample_rmse:.3f} mm", fontsize=15)
    plt.axis("equal")
    plt.axis("off")
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()

    figure_path = os.path.join(OUTPUT_DIR, f"test_prediction_sample_{sample_index:04d}.png")
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.show()
    
    
    plt.figure(figsize=(9, 9))
    plt.triplot(nodes[:, 0], nodes[:, 1], tri, linewidth=0.35, alpha=0.25)

    if has_skin:
        plt.fill(calf_boundary[:, 0], calf_boundary[:, 1], color="mistyrose", alpha=0.8, label="Skin")
        plt.fill(gt_skin_fat_boundary[:, 0], gt_skin_fat_boundary[:, 1], color="khaki", alpha=0.8, label="Fat")
        plt.fill(gt_fat_boundary[:, 0], gt_fat_boundary[:, 1], color="lightcoral", alpha=0.35, label="Muscle")
    else:
        plt.fill(calf_boundary[:, 0], calf_boundary[:, 1], color="khaki", alpha=0.55, label="Fat")
        plt.fill(gt_fat_boundary[:, 0], gt_fat_boundary[:, 1], color="lightcoral", alpha=0.35, label="Muscle")

    plt.fill(tibia_boundary[:, 0], tibia_boundary[:, 1], color="lightsteelblue", alpha=0.9)
    plt.fill(fibula_boundary[:, 0], fibula_boundary[:, 1], color="lightsteelblue", alpha=0.9)
    plt.plot(calf_boundary[:, 0], calf_boundary[:, 1], "k-", linewidth=2.5, label="Calf surface")

    if has_skin:
        plt.plot(gt_skin_fat_boundary[:, 0], gt_skin_fat_boundary[:, 1], "m-", linewidth=2.2, label="Skin-fat boundary")

    plt.plot(gt_fat_boundary[:, 0], gt_fat_boundary[:, 1], "r-", linewidth=2.5, label="GT fat-muscle")
    plt.plot(tibia_boundary[:, 0], tibia_boundary[:, 1], "b-", linewidth=1.8, label="Tibia")
    plt.plot(fibula_boundary[:, 0], fibula_boundary[:, 1], "c-", linewidth=1.8, label="Fibula")

    if has_skin:
        for p0, p1 in zip(gt_outer_points, gt_skin_inner_points):
            plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "m-", linewidth=1.3, alpha=0.8)

    for electrode_index, (p0, p1) in enumerate(zip(fat_line_start, gt_inner_points)):
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "r-", linewidth=1.8)
        plt.text(p1[0], p1[1], f"E{electrode_index + 1}\nGT: {gt_effective_mm[electrode_index]:.1f}", fontsize=11, ha="center", va="center", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5})

    plt.scatter(electrode_xy[:, 0], electrode_xy[:, 1], edgecolors="black", linewidths=1.2, s=110, zorder=5)

    anatomy_text = "Skin + Fat" if has_skin else "Fat only"
    plt.title(f"{MODEL_TYPE} Fat-Thickness Prediction on Remeshed Geometry\n{anatomy_text} | Sample {sample_index} | C = {circumference_mm:.1f} mm | MAE = {sample_mae:.3f} mm | RMSE = {sample_rmse:.3f} mm", fontsize=15)
    plt.axis("equal")
    plt.axis("off")
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.show()

plot_end = min(len(test_dataset), TEST_PLOT_START_INDEX + NUMBER_OF_TEST_PLOTS)

for sample_index in range(TEST_PLOT_START_INDEX, plot_end):
    plot_test_prediction(sample_index)

train_dataset.close()
val_dataset.close()
test_dataset.close()
