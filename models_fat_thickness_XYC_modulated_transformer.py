# -*- coding: utf-8 -*-
"""
Models for calf fat-thickness estimation from multi-frequency EIT.

Included models
---------------
1. FatThicknessMLP
   Circumference-modulated MLP using FiLM rather than simple concatenation.

2. ProtocolAwareFatThicknessTransformer
   Treats each EIT measurement channel as one token. Each token contains:
       - multi-frequency real/imaginary voltage features;
       - the injection electrode pair;
       - the voltage-measurement electrode pair;
       - a learned measurement-channel embedding.

   Sixteen learned electrode queries decode one fat-thickness value per
   electrode location.

Expected EIT input
------------------
x: [batch, n_measurements, 2, n_frequencies]

The MLP also accepts a flattened input:
x: [batch, n_measurements * 2 * n_frequencies]

Circumference input
-------------------
circumference: [batch, circumference_dim]

The training script should standardize circumference using statistics obtained
only from the training dataset.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[Sequence[int]]]


#====================
# basic MLP
#===================

class FatThicknessMLP_basic(nn.Module):
    """Predict 16 fat-thickness values from EIT voltage and circumference."""

    def __init__(
        self,
        input_dim,
        circumference_dim=1,
        output_dim=16,
        dropout=0.2,
    ):
        super().__init__()

        self.voltage_encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.circumference_encoder = nn.Sequential(
            nn.Linear(circumference_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.output_head = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x, circumference):
        x = x.reshape(x.shape[0], -1)
        circumference = circumference.reshape(circumference.shape[0], -1)

        voltage_features = self.voltage_encoder(x)
        circumference_features = self.circumference_encoder(circumference)
        features = torch.cat((voltage_features, circumference_features), dim=1)
        return self.output_head(features)

# =========================================================
# Protocol utilities
# =========================================================
def load_protocol_indices_csv(
    csv_path: Union[str, Path],
    expected_measurements: Optional[int] = 208,
) -> np.ndarray:
    """
    Load the four electrode indices for every EIT measurement.

    Expected logical column order:
        [high-current, low-current, high-potential, low-potential]

    The loader first searches for common column names. If they are unavailable,
    it uses the first four numeric columns.

    Returns
    -------
    protocol_indices : np.ndarray, shape [n_measurements, 4]
        Integer electrode indices. Both zero-based and one-based CSV files are
        accepted; conversion to zero-based indexing is performed later by
        ``protocol_indices_to_features``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Protocol CSV was not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)

    normalized_columns = {
        str(column).strip().lower().replace(" ", "").replace("_", ""): column
        for column in dataframe.columns
    }

    candidate_groups = (
        ("hc", "lc", "hp", "lp"),
        ("hcur", "lcur", "hpot", "lpot"),
        ("source+", "source-", "measure+", "measure-"),
        ("inj+", "inj-", "meas+", "meas-"),
        ("a", "b", "m", "n"),
    )

    selected_columns = None
    for group in candidate_groups:
        normalized_group = tuple(
            name.lower().replace(" ", "").replace("_", "")
            for name in group
        )
        if all(name in normalized_columns for name in normalized_group):
            selected_columns = [
                normalized_columns[name] for name in normalized_group
            ]
            break

    if selected_columns is None:
        numeric = dataframe.select_dtypes(include=[np.number])
        if numeric.shape[1] < 4:
            raise ValueError(
                "The protocol CSV must contain at least four numeric columns "
                "for [hc, lc, hp, lp]."
            )
        selected_columns = list(numeric.columns[:4])

    protocol_indices = (
        dataframe[selected_columns]
        .to_numpy(dtype=np.int64, copy=True)
    )

    if protocol_indices.ndim != 2 or protocol_indices.shape[1] != 4:
        raise ValueError(
            "Protocol indices must have shape [n_measurements, 4], "
            f"received {protocol_indices.shape}."
        )

    if expected_measurements is not None:
        if len(protocol_indices) != expected_measurements:
            raise ValueError(
                f"Expected {expected_measurements} protocol rows, "
                f"found {len(protocol_indices)}."
            )

    return protocol_indices


def _to_zero_based_protocol(
    protocol_indices: ArrayLike,
    n_electrodes: int,
) -> torch.Tensor:
    """Validate protocol indices and convert one-based indices when needed."""
    protocol = torch.as_tensor(protocol_indices, dtype=torch.long).clone()

    if protocol.ndim != 2 or protocol.shape[1] != 4:
        raise ValueError(
            "protocol_indices must have shape [n_measurements, 4], "
            f"received {tuple(protocol.shape)}."
        )

    minimum = int(protocol.min().item())
    maximum = int(protocol.max().item())

    # Detect conventional one-based electrode numbering.
    if minimum >= 1 and maximum <= n_electrodes:
        protocol -= 1
        minimum -= 1
        maximum -= 1

    if minimum < 0 or maximum >= n_electrodes:
        raise ValueError(
            "Protocol electrode indices are outside the valid range. "
            f"After zero-base conversion: min={minimum}, max={maximum}, "
            f"expected 0 to {n_electrodes - 1}."
        )

    return protocol


def protocol_indices_to_features(
    protocol_indices: ArrayLike,
    n_electrodes: int = 16,
) -> torch.Tensor:
    """
    Convert [hc, lc, hp, lp] indices into circular geometry features.

    Features for each measurement token
    -----------------------------------
    For every one of the four electrode indices:
        sin(theta), cos(theta)                         -> 8 features

    Circular pair relationships:
        injection-pair circular distance
        measurement-pair circular distance
        hc-to-hp circular distance
        lc-to-lp circular distance                    -> 4 features

    Signed circular relationships:
        signed hc-to-lc displacement
        signed hp-to-lp displacement
        signed injection-midpoint to measurement-midpoint displacement
                                                         -> 3 features

    Total protocol feature dimension = 15.
    """
    protocol = _to_zero_based_protocol(
        protocol_indices=protocol_indices,
        n_electrodes=n_electrodes,
    )

    protocol_float = protocol.to(torch.float32)
    angles = 2.0 * math.pi * protocol_float / float(n_electrodes)

    circular_features = torch.stack(
        (torch.sin(angles), torch.cos(angles)),
        dim=-1,
    ).reshape(protocol.shape[0], -1)

    hc, lc, hp, lp = [
        protocol_float[:, index] for index in range(4)
    ]

    def unsigned_circular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raw = torch.abs(a - b)
        wrapped = torch.minimum(raw, float(n_electrodes) - raw)
        return wrapped / (0.5 * float(n_electrodes))

    def signed_circular_displacement(
        source: torch.Tensor,
        destination: torch.Tensor,
    ) -> torch.Tensor:
        half = 0.5 * float(n_electrodes)
        displacement = torch.remainder(
            destination - source + half,
            float(n_electrodes),
        ) - half
        return displacement / half

    distance_features = torch.stack(
        (
            unsigned_circular_distance(hc, lc),
            unsigned_circular_distance(hp, lp),
            unsigned_circular_distance(hc, hp),
            unsigned_circular_distance(lc, lp),
        ),
        dim=1,
    )

    # Circular midpoint represented through vector averaging, followed by the
    # signed angular difference between injection and measurement midpoints.
    injection_mid_sin = torch.sin(angles[:, 0]) + torch.sin(angles[:, 1])
    injection_mid_cos = torch.cos(angles[:, 0]) + torch.cos(angles[:, 1])
    measurement_mid_sin = torch.sin(angles[:, 2]) + torch.sin(angles[:, 3])
    measurement_mid_cos = torch.cos(angles[:, 2]) + torch.cos(angles[:, 3])

    injection_mid_angle = torch.atan2(
        injection_mid_sin,
        injection_mid_cos,
    )
    measurement_mid_angle = torch.atan2(
        measurement_mid_sin,
        measurement_mid_cos,
    )
    midpoint_difference = torch.atan2(
        torch.sin(measurement_mid_angle - injection_mid_angle),
        torch.cos(measurement_mid_angle - injection_mid_angle),
    ) / math.pi

    signed_features = torch.stack(
        (
            signed_circular_displacement(hc, lc),
            signed_circular_displacement(hp, lp),
            midpoint_difference,
        ),
        dim=1,
    )

    return torch.cat(
        (
            circular_features,
            distance_features,
            signed_features,
        ),
        dim=1,
    )


# =========================================================
# Shared building blocks
# =========================================================
class FiLM(nn.Module):
    """
    Feature-wise linear modulation.

    Given feature vector h and condition c:
        h_modulated = (1 + gamma(c)) * h + beta(c)

    The final conditioning layer is initialized to zero, so the module starts
    as an identity transformation. This makes optimization more stable.
    """

    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.condition_network = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * feature_dim),
        )

        final_layer = self.condition_network[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.condition_network(condition).chunk(2, dim=-1)

        # Support both [B, D] and [B, T, D].
        while gamma.ndim < features.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return (1.0 + gamma) * features + beta


class ModulatedMLPBlock(nn.Module):
    """Linear-normalization-activation block followed by circumference FiLM."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        condition_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.normalization = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.modulation = FiLM(
            condition_dim=condition_dim,
            feature_dim=output_dim,
            hidden_dim=max(32, condition_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
            ) -> torch.Tensor:
        x = self.linear(x)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.modulation(x, condition)
        return self.dropout(x)


# =========================================================
# Circumference-modulated MLP
# =========================================================
class FatThicknessMLP(nn.Module):
    """
    Circumference-modulated MLP.

    Unlike the previous model, circumference is not merely concatenated at the
    output head. It controls the scale and bias of hidden voltage features at
    all three encoder stages through FiLM.

    The class name is kept as ``FatThicknessMLP`` so existing training scripts
    can import it without changing the import statement.
    """

    def __init__(
        self,
        input_dim: int,
        circumference_dim: int = 1,
        output_dim: int = 16,
        dropout: float = 0.2,
        condition_dim: int = 64,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.circumference_dim = int(circumference_dim)
        self.output_dim = int(output_dim)

        self.circumference_encoder = nn.Sequential(
            nn.Linear(circumference_dim, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
        )

        self.block_1 = ModulatedMLPBlock(
            input_dim=input_dim,
            output_dim=1024,
            condition_dim=condition_dim,
            dropout=dropout,
        )
        self.block_2 = ModulatedMLPBlock(
            input_dim=1024,
            output_dim=512,
            condition_dim=condition_dim,
            dropout=dropout,
        )
        self.block_3 = ModulatedMLPBlock(
            input_dim=512,
            output_dim=256,
            condition_dim=condition_dim,
            dropout=dropout,
        )

        self.output_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        circumference: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        circumference = circumference.reshape(batch_size, -1)

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Flattened EIT input dimension is {x.shape[1]}, "
                f"but the model expects {self.input_dim}."
            )

        if circumference.shape[1] != self.circumference_dim:
            raise ValueError(
                f"Circumference dimension is {circumference.shape[1]}, "
                f"but the model expects {self.circumference_dim}."
            )

        condition = self.circumference_encoder(circumference)

        features = self.block_1(x, condition)
        features = self.block_2(features, condition)
        features = self.block_3(features, condition)

        return self.output_head(features)


# Explicit descriptive alias.
CircumferenceModulatedFatThicknessMLP = FatThicknessMLP


# =========================================================
# Protocol-aware transformer
# =========================================================
class ProtocolAwareFatThicknessTransformer(nn.Module):
    """
    Injection-protocol-aware transformer for 16-point fat prediction.

    Measurement tokens
    ------------------
    Each of the 208 measurement tokens combines:
        1. voltage features [real/imaginary x frequencies],
        2. protocol geometry derived from [hc, lc, hp, lp],
        3. a learned channel-position embedding.

    Transformer encoder
    -------------------
    Learns relationships between all measurement channels. Because protocol
    features are attached to every channel, attention can distinguish channels
    that have different current-injection and voltage-measurement patterns.

    Electrode-query decoder
    -----------------------
    Sixteen learned queries attend to the encoded 208 measurement tokens.
    Query i predicts the fat thickness associated with electrode location i.

    Circumference
    -------------
    Circumference modulates both measurement tokens and electrode queries using
    FiLM. It is not simply concatenated.
    """

    def __init__(
        self,
        protocol_indices: ArrayLike,
        n_frequencies: int = 10,
        n_components: int = 2,
        n_electrodes: int = 16,
        output_dim: int = 16,
        circumference_dim: int = 1,
        d_model: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 2,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        condition_dim: int = 64,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}."
            )

        protocol_zero_based = _to_zero_based_protocol(
            protocol_indices=protocol_indices,
            n_electrodes=n_electrodes,
        )
        protocol_features = protocol_indices_to_features(
            protocol_indices=protocol_zero_based,
            n_electrodes=n_electrodes,
        )

        self.n_measurements = int(protocol_zero_based.shape[0])
        self.n_frequencies = int(n_frequencies)
        self.n_components = int(n_components)
        self.n_electrodes = int(n_electrodes)
        self.output_dim = int(output_dim)
        self.circumference_dim = int(circumference_dim)
        self.d_model = int(d_model)

        if output_dim != n_electrodes:
            raise ValueError(
                "This electrode-query model expects output_dim to equal "
                f"n_electrodes. Received output_dim={output_dim}, "
                f"n_electrodes={n_electrodes}."
            )

        self.register_buffer(
            "protocol_indices",
            protocol_zero_based,
            persistent=True,
        )
        self.register_buffer(
            "protocol_features",
            protocol_features,
            persistent=True,
        )

        signal_dim = n_components * n_frequencies
        protocol_dim = int(protocol_features.shape[1])

        self.signal_projection = nn.Sequential(
            nn.Linear(signal_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.protocol_projection = nn.Sequential(
            nn.Linear(protocol_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.measurement_embedding = nn.Embedding(
            self.n_measurements,
            d_model,
        )

        self.circumference_encoder = nn.Sequential(
            nn.Linear(circumference_dim, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
        )

        self.measurement_film = FiLM(
            condition_dim=condition_dim,
            feature_dim=d_model,
            hidden_dim=condition_dim,
        )
        self.query_film = FiLM(
            condition_dim=condition_dim,
            feature_dim=d_model,
            hidden_dim=condition_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.electrode_queries = nn.Parameter(
            torch.empty(n_electrodes, d_model)
        )
        nn.init.normal_(
            self.electrode_queries,
            mean=0.0,
            std=0.02,
        )

        self.token_dropout = nn.Dropout(dropout)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _reshape_eit_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input to [batch, measurement, component*frequency].

        Supported shapes:
            [B, M, 2, F]
            [B, M, 2*F]
            [B, M*2*F]
        """
        batch_size = x.shape[0]
        expected_flat_dim = (
            self.n_measurements
            * self.n_components
            * self.n_frequencies
        )

        if x.ndim == 4:
            expected_shape = (
                self.n_measurements,
                self.n_components,
                self.n_frequencies,
            )
            if tuple(x.shape[1:]) != expected_shape:
                raise ValueError(
                    f"Expected EIT input [B, {expected_shape[0]}, "
                    f"{expected_shape[1]}, {expected_shape[2]}], "
                    f"received {tuple(x.shape)}."
                )
            return x.reshape(
                batch_size,
                self.n_measurements,
                -1,
            )

        if x.ndim == 3:
            expected_token_dim = (
                self.n_components * self.n_frequencies
            )
            if x.shape[1] != self.n_measurements:
                raise ValueError(
                    f"Expected {self.n_measurements} measurement tokens, "
                    f"received {x.shape[1]}."
                )
            if x.shape[2] != expected_token_dim:
                raise ValueError(
                    f"Expected token feature dimension {expected_token_dim}, "
                    f"received {x.shape[2]}."
                )
            return x

        if x.ndim == 2:
            if x.shape[1] != expected_flat_dim:
                raise ValueError(
                    f"Expected flattened input dimension {expected_flat_dim}, "
                    f"received {x.shape[1]}."
                )
            return x.reshape(
                batch_size,
                self.n_measurements,
                self.n_components * self.n_frequencies,
            )

        raise ValueError(
            "Unsupported EIT input shape. Expected 2D, 3D, or 4D input, "
            f"received {tuple(x.shape)}."
        )

    def forward(
        self,
        x: torch.Tensor,
        circumference: torch.Tensor,
    ) -> torch.Tensor:
        x = self._reshape_eit_input(x)
        batch_size = x.shape[0]

        circumference = circumference.reshape(batch_size, -1)
        if circumference.shape[1] != self.circumference_dim:
            raise ValueError(
                f"Circumference dimension is {circumference.shape[1]}, "
                f"but the model expects {self.circumference_dim}."
            )

        condition = self.circumference_encoder(circumference)

        signal_tokens = self.signal_projection(x)

        protocol_tokens = self.protocol_projection(
            self.protocol_features
        ).unsqueeze(0)

        measurement_ids = torch.arange(
            self.n_measurements,
            device=x.device,
        )
        position_tokens = self.measurement_embedding(
            measurement_ids
        ).unsqueeze(0)

        tokens = (
            signal_tokens
            + protocol_tokens
            + position_tokens
        )
        tokens = self.measurement_film(tokens, condition)
        tokens = self.token_dropout(tokens)

        memory = self.encoder(tokens)

        queries = self.electrode_queries.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        queries = self.query_film(queries, condition)

        decoded_queries = self.decoder(
            tgt=queries,
            memory=memory,
        )

        prediction = self.output_head(
            decoded_queries
        ).squeeze(-1)

        return prediction


# Short alias for convenient imports.
FatThicknessProtocolTransformer = ProtocolAwareFatThicknessTransformer


# =========================================================
# Smoke test
# =========================================================
if __name__ == "__main__":
    batch_size = 4
    n_measurements = 208
    n_frequencies = 10
    n_electrodes = 16

    # Example protocol only for shape testing. Real training must use the
    # exact protocol rows corresponding to the X measurement-channel order.
    example_protocol = np.zeros((n_measurements, 4), dtype=np.int64)
    for index in range(n_measurements):
        example_protocol[index] = (
            index % n_electrodes,
            (index + 1) % n_electrodes,
            (index + 2) % n_electrodes,
            (index + 3) % n_electrodes,
        )

    x = torch.randn(
        batch_size,
        n_measurements,
        2,
        n_frequencies,
    )
    circumference = torch.randn(batch_size, 1)

    mlp = FatThicknessMLP(
        input_dim=n_measurements * 2 * n_frequencies,
        circumference_dim=1,
        output_dim=n_electrodes,
    )
    mlp_output = mlp(x, circumference)
    print("Modulated MLP output:", mlp_output.shape)

    transformer = ProtocolAwareFatThicknessTransformer(
        protocol_indices=example_protocol,
        n_frequencies=n_frequencies,
        n_components=2,
        n_electrodes=n_electrodes,
        output_dim=n_electrodes,
    )
    transformer_output = transformer(x, circumference)
    print("Protocol transformer output:", transformer_output.shape)
