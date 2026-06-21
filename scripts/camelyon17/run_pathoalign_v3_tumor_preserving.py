#!/usr/bin/env python3
"""PathoAlign v3: tumor-preserving low-rank nuisance removal.

V2 showed that unconditional site-component subtraction reduced center leakage
but removed strongly tumor-predictive information. V3 constrains the removable
component using:

1. A low-rank nuisance encoder and decoder.
2. A center-prediction objective on the nuisance representation.
3. A gradient-reversed tumor adversary on the nuisance representation.
4. A frozen tumor teacher trained on the original standardized features.
5. A KL preservation loss between teacher and cleaned-feature predictions.
6. A center adversary on the cleaned diagnostic representation.
7. A magnitude penalty on the removable component.

Development selection uses center-1 validation only. Held-out center-2 test
performance is evaluated only for the baseline teacher and the selected
eligible configuration.

This is a feature-level mechanism experiment, not a clinical model.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset

from run_pathoalign_head_v01_sweep import (
    extract_features,
    load_feature_model,
    stratified_sample,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_from_source_train(
    x: np.ndarray,
    meta: pd.DataFrame,
) -> np.ndarray:
    train_mask = meta["split"].eq("train").to_numpy()

    mean = x[train_mask].mean(axis=0, keepdims=True)
    std = x[train_mask].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    return ((x - mean) / std).astype(np.float32)


class GradientReversalFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        coefficient: float,
    ) -> torch.Tensor:
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(
        ctx,
        gradient: torch.Tensor,
    ):
        return -ctx.coefficient * gradient, None


def gradient_reverse(
    x: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    return GradientReversalFunction.apply(
        x,
        float(coefficient),
    )


class DiagnosticModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_sites: int,
        hidden_dim: int = 256,
        representation_dim: int = 128,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, representation_dim),
            nn.LayerNorm(representation_dim),
            nn.GELU(),
        )

        self.tumor_head = nn.Linear(
            representation_dim,
            2,
        )

        self.center_adversary = nn.Sequential(
            nn.Linear(representation_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_sites),
        )

    def forward(
        self,
        x: torch.Tensor,
        center_grl: float = 0.0,
    ):
        representation = self.encoder(x)

        center_logits = self.center_adversary(
            gradient_reverse(
                representation,
                center_grl,
            )
        )

        return {
            "representation": representation,
            "tumor_logits": self.tumor_head(
                representation
            ),
            "center_logits": center_logits,
        }


class TumorPreservingNuisanceModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nuisance_dim: int,
        n_sites: int,
        dropout: float,
    ):
        super().__init__()

        self.nuisance_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, nuisance_dim),
            nn.LayerNorm(nuisance_dim),
            nn.GELU(),
        )

        self.center_head = nn.Sequential(
            nn.Linear(nuisance_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_sites),
        )

        self.tumor_adversary = nn.Sequential(
            nn.Linear(nuisance_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(nuisance_dim, 128),
            nn.GELU(),
            nn.Linear(128, input_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        tumor_grl: float,
    ):
        z_nuisance = self.nuisance_encoder(x)

        return {
            "z_nuisance": z_nuisance,
            "nuisance_center_logits": self.center_head(
                z_nuisance
            ),
            "tumor_adversary_logits":
                self.tumor_adversary(
                    gradient_reverse(
                        z_nuisance,
                        tumor_grl,
                    )
                ),
            "nuisance_component":
                self.decoder(z_nuisance),
        }


class PathoAlignV3Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nuisance_dim: int,
        n_sites: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()

        self.alpha = float(alpha)

        self.nuisance = TumorPreservingNuisanceModel(
            input_dim=input_dim,
            nuisance_dim=nuisance_dim,
            n_sites=n_sites,
            dropout=dropout,
        )

        self.diagnostic = DiagnosticModel(
            input_dim=input_dim,
            n_sites=n_sites,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        tumor_grl: float,
        center_grl: float,
    ):
        nuisance_output = self.nuisance(
            x=x,
            tumor_grl=tumor_grl,
        )

        cleaned = (
            x
            - self.alpha
            * nuisance_output[
                "nuisance_component"
            ]
        )

        diagnostic_output = self.diagnostic(
            x=cleaned,
            center_grl=center_grl,
        )

        return {
            **nuisance_output,
            **diagnostic_output,
            "cleaned_features": cleaned,
        }


@dataclass(frozen=True)
class CandidateConfig:
    alpha: float
    nuisance_dim: int
    tumor_adv_weight: float
    preserve_weight: float

    @property
    def name(self) -> str:
        return (
            f"a{self.alpha:g}"
            f"_d{self.nuisance_dim}"
            f"_ta{self.tumor_adv_weight:g}"
            f"_p{self.preserve_weight:g}"
        )


@dataclass
class TrainedCandidate:
    config: CandidateConfig
    best_epoch: int
    model_state: Dict[str, torch.Tensor]


def binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "n": int(len(y_true)),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
    }


def site_labels_from_meta(
    meta: pd.DataFrame,
) -> tuple[np.ndarray, List[int]]:
    source_mask = meta["split"].isin(
        ["train", "id_val"]
    )

    source_centers = sorted(
        meta[source_mask][
            "center"
        ].unique().tolist()
    )

    center_to_label = {
        center: index
        for index, center
        in enumerate(source_centers)
    }

    labels = (
        meta["center"]
        .map(
            lambda center:
                center_to_label.get(
                    center,
                    -1,
                )
        )
        .to_numpy(dtype=int)
    )

    return labels, source_centers


def train_teacher(
    x: np.ndarray,
    y: np.ndarray,
    site_labels: np.ndarray,
    meta: pd.DataFrame,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    device: torch.device,
):
    set_seed(seed)

    train_mask = meta[
        "split"
    ].eq("train").to_numpy()

    n_sites = (
        int(
            site_labels[
                train_mask
            ].max()
        )
        + 1
    )

    model = DiagnosticModel(
        input_dim=x.shape[1],
        n_sites=n_sites,
        dropout=dropout,
    ).to(device)

    dataset = TensorDataset(
        torch.tensor(
            x[train_mask],
            dtype=torch.float32,
        ),
        torch.tensor(
            y[train_mask],
            dtype=torch.long,
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loss_function = nn.CrossEntropyLoss()

    best_state = None
    best_val_auc = -math.inf
    best_id_val_auc = -math.inf
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                xb,
                center_grl=0.0,
            )

            loss = loss_function(
                output["tumor_logits"],
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()
            losses.append(float(loss.item()))

        metrics, _ = (
            evaluate_diagnostic_model(
                model=model,
                x=x,
                y=y,
                meta=meta,
                splits=[
                    "train",
                    "id_val",
                    "val",
                ],
                batch_size=batch_size,
                device=device,
            )
        )

        val_row = metrics[
            metrics["split"].eq("val")
        ].iloc[0]

        id_val_row = metrics[
            metrics["split"].eq("id_val")
        ].iloc[0]

        val_auc = float(
            val_row["auc"]
        )

        id_val_auc = float(
            id_val_row["auc"]
        )

        history.append({
            "seed": seed,
            "epoch": epoch,
            "mean_train_loss": float(
                np.mean(losses)
            ),
            "val_auc": val_auc,
            "id_val_auc": id_val_auc,
        })

        if (
            val_auc > best_val_auc
            or (
                math.isclose(
                    val_auc,
                    best_val_auc,
                )
                and id_val_auc
                > best_id_val_auc
            )
        ):
            best_val_auc = val_auc
            best_id_val_auc = id_val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )

    if best_state is None:
        raise RuntimeError(
            "No teacher checkpoint selected"
        )

    model.load_state_dict(best_state)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return (
        model,
        best_epoch,
        pd.DataFrame(history),
    )


@torch.no_grad()
def teacher_probabilities(
    teacher: DiagnosticModel,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(
            torch.tensor(
                x,
                dtype=torch.float32,
            )
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    probabilities = []

    teacher.eval()

    for (xb,) in loader:
        output = teacher(
            xb.to(device),
            center_grl=0.0,
        )

        probabilities.append(
            torch.softmax(
                output["tumor_logits"],
                dim=1,
            ).cpu().numpy()
        )

    return np.vstack(
        probabilities
    ).astype(np.float32)


@torch.no_grad()
def evaluate_diagnostic_model(
    model: DiagnosticModel,
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    splits: List[str],
    batch_size: int,
    device: torch.device,
):
    loader = DataLoader(
        TensorDataset(
            torch.tensor(
                x,
                dtype=torch.float32,
            )
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    probabilities = []
    representations = []

    model.eval()

    for (xb,) in loader:
        output = model(
            xb.to(device),
            center_grl=0.0,
        )

        probabilities.append(
            torch.softmax(
                output["tumor_logits"],
                dim=1,
            )[:, 1].cpu().numpy()
        )

        representations.append(
            output[
                "representation"
            ].cpu().numpy()
        )

    probabilities_array = (
        np.concatenate(probabilities)
    )

    representations_array = (
        np.vstack(representations)
    )

    rows = []

    for split in splits:
        mask = meta[
            "split"
        ].eq(split).to_numpy()

        if not mask.any():
            continue

        rows.append({
            "split": split,
            **binary_metrics(
                y[mask],
                probabilities_array[mask],
            ),
        })

    return (
        pd.DataFrame(rows),
        representations_array,
    )


def kl_preservation_loss(
    student_logits: torch.Tensor,
    teacher_probabilities_batch:
        torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    teacher = (
        teacher_probabilities_batch
        .clamp_min(1e-7)
    )

    if not math.isclose(
        temperature,
        1.0,
    ):
        teacher = (
            teacher.pow(
                1.0 / temperature
            )
        )

        teacher = (
            teacher
            / teacher.sum(
                dim=1,
                keepdim=True,
            )
        )

    student_log_probabilities = (
        F.log_softmax(
            student_logits
            / temperature,
            dim=1,
        )
    )

    return (
        F.kl_div(
            student_log_probabilities,
            teacher,
            reduction="batchmean",
        )
        * temperature
        * temperature
    )


def train_candidate(
    x: np.ndarray,
    y: np.ndarray,
    site_labels: np.ndarray,
    teacher_probs: np.ndarray,
    meta: pd.DataFrame,
    config: CandidateConfig,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    center_loss_weight: float,
    diagnostic_center_adv_weight: float,
    component_l2_weight: float,
    temperature: float,
    device: torch.device,
):
    set_seed(seed)

    train_mask = meta[
        "split"
    ].eq("train").to_numpy()

    n_sites = (
        int(
            site_labels[
                train_mask
            ].max()
        )
        + 1
    )

    model = PathoAlignV3Model(
        input_dim=x.shape[1],
        nuisance_dim=config.nuisance_dim,
        n_sites=n_sites,
        alpha=config.alpha,
        dropout=dropout,
    ).to(device)

    dataset = TensorDataset(
        torch.tensor(
            x[train_mask],
            dtype=torch.float32,
        ),
        torch.tensor(
            y[train_mask],
            dtype=torch.long,
        ),
        torch.tensor(
            site_labels[train_mask],
            dtype=torch.long,
        ),
        torch.tensor(
            teacher_probs[train_mask],
            dtype=torch.float32,
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    tumor_loss_function = (
        nn.CrossEntropyLoss()
    )

    center_loss_function = (
        nn.CrossEntropyLoss()
    )

    best_state = None
    best_val_auc = -math.inf
    best_val_accuracy = -math.inf
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        total_losses = []
        tumor_losses = []
        center_losses = []
        tumor_adv_losses = []
        preserve_losses = []
        diagnostic_center_losses = []
        component_losses = []

        for (
            xb,
            yb,
            sb,
            teacher_pb,
        ) in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)
            teacher_pb = (
                teacher_pb.to(device)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                x=xb,
                tumor_grl=(
                    config.tumor_adv_weight
                ),
                center_grl=(
                    diagnostic_center_adv_weight
                ),
            )

            tumor_loss = (
                tumor_loss_function(
                    output["tumor_logits"],
                    yb,
                )
            )

            center_loss = (
                center_loss_function(
                    output["nuisance_center_logits"],
                    sb,
                )
            )

            tumor_adv_loss = (
                tumor_loss_function(
                    output[
                        "tumor_adversary_logits"
                    ],
                    yb,
                )
            )

            preserve_loss = (
                kl_preservation_loss(
                    student_logits=output[
                        "tumor_logits"
                    ],
                    teacher_probabilities_batch=
                        teacher_pb,
                    temperature=temperature,
                )
            )

            diagnostic_center_loss = (
                center_loss_function(
                    output[
                        "center_logits"
                    ],
                    sb,
                )
            )

            component_l2 = (
                output[
                    "nuisance_component"
                ]
                .pow(2)
                .mean()
            )

            total_loss = (
                tumor_loss
                + center_loss_weight
                * center_loss
                + tumor_adv_loss
                + config.preserve_weight
                * preserve_loss
                + diagnostic_center_loss
                + component_l2_weight
                * component_l2
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_losses.append(
                float(
                    total_loss.item()
                )
            )

            tumor_losses.append(
                float(
                    tumor_loss.item()
                )
            )

            center_losses.append(
                float(
                    center_loss.item()
                )
            )

            tumor_adv_losses.append(
                float(
                    tumor_adv_loss.item()
                )
            )

            preserve_losses.append(
                float(
                    preserve_loss.item()
                )
            )

            diagnostic_center_losses.append(
                float(
                    diagnostic_center_loss.item()
                )
            )

            component_losses.append(
                float(
                    component_l2.item()
                )
            )

        development, _, _, _ = (
            evaluate_candidate(
                model=model,
                x=x,
                y=y,
                meta=meta,
                splits=[
                    "train",
                    "id_val",
                    "val",
                ],
                batch_size=batch_size,
                device=device,
            )
        )

        val_row = development[
            development["split"].eq(
                "val"
            )
        ].iloc[0]

        val_auc = float(
            val_row["auc"]
        )

        val_accuracy = float(
            val_row["accuracy"]
        )

        history.append({
            "seed": seed,
            "config": config.name,
            "epoch": epoch,
            "mean_total_loss": float(
                np.mean(total_losses)
            ),
            "mean_tumor_loss": float(
                np.mean(tumor_losses)
            ),
            "mean_center_loss": float(
                np.mean(center_losses)
            ),
            "mean_tumor_adversary_loss":
                float(
                    np.mean(
                        tumor_adv_losses
                    )
                ),
            "mean_preservation_loss":
                float(
                    np.mean(
                        preserve_losses
                    )
                ),
            "mean_diagnostic_center_loss":
                float(
                    np.mean(
                        diagnostic_center_losses
                    )
                ),
            "mean_component_l2":
                float(
                    np.mean(
                        component_losses
                    )
                ),
            "val_accuracy": val_accuracy,
            "val_auc": val_auc,
        })

        if (
            val_auc > best_val_auc
            or (
                math.isclose(
                    val_auc,
                    best_val_auc,
                )
                and val_accuracy
                > best_val_accuracy
            )
        ):
            best_val_auc = val_auc
            best_val_accuracy = (
                val_accuracy
            )

            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )

    if best_state is None:
        raise RuntimeError(
            "No checkpoint selected for "
            f"{config.name}"
        )

    model.load_state_dict(best_state)

    return (
        model,
        pd.DataFrame(history),
        TrainedCandidate(
            config=config,
            best_epoch=best_epoch,
            model_state=best_state,
        ),
    )


@torch.no_grad()
def evaluate_candidate(
    model: PathoAlignV3Model,
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    splits: List[str],
    batch_size: int,
    device: torch.device,
):
    loader = DataLoader(
        TensorDataset(
            torch.tensor(
                x,
                dtype=torch.float32,
            )
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    probabilities = []
    diagnostic_representations = []
    nuisance_representations = []
    nuisance_components = []

    model.eval()

    for (xb,) in loader:
        output = model(
            x=xb.to(device),
            tumor_grl=0.0,
            center_grl=0.0,
        )

        probabilities.append(
            torch.softmax(
                output["tumor_logits"],
                dim=1,
            )[:, 1].cpu().numpy()
        )

        diagnostic_representations.append(
            output[
                "representation"
            ].cpu().numpy()
        )

        nuisance_representations.append(
            output[
                "z_nuisance"
            ].cpu().numpy()
        )

        nuisance_components.append(
            output[
                "nuisance_component"
            ].cpu().numpy()
        )

    probabilities_array = (
        np.concatenate(probabilities)
    )

    diagnostic_array = (
        np.vstack(
            diagnostic_representations
        )
    )

    nuisance_array = np.vstack(
        nuisance_representations
    )

    component_array = np.vstack(
        nuisance_components
    )

    rows = []

    for split in splits:
        mask = meta[
            "split"
        ].eq(split).to_numpy()

        if not mask.any():
            continue

        rows.append({
            "split": split,
            **binary_metrics(
                y[mask],
                probabilities_array[mask],
            ),
        })

    return (
        pd.DataFrame(rows),
        diagnostic_array,
        nuisance_array,
        component_array,
    )


def posthoc_center_probe(
    representations: np.ndarray,
    meta: pd.DataFrame,
    source_centers: List[int],
) -> Dict[str, float]:
    center_to_label = {
        center: index
        for index, center
        in enumerate(source_centers)
    }

    train_mask = meta[
        "split"
    ].eq("train").to_numpy()

    id_val_mask = meta[
        "split"
    ].eq("id_val").to_numpy()

    labels = (
        meta["center"]
        .map(center_to_label)
        .to_numpy()
    )

    scaler = StandardScaler()

    x_train = scaler.fit_transform(
        representations[train_mask]
    )

    x_id_val = scaler.transform(
        representations[id_val_mask]
    )

    probe = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=0,
    )

    probe.fit(
        x_train,
        labels[train_mask],
    )

    return {
        "probe_train_accuracy": float(
            accuracy_score(
                labels[train_mask],
                probe.predict(x_train),
            )
        ),
        "probe_id_val_accuracy": float(
            accuracy_score(
                labels[id_val_mask],
                probe.predict(x_id_val),
            )
        ),
    }


def posthoc_tumor_probe(
    representations: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
) -> Dict[str, float]:
    train_mask = meta[
        "split"
    ].eq("train").to_numpy()

    id_val_mask = meta[
        "split"
    ].eq("id_val").to_numpy()

    scaler = StandardScaler()

    x_train = scaler.fit_transform(
        representations[train_mask]
    )

    x_id_val = scaler.transform(
        representations[id_val_mask]
    )

    probe = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=0,
    )

    probe.fit(
        x_train,
        y[train_mask],
    )

    probabilities = (
        probe.predict_proba(
            x_id_val
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "nuisance_tumor_accuracy":
            float(
                accuracy_score(
                    y[id_val_mask],
                    predictions,
                )
            ),
        "nuisance_tumor_auc":
            float(
                roc_auc_score(
                    y[id_val_mask],
                    probabilities,
                )
            ),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    repeats: int = 20000,
    seed: int = 20260605,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    rng = np.random.default_rng(seed)

    bootstrap = rng.choice(
        values,
        size=(
            repeats,
            len(values),
        ),
        replace=True,
    ).mean(axis=1)

    return (
        float(
            np.quantile(
                bootstrap,
                0.025,
            )
        ),
        float(
            np.quantile(
                bootstrap,
                0.975,
            )
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/wilds"),
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "results/camelyon17/"
            "metadata_audit.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results/"
            "camelyon17_supervised_resnet18/"
            "resnet18_source_epoch_2.pt"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/"
            "camelyon17_pathoalign_v3"
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )

    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.10, 0.25],
    )

    parser.add_argument(
        "--nuisance-dims",
        type=int,
        nargs="+",
        default=[8, 16, 32],
    )

    parser.add_argument(
        "--tumor-adv-weights",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
    )

    parser.add_argument(
        "--preserve-weights",
        type=float,
        nargs="+",
        default=[1.0, 5.0],
    )

    parser.add_argument(
        "--teacher-epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--candidate-epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-per-split-center-class",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--teacher-learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--candidate-learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--center-loss-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--diagnostic-center-adv-weight",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--component-l2-weight",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-validation-accuracy-loss",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--max-validation-auc-loss",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--max-nuisance-tumor-auc",
        type=float,
        default=0.70,
    )

    args = parser.parse_args()

    from wilds import get_dataset

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = pd.read_csv(
        args.metadata
    )

    dataset = get_dataset(
        dataset="camelyon17",
        root_dir=str(args.root),
        download=False,
    )

    feature_model = load_feature_model(
        args.checkpoint,
        str(device),
    )

    configurations = [
        CandidateConfig(
            alpha=float(alpha),
            nuisance_dim=int(
                nuisance_dim
            ),
            tumor_adv_weight=float(
                tumor_adv_weight
            ),
            preserve_weight=float(
                preserve_weight
            ),
        )
        for (
            alpha,
            nuisance_dim,
            tumor_adv_weight,
            preserve_weight,
        ) in itertools.product(
            args.alphas,
            args.nuisance_dims,
            args.tumor_adv_weights,
            args.preserve_weights,
        )
    ]

    development_rows = []
    teacher_rows = []
    teacher_history_frames = []
    candidate_history_frames = []
    payloads = {}

    for seed in args.seeds:
        print(
            f"=== Seed {seed} ===",
            flush=True,
        )

        sample = stratified_sample(
            metadata,
            args.max_per_split_center_class,
            seed,
        )

        x, y, indices = extract_features(
            feature_model,
            dataset,
            sample["index"].tolist(),
            args.batch_size,
            str(device),
        )

        meta = pd.DataFrame({
            "index": indices,
            "y": y,
        }).merge(
            sample[
                [
                    "index",
                    "center",
                    "split",
                ]
            ],
            on="index",
            how="left",
        )

        x = standardize_from_source_train(
            x=x,
            meta=meta,
        )

        (
            site_labels,
            source_centers,
        ) = site_labels_from_meta(meta)

        print(
            "Training tumor teacher",
            flush=True,
        )

        (
            teacher,
            teacher_best_epoch,
            teacher_history,
        ) = train_teacher(
            x=x,
            y=y,
            site_labels=site_labels,
            meta=meta,
            seed=seed,
            epochs=args.teacher_epochs,
            batch_size=args.batch_size,
            learning_rate=(
                args.teacher_learning_rate
            ),
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            device=device,
        )

        teacher_history_frames.append(
            teacher_history
        )

        teacher_probs = (
            teacher_probabilities(
                teacher=teacher,
                x=x,
                batch_size=args.batch_size,
                device=device,
            )
        )

        (
            teacher_metrics,
            teacher_representations,
        ) = evaluate_diagnostic_model(
            model=teacher,
            x=x,
            y=y,
            meta=meta,
            splits=[
                "train",
                "id_val",
                "val",
                "test",
            ],
            batch_size=args.batch_size,
            device=device,
        )

        teacher_probe = (
            posthoc_center_probe(
                representations=
                    teacher_representations,
                meta=meta,
                source_centers=
                    source_centers,
            )
        )

        for _, row in (
            teacher_metrics.iterrows()
        ):
            teacher_rows.append({
                "seed": seed,
                "split": row["split"],
                "accuracy":
                    row["accuracy"],
                "balanced_accuracy":
                    row[
                        "balanced_accuracy"
                    ],
                "macro_f1":
                    row["macro_f1"],
                "auc": row["auc"],
                "best_epoch":
                    teacher_best_epoch,
                "diagnostic_probe_id_val_accuracy":
                    teacher_probe[
                        "probe_id_val_accuracy"
                    ],
            })

        payloads[seed] = {
            "x": x,
            "y": y,
            "meta": meta,
            "site_labels": site_labels,
            "source_centers":
                source_centers,
            "teacher_state":
                copy.deepcopy(
                    teacher.state_dict()
                ),
            "candidate_states": {},
        }

        for index, config in enumerate(
            configurations,
            start=1,
        ):
            print(
                "Training "
                f"{config.name} "
                f"({index}/"
                f"{len(configurations)})",
                flush=True,
            )

            (
                candidate,
                history,
                trained,
            ) = train_candidate(
                x=x,
                y=y,
                site_labels=site_labels,
                teacher_probs=
                    teacher_probs,
                meta=meta,
                config=config,
                seed=seed,
                epochs=
                    args.candidate_epochs,
                batch_size=args.batch_size,
                learning_rate=
                    args.candidate_learning_rate,
                weight_decay=
                    args.weight_decay,
                dropout=args.dropout,
                center_loss_weight=
                    args.center_loss_weight,
                diagnostic_center_adv_weight=
                    args.diagnostic_center_adv_weight,
                component_l2_weight=
                    args.component_l2_weight,
                temperature=
                    args.temperature,
                device=device,
            )

            candidate_history_frames.append(
                history
            )

            payloads[seed][
                "candidate_states"
            ][config.name] = (
                trained.model_state
            )

            (
                metrics,
                diagnostic_representation,
                nuisance_representation,
                nuisance_component,
            ) = evaluate_candidate(
                model=candidate,
                x=x,
                y=y,
                meta=meta,
                splits=[
                    "train",
                    "id_val",
                    "val",
                ],
                batch_size=args.batch_size,
                device=device,
            )

            diagnostic_probe = (
                posthoc_center_probe(
                    representations=
                        diagnostic_representation,
                    meta=meta,
                    source_centers=
                        source_centers,
                )
            )

            nuisance_center_probe = (
                posthoc_center_probe(
                    representations=
                        nuisance_representation,
                    meta=meta,
                    source_centers=
                        source_centers,
                )
            )

            nuisance_tumor_probe = (
                posthoc_tumor_probe(
                    representations=
                        nuisance_representation,
                    y=y,
                    meta=meta,
                )
            )

            component_norm = float(
                np.mean(
                    nuisance_component
                    ** 2
                )
            )

            for _, row in (
                metrics.iterrows()
            ):
                development_rows.append({
                    "seed": seed,
                    "config": config.name,
                    "alpha":
                        config.alpha,
                    "nuisance_dim":
                        config.nuisance_dim,
                    "tumor_adv_weight":
                        config.tumor_adv_weight,
                    "preserve_weight":
                        config.preserve_weight,
                    "split":
                        row["split"],
                    "accuracy":
                        row["accuracy"],
                    "balanced_accuracy":
                        row[
                            "balanced_accuracy"
                        ],
                    "macro_f1":
                        row["macro_f1"],
                    "auc": row["auc"],
                    "best_epoch":
                        trained.best_epoch,
                    "diagnostic_probe_id_val_accuracy":
                        diagnostic_probe[
                            "probe_id_val_accuracy"
                        ],
                    "nuisance_center_probe_id_val_accuracy":
                        nuisance_center_probe[
                            "probe_id_val_accuracy"
                        ],
                    **nuisance_tumor_probe,
                    "nuisance_component_l2":
                        component_norm,
                })

    teacher_results = pd.DataFrame(
        teacher_rows
    )

    development = pd.DataFrame(
        development_rows
    )

    teacher_summary = (
        teacher_results
        .groupby("split")
        [[
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "auc",
            "diagnostic_probe_id_val_accuracy",
        ]]
        .agg(["mean", "std"])
        .reset_index()
    )

    teacher_summary.columns = [
        "_".join(
            str(value)
            for value in column
            if str(value)
        )
        for column in (
            teacher_summary.columns
        )
    ]

    summary = (
        development
        .groupby(
            [
                "config",
                "alpha",
                "nuisance_dim",
                "tumor_adv_weight",
                "preserve_weight",
                "split",
            ]
        )
        [[
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "auc",
            "diagnostic_probe_id_val_accuracy",
            "nuisance_center_probe_id_val_accuracy",
            "nuisance_tumor_accuracy",
            "nuisance_tumor_auc",
            "nuisance_component_l2",
        ]]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(
            str(value)
            for value in column
            if str(value)
        )
        for column in summary.columns
    ]

    validation = summary[
        summary["split"].eq("val")
    ].copy()

    teacher_validation = (
        teacher_summary[
            teacher_summary[
                "split"
            ].eq("val")
        ].iloc[0]
    )

    teacher_probe_mean = float(
        teacher_validation[
            "diagnostic_probe_id_val_accuracy_mean"
        ]
    )

    validation[
        "val_accuracy_delta_vs_teacher"
    ] = (
        validation["accuracy_mean"]
        - teacher_validation[
            "accuracy_mean"
        ]
    )

    validation[
        "val_auc_delta_vs_teacher"
    ] = (
        validation["auc_mean"]
        - teacher_validation[
            "auc_mean"
        ]
    )

    validation[
        "diagnostic_probe_delta_vs_teacher"
    ] = (
        validation[
            "diagnostic_probe_id_val_accuracy_mean"
        ]
        - teacher_probe_mean
    )

    validation[
        "passes_accuracy_guard"
    ] = (
        validation[
            "val_accuracy_delta_vs_teacher"
        ]
        >= -args.max_validation_accuracy_loss
    )

    validation[
        "passes_auc_guard"
    ] = (
        validation[
            "val_auc_delta_vs_teacher"
        ]
        >= -args.max_validation_auc_loss
    )

    validation[
        "passes_nuisance_tumor_guard"
    ] = (
        validation[
            "nuisance_tumor_auc_mean"
        ]
        <= args.max_nuisance_tumor_auc
    )

    validation[
        "reduces_diagnostic_center_probe"
    ] = (
        validation[
            "diagnostic_probe_delta_vs_teacher"
        ]
        < 0
    )

    validation["eligible"] = (
        validation[
            "passes_accuracy_guard"
        ]
        & validation[
            "passes_auc_guard"
        ]
        & validation[
            "passes_nuisance_tumor_guard"
        ]
        & validation[
            "reduces_diagnostic_center_probe"
        ]
    )

    candidates = validation[
        validation["eligible"]
    ].copy()

    candidates["selection_score"] = (
        candidates["auc_mean"]
        + 0.25
        * candidates["accuracy_mean"]
        - 0.10
        * candidates[
            "diagnostic_probe_id_val_accuracy_mean"
        ]
        + 0.05
        * candidates[
            "nuisance_center_probe_id_val_accuracy_mean"
        ]
        - 0.05
        * candidates[
            "nuisance_tumor_auc_mean"
        ]
    )

    candidates = candidates.sort_values(
        [
            "selection_score",
            "auc_mean",
            "accuracy_mean",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    selected_name = None

    if not candidates.empty:
        selected_name = str(
            candidates.iloc[0]["config"]
        )

    test_rows = []

    if selected_name is not None:
        selected_config = next(
            config
            for config in configurations
            if config.name
            == selected_name
        )

        for seed in args.seeds:
            payload = payloads[seed]

            teacher = DiagnosticModel(
                input_dim=
                    payload["x"].shape[1],
                n_sites=len(
                    payload[
                        "source_centers"
                    ]
                ),
                dropout=args.dropout,
            ).to(device)

            teacher.load_state_dict(
                payload[
                    "teacher_state"
                ]
            )

            teacher_metrics, _ = (
                evaluate_diagnostic_model(
                    model=teacher,
                    x=payload["x"],
                    y=payload["y"],
                    meta=payload["meta"],
                    splits=["test"],
                    batch_size=
                        args.batch_size,
                    device=device,
                )
            )

            teacher_row = (
                teacher_metrics.iloc[0]
            )

            test_rows.append({
                "seed": seed,
                "method": "teacher",
                "config": "teacher",
                "accuracy":
                    teacher_row[
                        "accuracy"
                    ],
                "balanced_accuracy":
                    teacher_row[
                        "balanced_accuracy"
                    ],
                "macro_f1":
                    teacher_row[
                        "macro_f1"
                    ],
                "auc":
                    teacher_row["auc"],
            })

            candidate = PathoAlignV3Model(
                input_dim=
                    payload["x"].shape[1],
                nuisance_dim=
                    selected_config.nuisance_dim,
                n_sites=len(
                    payload[
                        "source_centers"
                    ]
                ),
                alpha=
                    selected_config.alpha,
                dropout=args.dropout,
            ).to(device)

            candidate.load_state_dict(
                payload[
                    "candidate_states"
                ][selected_name]
            )

            (
                candidate_metrics,
                diagnostic_representation,
                nuisance_representation,
                _,
            ) = evaluate_candidate(
                model=candidate,
                x=payload["x"],
                y=payload["y"],
                meta=payload["meta"],
                splits=["test"],
                batch_size=
                    args.batch_size,
                device=device,
            )

            candidate_row = (
                candidate_metrics.iloc[0]
            )

            diagnostic_probe = (
                posthoc_center_probe(
                    representations=
                        diagnostic_representation,
                    meta=payload["meta"],
                    source_centers=
                        payload[
                            "source_centers"
                        ],
                )
            )

            nuisance_tumor_probe = (
                posthoc_tumor_probe(
                    representations=
                        nuisance_representation,
                    y=payload["y"],
                    meta=payload["meta"],
                )
            )

            test_rows.append({
                "seed": seed,
                "method":
                    "pathoalign_v3",
                "config":
                    selected_name,
                "accuracy":
                    candidate_row[
                        "accuracy"
                    ],
                "balanced_accuracy":
                    candidate_row[
                        "balanced_accuracy"
                    ],
                "macro_f1":
                    candidate_row[
                        "macro_f1"
                    ],
                "auc":
                    candidate_row["auc"],
                "diagnostic_probe_id_val_accuracy":
                    diagnostic_probe[
                        "probe_id_val_accuracy"
                    ],
                **nuisance_tumor_probe,
            })

    test_results = pd.DataFrame(
        test_rows
    )

    paired = pd.DataFrame()

    if selected_name is not None:
        teacher_test = (
            test_results[
                test_results[
                    "method"
                ].eq("teacher")
            ]
            .set_index("seed")
            .sort_index()
        )

        candidate_test = (
            test_results[
                test_results[
                    "method"
                ].eq("pathoalign_v3")
            ]
            .set_index("seed")
            .sort_index()
        )

        paired_rows = []

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "auc",
        ]:
            delta = (
                candidate_test[
                    metric
                ].to_numpy()
                - teacher_test[
                    metric
                ].to_numpy()
            )

            ci_low, ci_high = (
                bootstrap_mean_ci(delta)
            )

            paired_rows.append({
                "metric": metric,
                "candidate_mean":
                    float(
                        candidate_test[
                            metric
                        ].mean()
                    ),
                "teacher_mean":
                    float(
                        teacher_test[
                            metric
                        ].mean()
                    ),
                "paired_delta_mean":
                    float(delta.mean()),
                "paired_delta_std":
                    float(
                        delta.std(ddof=1)
                    ),
                "bootstrap_95_ci_low":
                    ci_low,
                "bootstrap_95_ci_high":
                    ci_high,
                "positive_seed_fraction":
                    float(
                        (delta > 0).mean()
                    ),
            })

        paired = pd.DataFrame(
            paired_rows
        )

    teacher_results.to_csv(
        args.out_dir
        / "pathoalign_v3_teacher_runs.csv",
        index=False,
    )

    teacher_summary.to_csv(
        args.out_dir
        / "pathoalign_v3_teacher_summary.csv",
        index=False,
    )

    development.to_csv(
        args.out_dir
        / "pathoalign_v3_development_runs.csv",
        index=False,
    )

    summary.to_csv(
        args.out_dir
        / "pathoalign_v3_development_summary.csv",
        index=False,
    )

    validation.to_csv(
        args.out_dir
        / "pathoalign_v3_validation_all.csv",
        index=False,
    )

    candidates.to_csv(
        args.out_dir
        / "pathoalign_v3_validation_ranking.csv",
        index=False,
    )

    test_results.to_csv(
        args.out_dir
        / "pathoalign_v3_selected_test.csv",
        index=False,
    )

    paired.to_csv(
        args.out_dir
        / "pathoalign_v3_paired_test.csv",
        index=False,
    )

    pd.concat(
        teacher_history_frames,
        ignore_index=True,
    ).to_csv(
        args.out_dir
        / "pathoalign_v3_teacher_history.csv",
        index=False,
    )

    pd.concat(
        candidate_history_frames,
        ignore_index=True,
    ).to_csv(
        args.out_dir
        / "pathoalign_v3_candidate_history.csv",
        index=False,
    )

    if selected_name is None:
        test_section = """
## Held-out test

No candidate passed all validation guards. The held-out test center was not evaluated for a PathoAlign v3 candidate.
"""
    else:
        test_section = f"""
## Validation-selected configuration

    {selected_name}

## Held-out test comparison

{paired.round(6).to_markdown(index=False)}
"""

    report = f"""# PathoAlign v3 tumor-preserving nuisance-removal experiment

## Architecture

1. Train a frozen tumor teacher on original standardized pathology features.
2. Learn a low-rank nuisance representation that predicts center.
3. Suppress tumor prediction from the nuisance representation with gradient reversal.
4. Decode a removable nuisance component.
5. Form cleaned features:

       x_clean = x - alpha * nuisance_component(x)

6. Train a diagnostic classifier on cleaned features.
7. Preserve teacher predictions using KL divergence.
8. Adversarially reduce center prediction from the diagnostic representation.

## Protocol

- Seeds: {args.seeds}
- Alpha values: {args.alphas}
- Nuisance dimensions: {args.nuisance_dims}
- Tumor adversary weights: {args.tumor_adv_weights}
- Preservation weights: {args.preserve_weights}
- Teacher epochs: {args.teacher_epochs}
- Candidate epochs: {args.candidate_epochs}
- Maximum examples per split/center/class: {args.max_per_split_center_class}
- Validation center: 1
- Held-out test center: 2

## Teacher validation baseline

{teacher_summary[teacher_summary["split"].eq("val")].round(6).to_markdown(index=False)}

## Validation results

{validation.round(6).to_markdown(index=False)}

## Ranked eligible candidates

{candidates.round(6).to_markdown(index=False)}

## Eligibility guards

A candidate is eligible only if:

- validation accuracy delta is at least {-args.max_validation_accuracy_loss:+.6f},
- validation AUC delta is at least {-args.max_validation_auc_loss:+.6f},
- nuisance tumor-probe AUC is no greater than {args.max_nuisance_tumor_auc:.3f},
- and diagnostic center-probe accuracy is lower than the teacher baseline.

{test_section}

## Interpretation boundary

V3 receives preliminary support only if the selected candidate preserves or improves held-out tumor performance while reducing diagnostic center leakage and keeping the removable nuisance representation weakly tumor-predictive.

This is a feature-level mechanism experiment, not a clinical or full federated-learning result.
"""

    (
        args.out_dir
        / "pathoalign_v3_report.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    config_payload = {
        "selected_config":
            selected_name,
        "seeds": args.seeds,
        "alphas": args.alphas,
        "nuisance_dims":
            args.nuisance_dims,
        "tumor_adv_weights":
            args.tumor_adv_weights,
        "preserve_weights":
            args.preserve_weights,
        "teacher_epochs":
            args.teacher_epochs,
        "candidate_epochs":
            args.candidate_epochs,
        "center_loss_weight":
            args.center_loss_weight,
        "diagnostic_center_adv_weight":
            args.diagnostic_center_adv_weight,
        "component_l2_weight":
            args.component_l2_weight,
        "temperature":
            args.temperature,
        "max_validation_accuracy_loss":
            args.max_validation_accuracy_loss,
        "max_validation_auc_loss":
            args.max_validation_auc_loss,
        "max_nuisance_tumor_auc":
            args.max_nuisance_tumor_auc,
    }

    (
        args.out_dir
        / "pathoalign_v3_config.json"
    ).write_text(
        json.dumps(
            config_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()





