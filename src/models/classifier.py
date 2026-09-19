"""Frame-level MLP head on frozen embeddings, retrained from the same initial weights at every active learning round."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .metrics import average_precision_tensors, precision_recall_curves

# Recorded in config.json so results produced under different settings land in different directories.
OPTIMIZER_NAME = "adam"
MODEL_SELECTION_METRIC = "val_mAP"
LR_SCHEDULER_METRIC = "val_mAP"
EARLY_STOPPING_PATIENCE = 10

# ReduceLROnPlateau settings.
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-6


def default_hidden_features(input_dim: int, num_classes: int) -> int:
    """Return the geometric mean of the input and output dimensions, clamped and rounded to a multiple of 64."""
    geometric_mean = math.sqrt(input_dim * num_classes)
    lower_bound = max(64, min(input_dim, num_classes))
    clamped = min(max(geometric_mean, lower_bound), 2048)
    return int(round(clamped / 64.0) * 64)


class FrameMLP(nn.Module):
    """Two-layer head applied to every frame independently.

    Args:
        input_dim: Frame embedding dimension.
        num_classes: Number of call types.
        hidden_features: Hidden width, or None for the default rule.
        drop: Dropout probability applied after the activation and after the output layer.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_features: int | None = None, drop: float = 0.25):
        super().__init__()
        self.hidden_features = hidden_features or default_hidden_features(input_dim, num_classes)
        self.fc1 = nn.Linear(input_dim, self.hidden_features)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.BatchNorm1d(self.hidden_features)
        self.fc2 = nn.Linear(self.hidden_features, num_classes)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map frame embeddings ``[B, F, D]`` to ``(logit [B, F, C], last_feature [B, F, H])``."""
        x = self.drop1(self.act(self.fc1(x)))
        # BatchNorm1d normalizes over the channel axis, so the frame axis moves to the end.
        last_feature = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        return self.drop2(self.fc2(last_feature)), last_feature


@dataclass(frozen=True, eq=False)
class RoundResult:
    """Validation outcome of the selected epoch of one active learning round.

    Attributes:
        thresholds: Per-class decision thresholds of the selected epoch, maximizing validation F1 among
            the epochs run up to it, float32 ``[C]``.
        val_map: Macro validation mAP at the selected epoch.
        num_epochs: Number of epochs actually run before early stopping.
    """

    thresholds: np.ndarray
    val_map: float
    num_epochs: int


def _iter_batches(num_rows: int, batch_size: int, order=None):
    """Yield contiguous slices, or slices of ``order`` when a permutation is given."""
    for start in range(0, num_rows, batch_size):
        stop = min(start + batch_size, num_rows)
        yield slice(start, stop) if order is None else order[start:stop]


def initial_head_state(input_dim: int, num_classes: int, hidden_features: int | None, device) -> dict:
    """Draw the initial weights every round of one experiment restarts from."""
    return FrameMLP(input_dim, num_classes, hidden_features).to(device).state_dict()


@torch.no_grad()
def predict_frames(model: FrameMLP, embeddings, *, batch_size: int, device, want_features: bool = False):
    """Run the head over frame embeddings in batches, leaving the outputs on the compute device.

    Args:
        model: Trained head, switched to eval mode by this function.
        embeddings: Frame embeddings, ``[N, F, D]``, either a mmap array streamed to the device in
            batches or a tensor already on the device.
        batch_size: Number of segments per inference batch.
        device: torch device.
        want_features: Also return the penultimate features BADGE differentiates against.

    Returns:
        ``proba`` float32 ``[N, F, C]``, or ``(proba, last_feature)`` when want_features is set.
    """
    model.eval()
    probas, features = [], []
    for rows in _iter_batches(embeddings.shape[0], batch_size):
        batch = embeddings[rows]
        if not torch.is_tensor(batch):
            batch = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
        logit, last_feature = model(batch)
        probas.append(torch.sigmoid(logit))
        if want_features:
            features.append(last_feature)
    proba = torch.cat(probas)
    return (proba, torch.cat(features)) if want_features else proba


def _best_f1_thresholds(curves) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-class thresholds maximizing F1 ``[C]`` and their mean F1 ``[]``, both left on the device."""
    best_thresholds, best_f1_scores = [], []
    for p, r, t in zip(*curves):
        f1 = torch.nan_to_num((2 * p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-8), nan=0.0)
        best = torch.argmax(f1)
        best_thresholds.append(t[best])
        best_f1_scores.append(f1[best])
    return torch.stack(best_thresholds), torch.stack(best_f1_scores).mean()


def _validation_summary(val_proba: torch.Tensor, val_targets: torch.Tensor) -> tuple[float, np.ndarray, float]:
    """Derive the validation mAP and the F1 thresholds of one epoch from a single pass over the curves.

    Args:
        val_proba: Probabilities of all validation frames, ``[M, C]``.
        val_targets: int64 targets of the same frames on the same device, ``[M, C]``.

    Returns:
        ``(val_map, thresholds, mean_f1)``, with the thresholds as float32 ``[C]``.
    """
    curves = precision_recall_curves(val_proba, val_targets)
    _, val_map = average_precision_tensors(curves)
    thresholds, mean_f1 = _best_f1_thresholds(curves)
    # Pack all values into one tensor so that the epoch costs a single transfer to the host.
    packed = torch.cat([val_map.reshape(1), mean_f1.reshape(1), thresholds]).cpu().numpy()
    return float(packed[0]), packed[2:], float(packed[1])


def train_round(
    *,
    train_embedding: torch.Tensor,
    train_label: torch.Tensor,
    val_embedding: torch.Tensor,
    val_targets: torch.Tensor,
    initial_state: dict,
    num_classes: int,
    hidden_features: int | None,
    lr: float,
    max_epochs: int,
    batch_size: int,
    infer_batch_size: int,
    device,
) -> tuple[FrameMLP, RoundResult]:
    """Train the head on the current labeled set and keep the epoch with the highest validation mAP.

    Args:
        train_embedding: Labeled frame embeddings on the device, ``[n_labeled, F, D]``.
        train_label: Labeled frame targets on the device, ``[n_labeled, F, C]``.
        val_embedding: Validation frame embeddings on the device, ``[M, F, D]``.
        val_targets: int64 targets of all validation frames on the device, ``[M * F, C]``.
        initial_state: Initial weights shared by every round of the experiment.
        num_classes: Number of call types.
        hidden_features: Hidden width of the head, or None for the default rule.
        lr: Initial learning rate.
        max_epochs: Upper bound on epochs.
        batch_size: Training batch size.
        infer_batch_size: Validation batch size.
        device: torch device.

    Returns:
        ``(model, round_result)`` with the model already restored to the selected weights.
    """
    model = FrameMLP(train_embedding.shape[-1], num_classes, hidden_features).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR
    )

    best_map, best_state = -math.inf, None
    best_f1, thresholds = -math.inf, np.full(num_classes, 0.5, dtype=np.float32)
    # The thresholds are paired with the selected weights, so freeze the running best at the epoch that is kept.
    selected_thresholds = thresholds
    epochs_without_improvement, epochs_run = 0, 0

    for _ in range(max_epochs):
        model.train()
        # Draw the permutation on the host so the shuffling order does not depend on the accelerator.
        order = torch.randperm(train_embedding.shape[0]).to(device)
        for rows in _iter_batches(train_embedding.shape[0], batch_size, order):
            logit, _ = model(train_embedding[rows])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logit.reshape(-1, num_classes), train_label[rows].reshape(-1, num_classes)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_proba = predict_frames(model, val_embedding, batch_size=infer_batch_size, device=device)
        val_map, epoch_thresholds, epoch_f1 = _validation_summary(val_proba.reshape(-1, num_classes), val_targets)
        scheduler.step(val_map)
        epochs_run += 1

        if epoch_f1 > best_f1:
            best_f1, thresholds = epoch_f1, epoch_thresholds

        if val_map > best_map:
            best_map, selected_thresholds = val_map, thresholds
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, RoundResult(thresholds=selected_thresholds, val_map=best_map, num_epochs=epochs_run)
