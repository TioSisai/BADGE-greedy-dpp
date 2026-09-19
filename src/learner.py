"""Active learning loop for a single dataset, strategy, and seed, including the timed query stage."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tqdm.auto import tqdm

from .data.arrays import FrameData
from .models.classifier import (
    EARLY_STOPPING_PATIENCE,
    LR_SCHEDULER_METRIC,
    MODEL_SELECTION_METRIC,
    OPTIMIZER_NAME,
    initial_head_state,
    predict_frames,
    train_round,
)
from .models.metrics import cumulative_aulc, frame_wise_map
from .strategies import COLD_START_STRATEGY, QueryContext, build_query_strategy
from .strategies.backends import (
    array_module,
    backend_signature,
    require_backend,
    seed_backend,
    to_backend,
)
from .strategies.badge import badge_gradient

CONFIG_HASH_LEN = 12

# Overall metric columns are also used to validate the CSV header when resuming.
OVERALL_FIELDNAMES = [
    "iteration", "num_labeled_samples", "query_time", "val_mAP", "test_mAP", "test_AULC",
]
CONFIG_FILENAME = "config.json"
RESULTS_FILENAME = "results.csv"
SELECTION_COLUMN = "queried_idxes_in_latest_iteration"


def strategy_config(
    *, dataset, strategy, step_size, max_iter, hidden_features, lr, max_epochs,
    batch_size, infer_batch_size, num_classes, n_train, device, backend,
) -> dict:
    """Generate a configuration shared across seeds, written to config.json and determining the output directory hash.

    Args:
        dataset: Dataset name.
        strategy: Strategy name.
        step_size: Number of segments selected per round, including the shared cold start.
        max_iter: Total number of rounds.
        hidden_features: Hidden width of the classification head, or 0 for the default rule.
        lr: Initial learning rate.
        max_epochs: Upper bound on epochs per round.
        batch_size: Training batch size.
        infer_batch_size: Inference batch size.
        num_classes: Number of classes.
        n_train: Number of segments in the training pool.
        device: torch device.
        backend: Selection backend signature.

    Returns:
        JSON-serializable dictionary containing all result-related settings.
    """
    return {
        "dataset": dataset,
        "strategy": strategy,
        "step_size": int(step_size),
        "max_iter": int(max_iter),
        "hidden_features": int(hidden_features),
        "optimizer": OPTIMIZER_NAME,
        "model_selection": MODEL_SELECTION_METRIC,
        "lr_scheduler": f"reduce_on_plateau({LR_SCHEDULER_METRIC})",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "lr": float(lr),
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "infer_batch_size": int(infer_batch_size),
        "num_classes": int(num_classes),
        "n_train": int(n_train),
        "device": str(device),
        "backend": backend,
    }


def config_hash(config: dict) -> str:
    """Return the first CONFIG_HASH_LEN characters of the configuration's SHA-256, independent of field order."""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:CONFIG_HASH_LEN]


def config_dir(output_root, config: dict) -> Path:
    """Return ``{output_root}/{dataset}/{strategy}/{config_hash}`` without creating the directory."""
    return Path(output_root) / config["dataset"] / config["strategy"] / config_hash(config)


def check_or_write_config(cfg_dir: Path, config: dict) -> None:
    """Validate or atomically write config.json, allowing multiple processes to share the output directory.

    Raises:
        ValueError: The existing configuration differs from the supplied configuration.
    """
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / CONFIG_FILENAME
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"config.json does not match the current parameters (config hash collision): {path}\n"
                f"existing: {existing}\ncurrent: {config}"
            )
    else:
        tmp_path = cfg_dir / f".config.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)


def results_csv_complete(results_csv: Path, max_iter: int) -> bool:
    """Check that the header starts with OVERALL_FIELDNAMES and the results contain exactly max_iter rows."""
    if not results_csv.is_file():
        return False
    with results_csv.open(newline="") as fh:
        # Rows are counted as lines because the full-supervised selection field exceeds the csv field size limit.
        header = next(csv.reader(fh), [])
        return header[:len(OVERALL_FIELDNAMES)] == OVERALL_FIELDNAMES and sum(1 for _ in fh) == max_iter


class ActiveLearner:
    """Active learning loop for one dataset, strategy, and seed.

    Args:
        data: FrameData reusable across strategies and seeds.
        dataset: Dataset name.
        strategy_name: Query strategy name.
        output_root: Output root directory.
        step_size: Number of segments selected per round.
        max_iter: Total number of rounds.
        hidden_features: Hidden width of the classification head, or 0 for the default rule.
        lr: Initial learning rate.
        max_epochs: Upper bound on epochs per round.
        seed: Seed for global random numbers, the cold start, the initial weights, and the query strategy.
        device: torch device.
        batch_size: Training batch size.
        infer_batch_size: Inference batch size for validation, testing, and querying.
        backend: Selection backend the results must be produced with. The current process is probed when None.
        pbar_position: Progress bar line number, with 0 reserved for the overall progress bar.

    Raises:
        ValueError: The total selection budget exceeds the training pool size.
    """

    def __init__(
        self,
        *,
        data: FrameData,
        dataset: str,
        strategy_name: str,
        output_root,
        step_size: int,
        max_iter: int,
        hidden_features: int,
        lr: float,
        max_epochs: int,
        seed: int,
        device,
        batch_size: int,
        infer_batch_size: int,
        backend: str | None = None,
        pbar_position: int = 1,
    ) -> None:
        self.data = data
        self.dataset = dataset
        self.strategy_name = strategy_name
        self.step_size = step_size
        self.max_iter = max_iter
        self.hidden_features = hidden_features
        self.lr = lr
        self.max_epochs = max_epochs
        self.seed = seed
        self.device = device
        self.xp = array_module(device)
        self.batch_size = batch_size
        self.infer_batch_size = infer_batch_size
        self.backend = backend or backend_signature(device)
        self.pbar_position = pbar_position

        required = self.max_iter * self.step_size
        if required > data.n_train:
            raise ValueError(
                f"not enough samples: max_iter*step_size={required} required, train pool only has {data.n_train}"
            )

        self.config = strategy_config(
            dataset=self.dataset, strategy=self.strategy_name, step_size=self.step_size,
            max_iter=self.max_iter, hidden_features=self.hidden_features, lr=self.lr,
            max_epochs=self.max_epochs, batch_size=self.batch_size,
            infer_batch_size=self.infer_batch_size, num_classes=data.num_classes,
            n_train=data.n_train, device=self.device, backend=self.backend,
        )
        self.config_dir = config_dir(output_root, self.config)
        self.seed_dir = self.config_dir / f"seed_{self.seed}"
        self.results_csv = self.seed_dir / RESULTS_FILENAME

    def _seed_everything(self) -> None:
        """Initialize the numpy, cupy and torch random streams, called once per experiment."""
        # Strategies are rebuilt every round but share this stream, so no round replays the draws of another.
        self.random_state = np.random.RandomState(self.seed)
        seed_backend(self.seed)
        torch.manual_seed(self.seed)

    def _to_device(self, array) -> torch.Tensor:
        """Upload a host array to the torch device."""
        return torch.from_numpy(np.ascontiguousarray(array)).to(self.device)

    def _frame_targets(self, frame_labels) -> torch.Tensor:
        """Upload frame labels ``[N, F, C]`` as the int64 targets ``[N * F, C]`` the metrics read."""
        return self._to_device(frame_labels).reshape(-1, self.data.num_classes).long()

    def _prepare_properties(self, names: tuple[str, ...], model, thresholds) -> dict:
        """Materialize exactly the properties a strategy declared, on the selection backend.

        Args:
            names: Property names declared by the strategy.
            model: Head trained in the previous round, or None on the cold start.
            thresholds: Per-class decision thresholds of that model, or None on the cold start.

        Returns:
            Mapping from property name to the prepared array.
        """
        xp = self.xp
        properties: dict = {}
        if "seg_embedding" in names:
            properties["seg_embedding"] = to_backend(self.data.seg_embedding, xp)
        if not {"prediction_soft", "prediction_hard", "gradient"} & set(names):
            return properties

        want_features = "gradient" in names
        outputs = predict_frames(
            model, self.data.train_embedding, batch_size=self.infer_batch_size,
            device=self.device, want_features=want_features,
        )
        proba, last_feature = outputs if want_features else (outputs, None)
        if "prediction_soft" in names:
            properties["prediction_soft"] = to_backend(proba, xp)
        if "prediction_hard" in names:
            cut = torch.as_tensor(thresholds, device=proba.device, dtype=proba.dtype).view(1, 1, -1)
            properties["prediction_hard"] = to_backend((proba >= cut).to(torch.int8), xp)
        if want_features:
            properties["gradient"] = badge_gradient(proba, last_feature, xp)
        return properties

    def _select_samples(self, labeled: np.ndarray, iteration: int, model, thresholds) -> tuple[list[int], float]:
        """Run one query stage and return the selection with the time spent on property preparation and selection."""
        name = COLD_START_STRATEGY if iteration == 0 else self.strategy_name
        strategy = build_query_strategy(name, random_state=self.random_state)

        start = perf_counter()
        properties = self._prepare_properties(strategy.required_properties, model, thresholds)
        ctx = QueryContext(
            labeled=labeled,
            unlabeled=np.setdiff1d(np.arange(self.data.n_train), labeled),
            step_size=self.step_size,
            n_train=self.data.n_train,
            frame_embedding=self.data.train_embedding,
            frame_labels=self.data.train_label,
            xp=self.xp,
            properties=properties,
        )
        selected = strategy.query(ctx)
        query_time = perf_counter() - start
        return selected.tolist(), query_time

    def _assert_selection(self, selected: list[int], labeled: np.ndarray) -> None:
        """Check the selection count, uniqueness within the batch, and disjointness from the labeled set."""
        unique = set(selected)
        if (
            len(selected) != self.step_size
            or len(unique) != len(selected)
            or not unique.isdisjoint(labeled.tolist())
        ):
            raise RuntimeError(
                f"strategy {self.strategy_name} returned an invalid selection (expected {self.step_size}"
                f" distinct unlabeled indices)"
            )

    def _csv_fieldnames(self) -> list[str]:
        """Return the results.csv header: overall metrics, per-class test mAP, then the selection column."""
        return (
            OVERALL_FIELDNAMES
            + [f"test_mAP_{c}" for c in self.data.class_names]
            + [SELECTION_COLUMN]
        )

    @staticmethod
    def _fmt(value: float) -> str:
        """Format metrics to six decimal places, leaving NaN blank."""
        return "" if math.isnan(value) else f"{value:.6f}"

    def _write_row(self, writer, *, iteration, num_labeled, query_time, val_map,
                   test_map, test_aulc, test_per_class, selected) -> None:
        """Write the results.csv row of one round, with the selection serialized as a JSON list."""
        row = {
            "iteration": iteration,
            "num_labeled_samples": num_labeled,
            "query_time": self._fmt(query_time),
            "val_mAP": self._fmt(val_map),
            "test_mAP": self._fmt(test_map),
            "test_AULC": self._fmt(test_aulc),
        }
        for cls, value in zip(self.data.class_names, test_per_class):
            row[f"test_mAP_{cls}"] = self._fmt(float(value))
        row[SELECTION_COLUMN] = json.dumps(selected)
        writer.writerow(row)

    def run(self) -> Path:
        """Run the experiment and return the seed directory, skipping completed seeds and rerunning incomplete ones.

        Raises:
            ValueError: The existing configuration differs from the current parameters.
            RuntimeError: The strategy returns an invalid selection.
        """
        # Fail before writing anything when this process cannot provide the backend named in the configuration.
        require_backend(self.backend, self.device)
        check_or_write_config(self.config_dir, self.config)
        if results_csv_complete(self.results_csv, self.max_iter):
            return self.seed_dir
        if self.seed_dir.exists():
            shutil.rmtree(self.seed_dir)
        self.seed_dir.mkdir(parents=True)

        # Query, training and testing consume random numbers in this order every round, so reordering changes results.
        self._seed_everything()
        data = self.data
        hidden_features = self.hidden_features or None
        initial_state = initial_head_state(
            data.train_embedding.shape[-1], data.num_classes, hidden_features, self.device
        )
        val_embedding = self._to_device(data.val_embedding)
        val_targets = self._frame_targets(data.val_label)
        test_targets = self._frame_targets(data.test_label)
        labeled = np.empty(0, dtype=np.int64)
        model, thresholds = None, None
        num_labeled_hist: list[int] = []
        test_map_hist: list[float] = []

        with self.results_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._csv_fieldnames())
            writer.writeheader()
            pbar = tqdm(
                range(self.max_iter),
                desc=f"[{self.dataset}/{self.strategy_name}/seed={self.seed}]",
                unit="iter", position=self.pbar_position, leave=False,
            )
            for iteration in pbar:
                selected, query_time = self._select_samples(labeled, iteration, model, thresholds)
                self._assert_selection(selected, labeled)
                labeled = np.concatenate([labeled, np.asarray(selected, dtype=np.int64)])

                model, round_result = train_round(
                    train_embedding=self._to_device(data.train_embedding[labeled]),
                    train_label=self._to_device(data.train_label[labeled]),
                    val_embedding=val_embedding,
                    val_targets=val_targets,
                    initial_state=initial_state,
                    num_classes=data.num_classes,
                    hidden_features=hidden_features,
                    lr=self.lr,
                    max_epochs=self.max_epochs,
                    batch_size=self.batch_size,
                    infer_batch_size=self.infer_batch_size,
                    device=self.device,
                )
                thresholds = round_result.thresholds
                test_proba = predict_frames(
                    model, data.test_embedding, batch_size=self.infer_batch_size, device=self.device
                )
                test_per_class, test_map = frame_wise_map(test_proba.reshape(-1, data.num_classes), test_targets)

                num_labeled_hist.append(int(labeled.size))
                test_map_hist.append(test_map)
                test_aulc = cumulative_aulc(num_labeled_hist, test_map_hist)

                self._write_row(
                    writer, iteration=iteration, num_labeled=int(labeled.size), query_time=query_time,
                    val_map=round_result.val_map, test_map=test_map, test_aulc=test_aulc,
                    test_per_class=test_per_class, selected=selected,
                )
                fh.flush()
                pbar.set_postfix({"test_mAP": f"{test_map:.4f}", "query_s": f"{query_time:.1f}"})
        return self.seed_dir
