"""Centralized and federated baseline training algorithms.

FedAvg, FedProx, and SCAFFOLD are implemented here rather than through the
project's federated package so benchmark behaviour cannot change main.py.
"""

from __future__ import annotations

import copy
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import BenchmarkConfig
from .data import BenchmarkData, build_benchmark_data
from .models import build_fusion_model, build_model, count_parameters


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    architecture: str
    algorithm: str
    description: str
    use_tcn: bool = True


AVAILABLE_BASELINES: Mapping[str, BaselineSpec] = {
    "centralized_mobilenet": BaselineSpec(
        "centralized_mobilenet", "mobilenet_v2", "centralized",
        "Centralized MobileNetV2 trained on the union of client training data.",
    ),
    "mobilenet_without_tcn": BaselineSpec(
        "mobilenet_without_tcn", "mobilenet_v2", "fedavg",
        "MobileNetV2 fusion baseline with mean-pooled video-frame features and no TCN.",
        use_tcn=False,
    ),
    "fedavg_mobilenet": BaselineSpec(
        "fedavg_mobilenet", "mobilenet_v2", "fedavg",
        "Federated MobileNetV2 trained with vanilla FedAvg and no DP.",
    ),
    "fedprox_mobilenet": BaselineSpec(
        "fedprox_mobilenet", "mobilenet_v2", "fedprox",
        "Federated MobileNetV2 trained with the FedProx proximal objective.",
    ),
    "scaffold_mobilenet": BaselineSpec(
        "scaffold_mobilenet", "mobilenet_v2", "scaffold",
        "Federated MobileNetV2 trained with SCAFFOLD control variates.",
    ),
    "resnet18": BaselineSpec(
        "resnet18", "resnet18", "fedavg",
        "ResNet18 trained with the same FedAvg protocol as the federated baselines.",
    ),
    "efficientnet_b0": BaselineSpec(
        "efficientnet_b0", "efficientnet_b0", "fedavg",
        "EfficientNet-B0 trained with the same FedAvg protocol as the federated baselines.",
    ),
    "tinyvit": BaselineSpec(
        "tinyvit", "tinyvit", "fedavg",
        "TinyViT (16x16 patches) trained with the same FedAvg protocol.",
    ),
}


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _reset_loader_generators(data: BenchmarkData, seed: int) -> None:
    """Give every baseline the same deterministic loader ordering.

    DataLoader owns a sampler generator whose state advances during an earlier
    baseline.  Resetting it here prevents benchmark order from changing a later
    model's mini-batch sequence.
    """
    loaders = [data.central_train_loader, *data.client_train_loaders]
    for offset, loader in enumerate(loaders):
        if loader.generator is not None:
            loader.generator.manual_seed(seed + offset)


def _clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _clone_named_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def _class_weights(labels: Sequence[int], num_classes: int, device: torch.device) -> Optional[torch.Tensor]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes)
    if not np.all(counts):
        return None
    weights = counts.sum() / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _make_criterion(
    labels: Sequence[int], num_classes: int, config: BenchmarkConfig, device: torch.device
) -> nn.Module:
    weights = _class_weights(labels, num_classes, device) if config.class_weighted_loss else None
    return nn.CrossEntropyLoss(weight=weights)


def _move_inputs(inputs: Any, device: torch.device) -> Any:
    if isinstance(inputs, (tuple, list)):
        return tuple(tensor.to(device) for tensor in inputs)
    return inputs.to(device)


def _forward_model(model: nn.Module, inputs: Any) -> torch.Tensor:
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def _weighted_average_states(
    states: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[int]
) -> Dict[str, torch.Tensor]:
    if not states or len(states) != len(weights):
        raise ValueError("State and weight lists must be equally non-empty.")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Aggregation weights must sum to a positive value.")
    result: Dict[str, torch.Tensor] = {}
    for name in states[0]:
        values = [state[name] for state in states]
        if torch.is_floating_point(values[0]):
            aggregate = torch.zeros_like(values[0], dtype=torch.float32)
            for value, weight in zip(values, weights):
                aggregate.add_(value.float(), alpha=weight / total)
            result[name] = aggregate.to(dtype=values[0].dtype)
        else:
            # Integer buffers (e.g. BatchNorm's counter) cannot be averaged.
            result[name] = values[0].clone()
    return result


def _load_state(model: nn.Module, state: Mapping[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({name: value.to(device) for name, value in state.items()}, strict=True)


def _train_local(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    config: BenchmarkConfig,
    device: torch.device,
    algorithm: str,
    global_parameters: Optional[Mapping[str, torch.Tensor]] = None,
    server_control: Optional[Mapping[str, torch.Tensor]] = None,
    client_control: Optional[Mapping[str, torch.Tensor]] = None,
) -> Tuple[float, int]:
    """Run local SGD/Adam updates and return mean loss plus optimizer steps."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_loss = 0.0
    samples = 0
    steps = 0
    for _ in range(config.local_epochs):
        for inputs, targets in loader:
            inputs, targets = _move_inputs(inputs, device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward_model(model, inputs)
            loss = criterion(logits, targets)
            if algorithm == "fedprox":
                if global_parameters is None:
                    raise RuntimeError("FedProx requires the incoming global parameters.")
                proximal = torch.zeros((), device=device)
                for name, parameter in model.named_parameters():
                    proximal = proximal + torch.sum(
                        (parameter - global_parameters[name].to(device)) ** 2
                    )
                loss = loss + 0.5 * config.fedprox_mu * proximal
            loss.backward()
            if algorithm == "scaffold":
                if server_control is None or client_control is None:
                    raise RuntimeError("SCAFFOLD requires server and client control variates.")
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        parameter.grad.add_(server_control[name].to(device) - client_control[name].to(device))
            optimizer.step()
            total_loss += loss.detach().item() * targets.size(0)
            samples += targets.size(0)
            steps += 1
    if samples == 0 or steps == 0:
        raise RuntimeError("A client loader produced no training batches.")
    return total_loss / samples, steps


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, class_names: Sequence[str]
) -> Dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    targets_all: List[int] = []
    predictions_all: List[int] = []
    positive_scores: List[float] = []
    for inputs, targets in loader:
        inputs, targets = _move_inputs(inputs, device), targets.to(device)
        logits = _forward_model(model, inputs)
        total_loss += criterion(logits, targets).item()
        probabilities = torch.softmax(logits, dim=1)
        predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
        targets_all.extend(targets.cpu().tolist())
        if probabilities.shape[1] == 2:
            positive_scores.extend(probabilities[:, 1].cpu().tolist())
    if not targets_all:
        raise RuntimeError("Evaluation loader produced no samples.")
    labels = list(range(len(class_names)))
    metrics: Dict[str, Any] = {
        "loss": float(total_loss / len(targets_all)),
        "accuracy": float(accuracy_score(targets_all, predictions_all)),
        "balanced_accuracy": float(balanced_accuracy_score(targets_all, predictions_all)),
        "precision_macro": float(precision_score(targets_all, predictions_all, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(targets_all, predictions_all, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(targets_all, predictions_all, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(targets_all, predictions_all, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(targets_all, predictions_all, labels=labels).tolist(),
        "samples": len(targets_all),
    }
    if len(class_names) == 2 and len(set(targets_all)) == 2:
        metrics["auc_roc"] = float(roc_auc_score(targets_all, positive_scores))
    else:
        metrics["auc_roc"] = None
    return metrics


def _early_stop(
    validation: Dict[str, Any], best_accuracy: float, bad_rounds: int, min_delta: float = 0.0
) -> Tuple[bool, float, int]:
    accuracy = float(validation["accuracy"])
    if accuracy > best_accuracy + min_delta:
        return True, accuracy, 0
    return False, best_accuracy, bad_rounds + 1


def _train_centralized(
    model: nn.Module, data: BenchmarkData, config: BenchmarkConfig, device: torch.device
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    criterion = _make_criterion(data.central_train_loader.dataset.labels, data.num_classes, config, device)
    best_state = _clone_state_dict(model)
    best_accuracy, bad_rounds = float("-inf"), 0
    history: List[Dict[str, Any]] = []
    for round_number in range(1, config.rounds + 1):
        print(
            f"  [centralized] round {round_number}/{config.rounds}: training...",
            flush=True,
        )
        train_loss, _ = _train_local(
            model, data.central_train_loader, criterion, config, device, "centralized"
        )
        validation = evaluate(model, data.validation_loader, device, data.class_names)
        improved, best_accuracy, bad_rounds = _early_stop(validation, best_accuracy, bad_rounds)
        if improved:
            best_state = _clone_state_dict(model)
        history.append({"round": round_number, "train_loss": train_loss, "validation": validation})
        print(
            f"  [centralized] round {round_number}: loss={train_loss:.4f}, "
            f"val_acc={validation['accuracy']:.4f}",
            flush=True,
        )
        if bad_rounds >= config.early_stopping_patience:
            break
    return best_state, history


def _train_federated(
    model: nn.Module, data: BenchmarkData, config: BenchmarkConfig, device: torch.device, algorithm: str
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    global_state = _clone_state_dict(model)
    global_control = {name: torch.zeros_like(value) for name, value in _clone_named_parameters(model).items()}
    client_controls = [
        {name: torch.zeros_like(value) for name, value in global_control.items()}
        for _ in data.client_train_loaders
    ]
    best_state = copy.deepcopy(global_state)
    best_accuracy, bad_rounds = float("-inf"), 0
    history: List[Dict[str, Any]] = []
    for round_number in range(1, config.rounds + 1):
        print(f"  [{algorithm}] round {round_number}/{config.rounds}", flush=True)
        client_states: List[Dict[str, torch.Tensor]] = []
        client_losses: List[float] = []
        new_client_controls: List[Dict[str, torch.Tensor]] = []
        parameter_snapshot = {
            name: value for name, value in global_state.items() if name in global_control
        }
        for client_id, loader in enumerate(data.client_train_loaders):
            print(
                f"    client {client_id + 1}/{len(data.client_train_loaders)}: training...",
                flush=True,
            )
            local_model = copy.deepcopy(model).to(device)
            _load_state(local_model, global_state, device)
            criterion = _make_criterion(loader.dataset.labels, data.num_classes, config, device)
            loss, steps = _train_local(
                local_model,
                loader,
                criterion,
                config,
                device,
                algorithm,
                parameter_snapshot,
                global_control if algorithm == "scaffold" else None,
                client_controls[client_id] if algorithm == "scaffold" else None,
            )
            client_losses.append(loss)
            client_states.append(_clone_state_dict(local_model))
            print(f"    client {client_id + 1}: loss={loss:.4f}", flush=True)
            if algorithm == "scaffold":
                local_parameters = _clone_named_parameters(local_model)
                scale = float(steps) * config.learning_rate
                new_client_controls.append(
                    {
                        name: client_controls[client_id][name] - global_control[name]
                        + (parameter_snapshot[name] - local_parameters[name]) / scale
                        for name in global_control
                    }
                )
        global_state = _weighted_average_states(client_states, data.client_sample_counts)
        _load_state(model, global_state, device)
        if algorithm == "scaffold":
            total = float(sum(data.client_sample_counts))
            global_control = {
                name: global_control[name]
                + sum(
                    (new_client_controls[client_id][name] - client_controls[client_id][name])
                    * (data.client_sample_counts[client_id] / total)
                    for client_id in range(len(client_controls))
                )
                for name in global_control
            }
            client_controls = new_client_controls
        validation = evaluate(model, data.validation_loader, device, data.class_names)
        improved, best_accuracy, bad_rounds = _early_stop(validation, best_accuracy, bad_rounds)
        if improved:
            best_state = copy.deepcopy(global_state)
        weighted_loss = float(np.average(client_losses, weights=data.client_sample_counts))
        history.append(
            {"round": round_number, "train_loss": weighted_loss, "validation": validation}
        )
        print(
            f"  [{algorithm}] round {round_number}: loss={weighted_loss:.4f}, "
            f"val_acc={validation['accuracy']:.4f}",
            flush=True,
        )
        if bad_rounds >= config.early_stopping_patience:
            break
    return best_state, history


def _write_json(path: Path, content: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, indent=2)


def run_baseline(spec: BaselineSpec, data: BenchmarkData, config: BenchmarkConfig) -> Dict[str, Any]:
    """Run one named baseline and write only inside ``config.output_dir``."""
    set_reproducibility(config.seed)
    _reset_loader_generators(data, config.seed)
    device = torch.device(config.device)
    print(
        f"\nStarting {spec.name} | modality={config.modality} | "
        f"algorithm={spec.algorithm} | device={device}",
        flush=True,
    )
    if config.modality == "fusion":
        model = build_fusion_model(
            spec.architecture,
            data.num_classes,
            config.image_size,
            config.pretrained,
            spec.use_tcn,
        ).to(device)
    else:
        model = build_model(
            spec.architecture, data.num_classes, config.image_size, config.pretrained
        ).to(device)
    parameters = count_parameters(model)
    print(f"  model parameters: {parameters:,}", flush=True)
    if spec.algorithm == "centralized":
        best_state, history = _train_centralized(model, data, config, device)
    else:
        best_state, history = _train_federated(model, data, config, device, spec.algorithm)
    _load_state(model, best_state, device)
    test_metrics = evaluate(model, data.test_loader, device, data.class_names)
    result: Dict[str, Any] = {
        "baseline": spec.name,
        "description": spec.description,
        "architecture": spec.architecture,
        "algorithm": spec.algorithm,
        "differential_privacy": False,
        "temporal_modeling": bool(config.modality == "fusion" and spec.use_tcn),
        "note": (
            "Fusion pairs image and video samples by their client-local index and uses the facial "
            "ASD label; the source datasets do not provide subject-level matched pairs."
            if config.modality == "fusion"
            else "For video mode, image backbones are applied to each frame and logits are averaged. "
            "For facial mode, the MobileNet-without-TCN entry is an image-only ablation and is "
            "therefore architecturally identical to FedAvg MobileNet."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.as_dict(),
        "dataset": {
            "modality": config.modality,
            "input_modalities": (
                ["facial_image", "behavioral_video"]
                if config.modality == "fusion"
                else [config.modality]
            ),
            "target_source": (
                "facial ASD label (autistic/non_autistic)"
                if config.modality == "fusion"
                else "dataset label"
            ),
            "class_names": data.class_names,
            "train_samples": data.train_samples,
            "validation_samples": data.validation_samples,
            "test_samples": data.test_samples,
            "client_train_samples": data.client_sample_counts,
        },
        "model_parameters": parameters,
        "rounds_completed": len(history),
        "history": history,
        "test_metrics": test_metrics,
    }
    config.output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_path / f"{config.modality}_{spec.name}_seed_{config.seed}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "baseline": spec.name,
            "architecture": spec.architecture,
            "num_classes": data.num_classes,
            "class_names": data.class_names,
            "config": config.as_dict(),
        },
        checkpoint_path,
    )
    result["checkpoint"] = str(checkpoint_path)
    _write_json(
        config.output_path / f"{config.modality}_{spec.name}_seed_{config.seed}.json", result
    )
    print(
        f"Completed {spec.name}: test_acc={test_metrics['accuracy']:.4f}; "
        f"artifacts saved to {config.output_path}",
        flush=True,
    )
    return result


def _write_summary(results: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    rows = []
    for result in results:
        metrics = result["test_metrics"]
        rows.append(
            {
                "baseline": result["baseline"],
                "architecture": result["architecture"],
                "algorithm": result["algorithm"],
                "rounds_completed": result["rounds_completed"],
                "parameters": result["model_parameters"],
                "test_accuracy": metrics["accuracy"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_f1_macro": metrics["f1_macro"],
                "test_f1_weighted": metrics["f1_weighted"],
                "test_auc_roc": metrics["auc_roc"],
            }
        )
    modality = results[0]["config"]["modality"]
    with (output_dir / f"{modality}_benchmark_summary_seed_{results[0]['config']['seed']}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmarks(names: Sequence[str], config: BenchmarkConfig) -> List[Dict[str, Any]]:
    """Run requested benchmark names using one common split and partition."""
    if not names:
        raise ValueError("Select at least one baseline.")
    unknown = sorted(set(names) - set(AVAILABLE_BASELINES))
    if unknown:
        raise ValueError(f"Unknown baseline(s): {', '.join(unknown)}")
    set_reproducibility(config.seed)
    print(
        f"Preparing {config.modality} data for {len(names)} baseline(s) "
        f"with {config.num_clients} client(s)...",
        flush=True,
    )
    data = build_benchmark_data(config)
    print(
        f"Data ready: train={data.train_samples}, validation={data.validation_samples}, "
        f"test={data.test_samples}; client train samples={data.client_sample_counts}",
        flush=True,
    )
    results = [run_baseline(AVAILABLE_BASELINES[name], data, config) for name in names]
    _write_summary(results, config.output_path)
    _write_json(
        config.output_path / f"{config.modality}_benchmark_results_seed_{config.seed}.json",
        {"config": config.as_dict(), "results": results},
    )
    print(f"All benchmark results saved to {config.output_path}", flush=True)
    return results
