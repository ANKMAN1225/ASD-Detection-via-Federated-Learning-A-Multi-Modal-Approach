"""
Federated client logic - local training and model parameter management.
"""

import random
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch.utils.data


class FederatedClient:
    """Individual federated learning client."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cpu",
    ) -> None:
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = device

    def _compute_class_weights(self, num_classes: int = 2) -> Optional[torch.Tensor]:
        """Compute inverse-frequency class weights from this client's training data.

        Reduces bias when the client's local dataset is class-imbalanced.
        Returns None if labels cannot be extracted (e.g. dummy TensorDataset
        with random labels where balancing would be meaningless).
        """
        try:
            dataset = self.train_loader.dataset
            labels = [int(dataset[i][1]) for i in range(len(dataset))]
            counts = Counter(labels)
            if len(counts) < 2:
                return None  # Only one class — skip weighting
            total = sum(counts.values())
            weights = torch.tensor(
                [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
                dtype=torch.float32,
            ).to(self.device)
            return weights
        except Exception:
            return None

    def local_train(self, epochs: int = 1, lr: float = 0.001, dp=None) -> Dict:
        """Train model locally and return metrics.

        Improvements vs. naive implementation:
        - Differential learning rates: backbone gets lr*0.1, classifier gets lr.
          Pretrained backbone features should be fine-tuned slowly.
        - CosineAnnealingLR: smoothly decays LR within each local training
          session, improving convergence stability.
        - Class-weighted CrossEntropyLoss: corrects for per-client class
          imbalance without adding hyperparameters.
        - Optional dp (DifferentialPrivacy): clips + noises gradients after
          loss.backward() to provide differential privacy guarantees.
        """
        self.model.train()

        # ── 1. Differential learning rate param groups ─────────────────────
        backbone_params = [
            p for name, p in self.model.named_parameters()
            if "classifier" not in name and p.requires_grad
        ]
        classifier_params = [
            p for name, p in self.model.named_parameters()
            if "classifier" in name and p.requires_grad
        ]

        if backbone_params:
            optimizer = torch.optim.Adam([
                {"params": backbone_params, "lr": lr * 0.1},   # fine-tune slowly
                {"params": classifier_params, "lr": lr},        # new head — normal LR
            ])
        else:
            # All params are in classifier (or backbone is fully frozen)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # ── 2. Cosine LR decay within local session ─────────────────────────
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01
        )

        # ── 3. Class-balanced loss ──────────────────────────────────────────
        class_weights = self._compute_class_weights(num_classes=2)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        total_loss = 0.0
        total_samples = 0

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()

                if len(data.shape) == 5:  # Video data
                    output, _ = self.model(data)
                else:  # Image data
                    output, _ = self.model(data)

                loss = criterion(output, target)
                loss.backward()

                # ── 4. Apply DP noise to gradients if requested ─────────────
                if dp is not None:
                    dp.add_noise_to_gradients(self.model)

                optimizer.step()

                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

            scheduler.step()

        avg_loss = total_loss / total_samples
        return {"loss": avg_loss, "samples": total_samples}

    def _local_eval(self, loader: DataLoader, include_predictions: bool = True) -> Dict:
        """Evaluate model locally on a given loader."""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds: List = []
        all_targets: List = []

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)

                if len(data.shape) == 5:  # Video data
                    output, _ = self.model(data)
                else:  # Image data
                    output, _ = self.model(data)

                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                if include_predictions:
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

        accuracy = 100.0 * correct / total
        avg_loss = test_loss / total

        metrics: Dict[str, object] = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "samples": total,
        }
        if include_predictions:
            metrics["predictions"] = all_preds
            metrics["targets"] = all_targets
        return metrics  # type: ignore[return-value]

    def local_validate(self, include_predictions: bool = True) -> Dict:
        """Validate model locally and return metrics."""
        if self.val_loader is None:
            # Fallback to test_loader if validation isn't available.
            return self._local_eval(self.test_loader, include_predictions=include_predictions)
        return self._local_eval(self.val_loader, include_predictions=include_predictions)

    def local_test(self, include_predictions: bool = True) -> Dict:
        """Test model locally and return metrics."""
        return self._local_eval(self.test_loader, include_predictions=include_predictions)

    def get_model_params(self) -> Dict:
        """Get full model state_dict for federated averaging.

        Uses state_dict (not named_parameters) so that BatchNorm buffers
        (running_mean, running_var, num_batches_tracked) are included in
        FedAvg — critical for correct inference after aggregation.
        """
        return {k: v.clone().detach() for k, v in self.model.state_dict().items()}

    def set_model_params(self, params: Dict) -> None:
        """Load aggregated state_dict into local model (includes BN buffers)."""
        self.model.load_state_dict(params, strict=True)


def create_federated_data_splits(
    dataset: Dataset,
    num_clients: int,
    iid: bool = True,
    alpha: float = 0.5,
) -> List[Dataset]:
    """Split dataset among federated clients. IID or Non-IID via Dirichlet distribution."""
    if iid:
        # IID split
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        client_indices = np.array_split(indices, num_clients)
    else:
        # Non-IID split using Dirichlet distribution
        labels = [dataset[i][1] for i in range(len(dataset))]
        labels = [l.item() if isinstance(l, torch.Tensor) else l for l in labels]
        num_classes = len(set(labels))

        client_indices = [[] for _ in range(num_clients)]

        for class_id in range(num_classes):
            class_indices = [i for i, label in enumerate(labels) if label == class_id]

            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (proportions * len(class_indices)).astype(int)

            # Ensure we don't lose any samples
            proportions[-1] = len(class_indices) - proportions[:-1].sum()

            start_idx = 0
            for client_id, prop in enumerate(proportions):
                if prop > 0:
                    end_idx = start_idx + prop
                    client_indices[client_id].extend(class_indices[start_idx:end_idx])
                    start_idx = end_idx

    # Create subset datasets
    client_datasets = []
    for indices in client_indices:
        if len(indices) > 0:
            client_datasets.append(torch.utils.data.Subset(dataset, indices))
        else:
            client_datasets.append(torch.utils.data.Subset(dataset, [0]))  # Dummy dataset

    return client_datasets
