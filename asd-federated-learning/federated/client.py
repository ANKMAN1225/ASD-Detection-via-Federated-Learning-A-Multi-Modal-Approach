"""
Federated client logic - local training and model parameter management.
"""

import random
from typing import Dict, List

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
        device: str = "cpu",
    ) -> None:
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def local_train(self, epochs: int = 1, lr: float = 0.001) -> Dict:
        """Train model locally and return metrics."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

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
                optimizer.step()

                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        return {"loss": avg_loss, "samples": total_samples}

    def local_test(self) -> Dict:
        """Test model locally and return metrics."""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                if len(data.shape) == 5:  # Video data
                    output, _ = self.model(data)
                else:  # Image data
                    output, _ = self.model(data)

                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        accuracy = 100.0 * correct / total
        avg_loss = test_loss / total

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "predictions": all_preds,
            "targets": all_targets,
            "samples": total,
        }

    def get_model_params(self) -> Dict:
        """Get model parameters for federated averaging."""
        return {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def set_model_params(self, params: Dict) -> None:
        """Set model parameters from federated averaging."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(params[name])


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
