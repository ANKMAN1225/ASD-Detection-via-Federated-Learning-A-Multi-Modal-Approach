"""
Federated learning framework for ASD detection.
FedAvg aggregation, client training, and privacy mechanisms.
"""

from .client import FederatedClient, create_federated_data_splits
from .server import FederatedServer
from .privacy import DifferentialPrivacy, secure_aggregate

__all__ = [
    "FederatedClient",
    "FederatedServer",
    "create_federated_data_splits",
    "DifferentialPrivacy",
    "secure_aggregate",
]
