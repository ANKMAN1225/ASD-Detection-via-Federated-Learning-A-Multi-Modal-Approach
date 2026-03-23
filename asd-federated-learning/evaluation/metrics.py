"""
Accuracy, F1, AUC-ROC, confusion matrix, and comprehensive evaluation reporting.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    y_prob: Optional[Union[List, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    metrics: Dict[str, Any] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support.sum(),
    }

    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
            metrics["auc"] = auc
        except ValueError:
            metrics["auc"] = None

    return metrics


class ComprehensiveEvaluator:
    """Comprehensive evaluation of federated learning results."""

    def __init__(self) -> None:
        self.all_results: Dict[str, Dict] = {}

    def evaluate_experiment(
        self, experiment_name: str, results: Dict
    ) -> Dict[str, Dict]:
        """Evaluate single experiment results."""
        final_global_accuracy = (
            results.get("facial_experiment", {})
            .get("final_metrics", {})
            .get("global_accuracy", 0.0)
        )
        evaluation = {
            "experiment_name": experiment_name,
            "final_global_accuracy": final_global_accuracy,
            "convergence_analysis": self.analyze_convergence(results),
            "privacy_analysis": self.analyze_privacy(results),
            "communication_efficiency": self.analyze_communication(results),
            "fairness_analysis": self.analyze_fairness(results),
        }

        self.all_results[experiment_name] = evaluation
        return evaluation

    def analyze_convergence(self, results: Dict) -> Dict:
        """Analyze model convergence."""
        if "facial_experiment" not in results:
            return {"status": "no_data"}

        facial_experiment = results.get("facial_experiment", {})
        round_metrics = facial_experiment.get("round_metrics", [])
        if not round_metrics:
            return {"status": "no_data"}
        losses = [m["avg_loss"] for m in round_metrics]
        if not losses:
            return {"status": "no_data"}

        # Calculate convergence metrics
        initial_loss = losses[0]
        final_loss = losses[-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100

        # Check for convergence (loss reduction < 1% in last 3 rounds)
        if len(losses) >= 3:
            recent_losses = losses[-3:]
            loss_variance = np.var(recent_losses)
            converged = loss_variance < 0.01
        else:
            converged = False

        return {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "improvement_percent": improvement,
            "converged": converged,
            "total_rounds": len(losses),
        }

    def analyze_privacy(self, results: Dict) -> Dict:
        """Analyze privacy preservation."""
        config = results.get("facial_experiment", {}).get("config", {}) or {}

        privacy_analysis = {
            "differential_privacy_enabled": config.get("use_differential_privacy", False),
            "noise_multiplier": config.get("noise_multiplier", 0.0),
            "privacy_budget_estimate": "N/A",
            "data_locality": "Preserved - no raw data sharing",
        }

        if privacy_analysis["differential_privacy_enabled"]:
            noise_level = privacy_analysis["noise_multiplier"]
            rounds = len(
                results.get("facial_experiment", {}).get("round_metrics", [])
            )
            if noise_level > 0:
                privacy_analysis["privacy_budget_estimate"] = (
                    f"ε ≈ {rounds * (1 / noise_level):.2f}"
                )

        return privacy_analysis

    def analyze_communication(self, results: Dict) -> Dict:
        """Analyze communication efficiency."""
        config = results.get("facial_experiment", {}).get("config", {})

        num_clients = config.get("num_clients", 5)
        num_rounds = len(
            results.get("facial_experiment", {}).get("round_metrics", [])
        )

        estimated_params = 2.23e6
        param_size_mb = (estimated_params * 4) / (1024 * 1024)

        total_communication_mb = (
            num_clients * num_rounds * param_size_mb * 2
        )  # Upload + download

        return {
            "total_rounds": num_rounds,
            "num_clients": num_clients,
            "estimated_params": estimated_params,
            "param_size_mb": param_size_mb,
            "total_communication_mb": total_communication_mb,
            "avg_communication_per_round_mb": (
                total_communication_mb / num_rounds if num_rounds > 0 else 0
            ),
        }

    def analyze_fairness(self, results: Dict) -> Dict:
        """Analyze fairness across clients."""
        if "facial_experiment" not in results:
            return {"status": "no_data"}

        facial_experiment = results.get("facial_experiment", {}) or {}
        final_metrics = facial_experiment.get("final_metrics", None)
        if not final_metrics:
            return {"status": "no_data"}
        client_metrics = final_metrics.get("client_metrics", [])

        if not client_metrics:
            return {"status": "no_client_data"}

        client_accuracies = [m["accuracy"] for m in client_metrics]

        return {
            "mean_accuracy": np.mean(client_accuracies),
            "std_accuracy": np.std(client_accuracies),
            "min_accuracy": np.min(client_accuracies),
            "max_accuracy": np.max(client_accuracies),
            "accuracy_range": np.max(client_accuracies) - np.min(client_accuracies),
            "coefficient_of_variation": (
                np.std(client_accuracies) / np.mean(client_accuracies)
                if np.mean(client_accuracies) > 0
                else 0
            ),
            "fairness_score": 1 - (np.std(client_accuracies) / 100),
        }

    def generate_comparison_report(self) -> None:
        """Generate comprehensive comparison report."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE FEDERATED LEARNING EVALUATION REPORT")
        print("=" * 80)

        if not self.all_results:
            print("No results available for comparison.")
            return

        print(
            f"\n{'Experiment':<25} {'Global Acc':<12} {'Privacy':<10} {'Fairness':<10}"
        )
        print("-" * 80)

        for exp_name, eval_data in self.all_results.items():
            privacy = eval_data.get("privacy_analysis", {})
            fairness = eval_data.get("fairness_analysis", {})

            final_acc = f"{eval_data.get('final_global_accuracy', 0.0):.1f}%"
            privacy_enabled = (
                "Yes" if privacy.get("differential_privacy_enabled", False) else "No"
            )
            fairness_score = (
                f"{fairness.get('fairness_score', 0):.2f}"
                if isinstance(fairness, dict) and "fairness_score" in fairness
                else "N/A"
            )

            print(
                f"{exp_name:<25} {final_acc:<12} {privacy_enabled:<10} "
                f"{fairness_score:<10}"
            )
