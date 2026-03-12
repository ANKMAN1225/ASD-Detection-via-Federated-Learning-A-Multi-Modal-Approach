"""
Entry point for ASD Federated Learning project.
Runs all configured experiments end-to-end.
"""

import argparse

from experiments.experiment_runner import run_experiments
from visualization.plots import run_complete_analysis


def main() -> dict:
    """Run experiments and return results."""
    parser = argparse.ArgumentParser(description="ASD Federated Learning")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load saved results from disk and skip training (use after first run)",
    )
    args = parser.parse_args()

    results = run_experiments(skip_training_if_saved=args.skip_training)

    run_complete_analysis(results, save_graphs=True)

    return results


if __name__ == "__main__":
    main()
