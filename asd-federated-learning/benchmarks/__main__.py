"""Command line entry point for the standalone benchmark package."""

from __future__ import annotations

import argparse
import os
from typing import List

from .config import BenchmarkConfig
from .engine import AVAILABLE_BASELINES, run_benchmarks


def _parse_baselines(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(AVAILABLE_BASELINES)
    return [name.strip() for name in value.split(",") if name.strip()]


def main() -> List[dict]:
    parser = argparse.ArgumentParser(
        description="Independent ASD baseline benchmarking framework (does not run main.py)."
    )
    parser.add_argument("--baselines", default="all", help="Comma-separated baseline names or 'all'.")
    parser.add_argument("--modality", choices=["facial", "video", "fusion"], default="facial")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Facial root for facial/fusion mode, or SSBD video root for video mode.",
    )
    parser.add_argument(
        "--video-data-dir",
        default=None,
        help="SSBD video root. Required together with facial --data-dir for fusion mode.",
    )
    parser.add_argument("--num-clients", type=int, default=5)
    partition = parser.add_mutually_exclusive_group()
    partition.add_argument("--iid", dest="iid", action="store_true", default=True)
    partition.add_argument("--non-iid", dest="iid", action="store_false")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=0,
        help="Deterministic cap per class in every split (0 keeps all samples). Useful for smoke tests.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision ImageNet weights (may download them).")
    parser.add_argument("--no-class-weights", dest="class_weighted_loss", action="store_false")
    parser.set_defaults(class_weighted_loss=True)
    parser.add_argument("--patience", type=int, default=5, help="Validation rounds without improvement before stopping.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="", help="e.g. cpu, cuda, cuda:0; default selects CUDA when available.")
    parser.add_argument("--output-dir", default="benchmark_results")
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = (
            os.environ.get("ASD_VIDEO_DATA_PATH", "../SSBD-file/ssbd2")
            if args.modality == "video"
            else os.environ.get("ASD_FACIAL_DATA_PATH", "../autistic-children-facial-data-set")
        )
    if args.video_data_dir is None:
        args.video_data_dir = os.environ.get("ASD_VIDEO_DATA_PATH", "../SSBD-file/ssbd2")
    config = BenchmarkConfig(
        modality=args.modality,
        data_dir=args.data_dir,
        video_data_dir=args.video_data_dir,
        num_clients=args.num_clients,
        iid=args.iid,
        dirichlet_alpha=args.dirichlet_alpha,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fedprox_mu=args.fedprox_mu,
        image_size=args.image_size,
        sequence_length=args.sequence_length,
        max_samples_per_class=args.max_samples_per_class,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        pretrained=args.pretrained,
        class_weighted_loss=args.class_weighted_loss,
        early_stopping_patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        device=args.device,
    )
    names = _parse_baselines(args.baselines)
    results = run_benchmarks(names, config)
    for result in results:
        metrics = result["test_metrics"]
        print(
            f"{result['baseline']}: accuracy={metrics['accuracy']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}, rounds={result['rounds_completed']}"
        )
    print(f"Results written to {config.output_path}")
    return results


if __name__ == "__main__":
    main()
