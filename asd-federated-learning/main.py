"""
Entry point for ASD Federated Learning project.
Runs all configured experiments end-to-end.
"""

import argparse

from experiments.experiment_runner import run_experiments
from visualization.plots import run_complete_analysis
from inference.video_predict import run_video_prediction


def main() -> dict:
    parser = argparse.ArgumentParser(description="ASD Federated Learning")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load saved results from disk and skip training (use after first run)",
    )
    parser.add_argument(
        "--skip-facial",
        action="store_true",
        help="Skip the facial experiment.",
    )
    parser.add_argument(
        "--skip-fusion",
        action="store_true",
        help="Skip the multi-modal fusion experiment.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if saved model checkpoints exist.",
    )
    parser.add_argument(
        "--evaluate-saved-models",
        action="store_true",
        help=(
            "Evaluate the existing checkpoints without training. Fails if a "
            "required checkpoint is missing."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Print and save results without generating visualization files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for a single experiment run.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["all", "IID_Distribution", "With_Differential_Privacy"],
        default="all",
        help="Run one experiment config (useful when splitting local vs Kaggle).",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run video inference (autistic vs non_autistic) without training",
    )
    parser.add_argument(
        "--predict_path",
        type=str,
        default=None,
        help="Path to a face-cropped video (e.g. 1 crop.mp4). If not set, you'll be prompted.",
    )
    parser.add_argument(
        "--predict_stride",
        type=int,
        default=10,
        help="Take 1 frame every N frames during inference (default: 10).",
    )
    parser.add_argument(
        "--predict_max_frames",
        type=int,
        default=40,
        help="Max number of sampled frames to process (default: 40).",
    )
    args = parser.parse_args()

    if args.predict:
        video_path = args.predict_path
        if not video_path:
            video_path = input("Enter path to face-cropped video: ").strip()

        results = run_video_prediction(
            video_path=video_path,
            stride=args.predict_stride,
            max_frames=args.predict_max_frames,
        )
        return results  # type: ignore[return-value]

    experiment_names = None
    if args.experiment != "all":
        experiment_names = [args.experiment]

    results = run_experiments(
        skip_training_if_saved=args.skip_training,
        skip_facial=args.skip_facial,
        skip_fusion=args.skip_fusion,
        force_retrain=args.force_retrain,
        evaluate_saved_models_only=args.evaluate_saved_models,
        experiment_names=experiment_names,
        random_seed=args.seed,
    )

    run_complete_analysis(results, save_graphs=not args.no_plots)
    return results


if __name__ == "__main__":
    main()
