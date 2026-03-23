"""
Entry point for ASD Federated Learning project.
Runs all configured experiments end-to-end.
"""

import argparse

from experiments.experiment_runner import run_experiments
from visualization.plots import run_complete_analysis
from inference.video_predict import run_video_prediction


def main() -> dict:
    """Run experiments and return results."""
    parser = argparse.ArgumentParser(description="ASD Federated Learning")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load saved results from disk and skip training (use after first run)",
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
        # Return dict for programmatic use; not required by plots pipeline.
        return results  # type: ignore[return-value]

    results = run_experiments(
        skip_training_if_saved=args.skip_training,
        skip_facial=args.skip_facial
    )

    run_complete_analysis(results, save_graphs=True)

    return results


if __name__ == "__main__":
    main()
