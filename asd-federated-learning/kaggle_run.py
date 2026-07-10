"""
Kaggle GPU entry point — uses the SAME code as local main.py.

SETUP ON KAGGLE
---------------
1. Create a Kaggle Notebook with GPU (T4/P100) enabled.
2. Add input datasets:
   - ankithmanchale/autism          (facial images)
   - shradheypathak/ssbd3           (ssbd2 behavioral videos)
3. Upload this entire `asd-federated-learning` folder as a Kaggle dataset
   OR zip it and add to the notebook, then:
       !unzip -q asd-federated-learning.zip -d /kaggle/working/
4. In a notebook cell:
       %cd /kaggle/working/asd-federated-learning
       !pip install -q opacus scikit-learn
       !python kaggle_run.py --force-retrain --experiment With_Differential_Privacy

SPLIT WITH LOCAL MACHINE (recommended)
--------------------------------------
  Local CPU  : python main.py --force-retrain --experiment IID_Distribution
  Kaggle GPU : python kaggle_run.py --force-retrain --experiment With_Differential_Privacy

Both runs produce identical checkpoint names under saved_models/:
  {Experiment}_facial.pt
  {Experiment}_behavioral.pt
  {Experiment}_fusion.pt
"""

from __future__ import annotations

import argparse
import os
import sys


def _setup_kaggle_paths() -> str:
    """Configure Kaggle dataset paths and output directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    kaggle_facial = "/kaggle/input/datasets/ankithmanchale/autism/autistic-children-facial-data-set"
    kaggle_ssbd = "/kaggle/input/datasets/shradheypathak/ssbd3/ssbd2"
    output_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else script_dir

    import config

    if os.path.isdir(kaggle_facial):
        if kaggle_facial not in config.FACIAL_DATA_PATHS:
            config.FACIAL_DATA_PATHS.insert(0, kaggle_facial)
    if os.path.isdir(kaggle_ssbd):
        if kaggle_ssbd not in config.VIDEO_DATA_PATHS:
            config.VIDEO_DATA_PATHS.insert(0, kaggle_ssbd)

    # Save artifacts to /kaggle/working so they appear in Output tab
    if os.path.isdir("/kaggle/working"):
        import experiments.experiment_runner as er

        er.SAVED_MODELS_DIR = os.path.join(output_dir, "saved_models")
        er.SAVED_RESULTS_PATH = os.path.join(output_dir, "saved_results.pkl")
        os.makedirs(er.SAVED_MODELS_DIR, exist_ok=True)

    import torch

    print("=" * 60)
    print("Kaggle GPU Run — synced with local main.py")
    print("=" * 60)
    print(f"Project root     : {script_dir}")
    print(f"Facial path      : {config._resolve_facial_data_path()}")
    print(f"SSBD path        : {config._resolve_video_data_path()}")
    print(f"Output dir       : {output_dir}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    return output_dir


def _is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell in ("ZMQInteractiveShell", "Shell")
    except NameError:
        return False


def parse_and_run(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="ASD Federated Learning — Kaggle GPU (same as local main.py)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load saved_results.pkl and skip training if it exists.",
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
        "--experiment",
        type=str,
        choices=["all", "IID_Distribution", "With_Differential_Privacy"],
        default="With_Differential_Privacy",
        help=(
            "Which experiment to run. Default: With_Differential_Privacy "
            "(run IID_Distribution on your local machine)."
        ),
    )

    if _is_notebook() and argv is None:
        # Notebook default: DP experiment, force retrain
        args = parser.parse_args(["--force-retrain", "--experiment", "With_Differential_Privacy"])
    else:
        args = parser.parse_args(argv)

    _setup_kaggle_paths()

    from experiments.experiment_runner import run_experiments
    from visualization.plots import run_complete_analysis

    experiment_names = None
    if args.experiment != "all":
        experiment_names = [args.experiment]

    results = run_experiments(
        skip_training_if_saved=args.skip_training,
        skip_facial=args.skip_facial,
        skip_fusion=args.skip_fusion,
        force_retrain=args.force_retrain,
        experiment_names=experiment_names,
    )

    try:
        run_complete_analysis(results, save_graphs=True)
    except Exception as exc:
        print(f"WARNING: Plot generation skipped: {exc}")

    return results


if __name__ == "__main__":
    parse_and_run()
