"""
Multi-run experiment harness for statistical evaluation.

Repeats each experiment configuration over N independent runs with different
random seeds to produce mean ± std and 95% CI for all metrics, as described
in the revised manuscript (Section 5.3 Statistical Evaluation Protocol).

Reviewer response: Addresses the concern that single-run results lack
confidence intervals and cannot assess variability from random initialization.

Usage
-----
    from experiments.multi_run import run_multi_experiment
    results = run_multi_experiment(n_runs=5)
"""

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to sys.path so 'config' refers to the local config.py file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import get_experiment_configs
from experiments.experiment_runner import CompleteFederatedTrainer, set_random_seeds

# Seeds used for the 5 independent runs (manuscript Table A1, A2)
MANUSCRIPT_SEEDS: List[int] = [42, 123, 456, 789, 2024]


def _extract_metrics(results: Dict[str, Any], exp_key: str) -> Dict[str, float]:
    """Pull scalar metrics from a single run's result dict."""
    exp_data = results.get(exp_key, {})
    final = exp_data.get("final_metrics", {})

    return {
        "accuracy": float(final.get("global_accuracy", 0.0)),
        "precision": float(final.get("global_precision", 0.0)),
        "recall": float(final.get("global_recall", 0.0)),
        "f1": float(final.get("global_f1", 0.0)),
        "auc": float(final.get("global_auc", 0.0)),
        "client_accuracy_variance": float(final.get("client_accuracy_variance", 0.0)),
    }


def _compute_stats(
    values: List[float],
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Compute mean, std, and confidence interval for a list of values.

    Uses the t-distribution for the CI since sample sizes are small (n=5).

    Parameters
    ----------
    values : list of float
    confidence : float
        Confidence level (default 0.95 for 95% CI).

    Returns
    -------
    dict with keys: mean, std, ci_lower, ci_upper, ci_half_width
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_half_width": 0.0}

    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))  # sample std

    # t critical value for two-tailed CI (df = n-1)
    # For n=5, df=4: t* ≈ 2.776 at 95% CI
    # Using scipy if available; otherwise hard-code common values
    try:
        from scipy import stats  # type: ignore
        t_star = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    except ImportError:
        # Fallback: t* for common (n, confidence) combinations
        t_table = {
            (5, 0.95): 2.776,
            (5, 0.99): 4.604,
            (10, 0.95): 2.262,
            (10, 0.99): 3.250,
        }
        t_star = t_table.get((n, confidence), 2.776)

    se = std / math.sqrt(n)
    half_width = t_star * se

    return {
        "mean": mean,
        "std": std,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
        "ci_half_width": half_width,
    }


def _paired_ttest(
    values_a: List[float],
    values_b: List[float],
) -> Dict[str, float]:
    """Paired t-test between two matched sets of values.

    Used to test whether IID and DP-enabled models produce significantly
    different accuracy (manuscript Section 6.3.1).

    Returns
    -------
    dict with t_stat, p_value, mean_diff, std_diff
    """
    if len(values_a) != len(values_b):
        raise ValueError("Paired t-test requires equal-length lists.")

    diffs = [a - b for a, b in zip(values_a, values_b)]
    n = len(diffs)
    arr = np.array(diffs, dtype=float)
    mean_diff = float(np.mean(arr))
    std_diff = float(np.std(arr, ddof=1))
    se = std_diff / math.sqrt(n)
    t_stat = mean_diff / se if se > 0 else 0.0

    try:
        from scipy import stats  # type: ignore
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    except ImportError:
        # Approximate: for t(4) = 3.42 -> p ≈ 0.027
        p_value = float("nan")

    return {
        "n_pairs": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "df": n - 1,
        "p_value": p_value,
        "significant_at_05": p_value < 0.05 if not math.isnan(p_value) else False,
    }


def run_multi_experiment(
    n_runs: int = 5,
    seeds: Optional[List[int]] = None,
    skip_facial: bool = False,
    use_flower: bool = False,
) -> Dict[str, Any]:
    """Run all experiment configurations n_runs times with different seeds.

    Parameters
    ----------
    n_runs : int
        Number of independent repetitions (manuscript: 5).
    seeds : list of int, optional
        Random seeds, one per run.  Defaults to MANUSCRIPT_SEEDS.
    skip_facial : bool
        If True, skip the facial modality (useful for quick behavioral-only tests).
    use_flower : bool
        If True, run using the Flower simulation framework.

    Returns
    -------
    dict with keys:
        'per_run_results'    : raw per-run results
        'aggregate'          : mean ± std and 95% CI per metric per experiment
        'statistical_tests'  : paired t-test IID vs DP
        'seeds'              : seeds used
    """
    if seeds is None:
        seeds = MANUSCRIPT_SEEDS[:n_runs]
    if len(seeds) < n_runs:
        raise ValueError(f"Need {n_runs} seeds, got {len(seeds)}.")

    configs = get_experiment_configs()
    # Track per-run metrics for facial and behavioral separately
    per_run_facial: Dict[str, List[Dict[str, float]]] = {
        cfg["experiment_name"]: [] for cfg in configs
    }
    per_run_behavioral: Dict[str, List[Dict[str, float]]] = {
        cfg["experiment_name"]: [] for cfg in configs
    }
    per_run_fusion: Dict[str, List[Dict[str, float]]] = {
        cfg["experiment_name"]: [] for cfg in configs
    }

    print("\n" + "=" * 70)
    print(f"MULTI-RUN EXPERIMENT: {n_runs} runs × {len(configs)} configurations")
    print(f"Seeds: {seeds[:n_runs]}")
    print("Dataset: Augmented SSBD (ssbd2), 183 videos, 3 behavioral classes")
    print("=" * 70)

    for run_idx, seed in enumerate(seeds[:n_runs]):
        print(f"\n{'=' * 50}")
        print(f"RUN {run_idx + 1}/{n_runs}  (seed={seed})")
        print("=" * 50)
        set_random_seeds(seed)

        for cfg in configs:
            cfg["use_flower"] = use_flower
            exp_name = cfg["experiment_name"]
            print(f"\n  Experiment: {exp_name}")
            cfg_copy = {**cfg, "random_seed": seed, "skip_saved_models": False}

            try:
                trainer = CompleteFederatedTrainer(cfg_copy)

                if skip_facial:
                    facial_res: Dict = {"status": "skipped", "final_metrics": {}}
                    facial_train_loaders = None
                    facial_test_loaders = None
                else:
                    facial_res = trainer.run_facial_experiment()
                    facial_train_loaders, _, facial_test_loaders = (
                        trainer.prepare_facial_datasets()
                    )

                behavioral_res = trainer.run_behavioral_experiment()
                behavioral_train_subsets = getattr(trainer, "_behavioral_train_subsets", None)
                behavioral_test_subsets = getattr(trainer, "_behavioral_test_subsets", None)
                if behavioral_train_subsets is None:
                    _, _, behavioral_train_subsets, behavioral_test_subsets = (
                        trainer.prepare_behavioral_datasets()
                    )

                if skip_facial or facial_train_loaders is None:
                    fusion_res: Dict = {"status": "skipped", "final_metrics": {}}
                else:
                    fusion_res = trainer.run_fusion_experiment(
                        facial_train_loaders=facial_train_loaders,
                        facial_test_loaders=facial_test_loaders,
                        behavioral_train_subsets=behavioral_train_subsets,
                        behavioral_test_subsets=behavioral_test_subsets,
                    )

                run_metrics_facial = _extract_metrics(
                    {"facial_experiment": facial_res}, "facial_experiment"
                )
                run_metrics_behav = _extract_metrics(
                    {"behavioral_experiment": behavioral_res}, "behavioral_experiment"
                )
                run_metrics_fusion = _extract_metrics(
                    {"fusion_experiment": fusion_res}, "fusion_experiment"
                )
                per_run_facial[exp_name].append(run_metrics_facial)
                per_run_behavioral[exp_name].append(run_metrics_behav)
                per_run_fusion[exp_name].append(run_metrics_fusion)

                print(
                    f"  Facial: {run_metrics_facial['accuracy']:.2f}% | "
                    f"Behavioral: {run_metrics_behav['accuracy']:.2f}% | "
                    f"Fusion: {run_metrics_fusion['accuracy']:.2f}%"
                )

            except Exception as exc:  # pragma: no cover
                print(f"  ERROR in run {run_idx + 1} / {exp_name}: {exc}")
                import traceback
                traceback.print_exc()

    # -----------------------------------------------------------------------
    # Aggregate statistics
    # -----------------------------------------------------------------------
    metric_names = ["accuracy", "precision", "recall", "f1", "auc", "client_accuracy_variance"]
    
    def aggregate_runs(per_run_dict):
        agg: Dict[str, Dict[str, Any]] = {}
        for exp_name, runs in per_run_dict.items():
            agg[exp_name] = {}
            for metric in metric_names:
                values = [r[metric] for r in runs if metric in r]
                agg[exp_name][metric] = _compute_stats(values)
        return agg

    aggregate_facial = aggregate_runs(per_run_facial)
    aggregate_behavioral = aggregate_runs(per_run_behavioral)
    aggregate_fusion = aggregate_runs(per_run_fusion)

    # -----------------------------------------------------------------------
    # Statistical comparison: IID vs DP
    # -----------------------------------------------------------------------
    stat_tests: Dict[str, Any] = {}
    iid_name = "IID_Distribution"
    dp_name = "With_Differential_Privacy"

    def add_ttest(per_run_dict, label):
        if iid_name in per_run_dict and dp_name in per_run_dict:
            iid_acc = [r["accuracy"] for r in per_run_dict[iid_name]]
            dp_acc = [r["accuracy"] for r in per_run_dict[dp_name]]
            if len(iid_acc) == len(dp_acc) and len(iid_acc) > 1:
                stat_tests[label] = _paired_ttest(iid_acc, dp_acc)

    add_ttest(per_run_facial, "facial_iid_vs_dp")
    add_ttest(per_run_behavioral, "behavioral_iid_vs_dp")
    add_ttest(per_run_fusion, "fusion_iid_vs_dp")

    def print_summary(agg_dict, title):
        print("\n" + "=" * 70)
        print(f"{title} (mean ± std  [95% CI])")
        print("=" * 70)
        for exp_name, metrics in agg_dict.items():
            print(f"\n{exp_name}")
            print("-" * 40)
            for metric, stats in metrics.items():
                m, s = stats["mean"], stats["std"]
                lo, hi = stats["ci_lower"], stats["ci_upper"]
                print(f"  {metric:<30s}: {m:6.2f} ± {s:.2f}  [{lo:.2f}, {hi:.2f}]")

    if not skip_facial:
        print_summary(aggregate_facial, "MULTI-RUN SUMMARY: FACIAL")
    print_summary(aggregate_behavioral, "MULTI-RUN SUMMARY: BEHAVIORAL")
    if not skip_facial:
        print_summary(aggregate_fusion, "MULTI-RUN SUMMARY: FUSION")

    def print_ttest(key, label):
        if key in stat_tests:
            t = stat_tests[key]
            print(f"\nPaired t-test ({label} IID vs DP, accuracy):")
            print(f"  Mean Δ     : {t['mean_diff']:.2f}%")
            print(f"  t-statistic: t({t['df']}) = {t['t_stat']:.3f}")
            print(f"  p-value    : {t['p_value']:.4f}")
            sig = "SIGNIFICANT" if t["significant_at_05"] else "not significant"
            print(f"  Result     : {sig} at α=0.05")

    if not skip_facial:
        print_ttest("facial_iid_vs_dp", "Facial")
    print_ttest("behavioral_iid_vs_dp", "Behavioral")
    if not skip_facial:
        print_ttest("fusion_iid_vs_dp", "Fusion")

    return {
        "per_run_results": {
            "facial": per_run_facial,
            "behavioral": per_run_behavioral,
            "fusion": per_run_fusion,
        },
        "aggregate": {
            "facial": aggregate_facial,
            "behavioral": aggregate_behavioral,
            "fusion": aggregate_fusion,
        },
        "statistical_tests": stat_tests,
        "seeds": seeds[:n_runs],
        "n_runs": n_runs,
    }


if __name__ == "__main__":
    import argparse
    import json
    import os
    import pickle

    parser = argparse.ArgumentParser(description="Multi-run statistical evaluation.")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs (default 5)")
    parser.add_argument("--skip-facial", action="store_true", help="Skip facial experiment")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to run, for example: 789 or 42,123,456",
    )
    args = parser.parse_args()

    selected_seeds = None
    n_runs = args.runs
    if args.seeds:
        selected_seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
        n_runs = len(selected_seeds)

    results = run_multi_experiment(
        n_runs=n_runs,
        seeds=selected_seeds,
        skip_facial=args.skip_facial,
    )

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if selected_seeds is None:
        result_stem = "multi_run_results"
    else:
        seed_label = "_".join(str(seed) for seed in selected_seeds)
        result_stem = f"seed_{seed_label}_run_results"

    pkl_path = os.path.join(out_dir, f"{result_stem}.pkl")
    json_path = os.path.join(out_dir, f"{result_stem}.json")

    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved statistical results to:\n  {pkl_path}\n  {json_path}")
