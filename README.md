# ASD Detection via Federated Learning — Multimodal Approach

This directory is the runnable implementation used for the ASD federated-learning experiments. It trains simulated federated clients under two configurations:

- `IID_Distribution`: IID client data, without differential privacy.
- `With_Differential_Privacy`: non-IID client data with DP-SGD.

Each configuration trains three components: a facial MobileNetV2 classifier, a behavioral-video MobileNetV2 + TCN classifier, and a late-fusion multimodal model. The project saves per-run metrics, plots, and model checkpoints locally. It is a research implementation, not a clinical diagnostic tool.

## Code map

| Location | Purpose |
| --- | --- |
| `main.py` | Primary single-seed experiment entry point. |
| `config.py` | Shared dataset, model, federated-learning, privacy, and training settings. |
| `data/` | Facial-image, behavioral-video, and multimodal datasets. |
| `models/` | MobileNetV2 facial encoder, TCN video model, and fusion model. |
| `federated/` | Local client/server aggregation, DP utilities, and optional Flower simulation support. |
| `experiments/experiment_runner.py` | End-to-end data preparation, training, evaluation, checkpointing, and result persistence. |
| `experiments/multi_run.py` | Repeats experiments for the manuscript seeds and writes aggregate statistics. |
| `evaluation/` and `visualization/` | Metrics and plots. |
| `saved_models/`, `saved_results.pkl`, `multi_run_results.*` | Generated artifacts; these are outputs, not input data. |

## Requirements

- Python 3.10–3.12.
- PyTorch and torchvision appropriate for the collaborator's CPU or CUDA version.
- The two datasets are **not included** in the repository and must be obtained separately with permission:
  - a facial image dataset arranged as expected by `data/facial_dataset.py`;
  - the SSBD behavioral-video dataset containing `armFlapping/`, `headBanging/`, and `spinning/` directories (each may contain a nested directory with the same name).

Create an environment and install the project packages:

```powershell
cd asd-federated-learning
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install the PyTorch build matching the machine first if the command above does not provide the desired CUDA build; follow the PyTorch installation selector for that machine.

## Point the code at a local dataset

Set these environment variables before each terminal session. They take precedence over the developer-specific fallback paths in `config.py`, so no source edit is needed.

```powershell
$env:ASD_FACIAL_DATA_PATH = "D:\datasets\autistic-children-facial-data-set"
$env:ASD_VIDEO_DATA_PATH  = "D:\datasets\ssbd2"
```

On macOS/Linux:

```bash
export ASD_FACIAL_DATA_PATH=/path/to/autistic-children-facial-data-set
export ASD_VIDEO_DATA_PATH=/path/to/ssbd2
```

## Reproduce a single run (seed 789)

From this directory, run:

```powershell
python main.py --seed 789 --force-retrain
```

This runs both IID and DP configurations, including facial, behavioral, and fusion experiments, and overwrites `saved_results.pkl` and the corresponding files in `saved_models/`. `--force-retrain` is important: it prevents existing checkpoints from being reused. To run only one configuration, add one of:

```powershell
python main.py --seed 789 --force-retrain --experiment IID_Distribution
python main.py --seed 789 --force-retrain --experiment With_Differential_Privacy
```

To repeat the manuscript-style statistical harness for only seed 789 (which also writes `seed_789_run_results.json` and `.pkl`), run:

```powershell
python -m experiments.multi_run --seeds 789
```

The five manuscript seeds are `42, 123, 456, 789, 2024`:

```powershell
python -m experiments.multi_run --seeds 42,123,456,789,2024
```

## Reproducibility notes

The runner applies the selected seed to Python, NumPy, PyTorch, CUDA (when available), data-loader generators, and cuDNN deterministic mode. With the same code revision, dataset files and ordering, dependency versions, hardware class, and seed, data splits and random initialization are reproducible. GPU kernels and different PyTorch/CUDA versions can still cause small numerical differences; record `python --version`, `pip freeze`, and whether CUDA was used when comparing results.

For the exact code snapshot published to GitHub `main`, use commit `ca9aa1e` (or a later commit only if all collaborators agree to update):

```powershell
git clone https://github.com/ankithmanchale/ASD-Detection-via-Federated-Learning-A-Multi-Modal-Approach.git
cd ASD-Detection-via-Federated-Learning-A-Multi-Modal-Approach
git checkout ca9aa1e
cd asd-federated-learning
```

## Existing results and inference

- Use `python main.py --skip-training` to load `saved_results.pkl` and recreate plots without retraining.
- Use `python main.py --predict --predict_path "path\to\video.mp4"` for the provided face-video inference path.

Do not treat supplied checkpoints or reported metrics as a replacement for evaluation on an independent, appropriately governed dataset.
