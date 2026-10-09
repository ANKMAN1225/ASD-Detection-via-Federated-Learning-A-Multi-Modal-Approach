# Independent baseline benchmark framework

This directory is intentionally separate from the paper pipeline. It does not
import `main.py`, modify saved checkpoints, write `saved_results.pkl`, or alter
the project's data/model/federated modules. It independently reads the same
facial and SSBD layouts, creates one deterministic split per invocation, and
writes results only to `benchmark_results/` (or `--output-dir`).

For the end-to-end multimodal comparison, run from `asd-federated-learning`:

```powershell
python -m benchmarks --modality fusion --data-dir ..\autistic-children-facial-data-set --video-data-dir ..\SSBD-file\ssbd2 --baselines all --num-clients 5 --non-iid --dirichlet-alpha 0.5 --rounds 50 --local-epochs 1 --batch-size 2 --sequence-length 16 --pretrained --seed 42
```

For a quick fusion smoke test that does not download weights:

```powershell
python -m benchmarks --modality fusion --data-dir ..\autistic-children-facial-data-set --video-data-dir ..\SSBD-file\ssbd2 --baselines all --num-clients 2 --rounds 1 --local-epochs 1 --batch-size 2 --max-samples-per-class 4 --device cpu
```

The available benchmark names are:

- `centralized_mobilenet` — MobileNetV2 on the pooled training set.
- `mobilenet_without_tcn` — frame-wise MobileNetV2 with average-logit pooling; it has no temporal module.
- `fedavg_mobilenet` — FedAvg MobileNetV2, explicitly with no DP.
- `fedprox_mobilenet` — FedProx MobileNetV2 (`--fedprox-mu` controls the proximal coefficient).
- `scaffold_mobilenet` — SCAFFOLD MobileNetV2 with per-client and server control variates.
- `resnet18`, `efficientnet_b0`, and `tinyvit` — alternative vision backbones under the same FedAvg schedule.

All federated baselines use the same client partition, validation criterion,
number of rounds, local epochs, optimizer settings, and held-out test set in a
single invocation. The runner selects the best validation-accuracy checkpoint,
then evaluates it once on test data. JSON output includes per-round history,
configuration, class names, client sample counts, parameter count, confusion
matrix, accuracy, balanced accuracy, macro/weighted F1, and binary AUC-ROC.
The CSV summary is ready to use in a paper table. Each selected baseline also
writes a CPU-portable `.pt` checkpoint beside its JSON file; these benchmark
artifacts are never placed in `saved_models/`. Output filenames begin with the
modality (for example, `fusion_fedavg_mobilenet_seed_42.json`) so runs cannot
overwrite results from a different modality.

## Data modes

`--modality facial` (default) expects the existing layout:

```text
<data-dir>/train/{autistic,non_autistic}/
<data-dir>/valid/{autistic,non_autistic}/
<data-dir>/test/{autistic,non_autistic}/
```

`--modality video` expects the SSBD layout with `armFlapping`, `headBanging`,
and `spinning` directories (flat or duplicated nested directories). It creates
deterministic stratified train/validation/test video splits. Image backbones
classify every sampled frame and average their logits, so none of the requested
baselines silently adds a TCN or other temporal model.

```powershell
python -m benchmarks --modality video --data-dir ..\SSBD-file\ssbd2 --non-iid --dirichlet-alpha 0.5 --baselines mobilenet_without_tcn,fedprox_mobilenet,scaffold_mobilenet
```

`--modality fusion` is the end-to-end multimodal benchmark. It accepts the
facial root in `--data-dir` and the SSBD root in `--video-data-dir`. It pairs
the two datasets by a deterministic client-local index order, uses the facial
`autistic`/`non_autistic` label as the target, and sends both tensors to every
baseline. Facial samples are sampled round-robin by class before pairing so a
smaller video set cannot produce a single-class fusion split.
This follows the existing fusion experiment's independent per-client modality
partitioning, pairing, and supervision rule; it does not claim subject-level
pairing.

In fusion mode, all baselines use a face encoder, a per-frame video encoder,
and a late-fusion classifier. `mobilenet_without_tcn` mean-pools video-frame
features; all remaining baselines use a compact temporal convolution stack.
FedAvg, FedProx, and SCAFFOLD describe the optimization method applied to the
entire fusion model. `resnet18`, `efficientnet_b0`, and `tinyvit` replace both
visual encoders while retaining the same fusion protocol.

On the facial-image dataset, `mobilenet_without_tcn` is necessarily equivalent
to `fedavg_mobilenet` at the architecture level because an image has no temporal
axis. The result JSON records this fact so it is not misrepresented as a
meaningful temporal ablation. Use video mode for the actual no-TCN comparison.

`--pretrained` is opt-in to avoid an unexpected network download; use it when
the proposed model is also evaluated with ImageNet initialization. For a fair
comparison, retain the same seed, partition mode, rounds, local epochs,
augmentation policy, initialization setting, and data split across every row.
`--max-samples-per-class` is only for a quick functional smoke test; leave it
at its default (`0`) for reportable benchmark results.
