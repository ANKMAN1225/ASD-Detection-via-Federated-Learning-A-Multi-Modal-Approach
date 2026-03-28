"""
Flower (flwr) smoke test: 1 round, 2 clients.
- Facial dataset (real images)
- Video dataset (SSBD)

Run from asd-federated-learning/:
    python flwr_smoke_test.py
"""

import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import Config
from data import FacialDataset, BehavioralVideoDataset
from federated import create_federated_data_splits
from models import MobileNetFeatureExtractor, VideoTCNModel
from experiments.flwr_simulation import run_flwr_simulation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Force Ray to use CPU for clients if we don't have enough GPUs for parallel workers
CLIENT_DEVICE = "cpu" 
NUM_CLIENTS = 2
BATCH_SIZE  = 4
SEQ_LEN     = 4

FACIAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "autistic-children-facial-data-set"))
VIDEO_PATH  = Config().video_data_path

print("=" * 60)
print("FLOWER (flwr) FL SMOKE TEST — 1 round, 2 clients")
print("=" * 60)
print(f"Server Device: {DEVICE}")
print(f"Client Device: {CLIENT_DEVICE}")
print()

# ── Facial dataset ─────────────────────────────────────────────────────────
print("── FACIAL DATASET FLOWER TEST ───────────────────────────────")
tf_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_facial = FacialDataset(FACIAL_PATH, "train", transform=tf_eval)
valid_facial = FacialDataset(FACIAL_PATH, "valid", transform=tf_eval)

client_train = create_federated_data_splits(train_facial, NUM_CLIENTS, iid=True)
client_valid = create_federated_data_splits(valid_facial, NUM_CLIENTS, iid=True)

train_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_train]
valid_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) for ds in client_valid]

global_model = MobileNetFeatureExtractor(num_classes=2, pretrained=True).to(DEVICE)

try:
    history = run_flwr_simulation(
        global_model=global_model,
        client_train_loaders=train_loaders,
        client_val_loaders=valid_loaders,
        num_clients=NUM_CLIENTS,
        num_rounds=1,
        local_epochs=1,
        learning_rate=0.001,
        device=CLIENT_DEVICE,
        experiment_name="facial_smoke_test"
    )
    print("Facial Flower Simulation SUCCESS ✓")
except Exception as e:
    print(f"Facial Flower Simulation FAILED: {e}")

print()

# ── Video dataset ──────────────────────────────────────────────────────────
print("── VIDEO DATASET FLOWER TEST ────────────────────────────────")
video_ds = BehavioralVideoDataset(VIDEO_PATH, sequence_length=SEQ_LEN)

if len(video_ds) == 0:
    print("  ⚠ SSBD-file has no video files yet.")
else:
    from torch.utils.data import random_split
    n_train = max(int(0.8 * len(video_ds)), 1)
    n_test  = len(video_ds) - n_train
    tr_ds, te_ds = random_split(video_ds, [n_train, n_test], generator=torch.Generator().manual_seed(42))

    c_train = create_federated_data_splits(tr_ds, NUM_CLIENTS, iid=True)
    c_test  = create_federated_data_splits(te_ds, NUM_CLIENTS, iid=True)

    v_train_loaders = [DataLoader(ds, batch_size=2, shuffle=True) for ds in c_train]
    v_test_loaders  = [DataLoader(ds, batch_size=2, shuffle=False) for ds in c_test]

    vid_global = VideoTCNModel(num_classes=3, sequence_length=SEQ_LEN).to(DEVICE)
    
    try:
        vid_history = run_flwr_simulation(
            global_model=vid_global,
            client_train_loaders=v_train_loaders,
            client_val_loaders=v_test_loaders,
            num_clients=NUM_CLIENTS,
            num_rounds=1,
            local_epochs=1,
            learning_rate=0.001,
            device=CLIENT_DEVICE,
            experiment_name="video_smoke_test"
        )
        print("Video Flower Simulation SUCCESS ✓")
    except Exception as e:
        print(f"Video Flower Simulation FAILED: {e}")

print()
print("=" * 60)
print("FLOWER SMOKE TEST COMPLETE ✓")
print("=" * 60)
