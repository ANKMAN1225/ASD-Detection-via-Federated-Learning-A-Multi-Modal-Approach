"""
Smoke test: 1 round, 1 epoch, 2 clients.
- Facial dataset (real images from autistic-children-facial-data-set)
- Video dataset (SSBD) — will warn if no videos found

Run from asd-federated-learning/:
    python smoke_test.py
"""

import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import Config
from data import FacialDataset, BehavioralVideoDataset
from federated import FederatedClient, FederatedServer, create_federated_data_splits
from models import MobileNetFeatureExtractor, VideoTCNModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 2
BATCH_SIZE  = 4
SEQ_LEN     = 4   # short for speed

FACIAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "autistic-children-facial-data-set"))
VIDEO_PATH  = Config().video_data_path

print("=" * 60)
print("ASD FL SMOKE TEST — 1 round, 1 epoch")
print("=" * 60)
print(f"Device   : {DEVICE}")
print(f"Facial   : {FACIAL_PATH}")
print(f"Video    : {VIDEO_PATH}")
print()

# ── Facial dataset ─────────────────────────────────────────────────────────
print("── FACIAL DATASET TEST ──────────────────────────────────────")
assert os.path.exists(FACIAL_PATH), f"MISSING: {FACIAL_PATH}"

tf_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
tf_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_facial = FacialDataset(FACIAL_PATH, "train", transform=tf_train)
test_facial  = FacialDataset(FACIAL_PATH, "test",  transform=tf_eval)
valid_facial = FacialDataset(FACIAL_PATH, "valid", transform=tf_eval)

assert len(train_facial) > 0, "Train facial split is EMPTY — check folder structure."
assert len(test_facial)  > 0, "Test facial split is EMPTY."
assert len(valid_facial) > 0, "Valid facial split is EMPTY."
print(f"  train={len(train_facial)}, valid={len(valid_facial)}, test={len(test_facial)} samples ✓")

# Federated split
client_train = create_federated_data_splits(train_facial, NUM_CLIENTS, iid=True)
client_test  = create_federated_data_splits(test_facial,  NUM_CLIENTS, iid=True)
client_valid = create_federated_data_splits(valid_facial, NUM_CLIENTS, iid=True)

# Build server + clients
global_model = MobileNetFeatureExtractor(num_classes=2, pretrained=True).to(DEVICE)
clients = []
for i in range(NUM_CLIENTS):
    cm = MobileNetFeatureExtractor(num_classes=2, pretrained=True)
    tr = DataLoader(client_train[i], batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    te = DataLoader(client_test[i],  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    va = DataLoader(client_valid[i], batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    clients.append(FederatedClient(client_id=i, model=cm, train_loader=tr, test_loader=te,
                                   val_loader=va, device=DEVICE))

server = FederatedServer(global_model, clients)

print("  Running 1 FL round (1 local epoch) on facial data ...")
rnd = server.train_round(local_epochs=1, lr=0.001)
print(f"  Round loss: {rnd['avg_loss']:.4f}  samples: {rnd['total_samples']} ✓")

eval_m = server.evaluate_global_model(split="test")
print(f"  Global test accuracy: {eval_m['global_accuracy']:.2f}% ✓")

print()

# ── Video dataset ──────────────────────────────────────────────────────────
print("── VIDEO DATASET TEST ───────────────────────────────────────")
video_ds = BehavioralVideoDataset(VIDEO_PATH, sequence_length=SEQ_LEN)

if len(video_ds) == 0:
    print("  ⚠ SSBD-file has no video files yet.")
    print(f"    Download them by running: python {os.path.join(VIDEO_PATH, 'download_ssbd.py')}")
    print("    (requires yt-dlp and moviepy; many clips may be unavailable on YouTube)")
    print("  Skipping video FL smoke test.")
else:
    print(f"  Found {len(video_ds)} SSBD video clips ✓")
    from torch.utils.data import random_split
    n_train = max(int(0.8 * len(video_ds)), 1)
    n_test  = len(video_ds) - n_train
    tr_ds, te_ds = random_split(video_ds, [n_train, n_test],
                                 generator=torch.Generator().manual_seed(42))

    c_train = create_federated_data_splits(tr_ds, NUM_CLIENTS, iid=True)
    c_test  = create_federated_data_splits(te_ds, NUM_CLIENTS, iid=True)

    vid_global = VideoTCNModel(num_classes=3, sequence_length=SEQ_LEN).to(DEVICE)
    vid_clients = []
    for i in range(NUM_CLIENTS):
        vm = VideoTCNModel(num_classes=3, sequence_length=SEQ_LEN)
        tr = DataLoader(c_train[i], batch_size=2, shuffle=True,  drop_last=False)
        te = DataLoader(c_test[i],  batch_size=2, shuffle=False, drop_last=False)
        vid_clients.append(FederatedClient(client_id=i, model=vm, train_loader=tr,
                                           test_loader=te, device=DEVICE))

    vid_server = FederatedServer(vid_global, vid_clients)
    print("  Running 1 FL round (1 local epoch) on video data ...")
    vrnd = vid_server.train_round(local_epochs=1, lr=0.001)
    print(f"  Round loss: {vrnd['avg_loss']:.4f}  samples: {vrnd['total_samples']} ✓")
    veval = vid_server.evaluate_global_model()
    print(f"  Global test accuracy: {veval['global_accuracy']:.2f}% ✓")

print()
print("=" * 60)
print("SMOKE TEST COMPLETE — no dummy values, real data loaded ✓")
print("=" * 60)
