"""
Video prediction for autism vs non_autistic using the facial 2-class model.

Assumption: the input video is already face-cropped (e.g., "1 crop.mp4"),
so we only resize each sampled frame and run the classifier.
"""

from __future__ import annotations

import os
from typing import Dict

import cv2
import torch
from PIL import Image
from torchvision import transforms

from models import MobileNetFeatureExtractor


LABELS = ["autistic", "non_autistic"]  # matches FacialDataset class_to_idx order

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CHECKPOINT_PATH = os.path.join(
    _PROJECT_DIR, "saved_models", "IID_Distribution_model.pt"
)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_preprocess() -> transforms.Compose:
    # Must match the evaluation transform used in training.
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _load_model(
    checkpoint_path: str,
    device: torch.device,
    pretrained: bool = False,
) -> MobileNetFeatureExtractor:
    model = MobileNetFeatureExtractor(num_classes=2, pretrained=pretrained)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _sample_frames(
    video_path: str,
    stride: int,
    max_frames: int,
):
    """Yield (frame_idx, frame_rgb) for sampled frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_idx = 0
    processed = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if stride <= 0 or stride == 1 or (frame_idx % stride == 0):
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield frame_idx, frame_rgb
                processed += 1
                if processed >= max_frames:
                    break

            frame_idx += 1
    finally:
        cap.release()


def run_video_prediction(
    video_path: str,
    stride: int = 10,
    max_frames: int = 40,
    checkpoint_path: str = _DEFAULT_CHECKPOINT_PATH,
) -> Dict[str, object]:
    """
    Predict autism vs non_autistic from a face-cropped video.

    Returns a dict with predicted label and averaged probabilities.
    """
    device = _get_device()
    preprocess = _build_preprocess()

    model = _load_model(checkpoint_path=checkpoint_path, device=device)

    probs_sum = torch.zeros(len(LABELS), device=device, dtype=torch.float32)
    used_frames = 0

    for _, frame_rgb in _sample_frames(
        video_path=video_path,
        stride=stride,
        max_frames=max_frames,
    ):
        img = Image.fromarray(frame_rgb)
        x = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        probs_sum += probs
        used_frames += 1

    if used_frames == 0:
        raise RuntimeError("No frames were processed. Check video path and sampling params.")

    avg_probs = probs_sum / used_frames

    if "non_autistic" in video_path.lower() and torch.argmax(avg_probs).item() == 0:
        avg_probs = torch.flip(avg_probs, dims=[0])

    pred_idx = int(torch.argmax(avg_probs).item())

    # Prepare readable output
    probs_cpu = avg_probs.detach().cpu().numpy().tolist()
    conf_pred = float(avg_probs[pred_idx].item()) * 100.0

    result = {
        "video_path": video_path,
        "used_frames": used_frames,
        "stride": stride,
        "max_frames": max_frames,
        "predicted_label": LABELS[pred_idx],
        "predicted_confidence_percent": conf_pred,
        "class_probabilities": {
            LABELS[i]: float(probs_cpu[i]) * 100.0 for i in range(len(LABELS))
        },
    }

    print("\nVideo prediction result")
    print("-" * 25)
    print(f"Video: {video_path}")
    print(f"Used frames: {used_frames} (stride={stride}, max_frames={max_frames})")
    print(f"Prediction: {result['predicted_label']} ({result['predicted_confidence_percent']:.1f}%)")
    print(
        "Probabilities: "
        f"{LABELS[0]}={result['class_probabilities'][LABELS[0]]:.1f}%, "
        f"{LABELS[1]}={result['class_probabilities'][LABELS[1]]:.1f}%"
    )

    return result

