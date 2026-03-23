# ASD Federated Learning - Video Prediction

## Predict autism vs non_autistic from a face-cropped video

Run:

```bash
python main.py --predict
```

You will be prompted to enter the path to a face-cropped video (for example: `1 crop.mp4`).

Optional arguments:

- `--predict_path "path/to/video.mp4"` (skip the prompt)
- `--predict_stride 10` (process 1 frame every N frames)
- `--predict_max_frames 40` (max number of sampled frames)

