# Deepfake Video Sampler

This repository provides a simple script that samples frames from a video to decide whether the content is `real` or `fake`. It uses the [`prithivMLmods/deepfake-detector-model-v1`](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1) model hosted on Hugging Face.

## Setup

1. Ensure you have Python 3.9 or newer.
2. Install the dependencies:

```powershell
pip install -r requirements.txt
```

> **Note:** Installing `torch` may require selecting the correct wheel for your system. See [pytorch.org](https://pytorch.org/get-started/locally/) if the default installation fails. Video support requires `opencv-python`, which is already listed in `requirements.txt`.

## Configure

Open `Untitled-1.py` and edit the configuration block near the top:


```python
MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
VIDEO_PATH = r"C:\\path\\to\\video.mp4"
FRAME_STEP = 50
MAX_FRAMES: int | None = None
FRAME_PREVIEW_LIMIT = 5
DEVICE_CHOICE = "auto"
```

- **VIDEO_PATH** — set this to the video you want to analyze.
- **FRAME_STEP** — sample every Nth frame (smaller values mean more frames, more time, better accuracy).
- **MAX_FRAMES** — optionally cap the total number of frames to process.
- **DEVICE_CHOICE** — pick `"cpu"`, `"cuda"`, or keep `"auto"` to let the script decide.

## Run

```powershell
python Untitled-1.py
```

The script downloads the model on first run (internet connection required), processes the configured video, and prints averaged probabilities along with a small table of sampled frames.

## Output

Typical output:

```
Model: prithivMLmods/deepfake-detector-model-v1
Device: cuda
Input video: C:\path\to\clip.mp4
Frames sampled: 18 (every 50 frame)

Average predictions:
   fake: 0.732
   real: 0.268

Sample frame scores:
  frame     0: fake=0.651, real=0.349
  frame    50: fake=0.744, real=0.256
  frame   100: fake=0.802, real=0.198
```

## Troubleshooting

- **Slow or failing downloads** &mdash; the first run downloads model weights from Hugging Face; ensure you have an internet connection.
- **CUDA errors** &mdash; select `--device cpu` if your machine does not have a compatible NVIDIA GPU.
- **File not found** &mdash; verify the path to the image is correct and accessible.
- **Missing `opencv-python`** &mdash; install it with `pip install opencv-python`.
- **Video issues** &mdash; confirm the path is correct and encoded in a format OpenCV can decode (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.mpg`, `.mpeg`, `.m4v`).
