"""Video deepfake classification by sampling frames and averaging predictions."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

try:
	import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency when handling video
	cv2 = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration block – edit these values to suit your environment.
# ---------------------------------------------------------------------------

MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
VIDEO_PATH = r"C:\Users\Kshit\Downloads\ai-processing-chip-integrated-into-a-high-tech-circuit-board-the-chip-glows-with-blue-neon-lines-that-indicate-data-transfer-surrounded-by-a-modern-black-technological-environment-video.jpg"
FRAME_STEP = 20  # Process every Nth frame from the video.
MAX_FRAMES: int | None = None  # Set to an int to cap processed frames, or None for all.
FRAME_PREVIEW_LIMIT = 5  # Number of sampled frame scores to display in the summary.
DEVICE_CHOICE = "auto"  # One of {"auto", "cpu", "cuda"}


# ---------------------------------------------------------------------------
# Model metadata and cached state
# ---------------------------------------------------------------------------

ID2LABEL: Dict[str, str] = {
	"0": "fake",
	"1": "real",
}
LABEL_IDS: Tuple[str, ...] = tuple(sorted(ID2LABEL.keys(), key=int))

MODEL: SiglipForImageClassification | None = None
PROCESSOR: AutoImageProcessor | None = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_resources() -> Tuple[SiglipForImageClassification, AutoImageProcessor]:
	"""Load the model and processor once, caching them for reuse."""

	global MODEL, PROCESSOR

	if MODEL is None or PROCESSOR is None:
		model_source = MODEL_NAME
		if os.path.isfile(model_source):
			model_dir = os.path.dirname(model_source) or "."
			model_source = model_dir
		MODEL = SiglipForImageClassification.from_pretrained(model_source)
		PROCESSOR = AutoImageProcessor.from_pretrained(model_source)
		MODEL.to(DEVICE)
		MODEL.eval()

	return MODEL, PROCESSOR


def _prepare_inputs(image: Image.Image) -> Dict[str, torch.Tensor]:
	"""Convert an ``Image`` into tensors suitable for the model."""

	_, processor = _load_resources()
	inputs = processor(images=image, return_tensors="pt")
	return {name: tensor.to(DEVICE) for name, tensor in inputs.items()}


def _predict(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
	"""Run a forward pass and return probabilities."""

	model, _ = _load_resources()
	with torch.no_grad():
		logits = model(**inputs).logits
		probs = torch.nn.functional.softmax(logits, dim=1)
	return probs.squeeze(0)


def _image_to_probabilities(image: Image.Image) -> List[float]:
	probabilities = _predict(_prepare_inputs(image.convert("RGB"))).tolist()
	if len(probabilities) != len(LABEL_IDS):
		raise ValueError(
			"Unexpected number of logits returned by the model. "
			"Ensure the label mapping matches the model outputs."
		)
	return probabilities


def _format_probabilities(probabilities: List[float]) -> Dict[str, float]:
	return {
		ID2LABEL[label_id]: round(probability, 6)
		for label_id, probability in zip(LABEL_IDS, probabilities)
	}


def classify_video_path(
	video_path: str,
	*,
	frame_step: int,
	max_frames: int | None,
) -> Tuple[Dict[str, float], int, List[Tuple[int, Dict[str, float]]]]:
	"""Classify a video by sampling frames and averaging probabilities."""

	if frame_step < 1:
		raise ValueError("frame_step must be at least 1.")

	if cv2 is None:  # pragma: no cover - handled at runtime
		raise ImportError(
			"opencv-python is required for video classification. Install it via 'pip install opencv-python'."
		)

	if max_frames is not None and max_frames < 1:
		raise ValueError("max_frames must be at least 1 when provided.")

	if not os.path.isfile(video_path):
		raise FileNotFoundError(f"Video not found: {video_path}")

	capture = cv2.VideoCapture(video_path)
	if not capture.isOpened():  # pragma: no cover - depends on runtime files
		raise RuntimeError(f"Unable to open video file: {video_path}")

	accumulated = [0.0 for _ in LABEL_IDS]
	frame_predictions: List[Tuple[int, Dict[str, float]]] = []
	processed_frames = 0
	frame_index = 0

	try:
		while True:
			grabbed, frame = capture.read()
			if not grabbed:
				break

			if frame_index % frame_step == 0:
				rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				image = Image.fromarray(rgb_frame)
				probabilities = _image_to_probabilities(image)
				for index, value in enumerate(probabilities):
					accumulated[index] += value
				frame_predictions.append((frame_index, _format_probabilities(probabilities)))
				processed_frames += 1

				if max_frames is not None and processed_frames >= max_frames:
					break

			frame_index += 1
	finally:
		capture.release()

	if processed_frames == 0:
		raise ValueError(
			"No frames were sampled from the video. Adjust FRAME_STEP or verify the video file."
		)

	averaged = [value / processed_frames for value in accumulated]
	return _format_probabilities(averaged), processed_frames, frame_predictions


def _sorted_predictions(predictions: Dict[str, float]) -> Iterable[Tuple[str, float]]:
	return sorted(predictions.items(), key=lambda item: item[1], reverse=True)


def _resolve_device(choice: str) -> torch.device:
	if choice == "cpu":
		return torch.device("cpu")
	if choice == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA was requested but is not available on this machine.")
		return torch.device("cuda")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
	if cv2 is None:
		raise ImportError(
			"opencv-python is required but not installed. Install it via 'pip install opencv-python'."
		)

	if not VIDEO_PATH:
		raise ValueError("VIDEO_PATH is empty. Edit the script to point to a video file.")

	if not os.path.isfile(VIDEO_PATH):
		raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

	global DEVICE
	DEVICE = _resolve_device(DEVICE_CHOICE)

	print(f"Model: {MODEL_NAME}")
	print(f"Device: {DEVICE}")
	print(f"Video: {os.path.abspath(VIDEO_PATH)}")
	print(f"Frame step: {FRAME_STEP}")
	if MAX_FRAMES is not None:
		print(f"Max frames: {MAX_FRAMES}")

	average_predictions, frames_processed, frame_predictions = classify_video_path(
		VIDEO_PATH,
		frame_step=FRAME_STEP,
		max_frames=MAX_FRAMES,
	)

	top_predictions = list(_sorted_predictions(average_predictions))

	print(f"\nFrames sampled: {frames_processed}")
	print("Average predictions:")
	for label, probability in top_predictions:
		print(f"  {label:>4}: {probability:.3f}")

	preview_count = min(FRAME_PREVIEW_LIMIT, len(frame_predictions))
	if preview_count:
		print("\nSample frame scores:")
		for frame_index, scores in frame_predictions[:preview_count]:
			fake_score = scores[ID2LABEL[LABEL_IDS[0]]]
			real_score = scores[ID2LABEL[LABEL_IDS[1]]]
			print(
				f"  frame {frame_index:>5}: "
				f"fake={fake_score:.3f}, real={real_score:.3f}"
			)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
