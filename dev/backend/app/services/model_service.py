"""
Model inference service with a pluggable registry.

To add a new model:
1. Define its nn.Module class in architectures.py.
2. Write a build function and output extractor here.
3. Add one entry to MODEL_REGISTRY.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import torch
import torch.nn as nn

from .architectures import FairDetector, Xception


@dataclass
class PredictionResult:
	label: str
	confidence: float
	fake_probability: float
	threshold: float
	explanation: str
	model_name: str


@dataclass(frozen=True)
class ModelSpec:
	"""Everything InferenceService needs to build and run a model."""

	build_fn: Callable[[], nn.Module]
	image_size: int
	mean: tuple[float, ...]
	std: tuple[float, ...]
	description: str
	extract_prob: Callable[[Any], float]


def _build_vit() -> nn.Module:
	from torchvision import models

	model = models.vit_b_16(weights=None)
	model.heads[0] = nn.Linear(768, 1)
	return model


def _build_xception() -> nn.Module:
	return Xception(
		{
			"mode": "original",
			"num_classes": 1,
			"inc": 3,
			"dropout": False,
		}
	)


def _build_pg_fdd() -> nn.Module:
	return FairDetector()


def _extract_sigmoid_scalar(output: Any) -> float:
	"""ViT returns a (B, 1) logit tensor."""

	logits = output
	if logits.ndim > 1:
		logits = logits.squeeze(1)
	return torch.sigmoid(logits)[0].item()


def _extract_xception(output: Any) -> float:
	"""Xception.forward returns (logits, features), logits shape (B, 1)."""

	logits, _ = output
	if logits.ndim > 1:
		logits = logits.squeeze(1)
	return torch.sigmoid(logits)[0].item()


def _extract_pg_fdd(output: Any) -> float:
	"""FairDetector returns dict with key 'cls' containing fused logits."""

	logits = output["cls"]
	if logits.ndim > 1:
		logits = logits.squeeze(1)
	return torch.sigmoid(logits)[0].item()


MODEL_REGISTRY: dict[str, ModelSpec] = {
	"vit": ModelSpec(
		build_fn=_build_vit,
		image_size=224,
		mean=(0.485, 0.456, 0.406),
		std=(0.229, 0.224, 0.225),
		description="ViT-B/16 deepfake classifier",
		extract_prob=_extract_sigmoid_scalar,
	),
	"xception": ModelSpec(
		build_fn=_build_xception,
		image_size=256,
		mean=(0.5, 0.5, 0.5),
		std=(0.5, 0.5, 0.5),
		description="Xception deepfake classifier",
		extract_prob=_extract_xception,
	),
	"pg_fdd": ModelSpec(
		build_fn=_build_pg_fdd,
		image_size=256,
		mean=(0.5, 0.5, 0.5),
		std=(0.5, 0.5, 0.5),
		description="PG-FDD (Fair Deepfake Detector)",
		extract_prob=_extract_pg_fdd,
	),
}


def _resolve_spec(model_name: str) -> ModelSpec:
	"""Match a model_name to a registry entry using substring matching."""

	name_lower = model_name.lower()
	for key, spec in MODEL_REGISTRY.items():
		if key in name_lower:
			return spec
	supported = ", ".join(MODEL_REGISTRY.keys())
	raise ValueError(f"Unknown model '{model_name}'. Supported: {supported}")


class InferenceService:
	"""Generic inference service that works with any registered model."""

	def __init__(self, model_name: str, model_path: Path, threshold: float) -> None:
		self.model_name = model_name
		self.model_path = Path(model_path)
		self.threshold = threshold

		self._spec = _resolve_spec(model_name)
		self.image_size = self._spec.image_size
		self.mean = self._spec.mean
		self.std = self._spec.std

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self._model: nn.Module | None = None
		self._ready = False
		self._lock = Lock()

	@property
	def model_ready(self) -> bool:
		return self._ready

	def ensure_loaded(self) -> None:
		if self._ready and self._model is not None:
			return
		with self._lock:
			if self._ready and self._model is not None:
				return
			self._model = self._load_model()
			self._ready = True

	def _load_model(self) -> nn.Module:
		if not self.model_path.exists():
			raise FileNotFoundError(
				f"Checkpoint not found at '{self.model_path}'. "
				"Place the .pth file in models/pretrained/."
			)

		model = self._spec.build_fn()

		checkpoint = torch.load(self.model_path, map_location=self.device)
		if isinstance(checkpoint, dict):
			for wrapper_key in ("state_dict", "model_state_dict"):
				if wrapper_key in checkpoint and isinstance(checkpoint[wrapper_key], dict):
					checkpoint = checkpoint[wrapper_key]
					break

		model_keys = set(model.state_dict().keys())
		ckpt_keys = set(checkpoint.keys())
		if not model_keys & ckpt_keys and ckpt_keys:
			prefix = next(iter(ckpt_keys)).split(".")[0] + "."
			stripped = {
				k[len(prefix):]: v
				for k, v in checkpoint.items()
				if k.startswith(prefix)
			}
			if set(stripped.keys()) & model_keys:
				checkpoint = stripped

		model.load_state_dict(checkpoint, strict=False)
		model.to(self.device)
		model.eval()
		return model

	def predict(self, image_tensor: torch.Tensor) -> PredictionResult:
		if self._model is None:
			raise RuntimeError("Model is not loaded.")

		with torch.no_grad():
			if isinstance(self._model, FairDetector):
				output = self._model({"image": image_tensor}, inference=True)
			else:
				output = self._model(image_tensor)

			fake_probability = self._spec.extract_prob(output)

		is_fake = fake_probability >= self.threshold
		label = "FAKE" if is_fake else "REAL"
		confidence = fake_probability if is_fake else 1.0 - fake_probability

		return PredictionResult(
			label=label,
			confidence=round(confidence, 4),
			fake_probability=round(fake_probability, 4),
			threshold=self.threshold,
			explanation=(
				f"Prediction is based on {self._spec.description}. "
				"Explainability heatmaps are planned in the next milestone."
			),
			model_name=self.model_name,
		)


def model_service_from_settings(settings: Any) -> InferenceService:
	return InferenceService(
		model_name=settings.model_name,
		model_path=settings.model_path,
		threshold=settings.fake_threshold,
	)
