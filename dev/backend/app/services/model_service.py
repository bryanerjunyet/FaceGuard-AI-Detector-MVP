"""
Model inference and Grad-CAM explainability service with a pluggable registry.

To add a new model:
  1. Define its nn.Module class in architectures.py.
  2. Write a build function and an output-extractor function here.
  3. Add one entry to MODEL_REGISTRY.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from .architectures import FairDetector, Xception

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:  # pragma: no cover
    GradCAM = None  # type: ignore[assignment]
    show_cam_on_image = None  # type: ignore[assignment]


@dataclass
class PredictionResult:
    label: str
    confidence: float
    fake_probability: float
    threshold: float
    explanation: str
    model_name: str
    heatmap_overlay: str | None
    explainability_method: str


@dataclass(frozen=True)
class ModelSpec:
    """Everything the InferenceService needs to build and run a model."""

    build_fn: Callable[[], nn.Module]
    image_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    description: str
    extract_prob: Callable[[Any], float]


class _BinaryLogitTarget:
    """
    Target fake evidence when predicted FAKE, otherwise target real evidence.
    """

    def __init__(self, label: str) -> None:
        self._sign = 1.0 if label == "FAKE" else -1.0

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        logits = model_output
        if logits.ndim > 1:
            logits = logits[:, 0]
        return logits * self._sign


class _CamModelAdapter(nn.Module):
    """
    Adapts model outputs into a simple logits tensor expected by Grad-CAM.
    """

    def __init__(self, model: nn.Module, model_name: str) -> None:
        super().__init__()
        self.model = model
        self.model_name = model_name.lower()

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if "pg_fdd" in self.model_name and isinstance(self.model, FairDetector):
            output = self.model({"image": image_tensor}, inference=True)["cls"]
            return output

        output = self.model(image_tensor)
        if isinstance(output, tuple):
            output = output[0]
        elif isinstance(output, dict):
            if "cls" in output:
                output = output["cls"]
            elif "logits" in output:
                output = output["logits"]

        if not isinstance(output, torch.Tensor):
            raise TypeError("Grad-CAM adapter expected tensor logits output.")
        return output


def _build_vit() -> nn.Module:
    from torchvision import models

    model = models.vit_b_16(weights=None)
    model.heads[0] = nn.Linear(768, 1)
    return model


def _build_xception() -> nn.Module:
    return Xception({"mode": "original", "num_classes": 1, "inc": 3, "dropout": False})


def _build_pg_fdd() -> nn.Module:
    return FairDetector()


def _extract_sigmoid_scalar(output: Any) -> float:
    logits = output
    if logits.ndim > 1:
        logits = logits.squeeze(1)
    return torch.sigmoid(logits)[0].item()


def _extract_xception(output: Any) -> float:
    logits, _ = output
    if logits.ndim > 1:
        logits = logits.squeeze(1)
    return torch.sigmoid(logits)[0].item()


def _extract_pg_fdd(output: Any) -> float:
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
    name_lower = model_name.lower()
    for key, spec in MODEL_REGISTRY.items():
        if key in name_lower:
            return spec
    supported = ", ".join(MODEL_REGISTRY.keys())
    raise ValueError(f"Unknown model '{model_name}'. Supported: {supported}")