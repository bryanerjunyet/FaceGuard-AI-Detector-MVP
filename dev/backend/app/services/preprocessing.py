from __future__ import annotations

from io import BytesIO
from typing import Iterable

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps
import torch
from torchvision import transforms


def validate_upload(upload: UploadFile, allowed_mime_types: Iterable[str], max_upload_bytes: int) -> bytes:
    if upload.content_type not in allowed_mime_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {upload.content_type}",
        )

    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(raw) > max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds max allowed size of {max_upload_bytes // (1024 * 1024)} MB.",
        )

    return raw


def strip_exif_and_load_image(raw: bytes) -> Image.Image:
    # Apply EXIF orientation before stripping metadata to keep visual orientation stable.
    source = Image.open(BytesIO(raw))
    source = ImageOps.exif_transpose(source).convert("RGB")
    cleaned_buffer = BytesIO()
    source.save(cleaned_buffer, format="PNG")
    cleaned_buffer.seek(0)
    return Image.open(cleaned_buffer).convert("RGB")


def build_inference_transform(
    image_size: int,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def image_to_tensor(
    image: Image.Image,
    image_size: int,
    device: torch.device,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    transform = build_inference_transform(image_size=image_size, mean=mean, std=std)
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor