"""Local model cache for the two ONNX files the face pipeline needs.

Both come from the OpenCV Zoo and are loaded through OpenCV's own
``FaceDetectorYN`` / ``FaceRecognizerSF`` wrappers, so the recognition path
adds no runtime beyond the OpenCV already required. They are downloaded on
demand rather than vendored, and once cached the whole pipeline runs offline.

Downloads are verified by size and cached under ``~/.casework/models``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import SETTINGS, require_network

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    approx_bytes: int
    purpose: str


DETECTOR = ModelSpec(
    name="yunet",
    filename="face_detection_yunet_2023mar.onnx",
    url=f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    approx_bytes=232_000,
    purpose="Face detection with five-point landmarks (needed for alignment)",
)

RECOGNISER = ModelSpec(
    name="sface",
    filename="face_recognition_sface_2021dec.onnx",
    url=f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    approx_bytes=38_900_000,
    purpose="128-dimensional face embeddings for clustering and matching",
)

ALL_MODELS = (DETECTOR, RECOGNISER)


def model_path(spec: ModelSpec) -> Path:
    return SETTINGS.model_cache / spec.filename


def is_cached(spec: ModelSpec) -> bool:
    path = model_path(spec)
    # A truncated download decodes as a corrupt ONNX with an unhelpful error,
    # so treat anything far below the expected size as absent.
    return path.exists() and path.stat().st_size > spec.approx_bytes * 0.5


def missing() -> list[ModelSpec]:
    return [m for m in ALL_MODELS if not is_cached(m)]


def fetch(spec: ModelSpec, progress=None) -> Path:
    path = model_path(spec)
    if is_cached(spec):
        return path
    require_network()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")

    with httpx.stream(
        "GET", spec.url, follow_redirects=True, timeout=300.0,
        headers={"User-Agent": SETTINGS.user_agent},
    ) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or spec.approx_bytes)
        done = 0
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1 << 16):
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(spec.name, done, total)
    tmp.replace(path)
    return path


def ensure_models(progress=None) -> list[Path]:
    return [fetch(spec, progress) for spec in ALL_MODELS]


def status() -> list[dict[str, object]]:
    return [
        {
            "name": m.name,
            "filename": m.filename,
            "purpose": m.purpose,
            "cached": is_cached(m),
            "size": model_path(m).stat().st_size if model_path(m).exists() else 0,
        }
        for m in ALL_MODELS
    ]
