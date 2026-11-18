"""Detection, alignment and embedding, wrapped around OpenCV's ONNX models.

YuNet detects and returns five landmarks; SFace turns an aligned 112x112 crop
into a 128-d embedding. Using OpenCV's own ``FaceRecognizerSF.alignCrop``
rather than a hand-rolled similarity transform matters more than it sounds:
SFace was trained on crops produced by exactly that alignment, and feeding it
a plain bounding-box crop costs far more accuracy than the convenience saves.

Two practical traps are handled here:

**Detection scale.** YuNet's input size is set explicitly, and running it at
the native size of a 6000px photograph is slow and *less* accurate than
running it at ~1024px, because the model's anchors are tuned for that range.
Detection therefore runs on a downscaled copy and the coordinates are scaled
back to the original, so crops are still taken at full resolution.

**Tiny faces.** A 30px face yields an embedding that is confidently wrong
rather than uselessly vague, which is the failure mode that produces false
identifications. Faces below ``det_min_pixels`` are dropped at detection
time, not filtered later.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ..config import SETTINGS
from ..modelmgr import DETECTOR, RECOGNISER, is_cached, model_path

DETECT_MAX_DIM = 1024


class ModelsUnavailable(RuntimeError):
    """The ONNX models are not cached yet."""


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    score: float
    raw: np.ndarray  # the 15-vector YuNet row, in original-image coordinates


def load_image(path: Path) -> np.ndarray | None:
    """Read an image as BGR, falling back to Pillow for formats OpenCV lacks.

    HEIC and some progressive JPEGs return None from ``cv2.imread`` on a
    stock build; silently treating those as "no faces" would quietly drop
    whole phone-camera collections.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is not None:
        return bgr
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _detector() -> cv2.FaceDetectorYN:
    if not is_cached(DETECTOR):
        raise ModelsUnavailable(
            "Face detection model is not cached. Run `casework models fetch`."
        )
    return cv2.FaceDetectorYN.create(
        str(model_path(DETECTOR)), "", (320, 320),
        score_threshold=SETTINGS.det_min_score, nms_threshold=0.3, top_k=5000,
    )


@lru_cache(maxsize=1)
def _recogniser() -> cv2.FaceRecognizerSF:
    if not is_cached(RECOGNISER):
        raise ModelsUnavailable(
            "Face recognition model is not cached. Run `casework models fetch`."
        )
    return cv2.FaceRecognizerSF.create(str(model_path(RECOGNISER)), "")


def detect(bgr: np.ndarray) -> list[Detection]:
    height, width = bgr.shape[:2]
    scale = min(1.0, DETECT_MAX_DIM / max(height, width))
    small = (
        cv2.resize(bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else bgr
    )

    det = _detector()
    det.setInputSize((small.shape[1], small.shape[0]))
    _retval, faces = det.detect(small)
    if faces is None:
        return []

    out: list[Detection] = []
    for row in faces:
        original = row.astype(np.float32).copy()
        if scale < 1.0:
            # Columns 0..13 are box + five landmark xy pairs; column 14 is
            # the score and must not be rescaled with them.
            original[:14] = original[:14] / scale
        x, y, w, h = (round(float(v)) for v in original[:4])
        if min(w, h) < SETTINGS.det_min_pixels:
            continue
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        out.append(Detection(box=(x, y, w, h), score=float(original[14]), raw=original))
    return out


def embed(bgr: np.ndarray, detection: Detection) -> np.ndarray:
    """L2-normalised 128-d embedding for one detected face.

    Normalising here means cosine similarity downstream is a plain dot
    product, and that clustering distances are comparable across cases.
    """
    rec = _recogniser()
    aligned = rec.alignCrop(bgr, detection.raw.reshape(1, -1))
    feature = rec.feature(aligned).flatten().astype(np.float32)
    norm = float(np.linalg.norm(feature))
    return feature / norm if norm > 0 else feature


def embed_file(path: Path) -> list[tuple[Detection, np.ndarray]]:
    bgr = load_image(path)
    if bgr is None:
        return []
    return [(d, embed(bgr, d)) for d in detect(bgr)]


def crop(bgr: np.ndarray, box: tuple[int, int, int, int], pad: float = 0.35) -> np.ndarray:
    """Padded crop for review thumbnails.

    Padded because an analyst adjudicating a match needs hairline, ears and
    jaw -- a tight recognition crop discards most of what a human uses.
    """
    height, width = bgr.shape[:2]
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(width, x + w + px), min(height, y + h + py)
    return bgr[y0:y1, x0:x1]


def models_ready() -> bool:
    return is_cached(DETECTOR) and is_cached(RECOGNISER)
