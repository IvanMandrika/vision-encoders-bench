import io
from collections.abc import Callable

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

SEVERITIES = (1, 2, 3, 4, 5)

PARAMS: dict[str, dict[str, tuple[float, ...]]] = {
    "gaussian_noise": {"sigma_of_255": (6.0, 12.0, 18.0, 26.0, 36.0)},
    "gaussian_blur": {"sigma_px": (0.5, 0.75, 1.0, 1.4, 2.0)},
    "jpeg": {"quality": (75, 50, 30, 18, 10)},
    "darken": {"brightness_factor": (0.8, 0.6, 0.45, 0.32, 0.2)},
    "contrast": {"contrast_factor": (0.7, 0.5, 0.35, 0.22, 0.12)},
    "pixelate": {"side_px": (24, 20, 16, 12, 8)},
}

Corruption = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


def _param(name: str, severity: int) -> float:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be in 1..5, got {severity}")
    (values,) = PARAMS[name].values()
    return values[severity - 1]


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def gaussian_noise(images: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    sigma = _param("gaussian_noise", severity)
    return _to_uint8(images + rng.normal(0.0, sigma, size=images.shape))


def gaussian_blur(images: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    sigma = _param("gaussian_blur", severity)
    return _to_uint8(gaussian_filter(images.astype(np.float32), sigma=(0, sigma, sigma, 0), mode="nearest"))


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return np.asarray(Image.open(buffer).convert("RGB"))


def jpeg(images: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    quality = int(_param("jpeg", severity))
    return np.stack([_jpeg_roundtrip(image, quality) for image in images])


def darken(images: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    return _to_uint8(images * _param("darken", severity))


def contrast(images: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    factor = _param("contrast", severity)
    mean = images.mean(axis=(1, 2, 3), keepdims=True)
    return _to_uint8(mean + (images - mean) * factor)


def _pixelate_one(image: np.ndarray, side: int) -> np.ndarray:
    height, width = image.shape[:2]
    small = Image.fromarray(image).resize((side, side), Image.Resampling.BOX)
    return np.asarray(small.resize((width, height), Image.Resampling.NEAREST))


def pixelate(images: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    side = int(_param("pixelate", severity))
    return np.stack([_pixelate_one(image, side) for image in images])


CORRUPTIONS: dict[str, Corruption] = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "jpeg": jpeg,
    "darken": darken,
    "contrast": contrast,
    "pixelate": pixelate,
}


def corrupt(images: np.ndarray, name: str, severity: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng([seed, list(CORRUPTIONS).index(name), severity])
    return CORRUPTIONS[name](images, severity, rng)
