import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from veb.corruptions import CORRUPTIONS, PARAMS, SEVERITIES, corrupt

NAMES = list(CORRUPTIONS)


@pytest.fixture(scope="module")
def images() -> np.ndarray:
    rng = np.random.default_rng(0)
    smooth = gaussian_filter(rng.normal(size=(24, 32, 32, 3)), sigma=(0, 3, 3, 0))
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    textured = 30 + 195 * smooth + rng.normal(0, 6, size=smooth.shape)
    return np.clip(textured, 0, 255).astype(np.uint8)


def mean_abs_change(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def test_registry_matches_spec():
    assert NAMES == ["gaussian_noise", "gaussian_blur", "jpeg", "darken", "contrast", "pixelate"]
    assert set(PARAMS) == set(NAMES)
    assert all(len(values) == 5 for table in PARAMS.values() for values in table.values())


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("severity", SEVERITIES)
def test_shape_and_dtype_preserved(images, name, severity):
    before = images.copy()
    out = CORRUPTIONS[name](images, severity, np.random.default_rng(0))
    assert out.shape == images.shape
    assert out.dtype == np.uint8
    assert np.array_equal(images, before)


@pytest.mark.parametrize("name", NAMES)
def test_deterministic_given_rng(images, name):
    a = CORRUPTIONS[name](images, 3, np.random.default_rng(7))
    b = CORRUPTIONS[name](images, 3, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert np.array_equal(corrupt(images, name, 3, seed=1), corrupt(images, name, 3, seed=1))


def test_noise_depends_on_seed(images):
    assert not np.array_equal(corrupt(images, "gaussian_noise", 3, seed=0), corrupt(images, "gaussian_noise", 3, seed=1))


@pytest.mark.parametrize("name", NAMES)
def test_change_grows_with_severity(images, name):
    changes = [mean_abs_change(images, corrupt(images, name, severity)) for severity in SEVERITIES]
    assert changes[0] > 0.5
    assert all(low < high for low, high in zip(changes, changes[1:]))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("severity", [0, 6])
def test_invalid_severity_rejected(images, name, severity):
    with pytest.raises(ValueError):
        CORRUPTIONS[name](images, severity, np.random.default_rng(0))
