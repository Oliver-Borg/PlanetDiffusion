from opensimplex.internals import _noise3, _noise2, _init
import numpy as np

from .noise_settings import NoiseSettings

try:
    from numba import njit, prange
except ImportError:
    prange = range

    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper


def noise3coords(stacked_coords: np.ndarray, seed=0) -> np.ndarray:
    """
    Generate 3D OpenSimplex noise from X,Y,Z coordinates.
    :param stacked_coords: numpy array of shape (h, w, 3)
    :param seed: seed for the noise
    :return:  generated 3D noise as numpy array of shape (h, w)
    """
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
    _perm, _perm_grad_index3 = _init(seed)
    return _noise3coords(stacked_coords, _perm, _perm_grad_index3)


@njit(cache=True, parallel=True)
def _noise3coords(
    stacked_coords: np.ndarray, _perm: np.ndarray, _perm_grad_index3: np.ndarray
):
    h, w, _ = stacked_coords.shape
    noise = np.zeros((h, w))
    for i in prange(h):
        for j in prange(w):
            x, y, z = stacked_coords[i, j]
            if x == np.nan or y == np.nan or z == np.nan:
                continue
            noise[i, j] = _noise3(x, y, z, _perm, _perm_grad_index3)
    return noise


def stacked_noise(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    settings: NoiseSettings,
    amplitude_mask: np.ndarray | None = None,
):
    freq = settings.frequency
    freq_mult = settings.roughness
    amplitude = settings.amplitude
    persistence = settings.persistence
    min_value = settings.min_value
    max_value = settings.max_value
    octaves = settings.octaves
    total_noise = np.zeros_like(x, dtype=np.float32)

    if amplitude_mask is not None:
        gradient_map = amplitude_mask
    else:
        gradient_map = np.ones_like(total_noise)
    for _ in range(octaves):
        stacked = np.dstack((x, y, z)) * freq
        noise = noise3coords(stacked, settings.seed)
        noise = settings.apply_filter(noise)
        total_noise += noise * amplitude
        freq *= freq_mult
        amplitude *= persistence

    if not settings.additive:
        total_noise = total_noise * gradient_map
    total_noise = total_noise**settings.power
    total_noise = (total_noise - total_noise.min()) / (total_noise.max() - total_noise.min())
    total_noise = total_noise * (max_value - min_value) + min_value
    total_noise = np.maximum(np.zeros_like(total_noise), total_noise)
    if settings.use_max:
        total_noise = np.minimum(np.ones_like(total_noise) * max_value, total_noise)
    if settings.negate:
        total_noise = -total_noise
    total_noise *= settings.multiplier
    return total_noise


def stacked_multi_noise(
    surface_coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    settings: list[NoiseSettings],
    amplitude_mask: np.ndarray | None = None,
):
    x, y, z = surface_coords
    total_noise = np.zeros_like(x, dtype=np.float32)
    noise_mask = np.ones_like(x, dtype=bool)
    for setting in settings:
        next_noise = stacked_noise(x, y, z, setting, amplitude_mask)
        next_noise[~noise_mask] = 0
        total_noise += next_noise
        noise_mask = total_noise > 0
    return total_noise


def noise2coords(stacked_coords: np.ndarray, seed=0) -> np.ndarray:
    """
    Generate 2D OpenSimplex noise from X,Y coordinates.
    :param stacked_coords: numpy array of shape (h, w, 2)
    :param seed: seed for the noise
    :return:  generated 2D noise as numpy array of shape (h, w)
    """
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    _perm, _ = _init(seed)
    return _noise2coords(stacked_coords, _perm)


@njit(cache=True, parallel=True)
def _noise2coords(stacked_coords: np.ndarray, _perm: np.ndarray):
    h, w, _ = stacked_coords.shape
    noise = np.zeros((h, w))
    for i in prange(h):
        for j in prange(w):
            x, y = stacked_coords[i, j]
            noise[i, j] = _noise2(x, y, _perm)
    return noise
