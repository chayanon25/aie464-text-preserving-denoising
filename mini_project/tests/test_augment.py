"""Training augmentation tests: clean target is only ever flipped."""
import numpy as np

from src.augment import TrainAugment


def test_clean_only_flipped():
    rng = np.random.default_rng(0); n = np.random.rand(32, 48).astype(np.float32); c = np.random.rand(32, 48).astype(np.float32)
    n2, c2 = TrainAugment()(n, c, rng)
    assert np.array_equal(c2, c) or np.array_equal(c2, c[:, ::-1])


def test_range_shape_dtype():
    n2, c2 = TrainAugment(1, 1, 1, 1)(np.random.rand(40, 40).astype(np.float32), np.ones((40, 40), np.float32), np.random.default_rng(1))
    assert n2.shape == (40, 40) and n2.dtype == np.float32 and 0 <= n2.min() and n2.max() <= 1


def test_deterministic_given_rng():
    n = np.random.rand(40, 40).astype(np.float32); c = np.ones_like(n)
    a = TrainAugment()(n, c, np.random.default_rng(7)); b = TrainAugment()(n, c, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0])


def test_all_off_is_identity_or_flip():
    n = np.random.rand(20, 30).astype(np.float32); c = np.ones_like(n)
    n2, _ = TrainAugment(0, 0, 0, 0)(n, c, np.random.default_rng(3))
    assert np.array_equal(n2, n) or np.array_equal(n2, n[:, ::-1])
