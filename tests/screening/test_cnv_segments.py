"""Tests for deletion segment calling."""
import numpy as np
import pandas as pd

from screening.src.cnv.segments import (call_deletion_segments, filter_segments,
                              robust_z, segment_overlap_frac)


def _signal(dip_ranges):
    """Synthetic per-window signal: flat noise + deep negative dips at given rows."""
    n = 200
    rng = np.random.default_rng(0)
    score = rng.normal(0, 0.1, n)
    for lo, hi in dip_ranges:
        score[lo:hi] = -3.0
    return pd.DataFrame({"chr": ["chr1"] * 100 + ["chr7"] * 100,
                         "start": np.arange(n) * 10000,
                         "end": np.arange(n) * 10000 + 9999,
                         "score": score})


def test_recovers_planted_deletion():
    sig = _signal([(130, 140)])          # 10 consecutive dipping windows on chr7
    segs = call_deletion_segments(sig, z_threshold=1.5, min_windows=3)
    assert len(segs) == 1
    assert segs.iloc[0]["chr"] == "chr7"
    assert segs.iloc[0]["n_windows"] == 10
    assert segs.iloc[0]["start"] == 130 * 10000


def test_short_runs_filtered_by_min_windows():
    sig = _signal([(130, 131)])          # only 1 window below threshold
    segs = call_deletion_segments(sig, z_threshold=1.5, min_windows=3)
    assert segs.empty


def test_flat_signal_yields_nothing():
    segs = call_deletion_segments(_signal([]), z_threshold=1.5, min_windows=3)
    assert segs.empty


def test_robust_z_constant_input_no_nan():
    z = robust_z(np.ones(10))
    assert np.all(np.isfinite(z))


def test_overlap_frac():
    assert segment_overlap_frac("chr7", 100, 200, "chr7", 0, 400) == 0.25
    assert segment_overlap_frac("chr1", 100, 200, "chr7", 0, 400) == 0.0


def test_filter_segments_size_cap():
    segs = pd.DataFrame({"chr": ["chr7", "chr7"], "start": [0, 0],
                         "end": [5_000_000, 25_000_000],
                         "n_windows": [3, 4], "mean_z": [-2.0, -3.0]})
    kept = filter_segments(segs, max_bp=10_000_000)
    assert len(kept) == 1 and kept.iloc[0]["end"] == 5_000_000
    assert len(filter_segments(segs, None)) == 2
