#!/usr/bin/env python3
"""Call deletion segments from an ordered per-window CNV signal.

Pure numpy/pandas — deliberately free of scanpy/infercnvpy so the logic is
unit-testable and auditable.

Method: robust z-score (median/MAD) over ALL windows (a single small germline
deletion barely shifts the genome-wide median), then runs of consecutive
windows with z <= -z_threshold, kept if length >= min_windows.
"""
import numpy as np
import pandas as pd

SEGMENT_COLUMNS = ["chr", "start", "end", "n_windows", "mean_z"]


def robust_z(x):
    """Median/MAD-based z-score; falls back to std when MAD == 0."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale == 0:
        scale = float(np.std(x)) or 1.0
    return (x - med) / scale


def call_deletion_segments(signal, z_threshold=1.5, min_windows=3):
    """Find runs of negative deviation in an ordered CNV signal.

    signal: DataFrame with columns [chr, start, end, score], ordered along the
    genome; `score` is the per-window CNV deviation (negative = loss).
    Returns DataFrame[chr, start, end, n_windows, mean_z] (empty if none).
    """
    required = {"chr", "start", "end", "score"}
    if not required <= set(signal.columns):
        raise ValueError(f"signal must have columns: {sorted(required)}")
    df = signal.reset_index(drop=True).copy()
    df["z"] = robust_z(df["score"].to_numpy())

    segs = []
    for chrom, sub in df.groupby("chr", sort=False):
        sub = sub.reset_index(drop=True)
        below = (sub["z"] <= -z_threshold).to_numpy()
        i, n = 0, len(sub)
        while i < n:
            if not below[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and below[j + 1]:
                j += 1
            if j - i + 1 >= min_windows:
                run = sub.iloc[i:j + 1]
                segs.append({
                    "chr": chrom,
                    "start": int(run["start"].min()),
                    "end": int(run["end"].max()),
                    "n_windows": int(j - i + 1),
                    "mean_z": float(run["z"].mean()),
                })
            i = j + 1
    return pd.DataFrame(segs, columns=SEGMENT_COLUMNS)


def filter_segments(segs, max_bp=None):
    """Drop segments larger than max_bp (microdeletion scope: huge dips near
    centromeres/low-mappability regions are inferCNV artifacts, not deletions)."""
    if segs.empty or max_bp is None:
        return segs
    keep = (segs["end"] - segs["start"]) <= max_bp
    return segs[keep].reset_index(drop=True)


def segment_overlap_frac(seg_chr, seg_start, seg_end, iv_chr, iv_start, iv_end):
    """Fraction of the known interval [iv_start, iv_end] covered by a segment."""
    if str(seg_chr).removeprefix("chr") != str(iv_chr).removeprefix("chr"):
        return 0.0
    ov = max(0, min(seg_end, iv_end) - max(seg_start, iv_start))
    return ov / max(1, iv_end - iv_start)
