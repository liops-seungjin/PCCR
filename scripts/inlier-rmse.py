#!/usr/bin/env python3
"""Overlap-region (inlier) RMSE for recorded registration results.

The `rmse` column in experiments/registration-results.csv is the RMS of EVERY
source point's nearest-neighbour distance to the target after alignment. When
the source is a full scan and the target is only a crop, the non-overlapping
source points dominate that number, so it is a cross-algorithm comparison score,
NOT a precision metric.

This tool recomputes RMSE restricted to the OVERLAP: source points whose nearest
target point is within `--inlier-mult` x the target's point spacing (default 3x,
matching the C++ `inliers` definition in reg_common.cpp). That overlap-only RMSE
is the actual mm-grade alignment error over the region the two clouds share.

For each recorded transform on a chosen (source, target) pair it reports:
  inlier_rmse_mm, inlier_median_mm, n_inliers, overlap_pct, full_rmse_m.

Usage:
    scripts/inlier-rmse.py --source experiments/data/hap101_f0.ply \
        --target tests/data/crop.ply [--inlier-mult 3.0] [--out CSV]
stdlib + numpy + scipy + open3d.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
CSVS = [
    REPO / "experiments" / "registration-results.csv",
    REPO / "experiments" / "registration-results-archive-2026-06-11.csv",
]


def load_xyz(path):
    pc = o3d.io.read_point_cloud(str(REPO / path if not Path(path).is_absolute() else path))
    return np.asarray(pc.points, dtype=np.float64)


def point_spacing(xyz):
    """Median nearest-neighbour distance within a cloud (its resolution)."""
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=2)  # [:,0] is self (0)
    return float(np.median(d[:, 1]))


def parse_rows(source, target):
    """All recorded rows matching this (source, target) pair, newest-CSV first."""
    rows = []
    seen = set()
    for csv_path in CSVS:
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if not (r["source"].endswith(Path(source).name)
                        and r["target"].endswith(Path(target).name)):
                    continue
                if not r.get("transform", "").strip():
                    continue
                key = r["algo"]
                if key in seen:      # keep the newest (first CSV wins)
                    continue
                seen.add(key)
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--inlier-mult", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = load_xyz(args.source)
    tgt = load_xyz(args.target)
    spacing = point_spacing(tgt)
    inlier_r = args.inlier_mult * spacing
    tree = cKDTree(tgt)

    print(f"source {Path(args.source).name}: {len(src)} pts   "
          f"target {Path(args.target).name}: {len(tgt)} pts")
    print(f"target point spacing: {spacing * 1000:.2f} mm   "
          f"inlier radius: {args.inlier_mult}x = {inlier_r * 1000:.2f} mm\n")

    rows = parse_rows(args.source, args.target)
    if not rows:
        sys.exit("no matching recorded rows")

    out_rows = []
    hdr = (f"{'algo':<13} {'conv':<5} {'inlier_rmse':>12} {'median':>9} "
           f"{'n_inliers':>9} {'overlap%':>8} {'full_rmse':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        T = np.array([float(x) for x in r["transform"].split()], dtype=np.float64).reshape(4, 4)
        warped = src @ T[:3, :3].T + T[:3, 3]
        d, _ = tree.query(warped, k=1)        # nearest target distance per source pt
        inl = d < inlier_r
        n_inl = int(inl.sum())
        if n_inl:
            rmse_mm = float(np.sqrt(np.mean(d[inl] ** 2)) * 1000)
            med_mm = float(np.median(d[inl]) * 1000)
        else:
            rmse_mm = med_mm = float("nan")
        full_rmse = float(np.sqrt(np.mean(d ** 2)))
        overlap = 100.0 * n_inl / len(src)
        print(f"{r['algo']:<13} {r['converged']:<5} {rmse_mm:>10.2f}mm "
              f"{med_mm:>7.2f}mm {n_inl:>9d} {overlap:>7.1f}% {full_rmse:>9.3f}m")
        out_rows.append({
            "algo": r["algo"], "converged": r["converged"],
            "inlier_rmse_mm": f"{rmse_mm:.3f}", "inlier_median_mm": f"{med_mm:.3f}",
            "n_inliers": n_inl, "overlap_pct": f"{overlap:.2f}",
            "full_rmse_m": f"{full_rmse:.4f}",
        })

    if args.out:
        out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
