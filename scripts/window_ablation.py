#!/usr/bin/env python3
"""Ablation over window construction: how the survey is divided before per-window estimation.

    python scripts/window_ablation.py --sparse model/sparse/0 --imu rec.insv --period 0.1 [--truth 0.987]

Constructions: joint (whole survey as one system), contiguous runs, fixed windows of N intervals,
sliding windows of N intervals at a given stride. Each windowed variant estimates every window
independently and reports the mean. With --truth (a reference scale) the error of each is printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imuscale.colmap_io import load_frames          # noqa: E402
from imuscale.estimator import InertialEstimator    # noqa: E402
from imuscale.insv import load_imu                  # noqa: E402


def windows(runs, mode, n=25, stride=1):
    if mode == "contiguous":
        return runs
    out = []
    for rr in runs:
        if mode == "fixed":
            out += [rr[i:i + n] for i in range(0, len(rr), n)]
        elif mode == "sliding":
            out += [rr[i:i + n] for i in range(0, max(1, len(rr) - n + 1), stride)]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sparse", type=Path, required=True); ap.add_argument("--imu", type=Path, required=True)
    ap.add_argument("--period", type=float, required=True)
    ap.add_argument("--imu-offset", type=float, default=0.0, help="IMU time of video t=0 [s]")
    ap.add_argument("--group", default=None); ap.add_argument("--index-base", type=int, default=0)
    ap.add_argument("--truth", type=float, default=None, help="reference scale for the error column")
    args = ap.parse_args(argv)
    tg, gyr, acc, _ = load_imu(args.imu)
    tg = tg - args.imu_offset
    _, _, IDX, Rwc, P, _ = load_frames(args.sparse, args.group, index_base=args.index_base)
    est = InertialEstimator(tg, gyr, acc, IDX, Rwc, P)
    est.fit_time_map(args.period)
    runs, s_joint, *_ = est.robust_solve()
    variants = [("joint (whole survey)", None), ("contiguous runs", ("contiguous",)),
                ("fixed 15", ("fixed", 15)), ("fixed 25", ("fixed", 25)), ("fixed 30", ("fixed", 30)), ("fixed 50", ("fixed", 50)),
                ("sliding 15 stride 1", ("sliding", 15, 1)), ("sliding 30 stride 1", ("sliding", 30, 1)),
                ("sliding 30 stride 15", ("sliding", 30, 15)), ("sliding 60 stride 1", ("sliding", 60, 1))]
    print(f"{'construction':24s} {'windows':>8s} {'mean scale':>11s} {'sd %':>6s}" + ("  error %" if args.truth else ""))
    for name, spec in variants:
        if spec is None:
            vals = np.array([s_joint]); nwin = 1
        else:
            ws = windows(runs, *spec)
            vals, w, *_ = est.scale_per_run(ws)
            nwin = len(ws)
        if len(vals) == 0:
            print(f"{name:24s} {nwin:8d} {'-':>11s}"); continue
        line = f"{name:24s} {nwin:8d} {vals.mean():11.4f} {100*vals.std(ddof=1)/vals.mean() if len(vals) > 1 else 0:6.2f}"
        if args.truth:
            line += f"  {100*(vals.mean()/args.truth-1):+7.2f}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
