#!/usr/bin/env python3
"""Invariance test: apply a random similarity (scale, rotation, translation) to the input poses and
check that the recovered scale equals the original estimate divided by the applied factor.

    python scripts/sim3_invariance.py --sparse model/sparse/0 --imu rec.insv --period 0.1 [--trials 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imuscale.colmap_io import load_frames                    # noqa: E402
from imuscale.estimator import InertialEstimator, estimate    # noqa: E402
from imuscale.insv import load_imu                            # noqa: E402


def run(tg, gyr, acc, IDX, Rwc, P, period):
    est = InertialEstimator(tg, gyr, acc, IDX, Rwc, P)
    est.fit_time_map(period)
    return estimate(est, bootstrap=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sparse", type=Path, required=True); ap.add_argument("--imu", type=Path, required=True)
    ap.add_argument("--period", type=float, required=True)
    ap.add_argument("--group", default=None); ap.add_argument("--index-base", type=int, default=0)
    ap.add_argument("--trials", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    tg, gyr, acc, _ = load_imu(args.imu)
    _, _, IDX, Rwc, P, _ = load_frames(args.sparse, args.group, index_base=args.index_base)
    base = run(tg, gyr, acc, IDX, Rwc, P, args.period)
    print(f"original: scale {base['scale']:.5f}, |g| {base['gravity_norm']:.3f}")
    rng = np.random.default_rng(args.seed)
    for i in range(args.trials):
        k = float(np.exp(rng.uniform(np.log(0.2), np.log(8.0)))); R = Rot.random(random_state=rng.integers(1 << 30)).as_matrix()
        t = rng.uniform(-100, 100, 3)
        r = run(tg, gyr, acc, IDX, np.einsum("ij,njk->nik", R, Rwc), (P * k) @ R.T + t, args.period)
        exp = base["scale"] / k
        g_back = R.T @ np.asarray(r["gravity_model_frame"])
        print(f"trial {i}: k={k:.3f} rot={np.rad2deg(np.linalg.norm(Rot.from_matrix(R).as_rotvec())):.0f} deg -> "
              f"scale {r['scale']:.5f} expected {exp:.5f} diff {100*(r['scale']/exp-1):+.4f}% | "
              f"gravity rotated back differs {np.rad2deg(np.arccos(np.clip(g_back @ base['gravity_model_frame'] / (np.linalg.norm(g_back)*base['gravity_norm']), -1, 1))):.3f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
