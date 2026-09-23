"""Accelerometer scale factor and bias drift from a static recording (camera resting on a table).

    python -m imuscale.static_bias static.insv [--local-g 9.806] [--block 10] [--out bias.png]

Reports the mean specific-force magnitude against local gravity (accelerometer scale factor), the
bias excursion in `--block`-second averages over the recording, and the gyroscope bias. Bias
instability proper needs an Allan-variance analysis of a much longer recording (IEEE Std 952-1997).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .insv import load_imu


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--local-g", type=float, default=9.80665, help="local gravity [m/s^2]")
    ap.add_argument("--block", type=float, default=10.0, help="averaging block [s]")
    ap.add_argument("--skip", type=float, default=5.0, help="seconds skipped at start and end")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    t, gyr, acc, meta = load_imu(args.file)
    m = (t > t[0] + args.skip) & (t < t[-1] - args.skip)
    t, gyr, acc = t[m], gyr[m], acc[m]
    wn = np.rad2deg(np.linalg.norm(gyr, axis=1))
    print(f"{args.file}: {t[-1]-t[0]:.0f} s, |omega| p99 {np.percentile(wn, 99):.2f} deg/s (should be well below 1 for a static camera)")
    an = np.linalg.norm(acc, axis=1)
    print(f"|specific force| mean {an.mean():.4f} m/s^2 vs local g {args.local_g:.4f}: scale factor error {100*(an.mean()/args.local_g-1):+.3f}%")
    edges = np.arange(t[0], t[-1], args.block)
    idx = np.digitize(t, edges) - 1
    ba = np.array([acc[idx == i].mean(0) for i in range(len(edges) - 1)])
    bg = np.array([gyr[idx == i].mean(0) for i in range(len(edges) - 1)])
    g_dir = acc.mean(0) / an.mean()
    # bias drift: motion of the block means orthogonal and parallel to gravity, as a vector excursion
    exc = np.linalg.norm(ba[-1] - ba[0]); exc_max = np.linalg.norm(ba - ba[0], axis=1).max()
    print(f"accel block means ({args.block:.0f} s): first {ba[0].round(4)} last {ba[-1].round(4)}; vector excursion {exc:.4f} m/s^2 (max {exc_max:.4f}) over {edges[-1]-edges[0]:.0f} s")
    print(f"gyro bias mean {np.rad2deg(gyr.mean(0)).round(3)} deg/s, |bias| {np.rad2deg(np.linalg.norm(gyr.mean(0))):.3f}, excursion {np.rad2deg(np.linalg.norm(bg[-1]-bg[0])):.3f} deg/s")
    if args.out:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        tc = 0.5 * (edges[:-1] + edges[1:]) - t[0]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for k, c in enumerate("xyz"):
            ax[0].plot(tc, ba[:, k] - ba[0, k], label=f"accel {c}"); ax[1].plot(tc, np.rad2deg(bg[:, k]), label=f"gyro {c}")
        ax[0].set_ylabel("accel block mean - first [m/s^2]"); ax[1].set_ylabel("gyro block mean [deg/s]")
        for a in ax:
            a.set_xlabel("t [s]"); a.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(args.out, dpi=150); print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
