"""Amplitude spectral density of the gyroscope and accelerometer (Welch), from a moving or static recording.

    python -m imuscale.sensor_noise recording.insv [more.insv ...] --out asd.png

Operator motion occupies the band below ~50 Hz; above ~100 Hz only sensor noise remains, so the
white-noise floor can be read there even from a hand-held recording. A flat floor is white noise;
a spectrum that keeps falling reveals a firmware low-pass filter.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.signal import welch

from .insv import load_imu


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--axis", type=int, default=1, help="axis 0/1/2 (default 1)")
    ap.add_argument("--nperseg", type=int, default=4096)
    ap.add_argument("--out", default=None, help="PNG (default: print the noise floor only)")
    args = ap.parse_args(argv)
    curves = []
    for f in args.files:
        t, gyr, acc, meta = load_imu(f)
        fs = len(t) / (t[-1] - t[0])
        fg, Pg = welch(np.rad2deg(gyr[:, args.axis]), fs=fs, nperseg=args.nperseg)
        fa, Pa = welch(acc[:, args.axis], fs=fs, nperseg=args.nperseg)
        hi = fg > 100
        print(f"{f}: {fs:.0f} Hz | gyro ASD above 100 Hz: median {np.median(np.sqrt(Pg[hi])):.4f} deg/s/sqrt(Hz), "
              f"ratio 400 Hz / 150 Hz {np.sqrt(Pg[np.argmin(abs(fg-400))]/Pg[np.argmin(abs(fg-150))]):.2f} "
              f"(1 = white, <1 = filtered) | accel ASD above 100 Hz median {1e3*np.median(np.sqrt(Pa[fa>100])):.2f} mm/s^2/sqrt(Hz)")
        curves.append((f, fg, np.sqrt(Pg), fa, np.sqrt(Pa)))
    if args.out:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for f, fg, Ag, fa, Aa in curves:
            ax[0].loglog(fg[1:], Ag[1:], lw=.9, label=f); ax[1].loglog(fa[1:], Aa[1:], lw=.9, label=f)
        ax[0].set_title(f"gyro axis {args.axis} ASD [deg/s/sqrt(Hz)]"); ax[1].set_title(f"accel axis {args.axis} ASD [m/s^2/sqrt(Hz)]")
        for a in ax:
            a.set_xlabel("Hz"); a.axvspan(100, fg[-1], color="0.9", zorder=0); a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(args.out, dpi=150); print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
