"""Time offset and clock drift between two cameras mounted on the same rigid rig, from their gyroscopes.

    python -m imuscale.rig_sync left.insv right.insv --imu-offset 12.5 9.3 [--window 20] [--out sync.png]

On a rigid rig both gyroscopes see the same angular velocity, each in its own axes, so the magnitude
|omega| is the same signal on both and needs no knowledge of the relative mounting. The two magnitudes
are resampled at 1 kHz and cross-correlated, over the whole overlap and in sliding windows; a line
fitted to the windowed lags gives the offset at t=0 and the drift between the two clocks (ppm).

`--imu-offset A B` is, for each file, the IMU time of video t=0 (e.g. the cut point when the videos
were trimmed to a common start, since the IMU record keeps the original timeline). With the right
values the reported lag is the residual between the two video timelines.
Sign: lag > 0 means a motion appears later on the first camera than on the second.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .insv import load_imu

FS = 1000.0


def xcorr_lag(a, b, max_lag):
    """a: n samples, b: the same span with max_lag extra samples on each side. Lag in samples
    (sub-sample, parabolic) such that a[i] ~ b[i - lag], always over the full window; -> (lag, peak)."""
    n = len(a); a = (a - a.mean()) / a.std()
    cs, cs2 = np.r_[0, np.cumsum(b)], np.r_[0, np.cumsum(b * b)]
    mu = (cs[n:] - cs[:-n]) / n; sd = np.sqrt((cs2[n:] - cs2[:-n]) / n - mu ** 2)   # b stats per shift
    c = (np.correlate(b, a, "valid") / n / sd)[::-1]   # Pearson; index j <-> lag j - max_lag
    i = int(np.argmax(c)); d = 0.0
    if 0 < i < len(c) - 1:
        y0, y1, y2 = c[i - 1:i + 2]; d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return i - max_lag + d, float(c[i])


def rig_sync(ta, ma, tb, mb, window=20.0, step=10.0, max_lag=0.5):
    """|omega| of two gyros on their (offset-corrected) timelines -> dict with global lag, windows, fit [s]."""
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    if hi - lo < window + 2 * max_lag:
        raise RuntimeError(f"overlap of {hi - lo:.1f} s is shorter than one window: check --imu-offset")
    tg = np.arange(lo, hi, 1 / FS)
    a = np.interp(tg, ta, ma); b = np.interp(tg, tb, mb)
    L = int(max_lag * FS)
    lag, peak = xcorr_lag(a[L:-L], b, L)
    W, S = int(window * FS), int(step * FS)
    win = []
    for s in range(0, len(tg) - W - 2 * L + 1, S):
        l, p = xcorr_lag(a[s + L:s + L + W], b[s:s + W + 2 * L], L)
        win.append((tg[s + L + W // 2], l / FS, p))
    win = np.array(win)
    good = win[:, 2] > 0.8   # windows with too little rotation correlate poorly and are ignored in the fit
    if good.sum() >= 2:
        drift, off0 = np.polyfit(win[good, 0], win[good, 1], 1)
    else:
        drift, off0 = 0.0, lag / FS
    return {"lag_s": lag / FS, "peak": peak, "overlap_s": hi - lo - 2 * max_lag, "windows": win,
            "offset_at_t0_s": off0, "drift_ppm": drift * 1e6, "n_windows_fit": int(good.sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imu_a"); ap.add_argument("imu_b")
    ap.add_argument("--imu-offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("A", "B"),
                    help="IMU time of video t=0 for each file [s] (default 0 0)")
    ap.add_argument("--window", type=float, default=20.0, help="window length [s]")
    ap.add_argument("--max-lag", type=float, default=0.5, help="lag search range [s]; raise it for unaligned files")
    ap.add_argument("--out", default=None, help="PNG")
    args = ap.parse_args(argv)

    sig = []
    for path, off in ((args.imu_a, args.imu_offset[0]), (args.imu_b, args.imu_offset[1])):
        t, g, _, _ = load_imu(path)
        sig += [t - off, np.linalg.norm(g, axis=1)]
    r = rig_sync(*sig, window=args.window, step=args.window / 2, max_lag=args.max_lag)
    print(f"overlap {r['overlap_s']:.0f} s, global lag {r['lag_s']*1e3:+.2f} ms (peak {r['peak']:.3f})")
    for tc, l, p in r["windows"]:
        print(f"  t={tc:7.1f} s  lag {l*1e3:+7.2f} ms  peak {p:.3f}")
    print(f"fit on {r['n_windows_fit']} windows: offset at t=0 {r['offset_at_t0_s']*1e3:+.2f} ms, drift {r['drift_ppm']:+.1f} ppm")
    if args.out:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        w = r["windows"]; fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(w[:, 0], w[:, 1] * 1e3, "o", ms=4)
        ax.plot(w[:, 0], (r["offset_at_t0_s"] + r["drift_ppm"] * 1e-6 * w[:, 0]) * 1e3, "k--", lw=.8)
        ax.set_xlabel("t [s]"); ax.set_ylabel("lag [ms]"); fig.tight_layout(); fig.savefig(args.out, dpi=150)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
