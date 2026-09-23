"""Read the IMU record embedded in an Insta360 .insv file.

Uses the telemetry-parser Python binding from Gyroflow (pip install telemetry-parser).
Returns time in seconds on the video clock, gyroscope in rad/s and accelerometer in m/s^2.

Command line:
    python -m imuscale.insv recording.insv [imu.csv]
prints a short report and optionally writes a CSV with columns t_s,gx,gy,gz,ax,ay,az
(gyro in deg/s, accel in m/s^2).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


def load_insv_imu(path: str | Path):
    """-> (t [s], gyro [rad/s, n x 3], accel [m/s^2, n x 3], metadata dict)."""
    import telemetry_parser

    p = telemetry_parser.Parser(str(path))
    imu = p.normalized_imu()
    if not imu:
        raise RuntimeError(f"no IMU samples in {path} (this recording mode may not store the IMU, "
                           "or the file was re-exported and lost its metadata)")
    t = np.array([s["timestamp_ms"] for s in imu], float) / 1000.0
    gyr = np.array([s["gyro"] if s.get("gyro") is not None else [np.nan] * 3 for s in imu], float)
    acc = np.array([s["accl"] if s.get("accl") is not None else [np.nan] * 3 for s in imu], float)
    ok = ~(np.isnan(gyr).any(1) | np.isnan(acc).any(1))
    t, gyr, acc = t[ok], np.deg2rad(gyr[ok]), acc[ok]
    order = np.argsort(t, kind="stable")
    t, gyr, acc = t[order], gyr[order], acc[order]
    keep = np.r_[True, np.diff(t) > 0]
    t, gyr, acc = t[keep], gyr[keep], acc[keep]
    meta = {"camera": str(getattr(p, "model", "") or "")}
    try:
        md = p.telemetry()[0]["Default"]["Metadata"]
        meta["fw_version"] = str(md.get("fw_version", ""))
    except Exception:
        pass
    return t, gyr, acc, meta


def load_imu_csv(path: str | Path):
    """CSV with header t_s,gx,gy,gz,ax,ay,az (gyro deg/s, accel m/s^2) -> same tuple as load_insv_imu."""
    d = np.genfromtxt(str(path), delimiter=",", names=True)
    t = np.asarray(d["t_s"], float)
    gyr = np.deg2rad(np.c_[d["gx"], d["gy"], d["gz"]])
    acc = np.c_[d["ax"], d["ay"], d["az"]]
    return t, gyr, acc, {"source": str(path)}


def load_imu(path: str | Path):
    path = Path(path)
    return load_imu_csv(path) if path.suffix.lower() == ".csv" else load_insv_imu(path)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__); return 1
    t, gyr, acc, meta = load_insv_imu(argv[0])
    dt = np.diff(t)
    print(f"{argv[0]}: {meta}")
    print(f"{len(t)} samples, {t[0]:.3f}..{t[-1]:.3f} s, {len(t)/(t[-1]-t[0]):.1f} Hz, "
          f"dt median {np.median(dt)*1e3:.3f} ms, gaps > 3 dt: {(dt > 3*np.median(dt)).sum()}")
    an = np.linalg.norm(acc, axis=1)
    print(f"|accel| mean {an.mean():.3f} m/s^2, gyro rms {np.rad2deg(np.sqrt((gyr**2).mean(0))).round(2)} deg/s")
    if len(argv) > 1:
        with open(argv[1], "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["t_s", "gx", "gy", "gz", "ax", "ay", "az"])
            g = np.rad2deg(gyr)
            for i in range(len(t)):
                w.writerow([f"{t[i]:.6f}", *(f"{v:.6f}" for v in g[i]), *(f"{v:.6f}" for v in acc[i])])
        print("wrote", argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
