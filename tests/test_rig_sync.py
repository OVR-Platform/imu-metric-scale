"""Self-check for rig_sync: two gyros on one rigid rig (different mounting), second clock offset by
12.3 ms and running 6 ppm fast; the offset at t=0 and the drift must be recovered.

    python -m pytest tests/        or        python tests/test_rig_sync.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imuscale.rig_sync import rig_sync  # noqa: E402


def test_rig_sync(offset=0.0123, drift=6e-6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 180, 1e-3)
    k = np.exp(-0.5 * (np.arange(-120, 121) / 25.0) ** 2); k /= k.sum()   # smooth, walking-like rates
    w = np.c_[[np.convolve(rng.normal(0, 3, len(t)), k, "same") for _ in range(3)]].T
    w_b = w @ Rot.from_euler("xyz", [90, 0, 180], degrees=True).as_matrix().T   # other camera, other axes
    # camera A sees true time t at clock t + offset + drift*t; camera B at t
    ta = t + offset + drift * t
    r = rig_sync(ta, np.linalg.norm(w, axis=1), t, np.linalg.norm(w_b + rng.normal(0, 0.01, w.shape), axis=1))
    assert abs(r["offset_at_t0_s"] - offset) < 0.3e-3, r["offset_at_t0_s"]
    assert abs(r["drift_ppm"] - drift * 1e6) < 1.0, r["drift_ppm"]
    print(f"offset {r['offset_at_t0_s']*1e3:.2f} ms (true {offset*1e3}), drift {r['drift_ppm']:.2f} ppm (true {drift*1e6})")


if __name__ == "__main__":
    test_rig_sync()
