"""Self-check on synthetic data: a smooth 3D trajectory with varying orientation, a 1 kHz IMU derived
from it (specific force = R_WB^T (a - g), gyro = body rates), camera poses every 0.1 s in units scaled
by 1/s_true with an unknown camera-IMU rotation and a frame-time offset. The estimator must recover
s_true, gravity and R_CB.

    python -m pytest tests/        or        python tests/test_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imuscale.estimator import InertialEstimator, estimate  # noqa: E402

G = np.array([0.0, 0.0, -9.81])   # gravity acceleration, world frame (down = -Z)


def synth(duration=90.0, rate=1000.0, frame_dt=0.1, s_true=0.42, offset=0.37, seed=0, gaps=True):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / rate)
    # position: sum of sines, walking-like (~1 m/s) with vertical bob
    f = np.array([0.11, 0.07, 0.9]); A = np.array([6.0, 4.0, 0.04]); ph = rng.uniform(0, 2 * np.pi, 3)
    pos = np.c_[[A[i] * np.sin(2 * np.pi * f[i] * t + ph[i]) for i in range(3)]].T + np.c_[0.8 * t, 0.3 * t, np.zeros_like(t)]
    vel = np.gradient(pos, t, axis=0); accel = np.gradient(vel, t, axis=0)
    # orientation: yaw following the path plus swaying roll/pitch
    yaw = np.arctan2(vel[:, 1], vel[:, 0]); roll = 0.15 * np.sin(2 * np.pi * 0.8 * t); pitch = 0.1 * np.sin(2 * np.pi * 0.5 * t + 1)
    R_WB = Rot.from_euler("zyx", np.c_[yaw, pitch, roll]).as_matrix()
    # body rates from finite differences of orientation
    dR = np.einsum("nji,njk->nik", R_WB[:-1], R_WB[1:])       # R_k^T R_{k+1}
    gyr = Rot.from_matrix(dR).as_rotvec() * rate; gyr = np.vstack([gyr, gyr[-1]])
    acc = np.einsum("nji,nj->ni", R_WB, accel - G)             # specific force in body frame
    acc += rng.normal(0, 0.02, acc.shape); gyr += rng.normal(0, np.deg2rad(0.05), gyr.shape)
    # camera: fixed rotation w.r.t. body, poses every frame_dt, model units = metres / s_true
    R_CB = Rot.from_euler("xyz", [90, 0, -90], degrees=True).as_matrix()
    R_BC = R_CB.T
    fidx = np.arange(0, int(duration / frame_dt) - 3)
    # drop 15 frames every 100 (a sharpness filter would do this) so that the survey splits into contiguous runs
    if gaps:
        fidx = fidx[(fidx % 100) >= 15]
    tf = fidx * frame_dt + offset
    ii = np.searchsorted(t, tf)
    Rwc = np.einsum("nij,jk->nik", R_WB[ii], R_BC)
    P = pos[ii] / s_true + rng.normal(0, 0.003 / s_true, (len(ii), 3))   # 3 mm pose noise
    return t, gyr, acc, fidx.astype(float), Rwc, P, dict(s=s_true, offset=offset, R_CB=R_CB, frame_dt=frame_dt)


def test_recovers_scale_gravity_and_rotation(gaps=True):
    t, gyr, acc, IDX, Rwc, P, truth = synth(gaps=gaps)
    est = InertialEstimator(t, gyr, acc, IDX, Rwc, P)
    a, b, R_CB, he = est.fit_time_map(truth["frame_dt"] * 1.002)      # start 0.2% off the true period
    assert abs(a / truth["frame_dt"] - 1) < 2e-4, a
    assert abs(b - truth["offset"]) < 0.01, b
    assert he < 0.3, he
    assert np.rad2deg(np.linalg.norm(Rot.from_matrix(R_CB.T @ truth["R_CB"]).as_rotvec())) < 0.5
    r = estimate(est, bootstrap=0)
    assert abs(r["scale"] / truth["s"] - 1) < 0.01, r["scale"]
    g = np.asarray(r["gravity_model_frame"])
    assert abs(np.linalg.norm(g) - 9.81) < 0.1, g
    assert np.rad2deg(np.arccos(g @ G / (np.linalg.norm(g) * 9.81))) < 1.0
    assert r["quality"]["ok"], r["quality"]



def test_recording_without_gaps():
    """A single contiguous run must be cut into windows and pass, not fall back to the joint fit."""
    test_recovers_scale_gravity_and_rotation(gaps=False)


if __name__ == "__main__":
    test_recovers_scale_gravity_and_rotation()
    test_recording_without_gaps()
    print("ok")
