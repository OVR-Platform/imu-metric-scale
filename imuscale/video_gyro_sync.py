"""Camera-IMU time offset and axis mapping from the video alone, without a calibration target.

    python -m imuscale.video_gyro_sync recording.insv --start 60 --duration 20 [--focal 280] [--out sync.png]

Frames are decoded with ffmpeg (one lens of the dual-fisheye stream, downscaled to 960 px), Shi-Tomasi
corners are tracked with pyramidal Lucas-Kanade and a forward-backward check, tracks are unprojected
with an equidistant fisheye model and the inter-frame rotation is estimated by Kabsch inside RANSAC.
The resulting angular rate is compared with the gyroscope averaged over the same frame interval for
time shifts of -150..+150 ms and all signed axis permutations; the best shift is the camera-IMU
offset. The correlation curve is broad (each visual sample averages over a frame interval), so the
offset is bounded to a few ms rather than resolved exactly.

Requires ffmpeg on PATH and opencv-python. Use a segment with vigorous rotation.
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .insv import load_imu

W = 960


def decode_frames(video, start, duration, fps, stream=0):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start}", "-t", f"{duration}", "-i", str(video), "-map", f"0:v:{stream}",
           "-vf", f"scale={W}:{W}", "-r", f"{fps}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, W, W)


def kabsch(A, B):
    H = A.T @ B; U, _, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def ransac_rot(A, B, iters=200, th=np.deg2rad(0.5), rng=np.random.default_rng(0)):
    best = None
    for _ in range(iters):
        idx = rng.choice(len(A), 3, replace=False); R = kabsch(A[idx], B[idx])
        err = np.arccos(np.clip(((A @ R.T) * B).sum(1), -1, 1)); inl = err < th
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), inl)
    inl = best[1]
    if inl.sum() < 8:
        return None, 0
    return kabsch(A[inl], B[inl]), int(inl.sum())


def main(argv=None) -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--imu", default=None, help="IMU source (default: the video itself)")
    ap.add_argument("--start", type=float, default=0.0, help="start time in the video [s]")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=24.0, help="decode rate; frame interval = 1/fps")
    ap.add_argument("--focal", type=float, default=280.0, help=f"equidistant focal length in px at {W} px width")
    ap.add_argument("--stream", type=int, default=0, help="video stream index (lens)")
    ap.add_argument("--video-offset", type=float, default=0.0, help="video time of IMU t=0 [s], if not 0")
    ap.add_argument("--out", default=None, help="PNG")
    args = ap.parse_args(argv)

    fr = decode_frames(args.video, args.start, args.duration, args.fps, args.stream)
    N = len(fr); cx = cy = W / 2
    mask = np.zeros((W, W), np.uint8); cv2.circle(mask, (int(cx), int(cy)), int(0.46 * W), 255, -1)

    def unproj(p):
        x = p[:, 0] - cx; y = p[:, 1] - cy; r = np.hypot(x, y); th = r / args.focal
        s = np.sin(th) / np.maximum(r, 1e-9)
        return np.c_[x * s, y * s, np.cos(th)]

    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    omega, tvis = [], []
    for i in range(N - 1):
        p0 = cv2.goodFeaturesToTrack(fr[i], 600, 0.01, 12, mask=mask)
        R = None
        if p0 is not None:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(fr[i], fr[i + 1], p0, None, **lk)
            p0b, st2, _ = cv2.calcOpticalFlowPyrLK(fr[i + 1], fr[i], p1, None, **lk)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p0 - p0b, axis=2).ravel() < 0.5)
            if ok.sum() >= 8:
                R, _ = ransac_rot(unproj(p0[ok, 0]), unproj(p1[ok, 0]))
        omega.append(Rot.from_matrix(R.T).as_rotvec() * args.fps if R is not None else [np.nan] * 3)
        tvis.append(args.start - args.video_offset + (i + 0.5) / args.fps)
    omega = np.array(omega); tvis = np.array(tvis); good = ~np.isnan(omega).any(1)
    print(f"{N} frames, {good.sum()} pairs with rotation, visual |omega| median "
          f"{np.rad2deg(np.median(np.linalg.norm(omega[good], axis=1))):.1f} deg/s")

    tg, gyr, acc, _ = load_imu(args.imu or args.video)
    cg = np.vstack([np.zeros(3), np.cumsum(gyr[:-1] * np.diff(tg)[:, None], axis=0)])

    def gyro_mean(ta, tb):
        ia = np.c_[[np.interp(ta, tg, cg[:, k]) for k in range(3)]].T
        ib = np.c_[[np.interp(tb, tg, cg[:, k]) for k in range(3)]].T
        return (ib - ia) / (tb - ta)[:, None]

    ta = tvis - 0.5 / args.fps; tb = tvis + 0.5 / args.fps
    perms = list(itertools.permutations(range(3)))

    def score(tau):
        G = gyro_mean(ta + tau, tb + tau)
        C = np.array([[np.corrcoef(omega[good, a], G[good, b])[0, 1] for b in range(3)] for a in range(3)])
        best = max(perms, key=lambda p: sum(abs(C[a, p[a]]) for a in range(3)))
        return sum(abs(C[a, best[a]]) for a in range(3)) / 3, best, C

    taus = np.arange(-0.150, 0.1501, 0.001)
    sc = np.array([score(t)[0] for t in taus]); k = int(sc.argmax()); tau = taus[k]
    s, perm, C = score(tau)
    signs = [np.sign(C[a, perm[a]]) for a in range(3)]
    print(f"best time offset tau = {tau*1000:+.0f} ms (gyro time = video time + tau), mean |corr| = {s:.3f}, at tau=0: {score(0.0)[0]:.3f}")
    print("axis mapping visual(x,y,z) <- gyro:", [f"{'+' if signs[a] > 0 else '-'}{'xyz'[perm[a]]}" for a in range(3)])
    G = gyro_mean(ta + tau, tb + tau); Gm = np.c_[[signs[a] * G[:, perm[a]] for a in range(3)]].T
    ratio = np.median(np.linalg.norm(omega[good], axis=1) / np.maximum(np.linalg.norm(Gm[good], axis=1), 1e-3))
    print(f"|omega_vis|/|omega_gyro| median = {ratio:.3f} -> suggested --focal {args.focal*ratio:.1f} (used {args.focal})")
    if args.out:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2, 1]})
        dom = int(np.argmax(np.abs(omega[good]).mean(0)))
        ax[0].plot(tvis, np.rad2deg(omega[:, dom]), lw=.9, label="video"); ax[0].plot(tvis, np.rad2deg(Gm[:, dom]), lw=.9, label="gyro (mapped)")
        ax[0].set_xlabel("t [s]"); ax[0].set_ylabel(f"angular rate axis {'xyz'[dom]} [deg/s]"); ax[0].legend()
        ax[1].plot(taus * 1000, sc); ax[1].axvline(tau * 1000, color="k", lw=.7, ls="--"); ax[1].set_xlabel("gyro time shift [ms]"); ax[1].set_ylabel("mean |corr|")
        fig.tight_layout(); fig.savefig(args.out, dpi=150); print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
