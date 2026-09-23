"""Linear visual-inertial alignment of an SfM trajectory to IMU measurements.

Given camera poses (R_WC_k, C_k) in arbitrary model units with frame indices, and gyroscope and
accelerometer samples, the estimator
  1. finds the affine time map t = a*idx + b and the camera-IMU rotation R_CB by a hand-eye fit of
     SfM relative rotations against integrated gyroscope rotations;
  2. preintegrates the accelerometer over each frame interval (Lupton & Sukkarieh 2012;
     Forster et al. 2017, first order in the bias);
  3. solves the linear system in (scale, gravity, accelerometer bias, per-frame velocities)
     (Martinelli 2014; Mur-Artal & Tardos 2017; Qin et al. 2018), in batch over a survey;
  4. estimates the scale on each contiguous run of frames separately and reports the mean of the
     per-run scales. The joint solution over the whole survey, with gravity and bias shared, is
     biased high (3.4% on ten surveys with external reference) and is kept only as a diagnostic.

Conventions: `gravity` is the gravity acceleration vector expressed in the model frame (it points
DOWN). World-frame acceleration of the body is a_W = R_WB a_B + g.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

G_NOMINAL = 9.80665
# quality thresholds (see README, "Quality criteria")
Q_GNORM = (9.3, 10.3)         # accepted gravity magnitude of a run, m/s^2
Q_HALVES_PCT = 6.0            # max disagreement between first/second half per-run means
Q_HANDEYE_DEG = 2.5           # max median hand-eye residual per interval
Q_MIN_INTERVALS = 30
Q_MIN_SEGMENTS = 4            # runs needed for the per-run mean
Q_SEG_SEM_PCT = 2.5           # max standard error of the per-run mean
Q_TIME_DRIFT_MS = 60.0        # max drift of the optimal offset between start and end of the survey


def log(msg: str) -> None:
    print(f"[imuscale] {msg}", flush=True)


def kabsch(A, B):
    """Rotation R minimising |B - A R^T|, i.e. B ~ R A row-wise."""
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


class InertialEstimator:
    """tg, gyr [rad/s], acc [m/s^2]: IMU samples. IDX: frame indices (float ok). Rwc: (n,3,3) rotations
    camera->model. P: (n,3) camera centres in model units."""

    def __init__(self, tg, gyr, acc, IDX, Rwc, P, max_gap_s=1.2, min_run=5):
        self.tg, self.gyr, self.acc = np.asarray(tg, float), np.asarray(gyr, float), np.asarray(acc, float)
        self.IDX, self.Rwc, self.P = np.asarray(IDX, float), np.asarray(Rwc, float), np.asarray(P, float)
        self.max_gap_s, self.min_run = max_gap_s, min_run
        dqm = Rot.from_rotvec(self.gyr[:-1] * np.diff(self.tg)[:, None]).as_matrix()
        Rc = np.empty((len(self.tg), 3, 3)); Rc[0] = np.eye(3)
        for i in range(len(dqm)):                      # cumulative gyro orientation
            Rc[i + 1] = Rc[i] @ dqm[i]
        self.Rcum = Rc
        self.a = self.b = self.R_CB = self.handeye_deg = self.T = None
        self._pc = {}

    # ------------------------------------------------------------------ hand-eye / time map
    def gyro_rot(self, t0, t1):
        tg = self.tg
        i0 = min(max(np.searchsorted(tg, t0), 0), len(tg) - 1)
        i1 = min(max(np.searchsorted(tg, t1), 0), len(tg) - 1)
        return self.Rcum[i0].T @ self.Rcum[i1]

    def handeye(self, a, b):
        """Rotation vectors of consecutive-frame SfM rotations vs gyro rotations under t = a*idx + b.
        -> (R_CB, median residual [deg], residuals)."""
        IDX, Rwc = self.IDX, self.Rwc
        rc, rb = [], []
        for k in range(len(IDX) - 1):
            if (IDX[k + 1] - IDX[k]) * a > self.max_gap_s:
                continue
            t0, t1 = a * IDX[k] + b, a * IDX[k + 1] + b
            if t0 < self.tg[0] - 0.5 or t1 > self.tg[-1] + 0.5:
                continue
            rc.append(Rot.from_matrix(Rwc[k].T @ Rwc[k + 1]).as_rotvec())
            rb.append(Rot.from_matrix(self.gyro_rot(t0, t1)).as_rotvec())
        if len(rc) < 10:
            return None, np.inf, np.array([])
        rc, rb = np.array(rc), np.array(rb)
        R = kabsch(rb, rc)
        err = np.rad2deg(np.linalg.norm(rc - (R @ rb.T).T, axis=1))
        return R, float(np.median(err)), err

    def fit_time_map(self, period, offset_range=(-1.0, 1.0), period_tol=0.004):
        """Two-stage search of (a, b) around the nominal `period`: coarse offset grid, fine period grid
        (+-period_tol) with the offset re-centred and searched within offset_range, then a local simplex.
        The offset range must be wide enough to absorb the re-centring shift period_tol*a*span/2."""
        best = None
        for b in np.arange(offset_range[0], offset_range[1] + 1e-9, 0.05):
            _, e, _ = self.handeye(period, b)
            if best is None or e < best[0]:
                best = (e, period, b)
        e, a, b = best
        log(f"coarse time map: period {a:.5f} s, offset {b:+.3f} s, hand-eye median {e:.2f} deg")
        span = float(self.IDX[-1] - self.IDX[0])
        for da in np.arange(-period_tol, period_tol + 1e-9, period_tol / 16):
            a2 = a * (1 + da)
            b_c = b - da * a * span / 2                  # keep the centre of the survey fixed in time
            for db in np.arange(offset_range[0], offset_range[1] + 1e-9, 0.02):
                _, e2, _ = self.handeye(a2, b_c + db)
                if e2 < e:
                    e, best = e2, (e2, a2, b_c + db)
        _, a, b = best
        res = minimize(lambda x: self.handeye(x[0], x[1])[1], x0=[a, b], method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-4, "maxiter": 200,
                                "initial_simplex": [[a, b], [a * 1.0005, b], [a, b + 0.02]]})
        if np.isfinite(res.fun) and res.fun <= e and abs(res.x[0] / a - 1) < 0.02:
            a, b, e = float(res.x[0]), float(res.x[1]), float(res.fun)
        R_CB, e, errs = self.handeye(a, b)
        if R_CB is None:
            raise RuntimeError("hand-eye failed: no valid frame pairs (wrong period or IMU/video mismatch?)")
        log(f"time map: t = {a:.5f}*idx {b:+.3f} s ; hand-eye median {e:.2f} deg "
            f"(p90 {np.percentile(errs, 90):.2f}, {len(errs)} pairs)")
        self.a, self.b, self.R_CB, self.handeye_deg = a, b, R_CB, e
        self.T = a * self.IDX + b
        self._pc = {}
        return a, b, R_CB, e

    def set_time_map(self, a, b):
        """Use a given time map (e.g. per-frame timestamps with a=1) and fit only R_CB."""
        R_CB, e, errs = self.handeye(a, b)
        if R_CB is None:
            raise RuntimeError("hand-eye failed with the given time map")
        self.a, self.b, self.R_CB, self.handeye_deg = float(a), float(b), R_CB, e
        self.T = a * self.IDX + b
        self._pc = {}
        return R_CB, e

    def time_drift_ms(self):
        """Optimal offset on the first and last quarter of the survey; a difference means the period
        is wrong (or the video has gaps) and preintegration uses IMU from the wrong moment."""
        n = len(self.IDX) - 1
        if n < 40:
            return float("nan")
        T = self.T
        out = []
        for ks in (range(0, n // 4), range(3 * n // 4, n)):
            rc, kk = [], []
            for k in ks:
                if T[k + 1] - T[k] <= self.max_gap_s:
                    rc.append(Rot.from_matrix(self.Rwc[k].T @ self.Rwc[k + 1]).as_rotvec()); kk.append(k)
            if len(kk) < 10:
                return float("nan")
            rc = np.array(rc)
            best = None
            for db in np.arange(-0.5, 0.501, 0.01):
                rb = np.array([Rot.from_matrix(self.gyro_rot(T[k] + db, T[k + 1] + db)).as_rotvec() for k in kk])
                err = float(np.median(np.linalg.norm(rc - (self.R_CB @ rb.T).T, axis=1)))
                if best is None or err < best[0]:
                    best = (err, db)
            out.append(best[1])
        return float((out[1] - out[0]) * 1000)

    # ------------------------------------------------------------------ preintegration / linear system
    def preint(self, k):
        """Interval k -> (dt, dp, dv, J, K): double/single integrals of R_WB a and of R_WB (for the bias)."""
        if k in self._pc:
            return self._pc[k]
        tg, gyr, acc = self.tg, self.gyr, self.acc
        t0, t1 = self.T[k], self.T[k + 1]
        idx = np.where((tg >= t0) & (tg < t1))[0]
        R = self.Rwc[k] @ self.R_CB
        v = np.zeros(3); p = np.zeros(3); K = np.zeros((3, 3)); J = np.zeros((3, 3)); tprev = t0
        if len(idx) == 0:
            idx = np.array([min(max(np.searchsorted(tg, t0), 0), len(tg) - 1)])
        for i in idx:
            dt = tg[i] - tprev; tprev = tg[i]
            a_w = R @ acc[i]
            p += v * dt + 0.5 * a_w * dt * dt; v += a_w * dt
            J += K * dt + 0.5 * R * dt * dt; K += R * dt
            R = R @ Rot.from_rotvec(gyr[i] * dt).as_matrix()
        dt = t1 - tprev
        a_w = R @ acc[idx[-1]]
        p += v * dt + 0.5 * a_w * dt * dt; v += a_w * dt
        J += K * dt + 0.5 * R * dt * dt; K += R * dt
        self._pc[k] = (t1 - t0, p, v, J, K)
        return self._pc[k]

    def make_runs(self):
        """Contiguous runs: maximal sequences of intervals shorter than max_gap_s."""
        runs, cur = [], []
        for k in range(len(self.T) - 1):
            if self.T[k + 1] - self.T[k] <= self.max_gap_s:
                cur.append(k)
            else:
                if len(cur) >= self.min_run:
                    runs.append(cur)
                cur = []
        if len(cur) >= self.min_run:
            runs.append(cur)
        return runs

    def solve(self, runs):
        """Joint linear solve over the given runs -> (scale, gravity, accel bias, position rms)."""
        P = self.P
        nint = sum(len(r) for r in runs); nv = sum(len(r) + 1 for r in runs)
        rows, cols, vals = [], [], []
        bb = np.zeros(6 * nint); row = 0; voff = 7
        I3 = np.eye(3)

        def put(r0, c0, M):
            M = np.atleast_2d(M)
            for i, j in zip(*np.nonzero(M)):
                rows.append(r0 + i); cols.append(c0 + j); vals.append(M[i, j])

        for rr in runs:
            for j, k in enumerate(rr):
                dt, Dp, Dv, J, K = self.preint(k); r = row
                # s*(C_{k+1}-C_k) - v_k dt - 1/2 g dt^2 + J b = Dp
                put(r, 0, (P[k + 1] - P[k]).reshape(3, 1)); put(r, 1, -0.5 * dt * dt * I3); put(r, 4, J)
                put(r, voff + 3 * j, -dt * I3); bb[r:r + 3] = Dp
                r += 3
                # v_{k+1} - v_k - g dt + K b = Dv
                put(r, 1, -dt * I3); put(r, 4, K); put(r, voff + 3 * j, -I3); put(r, voff + 3 * j + 3, I3); bb[r:r + 3] = Dv
                row += 6
            voff += 3 * (len(rr) + 1)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(6 * nint, 7 + 3 * nv))
        x = spl.spsolve((A.T @ A).tocsc(), A.T @ bb)
        r = A @ x - bb
        rms = float(np.sqrt((r[0::6] ** 2 + r[1::6] ** 2 + r[2::6] ** 2).mean()))
        return float(x[0]), x[1:4], x[4:7], rms

    def interval_residuals(self, runs, s, g, ba):
        P, T = self.P, self.T
        res = {}
        for rr in runs:
            rs = set(rr)
            for k in rr:
                dt, Dp, Dv, J, K = self.preint(k)
                if k > 0 and (k - 1) in rs:
                    vk = s * (P[k + 1] - P[k - 1]) / (T[k + 1] - T[k - 1])
                else:
                    vk = s * (P[k + 1] - P[k]) / dt
                res[k] = np.linalg.norm(s * (P[k + 1] - P[k]) - (vk * dt + 0.5 * g * dt * dt + Dp - J @ ba))
        return res

    def robust_solve(self):
        """Joint solve, then drop intervals with residual > 5x median (misregistered SfM segments) and
        re-solve. -> (runs, s, g, ba, rms, n_removed)."""
        runs = self.make_runs()
        nint = sum(map(len, runs))
        log(f"contiguous runs: {len(runs)}, intervals {nint}")
        if nint < 10:
            raise RuntimeError(f"too few contiguous intervals ({nint})")
        s, g, ba, rms = self.solve(runs)
        res = self.interval_residuals(runs, s, g, ba)
        vals = np.array(list(res.values()))
        thr = max(5 * np.median(vals), 0.15)
        bad = {k for k, v in res.items() if v > thr}
        if bad:
            new_runs = []
            for rr in runs:
                cur = []
                for k in rr:
                    if k in bad:
                        if len(cur) >= self.min_run:
                            new_runs.append(cur)
                        cur = []
                    else:
                        cur.append(k)
                if len(cur) >= self.min_run:
                    new_runs.append(cur)
            if sum(map(len, new_runs)) >= 10:
                runs = new_runs
                s, g, ba, rms = self.solve(runs)
            log(f"robust pass: removed {len(bad)} intervals with residual > {thr*1000:.0f} mm "
                f"(median {np.median(vals)*1000:.0f} mm); runs now {len(runs)}")
        return runs, s, g, ba, rms, len(bad)

    def scale_per_run(self, runs, min_int=8):
        """Independent solve on every run. Runs shorter than min_int intervals, or whose gravity magnitude
        falls outside Q_GNORM, are discarded.
        -> (scales, weights = n intervals, gravity per run (n,3), indices of used runs, n intervals in
            runs discarded for gravity)."""
        vals, w, gs, idx, n_bad_g = [], [], [], [], 0
        for i, rr in enumerate(runs):
            if len(rr) < min_int:
                continue
            try:
                sr, gr, _, _ = self.solve([rr])
            except Exception:
                continue
            if not (Q_GNORM[0] <= float(np.linalg.norm(gr)) <= Q_GNORM[1]):
                n_bad_g += len(rr)
                continue
            if 0.3 < sr < 3.0:
                vals.append(float(sr)); w.append(len(rr)); gs.append(np.asarray(gr, float)); idx.append(i)
        return np.array(vals), np.array(w, float), np.array(gs).reshape(-1, 3), np.array(idx, int), n_bad_g


def estimate(est: InertialEstimator, bootstrap: int = 100) -> dict:
    """Full estimate after the time map is set: robust joint solve, per-run scales, quality criteria.
    Returns a JSON-serialisable dict; `quality.ok` is False when a criterion fails."""
    runs, s_joint, g_joint, ba, rms, n_removed = est.robust_solve()
    run_scales, run_w, run_g, run_idx, n_bad_g = est.scale_per_run(runs)
    n_seg = int(len(run_scales))
    nint = sum(map(len, runs))
    T = est.T
    L = float(np.linalg.norm(np.diff(est.P, axis=0), axis=1).sum())

    if n_seg >= Q_MIN_SEGMENTS:
        s = float(run_scales.mean())
        seg_sd = float(run_scales.std(ddof=1) / s * 100)
        seg_sem = seg_sd / np.sqrt(n_seg)
        g = (run_g * run_w[:, None]).sum(0) / run_w.sum()
        log(f"per-run scale: {n_seg} runs, mean {s:.4f}, dispersion {seg_sd:.1f}%, sem {seg_sem:.1f}% | "
            f"joint fit {s_joint:.4f} ({100*(s_joint/s-1):+.1f}%)")
    else:
        s, seg_sd, seg_sem, g = float(s_joint), float("nan"), float("nan"), g_joint
        log(f"per-run scale: only {n_seg} usable runs, falling back to the joint fit {s_joint:.4f}")
    gnorm = float(np.linalg.norm(g))

    half_a = half_b = float("nan")
    if n_seg >= 2 * Q_MIN_SEGMENTS:
        h = len(runs) // 2
        ma, mb = run_idx < h, run_idx >= h
        if ma.sum() >= 2 and mb.sum() >= 2:
            half_a, half_b = float(run_scales[ma].mean()), float(run_scales[mb].mean())
    boot_sd = float("nan")
    if len(runs) >= 3 and bootstrap > 0:
        rng = np.random.default_rng(0); bs = []
        for _ in range(bootstrap):
            pick = [runs[i] for i in rng.integers(0, len(runs), len(runs))]
            try:
                bs.append(est.solve(pick)[0])
            except Exception:
                pass
        if len(bs) > 5:
            boot_sd = float(np.std(bs) / s * 100)
    drift_ms = est.time_drift_ms()
    log(f"|g| {gnorm:.3f} (joint {np.linalg.norm(g_joint):.3f}) | halves {half_a:.4f}/{half_b:.4f} | "
        f"joint-fit bootstrap sd {boot_sd:.2f}% | time drift {drift_ms:+.0f} ms | trajectory {L*s:.1f} m in {T[-1]-T[0]:.0f} s")

    reasons = []
    if n_seg >= Q_MIN_SEGMENTS:
        if n_bad_g > 0.5 * nint:
            reasons.append(f"gravity outside {Q_GNORM} on {n_bad_g}/{nint} intervals")
    elif not (Q_GNORM[0] <= gnorm <= Q_GNORM[1]):
        reasons.append(f"|g| {gnorm:.2f} outside {Q_GNORM}")
    if np.isfinite(half_a) and np.isfinite(half_b) and half_b > 0 and abs(half_a / half_b - 1) * 100 > Q_HALVES_PCT:
        reasons.append(f"halves disagree {half_a:.3f} vs {half_b:.3f}")
    if est.handeye_deg > Q_HANDEYE_DEG:
        reasons.append(f"hand-eye {est.handeye_deg:.2f} deg > {Q_HANDEYE_DEG}")
    if np.isfinite(drift_ms) and abs(drift_ms) > Q_TIME_DRIFT_MS:
        reasons.append(f"time drift {drift_ms:+.0f} ms > {Q_TIME_DRIFT_MS} (wrong frame period or video gaps)")
    if n_seg >= Q_MIN_SEGMENTS and np.isfinite(seg_sem) and seg_sem > Q_SEG_SEM_PCT:
        reasons.append(f"runs disagree: sem {seg_sem:.1f}% > {Q_SEG_SEM_PCT}%")
    if n_seg < Q_MIN_SEGMENTS:
        reasons.append(f"only {n_seg} usable runs (min {Q_MIN_SEGMENTS})")
    if nint < Q_MIN_INTERVALS:
        reasons.append(f"only {nint} intervals")
    if s <= 0:
        reasons.append("negative scale")
    log("quality OK" if not reasons else "quality INSUFFICIENT: " + "; ".join(reasons))

    return {
        "scale": s, "scale_method": "mean of per-run scales", "scale_joint_fit": float(s_joint),
        "n_runs_used": n_seg, "run_scales": run_scales.tolist(), "run_scale_sd_pct": seg_sd, "run_scale_sem_pct": seg_sem,
        "joint_fit_bootstrap_sd_pct": boot_sd, "scale_half_a": half_a, "scale_half_b": half_b,
        "gravity_model_frame": np.asarray(g).tolist(), "gravity_norm": gnorm, "gravity_norm_joint_fit": float(np.linalg.norm(g_joint)),
        "accel_bias_joint_fit": np.asarray(ba).tolist(), "pos_rms_mm": rms * 1000,
        "handeye_median_deg": est.handeye_deg, "R_cam_imu": np.asarray(est.R_CB).tolist(),
        "time_map": {"period_s": est.a, "offset_s": est.b, "drift_ms": drift_ms},
        "n_frames": int(len(est.IDX)), "n_intervals": int(nint), "n_runs": len(runs), "n_intervals_removed": int(n_removed),
        "duration_s": float(T[-1] - T[0]), "trajectory_model_units": L, "trajectory_m": L * s,
        "quality": {"ok": not reasons, "reasons": reasons},
    }
