"""Command line: metric scale of a COLMAP model from the camera IMU.

    python -m imuscale.scale --sparse model/sparse/0 --imu recording.insv --period 0.1 --out scale.json

`--period` is the nominal duration, in seconds, of one unit of the frame index in the image names
(e.g. frames extracted at 10 fps -> 0.1). It is refined from the data together with the offset.
Alternatively give per-frame timestamps with `--frame-times times.csv` (columns: image_name,t_s).
For a rig of several cameras, run once per camera with its own `--imu`, `--group` and `--imu-offset`.

With `--write-model DIR` a scaled and levelled copy of the model is written (gravity onto -Z by default,
first frame at the origin), only if the quality criteria pass or `--force` is given.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from .colmap_io import load_frames, write_scaled_model
from .estimator import InertialEstimator, estimate, log
from .insv import load_imu


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sparse", type=Path, required=True, help="COLMAP model directory (cameras/images/points3D .bin or .txt)")
    ap.add_argument("--imu", type=Path, required=True, help=".insv recording or CSV t_s,gx,gy,gz,ax,ay,az (deg/s, m/s^2)")
    ap.add_argument("--imu-offset", type=float, default=0.0,
                    help="IMU time of video t=0 [s], subtracted from the IMU clock (e.g. the cut point of a trimmed video)")
    ap.add_argument("--period", type=float, default=None, help="nominal seconds per frame-index unit")
    ap.add_argument("--offset-range", type=float, nargs=2, default=(-1.0, 1.0), metavar=("MIN", "MAX"),
                    help="offset search range in seconds (default -1 1)")
    ap.add_argument("--frame-times", type=Path, default=None, help="CSV image_name,t_s with per-frame times on the IMU clock")
    ap.add_argument("--group", default=None, help="file-name template of the camera to use, e.g. 'lens0_#.jpg' (default: the one with most images)")
    ap.add_argument("--name-regex", default=r"(\d+)", help="regex whose last integer match in the file name is the frame index")
    ap.add_argument("--index-base", type=int, default=0, help="subtract this from the parsed index (1 for 1-based names)")
    ap.add_argument("--window", type=float, default=8.5,
                    help="cut contiguous runs into blocks of at most this many seconds (0: no cut; default 8.5)")
    ap.add_argument("--max-frames", type=int, default=2500, help="subsample frames above this count")
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None, help="result JSON")
    ap.add_argument("--write-model", type=Path, default=None, help="write scaled + levelled model copy here")
    ap.add_argument("--down", type=float, nargs=3, default=(0, 0, -1), help="direction gravity is mapped onto (default 0 0 -1)")
    ap.add_argument("--force", action="store_true", help="write the model even if quality criteria fail")
    args = ap.parse_args(argv)
    t0 = time.time()

    if args.frame_times is None and args.period is None:
        ap.error("give --period or --frame-times")
    frame_times = None
    if args.frame_times is not None:
        with open(args.frame_times) as fh:
            frame_times = {r[0]: float(r[1]) for r in csv.reader(fh) if r and not r[0].startswith(("#", "image_name"))}

    tg, gyr, acc, meta = load_imu(args.imu)
    tg = tg - args.imu_offset
    log(f"IMU: {len(tg)} samples, {tg[0]:.2f}..{tg[-1]:.2f} s, {len(tg)/(tg[-1]-tg[0]):.0f} Hz {meta}")
    rec, cam, IDX, Rwc, P, names = load_frames(args.sparse, args.group, args.name_regex, args.index_base, frame_times)
    if len(IDX) > args.max_frames:
        step = int(np.ceil(len(IDX) / args.max_frames))
        IDX, Rwc, P = IDX[::step], Rwc[::step], P[::step]
        log(f"subsampling 1 frame in {step}: {len(IDX)} frames")

    est = InertialEstimator(tg, gyr, acc, IDX, Rwc, P, window_s=args.window)
    if frame_times is not None:
        est.set_time_map(1.0, 0.0)
        est.fit_time_map(1.0, offset_range=(-0.2, 0.2), period_tol=0.0005)   # only a small offset/clock refinement
    else:
        est.fit_time_map(args.period, offset_range=tuple(args.offset_range))
    result = estimate(est, bootstrap=args.bootstrap)
    result.update({"sparse": str(args.sparse), "imu": str(args.imu), "group": cam, "elapsed_s": round(time.time() - t0, 1)})

    if args.write_model is not None:
        if result["quality"]["ok"] or args.force:
            R, t = write_scaled_model(rec, args.write_model, result["scale"], np.asarray(result["gravity_model_frame"]),
                                      down=args.down, origin=P[0])
            result["written_model"] = {"dir": str(args.write_model), "sim3": {"scale": result["scale"], "R": R.tolist(), "t": t.tolist()}}
            log(f"wrote scaled model to {args.write_model}")
        else:
            log("quality criteria failed: model not written (use --force to override)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
        log(f"result in {args.out}")
    else:
        print(json.dumps({k: result[k] for k in ("scale", "run_scale_sem_pct", "gravity_norm", "handeye_median_deg", "quality")}, indent=1))
    return 0 if result["quality"]["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
