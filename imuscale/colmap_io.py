"""Load camera poses from a COLMAP model and optionally write a scaled, levelled copy."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def load_frames(sparse_dir: str | Path, group: str | None = None, name_regex: str = r"(\d+)",
                index_base: int = 0, frame_times: dict | None = None):
    """-> (reconstruction, group, IDX, Rwc (n,3,3), P (n,3), names).

    The frame index is the LAST integer matched by `name_regex` in the image file name, minus
    `index_base`. Images are grouped by the file-name template obtained by replacing that integer with
    '#' (e.g. a rig gives 'lens0_#.jpg' and 'lens1_#.jpg'), so that one physical camera is used; by
    default the group with most registered images. If `frame_times` (name -> seconds) is given it
    replaces the index and the group is the COLMAP camera_id."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(sparse_dir))
    rx = re.compile(name_regex)
    per_cam: dict[str, list] = {}
    for im in rec.images.values():
        if not im.has_pose:
            continue
        name = Path(im.name).name
        if frame_times is not None:
            if im.name not in frame_times and name not in frame_times:
                continue
            idx = float(frame_times.get(im.name, frame_times.get(name)))
            key = f"camera {im.camera_id}"
        else:
            ms = list(rx.finditer(name))
            if not ms:
                continue
            m = ms[-1]
            idx = int(m.group(m.lastindex or 0)) - index_base
            key = name[:m.start()] + "#" + name[m.end():]
        cfw = im.cam_from_world()
        R_cw = np.asarray(cfw.rotation.matrix(), float)
        t = np.asarray(cfw.translation, float)
        per_cam.setdefault(key, []).append((idx, R_cw.T, -R_cw.T @ t, im.name))
    if not per_cam:
        raise RuntimeError(f"no registered image with a frame index in {sparse_dir}")
    if group is None or group not in per_cam:
        group = max(per_cam, key=lambda k: len(per_cam[k]))
    frames = sorted(per_cam[group], key=lambda x: x[0])
    if len(frames) < 10:
        raise RuntimeError(f"only {len(frames)} frames in group {group}")
    camera_id = group
    IDX = np.array([f[0] for f in frames], float)
    Rwc = np.array([f[1] for f in frames]); P = np.array([f[2] for f in frames])
    print(f"[imuscale] SfM: {len(rec.images)} images, using {len(frames)} frames of group {group!r} "
          f"(idx {IDX[0]:g}..{IDX[-1]:g}), groups {{{', '.join(f'{k}: {len(v)}' for k, v in sorted(per_cam.items()))}}}")
    return rec, camera_id, IDX, Rwc, P, [f[3] for f in frames]


def rotation_from_vectors(u, v):
    from scipy.spatial.transform import Rotation as Rot
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
    c = float(np.clip(u @ v, -1, 1)); w = np.cross(u, v); sn = np.linalg.norm(w)
    if sn < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.cross(u, [1, 0, 0]); axis = np.cross(u, [0, 1, 0]) if np.linalg.norm(axis) < 1e-6 else axis
        return Rot.from_rotvec(np.pi * axis / np.linalg.norm(axis)).as_matrix()
    return Rot.from_rotvec(w / sn * np.arctan2(sn, c)).as_matrix()


def write_scaled_model(rec, out_dir: str | Path, scale: float, gravity: np.ndarray | None, down=(0, 0, -1),
                       origin: np.ndarray | None = None):
    """Write a copy of `rec` scaled by `scale`, rotated so that `gravity` maps onto `down`, and with
    `origin` (model units) moved to zero. Default `down` is -Z (Z up)."""
    import pycolmap

    R = np.eye(3) if gravity is None else rotation_from_vectors(np.asarray(gravity, float), np.asarray(down, float))
    t = np.zeros(3) if origin is None else -scale * (R @ np.asarray(origin, float))
    sim = pycolmap.Sim3d(float(scale), pycolmap.Rotation3d(R), t)
    rec.transform(sim)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rec.write_binary(str(out))
    return R, t
