# imuscale: metric scale for SfM from the camera's embedded IMU

Companion code for *Metric Scale for Insta360 Structure-from-Motion from the Camera's Embedded IMU*.
Given a COLMAP model in arbitrary units and the gyroscope/accelerometer record that Insta360 X5/X6
cameras store at 1 kHz in the `.insv` file, it recovers the metric scale, the gravity direction and
the camera-IMU rotation, and reports quality diagnostics. It also contains the sensor
characterisation tools used in the paper.

What is **not** here: the reconstruction pipeline (any COLMAP-format model works), the survey data
and the reference measurements (LiDAR, fiducials, smartphone). A set of production surveys is
released separately; see the paper.

## Install

```
pip install -r requirements.txt
```
`telemetry-parser` is the Gyroflow binding that reads the IMU from `.insv`. `ffmpeg` on PATH is
needed only for `video_gyro_sync`.

## Scale a model

```
python -m imuscale.scale --sparse model/sparse/0 --imu recording.insv --period 0.1 --out scale.json
```

* `--period`: nominal seconds per unit of the frame index found in the image file names (frames
  extracted at 10 fps: 0.1, at 5 fps: 0.2). The value is
  refined from the data, within ±0.4%, together with the time offset (searched within `--offset-range`,
  default ±1 s). If you have per-frame timestamps on the IMU clock, pass `--frame-times times.csv`
  (`image_name,t_s`) instead.
* `--name-regex` / `--index-base`: how to parse the frame index from file names (default: last
  integer, 0-based).
* `--group`: for multi-camera rigs, which camera to use, given as the file-name template with the index
  replaced by `#` (e.g. `lens0_#.jpg`); default: the group with most images.
* `--window`: runs are cut into windows of at most this many seconds before the per-run estimation
  (default 8.5; 0 keeps the contiguous runs as they are).
* `--imu-offset`: IMU time of video t=0, in seconds. Needed when the video was trimmed without
  rewriting the IMU record (e.g. cut to a common start with an edit list): the IMU keeps the original
  timeline, so pass the cut point.
* `--write-model DIR`: write a scaled and levelled copy (gravity onto `--down`, default -Z; first frame
  at the origin), only if the quality criteria pass or `--force`.

Exit code 0: estimate accepted; 2: computed but a quality criterion failed (see `quality.reasons`).

### Rigs of several cameras

Each camera has its own IMU: run `scale` once per camera, with that camera's recording as `--imu`,
one of its lenses as `--group` and its cut point as `--imu-offset`. The per-camera scales must agree,
which is a useful check. To measure the residual time offset and the clock drift between two cameras
on the same rigid rig from their gyroscopes (no knowledge of the mounting needed):

```
python -m imuscale.rig_sync cam_a.insv cam_b.insv --imu-offset A B --out rig_sync.png
```

It cross-correlates the angular-rate magnitudes over the whole overlap and in 20 s windows, and fits a
line to the windowed lags (offset at t=0, drift in ppm). `video_gyro_sync` run on each camera tells
whether the camera-IMU delay is the same on both.

### Method in brief

1. **Time map and camera-IMU rotation.** Frame time is `t = a*idx + b`. Relative rotations between
   consecutive SfM frames must equal the gyroscope rotation integrated over the same interval,
   expressed through a fixed rotation `R_CB` (hand-eye, Kabsch on rotation vectors). `a`, `b` are found
   by minimising the median angular residual in two stages (coarse offset grid; fine period grid with
   the offset re-centred and re-searched), then a local simplex. A wrong period is detectable without
   ground truth: the optimal offset drifts between the first and last quarter of the survey.
2. **Preintegration** of the accelerometer over each frame interval, with first-order bias Jacobians.
3. **Linear system** in scale, gravity, accelerometer bias and per-frame velocities, solved through
   sparse normal equations; a robust pass drops intervals with residual above five times the median.
4. **Per-run estimation.** Velocities are chained only within contiguous runs of frames (gap < 1.2 s),
   and runs are cut into consecutive windows of at most 8.5 s (`--window`), so that a recording
   without gaps still gives several independent estimates. Much shorter windows bias the scale low.
   Solving the whole survey jointly, with gravity and bias shared, is biased high (3.4% on ten surveys
   with external reference); each run is therefore solved separately and the survey scale is the mean
   of the per-run scales. The joint solution is reported as a diagnostic.

### Quality criteria

All criteria are evaluated on per-run quantities. A survey is rejected when: runs whose recovered
gravity magnitude falls outside 9.3–10.3 m/s² hold more than half of the intervals; the per-run means
of the two halves of the survey disagree by more than 6%; the median hand-eye residual exceeds 2.5°;
the time offset drifts by more than 60 ms along the survey; the standard error of the per-run mean
exceeds 2.5%; fewer than 4 usable runs or 30 intervals remain; the scale is negative.

## Sensor characterisation

```
python -m imuscale.insv recording.insv imu.csv            # read/inspect/export the IMU
python -m imuscale.sensor_noise rec_x5.insv rec_x6.insv --out asd.png   # Welch ASD, noise floor above 100 Hz
python -m imuscale.video_gyro_sync recording.insv --start 60 --duration 20 --out sync.png   # time offset, axis map
python -m imuscale.static_bias static.insv --local-g 9.806 --out bias.png   # accel scale factor, bias drift
```

## Experiments

```
python scripts/window_ablation.py --sparse model/sparse/0 --imu rec.insv --period 0.1 --truth 0.987
python scripts/sim3_invariance.py --sparse model/sparse/0 --imu rec.insv --period 0.1
python tests/test_synthetic.py      # synthetic trajectory + IMU: recovers scale, gravity, R_CB
python tests/test_rig_sync.py       # two synthetic gyros: recovers offset and drift
```

## Conventions

`gravity_model_frame` is the gravity acceleration vector in the model frame, pointing down.
`R_cam_imu` maps IMU-frame vectors to camera-frame vectors. The IMU record read from `.insv` is on
the video clock; on the recordings we examined the residual camera-IMU offset is within ±5 ms.

## License

To be decided before publication.
