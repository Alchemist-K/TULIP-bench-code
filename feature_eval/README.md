# TULIP-Bench feature-error evaluation

Reproduces the clinical feature-error columns of Table 2 for both tasks, from
precomputed feature CSVs. Pose-level MPJPE / PA-MPJPE are computed by the
scripts in `pose_eval/` from the released pose pickles.

## Structure

```
eval_public_gait.py                 gait: MPJPE / PA-MPJPE + stride-feature error
eval_public_fist.py                 fist: MPJPE / PA-MPJPE + window-feature error
feature_csvs/
  gait/{GT,SAM3D,WHAM,MAGFPre,MAGFTulip}/features_stride.csv
  fist/{GT,SAM3D,Videopose3D}/features_window_making_a_fist_{Left,Right}.csv
```

One folder per pose source. Each CSV carries a `subject` column
(`Neurips_Sub{n}`) and a per-unit index (`stride_idx` for gait, `window_idx` for
fist); the scripts average per unit within subject, then across subjects.

## What each half consumes

- Pose metrics (MPJPE, PA-MPJPE): the released pose pickles. Runs directly
  against the dataset.
- Feature error (ST err %, kinematic err %): the feature CSVs in this folder.
  These are precomputed so the numbers can be regenerated without the
  third-party estimators.

## Selected Table 2 features

- Gait: `stride_length` (ST error), `arm2arm_ROM` (kinematic error)
- Fist: `speed_fingertip_mean` (ST error), `amp_mcp_range_rad_mean` (kinematic error)

## Running

```
python eval_public_gait.py    # uses feature_csvs/gait/ by default
python eval_public_fist.py    # uses feature_csvs/fist/ by default
```

Both accept command-line overrides for input directories and output paths; see
the argument defaults at the bottom of each script.
