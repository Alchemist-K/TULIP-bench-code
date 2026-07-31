# TULIP-Bench evaluation code

Reference code for the preprocessing, pose evaluation, feature extraction,
clinical feature evaluation, UPDRS prediction, and treatment-response analyses
in the paper. The dataset is released separately; see the paper for access.

## Setup

```bash
pip install -r requirements.txt
```

Each script has a configuration block or argument defaults at the top; set the
dataset paths there. Subjects are keyed `Neurips_Sub{n}` (observational) and
`Neurips_DBS_Sub{n}` (DBS) throughout.

## Layout

```
config.py                       shared dataset paths and subject naming

preprocessing/                  segmentation and normalisation
pose_eval/                      MPJPE / PA-MPJPE for gait and fist
features/                       clinical feature extraction
feature_eval/                   feature-error evaluation
  eval_public_gait.py           gait, from feature_csvs/gait/
  eval_public_fist.py           fist, from feature_csvs/fist/
  feature_csvs/
    gait/{GT,SAM3D,WHAM,MAGFPre,MAGFTulip}/features_stride.csv
    fist/{GT,SAM3D,Videopose3D}/features_window_making_a_fist_{Left,Right}.csv
updrs/                          UPDRS severity prediction (leave-one-subject-out)
task2/                          monocular pose (HaMeR, WiLoR, and others)
task3/                          DBS treatment-response analysis
```

## Notes

- UPDRS prediction uses leave-one-subject-out, so train and test folds are
  patient-disjoint by construction.
- Subject ids are derived from directory or pickle-key contents; no identifiers
  are embedded in the code.
- A GPU is needed for `task2/run_monocular_hand.py` and
  `task3/ssl_cohort_bridge.py`; other scripts run on CPU.
