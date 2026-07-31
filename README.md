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

## Which script reproduces which result

| Paper result | Script | Reads |
| --- | --- | --- |
| Table 2, gait MPJPE / PA-MPJPE | `pose_eval/eval_gait.py` | released gait pose pickles |
| Table 2, fist MPJPE / PA-MPJPE | `pose_eval/eval_fist.py` | released fist pose pickles |
| Table 2, gait feature error (ST %, kinematic %) | `feature_eval/eval_public_gait.py` | `feature_eval/feature_csvs/gait/` |
| Table 2, fist feature error (ST %, kinematic %) | `feature_eval/eval_public_fist.py` | `feature_eval/feature_csvs/fist/` |
| Table 4, monocular pose sources | `task2/run_monocular_hand.py` | released videos + pose pickles |
| UPDRS prediction (Task 1) | `updrs/train_feature_models.py` | extracted feature tables |
| Task 3 cohort-to-DBS transfer | `task3/ssl_cohort_bridge.py` | released DBS stride poses |

Selected Table 2 features: gait uses `stride_length` (ST) and `arm2arm_ROM`
(kinematic); fist uses `speed_fingertip_mean` (ST) and `amp_mcp_range_rad_mean`
(kinematic).

## Two evaluation halves

- **Pose metrics (MPJPE, PA-MPJPE)** run directly against the released pose
  pickles. No extra files needed.
- **Feature error (ST %, kinematic %)** is computed from precomputed feature
  CSVs included under `feature_eval/feature_csvs/`, so the reported numbers can
  be regenerated without the third-party estimators. Error is computed per unit
  (stride for gait, window for fist) within subject, then averaged across
  subjects.

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
