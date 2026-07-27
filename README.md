# TULIP-Bench evaluation code

Reference implementation of the preprocessing, pose evaluation, feature
extraction, clinical feature evaluation, UPDRS prediction, and treatment-response
analyses reported in the paper.

The dataset is released separately. See the paper for the access procedure.

## Setup

```bash
pip install -r requirements.txt
```

Each script reads a configuration block at the top of the file; there are no
command-line arguments. Set `DATA_ROOT` (and the shared `config.py` for the
Task 1/2/3 scripts) to point at the extracted dataset.

## Layout

```
config.py                       shared dataset paths and constants (Task 1/2/3)

features/                       clinical feature extraction
  extract_hand_features.py      hand closure features (Supplementary Table S2)
  select_features.py            bilateral-consistency hand feature selection
  extract_gait_features.py      gait stride features (Supplementary Table S5)
  transfer_to_h36m17.py         33-keypoint to Human3.6M-17 conversion, root-centering
  gait_feature_utils.py         gait geometry helpers

task1/                  segmentation and normalisation
  closure_scores.py             hand closure score c(t)
  repetition_detection.py       fist open-close repetition detection
  split_bouts_strides.py        bout and stride segmentation from raw gait
  slice_strides.py              stride slicing helpers
  organize_stride_sources.py    collate strides across pose sources
  stride_utils.py               projection, rotation, normalisation helpers

updrs/                          severity prediction (Task 1)
  train_feature_models.py       feature-based UPDRS, leave-one-subject-out
  posthoc_analysis.py           ensembling, ordinal grouping, bootstrap CIs

task2/                          monocular pose (Task 2)
  run_monocular_hand.py         off-the-shelf hand estimators (HaMeR, WiLoR, ...)
  restore_subject_names.py      map predictions to feature-extraction naming

task3/                          treatment response (Task 3)
  prepare_dbs_data.py           paired OFF/ON table construction
  feature_sensitivity.py        Cohen's d, Wilcoxon, convergent validity
  ssl_cohort_bridge.py          SSL pre-training and cohort-to-DBS transfer
```

## Subject naming

All scripts derive subject ids from directory names or pickle keys rather than
embedding any identifiers. The released de-identified convention is
`Neurips_Sub{n}` for the observational cohort and `Neurips_DBS_Sub{n}` for the
DBS cohort (with `_OFF` / `_ON` suffixes for treatment state). These prefixes
are defined once in `config.py` and used only where an id must be constructed or
displayed. If your local files use a different scheme, the scripts still work as
long as the ids are consistent across ground truth and predictions; adjust the
prefixes in `config.py` if any id needs to be built from parts.

### Segmentation notes

`preprocessing/split_bouts_strides.py` segments gait bouts and strides. Any
per-subject segmentation corrections live in the `SEGMENTATION_OVERRIDES` config
mapping (empty by default), keyed by de-identified subject id, rather than being
hardcoded.

### Cross-validation

UPDRS prediction uses leave-one-subject-out throughout, so train and test folds
are patient-disjoint by construction.

## Requirements

Python 3.9 or newer. See `requirements.txt`. A GPU is needed for the pose-based
models in `task2/run_monocular_hand.py` and `task3/ssl_cohort_bridge.py`; all
other scripts run on CPU.
