# ssl_cohort_bridge.py
"""
Task 3 benchmarks  with SSL pre-training as the cohort-DBS bridge.

Five parts:
    Part B: Data-efficiency curves. For each benchmark (Bench-1 stride OFF/ON;
            Bench-2 body-part severity; Bench-3 residualized response magnitude),
            train on k in {2,3,4,5,6,7,8,9} subjects and test on held-out
            remainder. Compare scratch vs SSL-pretrained ST-GCN. Five seeds.
            Shows whether SSL closes the small-N gap.
    Part C: Severity axis in SSL embedding space. Train linear regressor on
            cohort embeddings predicting UPDRS-gait -> "severity direction."
            Compute OFF->ON displacement per DBS subject and project onto
            severity direction.
    Part D: Zero-shot cohort-severity transfer. Apply cohort-learned severity
            regressor to DBS subjects (no DBS training data).
    Part E: UPDRS-blind OFF/ON classification. Restrict to the 8 DBS subjects
            with Delta UPDRS-gait = 0. Train OFF/ON classifier on their strides.
            Tests whether SSL embeddings see change UPDRS cannot.
    Part F: Displacement magnitude in "UPDRS-step units." Report OFF->ON
            distance per subject in SSL space, normalized by the median
            UPDRS=1 <-> UPDRS=2 distance on cohort.

Pre-training: same masked-reconstruction setup as  (15% mask on 56-subject
cohort strides), but we train 5 checkpoints (one per seed) so downstream
experiments can be averaged.

All baselines multi-seed. Figures get error bars/SD shading.

"""

import os

import config
import csv
import pickle
import re
import warnings
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.decomposition import PCA

import umap

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS & CONSTANTS
# =============================================================================
DBS_DIR = config.DBS_WORK_DIR
COHORT_DIR = config.GAIT_WORK_DIR

STRIDE_POSE_DIR = os.path.join(DBS_DIR, "DBS_strides_3Dposes")
COHORT_STRIDE_POSE_DIRS = [
    os.path.join(COHORT_DIR, "strides_3Dposes"),
    os.path.join(COHORT_DIR, "Gait_strides_3Dposes"),
    os.path.join(COHORT_DIR, "stride_poses_3d"),
    os.path.join("Gait_strides_3Dposes"),
]
DBS_STRIDE_FEAT_PKL = os.path.join(DBS_DIR, "stride_18features.pkl")
COHORT_STRIDE_FEAT_PKL = os.path.join(COHORT_DIR, "stride_18features.pkl")
PAIRED_CSV = os.path.join(DBS_DIR, "task3_dbs_paired.csv")
BODYPART_CSV = os.path.join(DBS_DIR, "Neurips2026_DBS_Gait_video_ratings_Categorized.csv")
COHORT_LABELS_CSV = config.GAIT_LABELS_CSV

FIG_ROOT = os.path.join(DBS_DIR, "figures")
FIG_DIR = os.path.join(FIG_ROOT, "benchmarks_ssl")
TAB_DIR = os.path.join(DBS_DIR, "tables")
CKPT_DIR = os.path.join(DBS_DIR, "ssl_checkpoints")

N_NORM_FRAMES = 120
N_JOINTS = 17
ROOT_IDX = 0
HEAD_IDX = 10

FEATURE_NAMES = [
    "stride_length", "step_length", "velocity", "hip_ROM", "knee_ROM", "leg2leg_ROM",
    "stride_time", "step_time", "cadance",
    "arm2arm_ROM", "shoulder_ROM", "elbow_ROM",
    "step_length_asy", "step_time_asy", "hip_ROM_asy",
    "knee_ROM_asy", "elbow_ROM_asy", "shoulder_ROM_asy",
]
BENCH3_TARGET_FEATURES = ["stride_length", "step_length", "velocity",
                           "hip_ROM", "knee_ROM", "leg2leg_ROM"]

BP_SEVERITY_MAP = {
    "normal": 0, "none": 0,
    "reduced (slight)": 1, "flexed (slight)": 1,
    "reduced (mild)": 2, "flexed (mild)": 2,
    "reduced (moderate)": 3, "flexed (moderate)": 3,
    "reduced (severe)": 4, "absent (severe)": 4,
}
BP_TASKS = [
    ("arm_swing_L", "Descriptions1_armswing_left_side"),
    ("arm_swing_R", "Descriptions1_armswing_right_side"),
    ("stride_L",    "Descriptions2_stride_left_side"),
    ("stride_R",    "Descriptions2_stride_right_side"),
    ("elbow_L",     "Descriptions3_flexed_elbows_left_side"),
    ("elbow_R",     "Descriptions3_flexed_elbows_right_side"),
]

BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

# UPDRS palette reused from  analysis
UPDRS_PALETTE = {0: "#2E7D32", 1: "#1976D2", 2: "#F57C00", 3: "#C62828", 4: "#6A1B9A"}

LABEL_SHORT_PREFIX = config.DBS_SUBJECT_PREFIX   # figures
LABEL_LONG_PREFIX = config.DBS_SUBJECT_PREFIX    # tables

# Pre-training
SSL_EPOCHS = 100
SSL_BATCH = 64
SSL_LR = 1e-3
SSL_MASK_RATIO = 0.15
SSL_VAL_FRAC = 0.1

# Fine-tuning
FT_EPOCHS = 60
FT_BATCH = 32
FT_LR = 5e-4

# Experiment controls
SEEDS = [42, 7, 123, 256, 2024]  # 5 seeds
DATA_SIZES_B = [2, 3, 4, 5, 6, 7, 8, 9]  # LOSO train sizes for Part B


# =============================================================================
# DATA LOADING & POSE NORMALIZATION (same as prior files)
# =============================================================================

def _strip_suffix(k): return k.removesuffix("_gait_pose")
def _base_from_dbs_key(k): return re.sub(r"_DBS(OFF|ON)_", "_", k)
def _state_from_dbs_key(k):
    if "DBSOFF" in k: return "OFF"
    if "DBSON" in k: return "ON"
    return None


def parse_bp_severity(val):
    if not val or not val.strip(): return None
    v = val.strip().lower()
    if v in BP_SEVERITY_MAP: return BP_SEVERITY_MAP[v]
    for k, sc in BP_SEVERITY_MAP.items():
        if k in v: return sc
    return None


def load_paired_rows():
    rows = []
    with open(PAIRED_CSV) as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v) if v not in ("", "nan", "NaN", "None") else float("nan")
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def load_bodypart_labels():
    out = {}
    if not os.path.exists(BODYPART_CSV):
        return out
    with open(BODYPART_CSV) as f:
        for row in csv.DictReader(f):
            name = row.get("subject_name", "").strip()
            state = _state_from_dbs_key(name)
            if state is None:
                continue
            base = _base_from_dbs_key(name)
            out[(base, state)] = {tn: parse_bp_severity(row.get(col, ""))
                                   for tn, col in BP_TASKS}
    return out


def load_feature_dict(pkl_path):
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    stripped = {_strip_suffix(k): v for k, v in raw.items()}
    out = defaultdict(dict)
    for key, feat_d in stripped.items():
        state = _state_from_dbs_key(key)
        base = _base_from_dbs_key(key)
        cleaned = {}
        for fn in FEATURE_NAMES:
            vals = feat_d.get(fn, [])
            clean = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
            cleaned[fn] = np.array(clean, dtype=np.float64)
        if state is None:
            out[base] = cleaned
        else:
            out[base][state] = cleaned
    return dict(out)


def load_cohort_updrs():
    """{cohort_subject_name: int UPDRS-gait}."""
    out = {}
    if not os.path.exists(COHORT_LABELS_CSV):
        return out
    with open(COHORT_LABELS_CSV) as f:
        for row in csv.DictReader(f):
            name = row.get("subject_name", "").strip()
            g = row.get("Gait", "")
            try:
                out[name] = int(float(g))
            except (ValueError, TypeError):
                pass
    return out


def load_dbs_stride_poses(base_subject):
    """Load OFF/ON stride poses for one DBS subject.

    Released files follow {DBS_SUBJECT_PREFIX}{n}_{STATE}. The base_subject id is
    used directly with the state suffix appended; a couple of filename variants
    are tried so a minor naming change does not break the lookup.
    """
    out = {}
    for state in ("OFF", "ON"):
        suffix = config.DBS_STATE_SUFFIXES.get(state, state)
        cands = [
            os.path.join(STRIDE_POSE_DIR, f"{base_subject}_{suffix}.pkl"),
            os.path.join(STRIDE_POSE_DIR, f"{base_subject}_{suffix}_gait_pose.pkl"),
            os.path.join(STRIDE_POSE_DIR, f"{base_subject}_{state}.pkl"),
        ]
        path = next((p for p in cands if os.path.exists(p)), None)
        if path is None:
            out[state] = []; continue
        with open(path, "rb") as f:
            out[state] = pickle.load(f)
    return out


def find_cohort_pose_dir():
    for d in COHORT_STRIDE_POSE_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".pkl"):
                    return d
    return None


def load_cohort_stride_poses_by_subject(cohort_pose_dir, cohort_subjects):
    out = {}
    for s in cohort_subjects:
        cand = os.path.join(cohort_pose_dir, f"{s}.pkl")
        if not os.path.exists(cand):
            continue
        with open(cand, "rb") as f:
            out[s] = pickle.load(f)
    return out


def time_normalize(stride, n_frames=N_NORM_FRAMES):
    T = len(stride)
    if T == n_frames:
        return stride.copy()
    t_orig = np.linspace(0, 1, T); t_new = np.linspace(0, 1, n_frames)
    flat = stride.reshape(T, -1)
    interp = np.empty((n_frames, flat.shape[1]), dtype=stride.dtype)
    for c in range(flat.shape[1]):
        interp[:, c] = np.interp(t_new, t_orig, flat[:, c])
    return interp.reshape((n_frames,) + stride.shape[1:])


def normalize_pose_shape_only(stride):
    stride = stride.astype(np.float64)
    pelvis = stride[:, ROOT_IDX:ROOT_IDX+1, :]
    s = stride - pelvis
    feet_mid = 0.5 * (s[:, 3, :] + s[:, 6, :])
    feet_mid_centered = feet_mid - feet_mid.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(feet_mid_centered, full_matrices=False)
        walk_dir = vt[0]
    except np.linalg.LinAlgError:
        walk_dir = np.array([1.0, 0.0, 0.0])
    if np.linalg.norm(feet_mid_centered, axis=1).max() < 10.0:
        walk_dir = np.array([1.0, 0.0, 0.0])
    head = s[:, HEAD_IDX, :].mean(axis=0)
    feet_avg = 0.5 * (s[:, 3, :] + s[:, 6, :]).mean(axis=0)
    vert = head - feet_avg; vert /= (np.linalg.norm(vert) + 1e-9)
    walk_dir = walk_dir - walk_dir.dot(vert) * vert
    walk_dir /= (np.linalg.norm(walk_dir) + 1e-9)
    proj = feet_mid_centered @ walk_dir
    t = np.arange(len(feet_mid))
    if np.polyfit(t, proj, 1)[0] < 0:
        walk_dir = -walk_dir
    lateral = np.cross(walk_dir, vert); lateral /= (np.linalg.norm(lateral) + 1e-9)
    R = np.column_stack([walk_dir, vert, lateral])
    s_rot = s @ R
    head_dist = np.linalg.norm(s_rot[:, HEAD_IDX, :], axis=1).mean()
    if head_dist < 1e-3: head_dist = 1.0
    return (s_rot / head_dist).astype(np.float32)


def build_subject_label_map(subjects):
    """Map released subject identifiers to display labels.

    Released identifiers are already de-identified (Subject_<N>). This orders
    them numerically and provides short labels for figures and long labels for
    tables, so that figure and table indices agree.
    """
    def index_of(name):
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else 0

    out = {}
    for full in sorted(subjects, key=index_of):
        n = index_of(full)
        out[full] = {"short": f"{LABEL_SHORT_PREFIX}{n}",
                     "long": f"{LABEL_LONG_PREFIX}{n}"}
    return out


def save_annotated_and_clean(fig, base_path, extra_hide=None):
    extra_hide = extra_hide or []
    fig.savefig(base_path + "_annotated.pdf", bbox_inches="tight")
    fig.savefig(base_path + "_annotated.png", bbox_inches="tight")
    hidden = []

    def _hide(a):
        if a is None: return
        try:
            was = a.get_visible(); hidden.append((a, was)); a.set_visible(False)
        except AttributeError:
            pass

    if fig._suptitle is not None: _hide(fig._suptitle)
    for ax in fig.axes:
        _hide(ax.title); _hide(ax.xaxis.label); _hide(ax.yaxis.label)
        lg = ax.get_legend()
        if lg: _hide(lg)
        for t in list(ax.texts): _hide(t)
    for lg in list(fig.legends): _hide(lg)
    for t in list(fig.texts):
        if t is fig._suptitle: continue
        _hide(t)
    for a in extra_hide: _hide(a)
    fig.savefig(base_path + "_clean.pdf", bbox_inches="tight")
    fig.savefig(base_path + "_clean.png", bbox_inches="tight")
    for a, was in hidden:
        try: a.set_visible(was)
        except AttributeError: pass
    print(f"  Saved: {os.path.basename(base_path)}_annotated.{{pdf,png}} + _clean.{{pdf,png}}")


def _save_csv(rows, path):
    if not rows: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: (f"{v:.6f}" if isinstance(v, float) and np.isfinite(v)
                       else "" if isinstance(v, float) else str(v))
                   for k, v in r.items()}
            w.writerow(out)
    print(f"  Saved: {path}")


# =============================================================================
# ST-GCN (same architecture)
# =============================================================================

def build_stgcn_adjacency():
    N = N_JOINTS
    bone_und = set()
    for a, b in BONES:
        bone_und.add((a, b)); bone_und.add((b, a))
    dist = np.full(N, -1, dtype=int); dist[ROOT_IDX] = 0
    frontier = [ROOT_IDX]
    while frontier:
        nxt = []
        for u in frontier:
            for v in range(N):
                if (u, v) in bone_und and dist[v] == -1:
                    dist[v] = dist[u] + 1; nxt.append(v)
        frontier = nxt
    A_self = np.eye(N); A_cent = np.zeros((N, N)); A_cfug = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j or (i, j) not in bone_und: continue
            (A_cent if dist[j] < dist[i] else A_cfug)[i, j] = 1.0
    def _norm(A):
        s = A.sum(axis=1, keepdims=True); s[s == 0] = 1
        return A / s
    return np.stack([_norm(A_self), _norm(A_cent), _norm(A_cfug)], axis=0).astype(np.float32)


STGCN_A = torch.tensor(build_stgcn_adjacency())


class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out, t_k=9, stride=1):
        super().__init__()
        self.K = 3
        self.conv_spatial = nn.Conv2d(c_in, c_out * self.K, kernel_size=1)
        self.bn_spatial = nn.BatchNorm2d(c_out)
        pad = (t_k - 1) // 2
        self.conv_temporal = nn.Sequential(
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=(t_k, 1), padding=(pad, 0),
                      stride=(stride, 1)),
            nn.BatchNorm2d(c_out),
        )
        self.residual = (nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=(stride, 1)),
            nn.BatchNorm2d(c_out)) if (c_in != c_out or stride != 1) else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        z = self.conv_spatial(x)
        B, CK, T, N = z.shape
        C_out = CK // self.K
        z = z.view(B, self.K, C_out, T, N)
        z = torch.einsum("bkctv,kvn->bctn", z, A)
        return self.relu(self.conv_temporal(self.relu(self.bn_spatial(z))) + res)


class STGCNEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * N_JOINTS)
        self.block1 = STGCNBlock(in_channels, 32, t_k=9, stride=1)
        self.block2 = STGCNBlock(32, 64, t_k=9, stride=2)
        self.block3 = STGCNBlock(64, 64, t_k=9, stride=2)
        self.out_channels = 64

    def forward(self, x, A):
        B, T, N, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x_bn = x.permute(0, 1, 3, 2).contiguous().view(B, C * N, T)
        x_bn = self.data_bn(x_bn)
        x = x_bn.view(B, C, N, T).permute(0, 1, 3, 2).contiguous()
        z1 = self.block1(x, A); z2 = self.block2(z1, A); z3 = self.block3(z2, A)
        return z1, z2, z3

    def embed(self, x, A):
        """Global-pooled embedding (B, 64) for downstream analysis."""
        _, _, z3 = self.forward(x, A)
        return z3.mean(dim=[2, 3])


class STGCNDecoder(nn.Module):
    def __init__(self, in_ch=64, out_ch=3):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 64, kernel_size=(3, 1), stride=(2, 1),
                                padding=(1, 0), output_padding=(1, 0)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 1), stride=(2, 1),
                                padding=(1, 0), output_padding=(1, 0)),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, kernel_size=1),
        )

    def forward(self, z):
        return self.upsample(z)


class STGCNHead(nn.Module):
    def __init__(self, in_ch=64, n_out=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1); self.head = nn.Linear(in_ch, n_out)

    def forward(self, z):
        return self.head(self.pool(z).flatten(1))


class STGCNFull(nn.Module):
    def __init__(self, encoder, n_out=1):
        super().__init__()
        self.encoder = encoder; self.head = STGCNHead(encoder.out_channels, n_out)

    def forward(self, x, A):
        _, _, z3 = self.encoder(x, A)
        return self.head(z3)


# =============================================================================
# SSL PRE-TRAINING (per-seed)
# =============================================================================

def mask_input(x, mask_ratio, rng_np):
    B, T, N, C = x.shape
    mask_np = rng_np.random((B, T, N)) < mask_ratio
    mask = torch.tensor(mask_np, device=x.device)
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


def pretrain_stgcn_seed(cohort_pose_X, seed):
    print(f"    Pretraining seed={seed}...")
    torch.manual_seed(seed)
    rng_np = np.random.default_rng(seed)
    n = len(cohort_pose_X)
    perm = rng_np.permutation(n)
    n_val = max(1, int(SSL_VAL_FRAC * n))
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    X_tr = torch.tensor(cohort_pose_X[tr_idx], dtype=torch.float32).to(DEVICE)
    X_val = torch.tensor(cohort_pose_X[val_idx], dtype=torch.float32).to(DEVICE)

    encoder = STGCNEncoder(in_channels=3).to(DEVICE)
    decoder = STGCNDecoder().to(DEVICE)
    A = STGCN_A.to(DEVICE)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                            lr=SSL_LR, weight_decay=1e-5)

    best_val = float("inf"); best_state = None
    for epoch in range(SSL_EPOCHS):
        encoder.train(); decoder.train()
        idx = torch.randperm(len(X_tr))
        for b in range(0, len(X_tr), SSL_BATCH):
            bi = idx[b:b + SSL_BATCH]
            x = X_tr[bi]
            x_masked, mask = mask_input(x, SSL_MASK_RATIO, rng_np)
            opt.zero_grad()
            _, _, z3 = encoder(x_masked, A)
            x_recon = decoder(z3).permute(0, 2, 3, 1)
            diff = ((x_recon - x) ** 2).mean(dim=-1)
            loss = (diff * mask.float()).sum() / (mask.float().sum() + 1e-6)
            loss.backward(); opt.step()

        encoder.eval(); decoder.eval()
        with torch.no_grad():
            x_m, mask_v = mask_input(X_val, SSL_MASK_RATIO, rng_np)
            _, _, z3v = encoder(x_m, A)
            x_r = decoder(z3v).permute(0, 2, 3, 1)
            d = ((x_r - X_val) ** 2).mean(dim=-1)
            vl = ((d * mask_v.float()).sum() / (mask_v.float().sum() + 1e-6)).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}

    ckpt_path = os.path.join(CKPT_DIR, f"stgcn_pretrained_seed{seed}.pt")
    torch.save({"encoder_state_dict": best_state, "best_val_loss": best_val,
                 "seed": seed}, ckpt_path)
    print(f"      seed={seed}: best_val_loss={best_val:.4f}  saved -> {os.path.basename(ckpt_path)}")
    return best_state


def pretrain_all_seeds(cohort_pose_X):
    os.makedirs(CKPT_DIR, exist_ok=True)
    states = {}
    for seed in SEEDS:
        path = os.path.join(CKPT_DIR, f"stgcn_pretrained_seed{seed}.pt")
        if os.path.exists(path):
            print(f"    Loading cached checkpoint for seed={seed}")
            ck = torch.load(path, map_location=DEVICE)
            states[seed] = ck["encoder_state_dict"]
        else:
            states[seed] = pretrain_stgcn_seed(cohort_pose_X, seed)
    return states


# =============================================================================
# FINE-TUNE FUNCTIONS
# =============================================================================

def finetune_clf(X_train, y_train, X_test, pretrained_state, seed,
                  epochs=FT_EPOCHS):
    torch.manual_seed(seed)
    A = STGCN_A.to(DEVICE)
    Xtr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    ytr = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    Xte = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    encoder = STGCNEncoder(in_channels=3).to(DEVICE)
    if pretrained_state is not None:
        encoder.load_state_dict(pretrained_state)
    model = STGCNFull(encoder, n_out=1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=FT_LR, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for b in range(0, n, FT_BATCH):
            bi = idx[b:b + FT_BATCH]
            opt.zero_grad()
            logit = model(Xtr[bi], A).squeeze(-1)
            loss = bce(logit, ytr[bi])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logit_te = model(Xte, A).squeeze(-1).cpu().numpy()
    return 1 / (1 + np.exp(-logit_te))


def finetune_reg(X_train, y_train, X_test, pretrained_state, seed,
                  epochs=FT_EPOCHS):
    torch.manual_seed(seed)
    A = STGCN_A.to(DEVICE)
    Xtr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    ytr = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    Xte = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    encoder = STGCNEncoder(in_channels=3).to(DEVICE)
    if pretrained_state is not None:
        encoder.load_state_dict(pretrained_state)
    model = STGCNFull(encoder, n_out=1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=FT_LR, weight_decay=1e-4)
    mse = nn.MSELoss()
    n = len(Xtr)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for b in range(0, n, FT_BATCH):
            bi = idx[b:b + FT_BATCH]
            opt.zero_grad()
            pred = model(Xtr[bi], A).squeeze(-1)
            loss = mse(pred, ytr[bi])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model(Xte, A).squeeze(-1).cpu().numpy()


def extract_embeddings(X, pretrained_state):
    """Return (N, 64) embedding vectors for input strides."""
    A = STGCN_A.to(DEVICE)
    encoder = STGCNEncoder(in_channels=3).to(DEVICE)
    if pretrained_state is not None:
        encoder.load_state_dict(pretrained_state)
    encoder.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    embs = []
    with torch.no_grad():
        for b in range(0, len(Xt), 128):
            embs.append(encoder.embed(Xt[b:b + 128], A).cpu().numpy())
    return np.concatenate(embs, axis=0)


# =============================================================================
# BUILD DATASETS
# =============================================================================

def build_dbs_stride_dataset(feat_dict, pose_by_subj):
    dataset = []
    for s in sorted(feat_dict.keys()):
        for state in ("OFF", "ON"):
            feats = feat_dict[s][state]
            poses = pose_by_subj.get(s, {}).get(state, [])
            n_strides = min(len(feats[fn]) for fn in FEATURE_NAMES)
            n_strides = min(n_strides, len(poses))
            for k in range(n_strides):
                fv = np.array([feats[fn][k] for fn in FEATURE_NAMES])
                if not np.all(np.isfinite(fv)): continue
                pose_raw = np.asarray(poses[k], dtype=np.float64)
                pose_shape = time_normalize(normalize_pose_shape_only(pose_raw))
                dataset.append({
                    "subject": s, "state": state, "stride_idx": k,
                    "feat_vec": fv, "pose": pose_shape.astype(np.float32),
                })
    return dataset


def build_cohort_pose_dataset(cohort_pose_dir, cohort_feat_dict, cohort_updrs):
    """Returns (stride_matrix (N,T,J,3), subj_list, updrs_list)."""
    cohort_subjects = list(cohort_feat_dict.keys())
    files = load_cohort_stride_poses_by_subject(cohort_pose_dir, cohort_subjects)
    all_strides = []; subj_list = []; updrs_list = []
    for sub, strides in files.items():
        #pass
        if sub not in cohort_updrs:
            continue
        for stride in strides:
            if len(stride) < 5: continue
            pose_norm = time_normalize(normalize_pose_shape_only(np.asarray(stride, dtype=np.float64)))
            all_strides.append(pose_norm.astype(np.float32))
            subj_list.append(sub); updrs_list.append(cohort_updrs[sub])
    return np.stack(all_strides), np.array(subj_list), np.array(updrs_list)


# =============================================================================
# PART B: DATA-EFFICIENCY CURVES
# =============================================================================

def _make_bench3_targets(dbs_dataset):
    """Return per-subject (feat_vec_off_mean, pose_off_mean, OFF_level[6], Y[6], resid_Y[6]).
       Returns dict with keys: subjects, pose_X (N,T,J,3), OFF_level, Y_raw, Y_resid."""
    per_subj = defaultdict(lambda: {"off_feat": [], "off_pose": [], "on_feat": []})
    for d in dbs_dataset:
        if d["state"] == "OFF":
            per_subj[d["subject"]]["off_feat"].append(d["feat_vec"])
            per_subj[d["subject"]]["off_pose"].append(d["pose"])
        else:
            per_subj[d["subject"]]["on_feat"].append(d["feat_vec"])
    subjects = sorted(per_subj.keys())
    valid = [s for s in subjects
             if per_subj[s]["off_feat"] and per_subj[s]["off_pose"] and per_subj[s]["on_feat"]]
    pose_X = np.array([np.mean(np.stack(per_subj[s]["off_pose"]), axis=0) for s in valid])
    fn_idx = {fn: i for i, fn in enumerate(FEATURE_NAMES)}
    OFF_level = np.zeros((len(valid), len(BENCH3_TARGET_FEATURES)))
    Y = np.zeros((len(valid), len(BENCH3_TARGET_FEATURES)))
    for si, s in enumerate(valid):
        off_m = np.mean(np.stack(per_subj[s]["off_feat"]), axis=0)
        on_m = np.mean(np.stack(per_subj[s]["on_feat"]), axis=0)
        for ti, tf in enumerate(BENCH3_TARGET_FEATURES):
            OFF_level[si, ti] = off_m[fn_idx[tf]]
            Y[si, ti] = on_m[fn_idx[tf]] - off_m[fn_idx[tf]]
    resid_Y = np.zeros_like(Y)
    for ti in range(len(BENCH3_TARGET_FEATURES)):
        lr = LinearRegression().fit(OFF_level[:, ti:ti+1], Y[:, ti])
        resid_Y[:, ti] = Y[:, ti] - lr.predict(OFF_level[:, ti:ti+1])
    return {"subjects": valid, "pose_X": pose_X, "OFF_level": OFF_level,
            "Y_raw": Y, "Y_resid": resid_Y}


def part_B_data_efficiency(dbs_dataset, bp_labels, pretrained_states, label_map):
    """For each benchmark, evaluate scratch vs pretrained at multiple training sizes."""
    print(f"\n{'='*70}\n  Part B: Data-efficiency curves\n{'='*70}")
    rng_outer = np.random.default_rng(2024)

    # ---------- Bench-1 at varying training sizes ----------
    print(f"\n  Part B.1: Bench-1 data efficiency")
    y_all = np.array([0 if d["state"] == "OFF" else 1 for d in dbs_dataset])
    g_all = np.array([d["subject"] for d in dbs_dataset])
    pose_X_all = np.array([d["pose"] for d in dbs_dataset])
    subjects_b1 = sorted(set(g_all.tolist()))
    n_subj = len(subjects_b1)

    # results[method][k] -> list of per-fold AUCs across seeds
    results_b1 = {"scratch": defaultdict(list), "pretrained": defaultdict(list)}
    for k in DATA_SIZES_B:
        if k >= n_subj: continue
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            # Strategy: randomly sample k subjects to train on; test on remaining (n_subj - k).
            # To keep variance controlled, repeat 2 times per seed with different sampled sets.
            for rep in range(2):
                sel_idx = rng.choice(n_subj, size=k, replace=False)
                train_subs = [subjects_b1[i] for i in sel_idx]
                test_subs = [s for s in subjects_b1 if s not in train_subs]
                tr_mask = np.isin(g_all, train_subs); te_mask = np.isin(g_all, test_subs)
                ytr, yte = y_all[tr_mask], y_all[te_mask]
                if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                    continue
                # scratch
                p_sc = finetune_clf(pose_X_all[tr_mask], ytr, pose_X_all[te_mask],
                                     None, seed)
                auc_sc = roc_auc_score(yte, p_sc)
                # pretrained: use seed-matched checkpoint
                p_pt = finetune_clf(pose_X_all[tr_mask], ytr, pose_X_all[te_mask],
                                     pretrained_states[seed], seed)
                auc_pt = roc_auc_score(yte, p_pt)
                results_b1["scratch"][k].append(auc_sc)
                results_b1["pretrained"][k].append(auc_pt)
        n_runs = len(results_b1["scratch"][k])
        sc_mean = float(np.mean(results_b1["scratch"][k]))
        pt_mean = float(np.mean(results_b1["pretrained"][k]))
        print(f"    k={k}: scratch={sc_mean:.3f}  pretrained={pt_mean:.3f}  "
              f"(n_runs={n_runs})")

    # ---------- Bench-2 at varying training sizes ----------
    print(f"\n  Part B.2: Bench-2 data efficiency")
    # Aggregate per (subject, state)
    agg = defaultdict(lambda: {"poses": []})
    for d in dbs_dataset:
        agg[(d["subject"], d["state"])]["poses"].append(d["pose"])
    obs = []
    for (s, st), v in agg.items():
        label = bp_labels.get((s, st), {})
        if not any(label.get(tn) is not None for tn, _ in BP_TASKS):
            continue
        obs.append({"subject": s, "state": st,
                    "pose": np.mean(np.stack(v["poses"]), axis=0).astype(np.float32),
                    "labels": {tn: label.get(tn) for tn, _ in BP_TASKS}})
    subjects_b2 = sorted(set(o["subject"] for o in obs))
    n_subj_b2 = len(subjects_b2)

    results_b2 = {"scratch": defaultdict(list), "pretrained": defaultdict(list)}
    for k in DATA_SIZES_B:
        if k >= n_subj_b2: continue
        for seed in SEEDS:
            rng = np.random.default_rng(seed + 1)
            for rep in range(2):
                sel_idx = rng.choice(n_subj_b2, size=k, replace=False)
                train_subs = [subjects_b2[i] for i in sel_idx]
                test_subs = [s for s in subjects_b2 if s not in train_subs]
                # Collect across all tasks: mean rho across tasks for this split
                rhos_sc = []; rhos_pt = []
                for tname, _ in BP_TASKS:
                    tr_obs = [o for o in obs if o["subject"] in train_subs
                              and o["labels"].get(tname) is not None]
                    te_obs = [o for o in obs if o["subject"] in test_subs
                              and o["labels"].get(tname) is not None]
                    if len(tr_obs) < 2 or len(te_obs) < 2: continue
                    pX_tr = np.stack([o["pose"] for o in tr_obs])
                    yY_tr = np.array([o["labels"][tname] for o in tr_obs], dtype=np.float32)
                    pX_te = np.stack([o["pose"] for o in te_obs])
                    yY_te = np.array([o["labels"][tname] for o in te_obs], dtype=float)
                    if len(np.unique(yY_te)) < 2: continue
                    p_sc = finetune_reg(pX_tr, yY_tr, pX_te, None, seed)
                    p_pt = finetune_reg(pX_tr, yY_tr, pX_te, pretrained_states[seed], seed)
                    try: r_sc, _ = spearmanr(yY_te, p_sc)
                    except Exception: r_sc = np.nan
                    try: r_pt, _ = spearmanr(yY_te, p_pt)
                    except Exception: r_pt = np.nan
                    if np.isfinite(r_sc): rhos_sc.append(r_sc)
                    if np.isfinite(r_pt): rhos_pt.append(r_pt)
                if rhos_sc: results_b2["scratch"][k].append(float(np.mean(rhos_sc)))
                if rhos_pt: results_b2["pretrained"][k].append(float(np.mean(rhos_pt)))
        n_runs = len(results_b2["scratch"][k])
        if n_runs > 0:
            sc_m = float(np.mean(results_b2["scratch"][k])); pt_m = float(np.mean(results_b2["pretrained"][k]))
            print(f"    k={k}: scratch={sc_m:+.3f}  pretrained={pt_m:+.3f}  (n_runs={n_runs})")

    # ---------- Bench-3 residualized at varying training sizes ----------
    print(f"\n  Part B.3: Bench-3 (residualized) data efficiency")
    b3 = _make_bench3_targets(dbs_dataset)
    pose_X_b3 = b3["pose_X"]; Y_resid = b3["Y_resid"]
    subjects_b3 = b3["subjects"]; n_subj_b3 = len(subjects_b3)

    results_b3 = {"scratch": defaultdict(list), "pretrained": defaultdict(list)}
    for k in DATA_SIZES_B:
        if k >= n_subj_b3: continue
        for seed in SEEDS:
            rng = np.random.default_rng(seed + 2)
            for rep in range(2):
                sel_idx = rng.choice(n_subj_b3, size=k, replace=False)
                test_idx = [i for i in range(n_subj_b3) if i not in sel_idx]
                rhos_sc = []; rhos_pt = []
                for ti, tf in enumerate(BENCH3_TARGET_FEATURES):
                    y_tr = Y_resid[sel_idx, ti].astype(np.float32)
                    y_te = Y_resid[test_idx, ti]
                    pX_tr = pose_X_b3[sel_idx]; pX_te = pose_X_b3[test_idx]
                    p_sc = finetune_reg(pX_tr, y_tr, pX_te, None, seed)
                    p_pt = finetune_reg(pX_tr, y_tr, pX_te, pretrained_states[seed], seed)
                    try: r_sc, _ = spearmanr(y_te, p_sc)
                    except Exception: r_sc = np.nan
                    try: r_pt, _ = spearmanr(y_te, p_pt)
                    except Exception: r_pt = np.nan
                    if np.isfinite(r_sc): rhos_sc.append(r_sc)
                    if np.isfinite(r_pt): rhos_pt.append(r_pt)
                if rhos_sc: results_b3["scratch"][k].append(float(np.mean(rhos_sc)))
                if rhos_pt: results_b3["pretrained"][k].append(float(np.mean(rhos_pt)))
        n_runs = len(results_b3["scratch"][k])
        if n_runs > 0:
            sc_m = float(np.mean(results_b3["scratch"][k])); pt_m = float(np.mean(results_b3["pretrained"][k]))
            print(f"    k={k}: scratch={sc_m:+.3f}  pretrained={pt_m:+.3f}  (n_runs={n_runs})")

    # ---------- Figure: 3 subplots, one per benchmark ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    all_results = [("Bench-1 (AUC)", results_b1, (0.4, 1.0)),
                   ("Bench-2 (mean ρ)", results_b2, (-0.5, 0.8)),
                   ("Bench-3 residualized (mean ρ)", results_b3, (-0.6, 0.6))]
    for ax, (title, res, ylim) in zip(axes, all_results):
        ks = sorted(res["scratch"].keys())
        sc_means = [np.mean(res["scratch"][k]) if res["scratch"][k] else np.nan for k in ks]
        sc_sds = [np.std(res["scratch"][k]) if len(res["scratch"][k]) > 1 else 0 for k in ks]
        pt_means = [np.mean(res["pretrained"][k]) if res["pretrained"][k] else np.nan for k in ks]
        pt_sds = [np.std(res["pretrained"][k]) if len(res["pretrained"][k]) > 1 else 0 for k in ks]
        ax.plot(ks, sc_means, "o-", color="#90A4AE", lw=2, label="Scratch", markersize=7)
        ax.fill_between(ks, np.array(sc_means) - np.array(sc_sds),
                         np.array(sc_means) + np.array(sc_sds), color="#90A4AE", alpha=0.2)
        ax.plot(ks, pt_means, "s-", color="#1565C0", lw=2, label="SSL-pretrained", markersize=7)
        ax.fill_between(ks, np.array(pt_means) - np.array(pt_sds),
                         np.array(pt_means) + np.array(pt_sds), color="#1565C0", alpha=0.2)
        ax.set_xlabel("# training subjects", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Part B: Data-efficiency curves (mean ± SD over 5 seeds × 2 reps)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_annotated_and_clean(fig, os.path.join(FIG_DIR, "partB_data_efficiency"))
    plt.close(fig)

    # Save tables
    for bench_name, res in [("b1", results_b1), ("b2", results_b2), ("b3", results_b3)]:
        tab = []
        for method in ["scratch", "pretrained"]:
            for k in sorted(res[method].keys()):
                vals = res[method][k]
                if not vals: continue
                tab.append({"bench": bench_name, "method": method, "k": k,
                            "mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                            "n_runs": len(vals)})
        _save_csv(tab, os.path.join(TAB_DIR, f"partB_{bench_name}_data_efficiency.csv"))

    return results_b1, results_b2, results_b3


# =============================================================================
# PART C: SEVERITY AXIS IN SSL SPACE
# =============================================================================

def part_C_severity_axis(cohort_pose_X, cohort_updrs_list, cohort_subj_list,
                          dbs_dataset, pretrained_states, label_map,
                          paired_rows):
    """Train linear model on cohort embeddings -> UPDRS. Get severity direction.
    Then project DBS OFF/ON centroids."""
    print(f"\n{'='*70}\n  Part C: Severity axis in SSL space\n{'='*70}")

    # Average across 5 seeds: fit once per seed, average direction.
    # But embeddings differ per seed, so we do per-seed analysis and aggregate stats.

    # DBS embeddings per seed
    dbs_subjects = sorted(set(d["subject"] for d in dbs_dataset))
    pose_X_dbs = np.array([d["pose"] for d in dbs_dataset])
    g_dbs = np.array([d["subject"] for d in dbs_dataset])
    state_dbs = np.array([d["state"] for d in dbs_dataset])

    projections_per_seed = {}  # seed -> {subject: (off_proj, on_proj, displacement)}
    cohort_rho_per_seed = []
    for seed in SEEDS:
        state_dict = pretrained_states[seed]
        # Cohort embeddings
        cohort_emb = extract_embeddings(cohort_pose_X, state_dict)
        # Fit linear regressor: cohort_emb -> UPDRS
        lr = LinearRegression().fit(cohort_emb, cohort_updrs_list.astype(np.float64))
        pred = lr.predict(cohort_emb)
        try: rho, _ = spearmanr(cohort_updrs_list, pred)
        except Exception: rho = np.nan
        cohort_rho_per_seed.append(rho)

        # Severity direction = coefficient vector
        direction = lr.coef_.astype(np.float64)
        direction_norm = direction / (np.linalg.norm(direction) + 1e-9)

        # DBS embeddings
        dbs_emb = extract_embeddings(pose_X_dbs, state_dict)
        # Per-subject OFF/ON median embeddings
        proj_per_subj = {}
        for s in dbs_subjects:
            m_off = (g_dbs == s) & (state_dbs == "OFF")
            m_on = (g_dbs == s) & (state_dbs == "ON")
            if m_off.sum() == 0 or m_on.sum() == 0:
                continue
            off_centroid = np.median(dbs_emb[m_off], axis=0)
            on_centroid = np.median(dbs_emb[m_on], axis=0)
            off_proj = float(off_centroid @ direction_norm)
            on_proj = float(on_centroid @ direction_norm)
            displacement = on_proj - off_proj
            proj_per_subj[s] = (off_proj, on_proj, displacement)
        projections_per_seed[seed] = proj_per_subj

    mean_rho = float(np.mean(cohort_rho_per_seed))
    std_rho = float(np.std(cohort_rho_per_seed))
    print(f"  Cohort UPDRS reconstruction ρ (linear probe on embeddings): "
          f"mean={mean_rho:.3f}, SD={std_rho:.3f}, seeds={cohort_rho_per_seed}")

    # For each DBS subject: aggregate displacement across seeds
    subj_summary = {}
    for s in dbs_subjects:
        disps = []; offs = []; ons = []
        for seed in SEEDS:
            if s in projections_per_seed[seed]:
                off_p, on_p, d = projections_per_seed[seed][s]
                disps.append(d); offs.append(off_p); ons.append(on_p)
        if not disps: continue
        subj_summary[s] = {"mean_displacement": float(np.mean(disps)),
                            "sd_displacement": float(np.std(disps)),
                            "mean_off_proj": float(np.mean(offs)),
                            "mean_on_proj": float(np.mean(ons)),
                            "frac_seeds_negative_disp": float(np.mean([d < 0 for d in disps]))}

    # Fraction of DBS subjects with displacement pointing toward "less severe"
    # (i.e., negative displacement, since higher UPDRS -> higher linear output).
    # BUT direction sign is arbitrary per seed; we define success as "consistent
    # sign across seeds, and the sign matches improvement direction."
    # Simpler: on the canonical sign (lr coefficient on UPDRS means higher
    # output -> higher UPDRS -> more severe), a negative displacement means OFF -> ON
    # reduced severity.
    mean_disps = np.array([subj_summary[s]["mean_displacement"] for s in subj_summary])
    n_correct = int(np.sum(mean_disps < 0))
    print(f"  Fraction of DBS subjects moving toward LESS severe on the "
          f"cohort-learned axis: {n_correct}/{len(mean_disps)}")

    # Figure: one bar per subject, mean disp with SD error bars. Color by clinician responder.
    resp_map = {r.get("subject"): r.get("responder") for r in paired_rows}
    delta_updrs_map = {r.get("subject"): r.get("delta_gait_updrs", np.nan) for r in paired_rows}

    subjects_sorted = sorted(subj_summary.keys(),
                             key=lambda s: subj_summary[s]["mean_displacement"])
    disps_mean = [subj_summary[s]["mean_displacement"] for s in subjects_sorted]
    disps_sd = [subj_summary[s]["sd_displacement"] for s in subjects_sorted]
    colors = []
    for s in subjects_sorted:
        r = resp_map.get(s)
        if isinstance(r, float) and np.isfinite(r):
            colors.append("#2E7D32" if int(r) == 1 else "#C62828")
        else:
            colors.append("#888")

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    xpos = np.arange(len(subjects_sorted))
    ax.barh(xpos, disps_mean, xerr=disps_sd, color=colors, edgecolor="white",
            height=0.75, capsize=4, alpha=0.9)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(xpos)
    ax.set_yticklabels([label_map[s]["short"] for s in subjects_sorted], fontsize=9)
    ax.set_xlabel("OFF→ON displacement on cohort-learned severity axis\n"
                  "(negative = toward less severe; mean ± SD over 5 seeds)",
                  fontsize=10)
    ax.axvspan(ax.get_xlim()[0], 0, alpha=0.08, color="#2E7D32")
    ax.axvspan(0, ax.get_xlim()[1], alpha=0.08, color="#C62828")
    ax.grid(True, axis="x", alpha=0.25)
    legend_handles = [
        mpatches.Patch(color="#2E7D32", label="Good responder (clinician)"),
        mpatches.Patch(color="#C62828", label="Poor responder"),
        mpatches.Patch(color="#888", label="Unknown"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=9)
    fig.suptitle(f"Part C: DBS OFF→ON displacement on the cohort-learned severity axis.  "
                 f"{n_correct}/{len(mean_disps)} subjects move toward less severe.  "
                 f"(Cohort linear probe ρ = {mean_rho:.2f} ± {std_rho:.2f})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_annotated_and_clean(fig, os.path.join(FIG_DIR, "partC_severity_axis_displacements"))
    plt.close(fig)

    # Save table
    tab = []
    for s in subjects_sorted:
        row = {"subject": s, "deid_short": label_map[s]["short"],
                "mean_displacement": subj_summary[s]["mean_displacement"],
                "sd_displacement": subj_summary[s]["sd_displacement"],
                "mean_off_proj": subj_summary[s]["mean_off_proj"],
                "mean_on_proj": subj_summary[s]["mean_on_proj"],
                "delta_updrs_gait": delta_updrs_map.get(s, np.nan),
                "responder": resp_map.get(s, "")}
        tab.append(row)
    _save_csv(tab, os.path.join(TAB_DIR, "partC_severity_axis_per_subject.csv"))

    # Also make a UMAP figure (using one representative seed) with cohort + DBS
    print(f"\n  Part C.2: UMAP figure")
    seed_ref = SEEDS[0]
    cohort_emb = extract_embeddings(cohort_pose_X, pretrained_states[seed_ref])
    dbs_emb = extract_embeddings(pose_X_dbs, pretrained_states[seed_ref])
    # UMAP jointly
    all_emb = np.vstack([cohort_emb, dbs_emb])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    all_umap = reducer.fit_transform(all_emb)
    cohort_umap = all_umap[:len(cohort_emb)]
    dbs_umap = all_umap[len(cohort_emb):]

    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
    # Cohort scatter
    unique_scores = sorted(set(cohort_updrs_list.tolist()))
    for sc in unique_scores:
        m = cohort_updrs_list == sc
        if m.sum() == 0: continue
        ax.scatter(cohort_umap[m, 0], cohort_umap[m, 1], c=[UPDRS_PALETTE.get(sc, "#888")],
                    s=10, alpha=0.35, edgecolor="none",
                    label=f"Cohort UPDRS-gait={sc} (n={m.sum()})", zorder=2)
    # DBS per-subject centroids + arrows
    for s in dbs_subjects:
        m_off = (g_dbs == s) & (state_dbs == "OFF")
        m_on = (g_dbs == s) & (state_dbs == "ON")
        if m_off.sum() == 0 or m_on.sum() == 0: continue
        off_c = np.median(dbs_umap[m_off], axis=0)
        on_c = np.median(dbs_umap[m_on], axis=0)
        ax.annotate("", xy=on_c, xytext=off_c,
                     arrowprops=dict(arrowstyle="-|>", color="#111", lw=2.0,
                                      mutation_scale=13, shrinkA=0, shrinkB=0), zorder=5)
        ax.scatter(*off_c, marker="X", color="#111", s=60,
                    edgecolor="white", linewidth=1.5, zorder=6)
        ax.scatter(*on_c, marker="*", color="#111", s=120,
                    edgecolor="white", linewidth=1.2, zorder=6)
        ax.annotate(label_map[s]["short"], off_c, fontsize=8, ha="center",
                    va="top", xytext=(0, -6), textcoords="offset points",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                    zorder=7)

    ax.set_xlabel("UMAP-1 (SSL embedding)", fontsize=10)
    ax.set_ylabel("UMAP-2 (SSL embedding)", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.suptitle("Part C: SSL embedding UMAP — cohort severity landscape + DBS OFF→ON arrows",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_annotated_and_clean(fig, os.path.join(FIG_DIR, "partC_ssl_umap_landscape"))
    plt.close(fig)

    return subj_summary, mean_rho, std_rho, cohort_emb, dbs_emb


# =============================================================================
# PART D: ZERO-SHOT COHORT-SEVERITY TRANSFER TO DBS
# =============================================================================

def part_D_zero_shot_transfer(cohort_pose_X, cohort_updrs_list, dbs_dataset,
                                pretrained_states, label_map, paired_rows):
    """Fit cohort severity regressor, apply zero-shot to DBS OFF/ON per seed."""
    print(f"\n{'='*70}\n  Part D: Zero-shot cohort severity transfer\n{'='*70}")

    dbs_subjects = sorted(set(d["subject"] for d in dbs_dataset))
    pose_X_dbs = np.array([d["pose"] for d in dbs_dataset])
    g_dbs = np.array([d["subject"] for d in dbs_dataset])
    state_dbs = np.array([d["state"] for d in dbs_dataset])

    # Per-seed: predict per-stride severity; aggregate to per-subject median.
    predictions_per_seed = {}
    for seed in SEEDS:
        state_dict = pretrained_states[seed]
        cohort_emb = extract_embeddings(cohort_pose_X, state_dict)
        dbs_emb = extract_embeddings(pose_X_dbs, state_dict)
        lr = LinearRegression().fit(cohort_emb, cohort_updrs_list.astype(np.float64))
        pred_dbs = lr.predict(dbs_emb)
        pred_per_subj_state = defaultdict(dict)
        for s in dbs_subjects:
            for st in ("OFF", "ON"):
                m = (g_dbs == s) & (state_dbs == st)
                if m.sum() == 0: continue
                pred_per_subj_state[s][st] = float(np.median(pred_dbs[m]))
        predictions_per_seed[seed] = pred_per_subj_state

    # Aggregate across seeds
    subj_agg = defaultdict(lambda: {"off_preds": [], "on_preds": []})
    for seed in SEEDS:
        for s, states in predictions_per_seed[seed].items():
            if "OFF" in states and "ON" in states:
                subj_agg[s]["off_preds"].append(states["OFF"])
                subj_agg[s]["on_preds"].append(states["ON"])

    # Compute per-subject mean predicted severity for OFF/ON, and Δ
    off_mean = {s: float(np.mean(v["off_preds"])) for s, v in subj_agg.items()}
    on_mean = {s: float(np.mean(v["on_preds"])) for s, v in subj_agg.items()}
    deltas = {s: on_mean[s] - off_mean[s] for s in off_mean}

    # Compare to actual DBS UPDRS-gait values
    dbs_off_updrs = {r.get("subject"): r.get("OFF_gait_updrs", np.nan) for r in paired_rows}
    dbs_on_updrs = {r.get("subject"): r.get("ON_gait_updrs", np.nan) for r in paired_rows}

    # Correlation: predicted severity vs actual UPDRS, pooled across subjects+states
    pooled_pred = []; pooled_actual = []
    for s in off_mean:
        if np.isfinite(dbs_off_updrs.get(s, np.nan)):
            pooled_pred.append(off_mean[s]); pooled_actual.append(dbs_off_updrs[s])
        if np.isfinite(dbs_on_updrs.get(s, np.nan)):
            pooled_pred.append(on_mean[s]); pooled_actual.append(dbs_on_updrs[s])
    try:
        r_pool, _ = spearmanr(pooled_pred, pooled_actual)
    except Exception:
        r_pool = np.nan
    print(f"  Pooled Spearman ρ (predicted severity vs actual UPDRS-gait): {r_pool:.3f}  "
          f"(n={len(pooled_pred)})")

    # Fraction of subjects whose predicted severity decreased OFF->ON
    n_decreased = int(sum(1 for s in deltas if deltas[s] < 0))
    print(f"  Fraction of DBS subjects with predicted severity decreasing OFF→ON: "
          f"{n_decreased}/{len(deltas)}")

    # Figure: predicted severity OFF vs ON per subject, alongside clinician UPDRS
    subjects_sorted = sorted(off_mean.keys(), key=lambda s: off_mean[s])
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    x = np.arange(len(subjects_sorted))
    ax.plot(x, [off_mean[s] for s in subjects_sorted], "o-", color="#C62828",
            lw=2, label="Predicted severity (OFF)", markersize=8)
    ax.plot(x, [on_mean[s] for s in subjects_sorted], "s-", color="#2E7D32",
            lw=2, label="Predicted severity (ON)", markersize=8)
    # Integer UPDRS reference as horizontal bands
    for sc in (1, 2, 3):
        ax.axhline(sc, color=UPDRS_PALETTE.get(sc), lw=0.5, alpha=0.3, linestyle="--")
        ax.text(len(subjects_sorted) - 0.3, sc, f"UPDRS={sc}", fontsize=7,
                 va="center", color=UPDRS_PALETTE.get(sc), alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([label_map[s]["short"] for s in subjects_sorted],
                                          fontsize=9)
    ax.set_ylabel("Predicted continuous severity\n(cohort-learned)", fontsize=10)
    ax.set_xlabel("DBS subject", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"Part D: Zero-shot cohort→DBS severity transfer.  "
                 f"Pooled ρ = {r_pool:.2f}; {n_decreased}/{len(deltas)} subjects show "
                 f"predicted severity decreasing OFF→ON.",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_annotated_and_clean(fig, os.path.join(FIG_DIR, "partD_zero_shot_transfer"))
    plt.close(fig)

    # Save table
    tab = []
    for s in subjects_sorted:
        tab.append({"subject": s, "deid_short": label_map[s]["short"],
                     "pred_severity_OFF": off_mean[s],
                     "pred_severity_ON": on_mean[s],
                     "delta_pred_severity": deltas[s],
                     "actual_UPDRS_OFF": dbs_off_updrs.get(s, np.nan),
                     "actual_UPDRS_ON": dbs_on_updrs.get(s, np.nan)})
    _save_csv(tab, os.path.join(TAB_DIR, "partD_zero_shot_transfer.csv"))

    return r_pool, n_decreased, deltas


# =============================================================================
# PART E: UPDRS-BLIND OFF/ON CLASSIFICATION
# =============================================================================

def part_E_updrs_blind_classification(dbs_dataset, pretrained_states, label_map,
                                        paired_rows):
    """Restrict to subjects with Delta UPDRS-gait = 0. Classify OFF vs ON.
    If AUC is high, SSL embeddings see change UPDRS doesn't."""
    print(f"\n{'='*70}\n  Part E: UPDRS-blind OFF/ON classification\n{'='*70}")

    delta_updrs = {r.get("subject"): r.get("delta_gait_updrs", np.nan) for r in paired_rows}
    blind_subjects = [s for s, d in delta_updrs.items() if np.isfinite(d) and d == 0]
    print(f"  UPDRS-blind subjects (Δ UPDRS-gait = 0): {len(blind_subjects)}")
    print(f"    {[label_map[s]['short'] for s in blind_subjects]}")

    # Filter dataset to these subjects
    blind_data = [d for d in dbs_dataset if d["subject"] in blind_subjects]
    if len(blind_data) == 0:
        print("  No blind subject data; skipping.")
        return

    y_all = np.array([0 if d["state"] == "OFF" else 1 for d in blind_data])
    g_all = np.array([d["subject"] for d in blind_data])
    pose_X_all = np.array([d["pose"] for d in blind_data])
    feat_X_all = np.array([d["feat_vec"] for d in blind_data])

    # LOSO: for each blind subject, train on remaining blind subjects, test on held-out.
    # Baselines: feature LogReg, SSL-pretrained ST-GCN (5 seeds).
    from sklearn.linear_model import LogisticRegression
    results = {"feat_logreg": {}, "pretrained_stgcn": defaultdict(list)}

    for held in blind_subjects:
        tr = g_all != held; te = g_all == held
        ytr, yte = y_all[tr], y_all[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            results["feat_logreg"][held] = np.nan
            for seed in SEEDS: results["pretrained_stgcn"][held].append(np.nan)
            continue
        # feat baseline
        sc = StandardScaler().fit(feat_X_all[tr])
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(sc.transform(feat_X_all[tr]), ytr)
        p = clf.predict_proba(sc.transform(feat_X_all[te]))[:, 1]
        try: results["feat_logreg"][held] = roc_auc_score(yte, p)
        except ValueError: results["feat_logreg"][held] = np.nan

        # SSL-pretrained ST-GCN per seed
        for seed in SEEDS:
            p_pt = finetune_clf(pose_X_all[tr], ytr, pose_X_all[te],
                                 pretrained_states[seed], seed)
            try: auc = roc_auc_score(yte, p_pt)
            except ValueError: auc = np.nan
            results["pretrained_stgcn"][held].append(auc)

        feat_auc = results["feat_logreg"][held]
        ssl_mean = float(np.mean([a for a in results["pretrained_stgcn"][held]
                                   if np.isfinite(a)]))
        print(f"    held-out {label_map[held]['short']:5s}  "
              f"feat={feat_auc:.3f}  SSL-ST-GCN={ssl_mean:.3f}")

    # Summary
    feat_aucs = [results["feat_logreg"][s] for s in blind_subjects
                  if np.isfinite(results["feat_logreg"][s])]
    ssl_flat = [a for s in blind_subjects for a in results["pretrained_stgcn"][s]
                 if np.isfinite(a)]
    ssl_per_subj_mean = {s: float(np.mean([a for a in results["pretrained_stgcn"][s]
                                            if np.isfinite(a)]))
                          for s in blind_subjects}
    feat_median = float(np.median(feat_aucs)) if feat_aucs else np.nan
    ssl_median = float(np.median([ssl_per_subj_mean[s] for s in blind_subjects
                                    if np.isfinite(ssl_per_subj_mean[s])]))
    print(f"\n  Median AUCs over UPDRS-blind subjects:")
    print(f"    Feature LogReg: {feat_median:.3f}")
    print(f"    SSL-ST-GCN (seed-avg): {ssl_median:.3f}")

    # Figure
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    x = np.arange(len(blind_subjects)); bar_w = 0.35
    feat_vals = [results["feat_logreg"][s] for s in blind_subjects]
    ssl_means = [ssl_per_subj_mean.get(s, np.nan) for s in blind_subjects]
    ssl_sds = [float(np.std([a for a in results["pretrained_stgcn"][s] if np.isfinite(a)]))
               if results["pretrained_stgcn"][s] else 0 for s in blind_subjects]
    ax.bar(x - bar_w/2, feat_vals, width=bar_w, color="#1976D2",
           edgecolor="white", label="Feature LogReg", alpha=0.9)
    ax.bar(x + bar_w/2, ssl_means, width=bar_w, yerr=ssl_sds,
           color="#C62828", edgecolor="white",
           label="SSL-ST-GCN (mean ± SD over 5 seeds)",
           alpha=0.9, capsize=3)
    ax.axhline(0.5, color="gray", linestyle="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([label_map[s]["short"] for s in blind_subjects],
                                          fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"Part E: UPDRS-blind OFF/ON classification  "
                 f"(medians: feat={feat_median:.2f}, SSL-ST-GCN={ssl_median:.2f})\n"
                 f"Subjects with Δ UPDRS-gait = 0 — if AUC > 0.5, SSL sees change UPDRS can't",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_annotated_and_clean(fig, os.path.join(FIG_DIR, "partE_updrs_blind_classification"))
    plt.close(fig)

    tab = []
    for s in blind_subjects:
        tab.append({"subject": s, "deid_short": label_map[s]["short"],
                     "feat_AUC": results["feat_logreg"][s],
                     "ssl_stgcn_AUC_mean": ssl_per_subj_mean.get(s, np.nan),
                     "ssl_stgcn_AUC_sd": float(np.std([a for a in results["pretrained_stgcn"][s]
                                                         if np.isfinite(a)]))
                                             if results["pretrained_stgcn"][s] else np.nan})
    _save_csv(tab, os.path.join(TAB_DIR, "partE_updrs_blind.csv"))


# =============================================================================
# PART F: DISPLACEMENT MAGNITUDE IN UPDRS-STEP UNITS
# =============================================================================

def part_F_updrs_step_units(cohort_pose_X, cohort_updrs_list, cohort_subj_list,
                              dbs_dataset, pretrained_states, label_map,
                              paired_rows):
    """For each seed:
        1. Compute cohort per-subject mean embedding.
        2. For each pair (UPDRS=1 subjects, UPDRS=2 subjects), compute Euclidean
           distances. Use median as "1-UPDRS-step" reference.
        3. For each DBS subject, compute OFF->ON centroid Euclidean distance.
        4. Express as a multiple of the 1-step reference.
       Aggregate across seeds."""
    print(f"\n{'='*70}\n  Part F: OFF→ON displacement in UPDRS-step units\n{'='*70}")

    dbs_subjects = sorted(set(d["subject"] for d in dbs_dataset))
    pose_X_dbs = np.array([d["pose"] for d in dbs_dataset])
    g_dbs = np.array([d["subject"] for d in dbs_dataset])
    state_dbs = np.array([d["state"] for d in dbs_dataset])

    # Cohort: aggregate strides per subject
    cohort_subj_unique = sorted(set(cohort_subj_list.tolist()))
    cohort_subj_updrs = {}
    for s in cohort_subj_unique:
        m = cohort_subj_list == s
        cohort_subj_updrs[s] = int(np.unique(cohort_updrs_list[m])[0])

    results_per_seed = {}
    for seed in SEEDS:
        state_dict = pretrained_states[seed]
        cohort_emb = extract_embeddings(cohort_pose_X, state_dict)
        # Per-cohort-subject mean embedding
        cohort_subj_emb = {}
        for s in cohort_subj_unique:
            m = cohort_subj_list == s
            cohort_subj_emb[s] = cohort_emb[m].mean(axis=0)

        # Compute 1-UPDRS-step reference distance: median of pairwise dists
        # between UPDRS=1 and UPDRS=2 cohort subjects.
        subs_u1 = [s for s, u in cohort_subj_updrs.items() if u == 1]
        subs_u2 = [s for s, u in cohort_subj_updrs.items() if u == 2]
        subs_u3 = [s for s, u in cohort_subj_updrs.items() if u == 3]
        d12 = []
        for a in subs_u1:
            for b in subs_u2:
                d12.append(np.linalg.norm(cohort_subj_emb[a] - cohort_subj_emb[b]))
        ref_step_12 = float(np.median(d12)) if d12 else np.nan
        d23 = []
        for a in subs_u2:
            for b in subs_u3:
                d23.append(np.linalg.norm(cohort_subj_emb[a] - cohort_subj_emb[b]))
        ref_step_23 = float(np.median(d23)) if d23 else np.nan

        # DBS embeddings
        dbs_emb = extract_embeddings(pose_X_dbs, state_dict)
        # Per-subject OFF->ON median centroid distance
        subj_disps = {}
        for s in dbs_subjects:
            m_off = (g_dbs == s) & (state_dbs == "OFF")
            m_on = (g_dbs == s) & (state_dbs == "ON")
            if m_off.sum() == 0 or m_on.sum() == 0:
                continue
            off_c = np.median(dbs_emb[m_off], axis=0)
            on_c = np.median(dbs_emb[m_on], axis=0)
            subj_disps[s] = float(np.linalg.norm(off_c - on_c))
        results_per_seed[seed] = {"ref_step_12": ref_step_12,
                                    "ref_step_23": ref_step_23,
                                    "subj_disps": subj_disps}

    # Aggregate
    ref12_vals = [results_per_seed[seed]["ref_step_12"] for seed in SEEDS]
    ref23_vals = [results_per_seed[seed]["ref_step_23"] for seed in SEEDS]
    print(f"  1-UPDRS-step reference (UPDRS=1↔2):  mean={np.mean(ref12_vals):.3f}, "
          f"SD={np.std(ref12_vals):.3f}")
    print(f"  1-UPDRS-step reference (UPDRS=2↔3):  mean={np.mean(ref23_vals):.3f}, "
          f"SD={np.std(ref23_vals):.3f}  (less reliable, few U3 subjects)")

    subj_units = {}  # subject -> list of (disp / ref) per seed
    for seed in SEEDS:
        ref = results_per_seed[seed]["ref_step_12"]
        if not np.isfinite(ref) or ref < 1e-6: continue
        for s, disp in results_per_seed[seed]["subj_disps"].items():
            subj_units.setdefault(s, []).append(disp / ref)

    subj_summary = {s: {"mean_units": float(np.mean(vs)),
                         "sd_units": float(np.std(vs))}
                     for s, vs in subj_units.items()}

    # Overall statistic: median OFF->ON distance in UPDRS units
    all_units = [subj_summary[s]["mean_units"] for s in subj_summary]
    median_units = float(np.median(all_units))
    print(f"\n  Median DBS OFF→ON displacement in 1-UPDRS-step units: {median_units:.2f}")

    # Fraction of subjects whose displacement >= 1 UPDRS step
    n_large = int(sum(1 for u in all_units if u >= 1.0))
    print(f"  Subjects with displacement ≥ 1 UPDRS step: {n_large}/{len(all_units)}")

    # Actual Δ UPDRS-gait for reference
    delta_updrs_map = {r.get("subject"): r.get("delta_gait_updrs", np.nan) for r in paired_rows}
    n_actual_moved = sum(1 for s in all_units if np.isfinite(delta_updrs_map.get(list(subj_summary.keys())[0], np.nan)))
    n_actual_moved = int(sum(1 for s in subj_summary
                              if np.isfinite(delta_updrs_map.get(s, np.nan)) and
                              abs(delta_updrs_map[s]) >= 1))

    # Figure
    subjects_sorted = sorted(subj_summary.keys(), key=lambda s: subj_summary[s]["mean_units"])
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    x = np.arange(len(subjects_sorted))
    vals = [subj_summary[s]["mean_units"] for s in subjects_sorted]
    sds = [subj_summary[s]["sd_units"] for s in subjects_sorted]
    delta_updrs_vals = [delta_updrs_map.get(s, np.nan) for s in subjects_sorted]
    colors = ["#111" if np.isfinite(d) and abs(d) >= 1 else "#888"
              for d in delta_updrs_vals]
    ax.bar(x, vals, yerr=sds, color=colors, edgecolor="white", capsize=3, alpha=0.9)
    ax.axhline(1.0, color="#C62828", lw=2, linestyle="--",
               label="1 UPDRS step (cohort UPDRS=1↔2 median distance)")
    ax.set_xticks(x); ax.set_xticklabels([label_map[s]["short"] for s in subjects_sorted],
                                          fontsize=9)
    ax.set_ylabel("OFF→ON displacement\n(units of cohort 1-UPDRS-step distance)",
                  fontsize=10)
    ax.set_xlabel("DBS subject", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    # Annotate each bar with actual Δ UPDRS-gait
    for xi, s in enumerate(subjects_sorted):
        d = delta_updrs_map.get(s, np.nan)
        tag = f"Δ={int(d):+d}" if np.isfinite(d) else "Δ=?"
        ax.text(xi, vals[xi] + sds[xi] + 0.05, tag, fontsize=7,
                ha="center", va="bottom",
                color="#111" if np.isfinite(d) and abs(d) >= 1 else "#888")
    fig.suptitle(f"Part F: DBS OFF→ON displacement in SSL space, measured in "
                 f"cohort 1-UPDRS-step units.\n"
                 f"Median = {median_units:.2f}; {n_large}/{len(all_units)} subjects "
                 f"exceed 1 step; only {n_actual_moved}/{len(all_units)} have "
                 f"actual UPDRS-gait change.",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_annotated_and_clean(fig, os.path.join(FIG_DIR, "partF_updrs_step_units"))
    plt.close(fig)

    tab = []
    for s in subjects_sorted:
        tab.append({"subject": s, "deid_short": label_map[s]["short"],
                     "displacement_mean_UPDRS_step_units": subj_summary[s]["mean_units"],
                     "displacement_sd_UPDRS_step_units": subj_summary[s]["sd_units"],
                     "actual_delta_UPDRS_gait": delta_updrs_map.get(s, np.nan)})
    _save_csv(tab, os.path.join(TAB_DIR, "partF_updrs_step_units.csv"))

    return median_units, n_large, n_actual_moved


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  Task 3 Benchmarks SSL   --  Parts B / C / D / E / F")
    print(f"  5 seeds: {SEEDS}")
    print(f"  Device: {DEVICE}")
    print(f"{'#'*70}")

    # Load DBS + cohort data
    print("\n  Loading DBS data...")
    paired_rows = load_paired_rows()
    dbs_feat = load_feature_dict(DBS_STRIDE_FEAT_PKL)
    pose_by_subj = {}
    for s in sorted(dbs_feat.keys()):
        pose_by_subj[s] = load_dbs_stride_poses(s)
    label_map = build_subject_label_map(sorted(dbs_feat.keys()))
    dbs_dataset = build_dbs_stride_dataset(dbs_feat, pose_by_subj)
    print(f"  DBS dataset: {len(dbs_dataset)} strides")

    print("\n  Loading cohort for SSL pre-training...")
    cohort_pose_dir = find_cohort_pose_dir()
    if cohort_pose_dir is None:
        print("  [!] Cohort pose directory not found."); raise SystemExit(1)
    cohort_feat = load_feature_dict(COHORT_STRIDE_FEAT_PKL)
    cohort_updrs = load_cohort_updrs()
    cohort_pose_X, cohort_subj_list, cohort_updrs_list = build_cohort_pose_dataset(
        cohort_pose_dir, cohort_feat, cohort_updrs)
    print(f"  Cohort pose bank: {cohort_pose_X.shape}")
    print(f"  Cohort UPDRS distribution: {dict(zip(*np.unique(cohort_updrs_list, return_counts=True)))}")

    # Pre-train with 5 seeds
    print(f"\n  SSL pre-training with {len(SEEDS)} seeds...")
    pretrained_states = pretrain_all_seeds(cohort_pose_X)

    # Part B
    part_B_data_efficiency(dbs_dataset, load_bodypart_labels(),
                            pretrained_states, label_map)

    # Part C
    subj_summary, mean_rho, std_rho, _, _ = part_C_severity_axis(
        cohort_pose_X, cohort_updrs_list, cohort_subj_list,
        dbs_dataset, pretrained_states, label_map, paired_rows)

    # Part D
    part_D_zero_shot_transfer(cohort_pose_X, cohort_updrs_list, dbs_dataset,
                                pretrained_states, label_map, paired_rows)

    # Part E
    part_E_updrs_blind_classification(dbs_dataset, pretrained_states, label_map,
                                        paired_rows)

    # Part F
    part_F_updrs_step_units(cohort_pose_X, cohort_updrs_list, cohort_subj_list,
                              dbs_dataset, pretrained_states, label_map,
                              paired_rows)

    print(f"\n{'#'*70}")
    print(f"  Task 3 SSL  Complete")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Tables:  {TAB_DIR}")
    print(f"  Checkpoints: {CKPT_DIR}")
    print(f"{'#'*70}\n")
