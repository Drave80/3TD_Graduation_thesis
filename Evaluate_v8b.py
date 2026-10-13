# =============================================================================
# Evaluate_v8b.py — Đánh giá model v8b (Gated Fusion, 2 công tắc audio)
#
# Model: TransformerFusionModel (giống hệt v8 gốc)
#   openface_encoder, i3d_attn, i3d_mlp, bert_proj, acou_proj, gate,
#   fusion (hidden*3 → hidden), regressor (NormalizedLinear)
#
# TỰ ĐỘNG khớp cấu hình audio từ checkpoint:
#   use_audio=True,  audio_flag_source="has_audio_text" → v8 nguyên bản (cờ 100%)
#   use_audio=True,  audio_flag_source="speech_flag"    → cờ lời nói thật (23,6%)
#   use_audio=False                                     → visual-only (audio = 0)
#
# Nếu checkpoint không có 2 trường này (v8 gốc cũ), mặc định về v8 nguyên bản.
#
# Report 3 lát cắt: FULL / SPEECH-ONLY / NO-SPEECH — dùng speech_flag để tách,
# độc lập với việc model có dùng cờ nào lúc train.
# =============================================================================

import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================
INPUT_ROOT = r"D:\Study\khoa_luan\CODE\processed"
MODEL_DIR  = os.path.join(INPUT_ROOT, "models")

DATA_DIRS = [
    ("MAX=300", os.path.join(INPUT_ROOT, "data_v4_300")),
    ("MAX=200", os.path.join(INPUT_ROOT, "data_v4_200")),
    ("MAX=100", os.path.join(INPUT_ROOT, "data_v4_100")),
]

# Bỏ comment dòng nào muốn chạy. Ba cấu hình v8b không đè nhau về tên file.
MODELS = [
    # {
    #     "name": "v8b_audsp (cờ lời nói thật)",
    #     "path": os.path.join(MODEL_DIR, "model_t_c_m_bacc_v8b_audsp_300_10_128_42_cp1.pth"),
    # },
    {
        "name": "v8b_aud_cp1 (v8 nguyên bản)",
        "path": os.path.join(MODEL_DIR, "model_t_c_m_bacc_v8b_aud_300_10_128_42_cp1.pth"),
    },
    # {
    #     "name": "v8b_aud_cp2 (v8 nguyên bản)",
    #     "path": os.path.join(MODEL_DIR, "model_t_c_m_bacc_v8b_aud_300_10_128_42_cp2.pth"),
    # },
    # {
    #     "name": "v8b_vis (visual-only)",
    #     "path": os.path.join(MODEL_DIR, "model_t_c_m_bacc_v8b_vis_300_10_128_42_cp1.pth"),
    # },
    # # v8 gốc (nếu muốn đối chiếu — checkpoint cũ, không có 2 trường công tắc)
    # {
    #     "name": "v8 gốc",
    #     "path": os.path.join(MODEL_DIR, "model_t_c_m_bacc_v8_300_10_128_42_cp1.pth"),
    # },
]

T_CHUNKS   = 10
HIDDEN_DIM = 128
BERT_DIM   = 768
N_HEADS    = 4
N_LAYERS   = 2
FF_DIM     = 256
DROPOUT    = 0.15

CLASS_NAMES = ["HD", "DE", "EG", "HE"]
NUM_CLASSES = 4
THRESHOLDS  = [-0.5, 0.0, 0.5]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# HELPERS
# =============================================================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def score_to_class(scores):
    if isinstance(scores, np.ndarray):
        scores = torch.tensor(scores, dtype=torch.float32)
    pred = torch.zeros_like(scores, dtype=torch.long)
    pred[scores >= THRESHOLDS[0]] = 1
    pred[scores >= THRESHOLDS[1]] = 2
    pred[scores >= THRESHOLDS[2]] = 3
    return pred


def chunk_openface(seq, T=T_CHUNKS):
    frames, D = seq.shape
    if frames < T:
        seq = np.tile(seq, ((T // max(frames, 1)) + 1, 1))
    chunks = np.array_split(seq, T, axis=0)
    out = []
    for c in chunks:
        if len(c) == 0:
            out.append(np.zeros(3 * D, dtype=np.float32))
        else:
            out.append(np.concatenate([c.min(axis=0), c.max(axis=0), c.var(axis=0)]).astype(np.float32))
    return np.stack(out, axis=0)


# =============================================================================
# DATASET — khớp cấu hình audio của checkpoint
# =============================================================================
class CMOSEDataset(Dataset):
    def __init__(self, data, norm_stats=None, use_audio=True, flag_source="has_audio_text"):
        self.data        = data
        self.norm_stats  = norm_stats
        self.use_audio   = use_audio
        self.flag_source = flag_source
        self.audio_dim   = data[0]["audio_text"].shape[0] if len(data) > 0 and "audio_text" in data[0] else 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]

        openface = chunk_openface(s["openface"], T=T_CHUNKS)
        i3d = s["i3d"].astype(np.float32)
        if i3d.shape[0] != 1024:
            i3d = np.zeros(1024, dtype=np.float32)

        if self.use_audio:
            audio_text = (s["audio_text"].astype(np.float32) if "audio_text" in s
                          else np.zeros(self.audio_dim, dtype=np.float32))
            if self.flag_source == "speech_flag":
                has_audio_text = float(s.get("speech_flag", 0))
            else:
                has_audio_text = float(s.get("has_audio_text", 0))
        else:
            audio_text     = np.zeros(self.audio_dim, dtype=np.float32)
            has_audio_text = 0.0

        if self.norm_stats is not None:
            i3d        = (i3d - self.norm_stats["i3d_mean"]) / self.norm_stats["i3d_std"]
            audio_text = (audio_text - self.norm_stats["audio_mean"]) / self.norm_stats["audio_std"]

        # speech_flag luôn trả về để tách subset (độc lập với cờ model dùng)
        speech_flag = float(s.get("speech_flag", 0))

        return (
            torch.tensor(openface,       dtype=torch.float32),
            torch.tensor(i3d,            dtype=torch.float32),
            torch.tensor(audio_text,     dtype=torch.float32),
            torch.tensor(has_audio_text, dtype=torch.float32),
            torch.tensor(speech_flag,    dtype=torch.float32),
            torch.tensor(int(s["label"]), dtype=torch.long),
        )


# =============================================================================
# MODEL — khớp CHÍNH XÁC v8 / v8b
# =============================================================================
class NormalizedLinear(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(x, dim=1)
        return (x @ w.T).squeeze(-1)


class OpenFaceTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, T=10, n_heads=4, n_layers=2, ff_dim=256, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, T, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.input_proj(x)
        h = h + self.pos_embed[:, :h.size(1), :]
        h = self.encoder(h)
        h = self.norm(h)
        return h


class TransformerFusionModel(nn.Module):
    def __init__(self, openface_dim, i3d_dim, audio_dim, hidden_dim=128, T=10):
        super().__init__()

        self.openface_encoder = OpenFaceTransformer(
            input_dim=openface_dim, hidden_dim=hidden_dim, T=T,
            n_heads=N_HEADS, n_layers=N_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT,
        )
        self.i3d_attn = nn.Sequential(
            nn.Linear(i3d_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, T),
        )
        self.i3d_mlp = nn.Sequential(
            nn.Linear(i3d_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        acou_dim = audio_dim - BERT_DIM
        self.bert_proj = nn.Sequential(
            nn.Linear(BERT_DIM, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.acou_proj = nn.Sequential(
            nn.Linear(acou_dim + 1, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
        )
        self.regressor = NormalizedLinear(hidden_dim)

    def forward(self, openface, i3d, audio_text, has_audio_text, return_embedding=False):
        openface_feat = self.openface_encoder(openface)
        attn = F.softmax(self.i3d_attn(i3d), dim=1).unsqueeze(-1)
        x_openface = (openface_feat * attn).sum(dim=1)
        x_i3d = self.i3d_mlp(i3d)

        has_audio_text = has_audio_text.unsqueeze(1)
        bert_feat  = self.bert_proj(audio_text[:, :BERT_DIM])
        acou_input = torch.cat([audio_text[:, BERT_DIM:], has_audio_text], dim=1)
        acou_feat  = self.acou_proj(acou_input)
        gate       = self.gate(torch.cat([bert_feat, acou_feat], dim=1))
        x_audio    = gate * bert_feat + (1.0 - gate) * acou_feat

        x = torch.cat([x_openface, x_i3d, x_audio], dim=1)
        embedding = self.fusion(x)
        score = self.regressor(embedding)
        score = torch.clamp(score, -1.0, 1.0)

        if return_embedding:
            return score, embedding
        return score


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(scores, labels):
    scores = np.asarray(scores, dtype=np.float32)
    true   = np.asarray(labels, dtype=np.int64)
    if len(true) == 0:
        return None
    pred = score_to_class(scores).numpy()
    cm   = confusion_matrix(true, pred, labels=list(range(NUM_CLASSES)))

    overall_acc = (pred == true).mean() * 100.0
    recalls, precisions, f1s = [], [], []
    for i in range(NUM_CLASSES):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        r  = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
        p  = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        recalls.append(r); precisions.append(p); f1s.append(f1)

    avg_acc  = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    return {
        "n": len(true),
        "combined": 0.5 * avg_acc + 0.5 * macro_f1,
        "overall_acc": overall_acc, "avg_acc": avg_acc, "macro_f1": macro_f1,
        "macro_precision": float(np.mean(precisions)),
        "group_acc": ((true >= 2) == (pred >= 2)).mean() * 100.0,
        "adjacent_acc": (np.abs(pred - true) <= 1).mean() * 100.0,
        "recalls": recalls, "precisions": precisions, "f1s": f1s, "cm": cm,
    }


def print_block(m, title):
    if m is None:
        print(f"\n  [{title}] — không có mẫu nào.")
        return
    print(f"\n  ┌─ {title}  (n={m['n']})")
    print(f"  │  Combined : {m['combined']:.2f}%")
    print(f"  │  Acc      : {m['overall_acc']:.2f}%")
    print(f"  │  AvgAcc   : {m['avg_acc']:.2f}%")
    print(f"  │  MacroF1  : {m['macro_f1']:.2f}%")
    print(f"  │  GrpAcc   : {m['group_acc']:.2f}%   AdjAcc: {m['adjacent_acc']:.2f}%")
    print(f"  │  Per-class Recall:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  │    {name}: R={m['recalls'][i]:6.2f} | P={m['precisions'][i]:6.2f} | F1={m['f1s'][i]:6.2f}")
    print(f"  │  Confusion Matrix (rows=true, cols=pred):")
    print(f"  │         " + "".join(f"{c:>6}" for c in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"  │    {name:>3} " + "".join(f"{m['cm'][i][j]:6d}" for j in range(NUM_CLASSES)))
    print(f"  └─")


# =============================================================================
# EVAL
# =============================================================================
@torch.no_grad()
def run_eval(model, loader):
    model.eval()
    sc_all, lb_all, fl_all = [], [], []
    for openface, i3d, audio_text, has_audio, speech_flag, y_label in tqdm(loader, desc="  Eval", leave=False):
        scores = model(openface.to(DEVICE), i3d.to(DEVICE),
                       audio_text.to(DEVICE), has_audio.to(DEVICE))
        sc_all.extend(scores.cpu().numpy().tolist())
        lb_all.extend(y_label.numpy().tolist())
        fl_all.extend(speech_flag.numpy().tolist())

    sc_all = np.array(sc_all); lb_all = np.array(lb_all); fl_all = np.array(fl_all)
    sp = fl_all >= 0.5
    return (
        compute_metrics(sc_all, lb_all),
        compute_metrics(sc_all[sp],  lb_all[sp]),
        compute_metrics(sc_all[~sp], lb_all[~sp]),
    )


def print_ckpt_info(ckpt):
    if not isinstance(ckpt, dict):
        return
    epoch   = ckpt.get("epoch", "?")
    val_acc = ckpt.get("val_accuracy", ckpt.get("val_acc"))
    val_avg = ckpt.get("val_avg_accuracy", ckpt.get("val_avg"))
    val_f1  = ckpt.get("val_macro_f1")
    msg = f"  Saved at epoch {epoch}"
    if val_acc is not None: msg += f" | val_acc={val_acc:.2f}%"
    if val_avg is not None: msg += f" | val_avg={val_avg:.2f}%"
    if val_f1  is not None: msg += f" | val_f1={val_f1:.2f}%"
    print(msg)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"[INFO] Device: {DEVICE}")
    summary_rows = []

    for data_tag, data_dir in DATA_DIRS:
        test_pkl = os.path.join(data_dir, "test.pkl")
        if not os.path.exists(test_pkl):
            print(f"\n[SKIP] {data_tag}: không thấy {test_pkl}")
            continue

        print(f"\n{'─'*72}")
        print(f"[DATA] {data_tag}  →  {data_dir}")
        print(f"{'─'*72}")

        test_data = load_pickle(test_pkl)
        n_speech  = sum(s.get("speech_flag", 0) for s in test_data)
        print(f"[INFO] Test samples: {len(test_data):,}  |  speech={n_speech} ({n_speech/len(test_data)*100:.1f}%)")

        for minfo in MODELS:
            name, path = minfo["name"], minfo["path"]
            print(f"\n[INFO] Loading: {name}")
            print(f"       Path: {path}")
            if not os.path.exists(path):
                print(f"  [SKIP] File không tồn tại.")
                continue

            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            print_ckpt_info(ckpt)

            norm_stats = ckpt.get("norm_stats", None)
            cfg        = ckpt.get("config", {})

            # ── TỰ ĐỘNG khớp cấu hình audio từ checkpoint ──
            use_audio   = cfg.get("use_audio", True)              # mặc định: v8 nguyên bản
            flag_source = cfg.get("audio_flag_source", "has_audio_text")
            mode_desc = ("VISUAL-ONLY (audio = 0)" if not use_audio
                         else f"AUDIO ({flag_source})")

            openface_dim = cfg.get("openface_dim")
            i3d_dim      = cfg.get("i3d_dim", 1024)
            audio_dim    = cfg.get("audio_dim")
            if openface_dim is None:
                openface_dim = chunk_openface(test_data[0]["openface"], T=T_CHUNKS).shape[-1]
            if audio_dim is None:
                audio_dim = test_data[0]["audio_text"].shape[0]

            model = TransformerFusionModel(
                openface_dim=openface_dim, i3d_dim=i3d_dim, audio_dim=audio_dim,
                hidden_dim=HIDDEN_DIM, T=T_CHUNKS,
            ).to(DEVICE)

            try:
                model.load_state_dict(ckpt["model_state"], strict=True)
            except RuntimeError as e:
                print(f"  [ERROR] Checkpoint KHÔNG khớp TransformerFusionModel (v8/v8b):")
                print(f"          {str(e)[:300]}")
                print(f"  [HINT] v19 → Evaluate_v19.py | v20 → Evaluate_v20.py")
                print(f"         v13/v16/v17/v18 → Evaluate_v18.py")
                continue

            print(f"  [INFO] Model loaded. Chế độ: {mode_desc}  "
                  f"norm_stats={'yes' if norm_stats else 'no'}")

            test_ds = CMOSEDataset(test_data, norm_stats=norm_stats,
                                   use_audio=use_audio, flag_source=flag_source)
            test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                                     num_workers=0, pin_memory=True, drop_last=False)

            m_full, m_speech, m_nospeech = run_eval(model, test_loader)

            print(f"\n{'═'*72}")
            print(f"  MODEL: {name} [{data_tag}]  —  {mode_desc}")
            print(f"{'═'*72}")
            print_block(m_full,     "FULL test set")
            print_block(m_speech,   "SPEECH-ONLY subset (paper Table 5)")
            print_block(m_nospeech, "NO-SPEECH subset")

            summary_rows.append({
                "name": f"{name} [{data_tag}]",
                "full": m_full, "speech": m_speech, "nospeech": m_nospeech,
            })

    # ── SUMMARY ──
    print(f"\n\n{'═'*104}")
    print(f"  SUMMARY  (so sánh với paper Table 5)")
    print(f"{'═'*104}")
    print(f"  {'Model':<40} {'Subset':<12} {'n':>5} {'Acc':>8} {'AvgAcc':>8} {'MacroF1':>8}")
    print(f"  {'-'*98}")
    for row in summary_rows:
        for tag, key in [("FULL", "full"), ("SPEECH", "speech"), ("NO-SPEECH", "nospeech")]:
            m = row[key]
            if m is None:
                continue
            print(f"  {row['name']:<40} {tag:<12} {m['n']:>5} "
                  f"{m['overall_acc']:>7.2f}% {m['avg_acc']:>7.2f}% {m['macro_f1']:>7.2f}%")
        print(f"  {'-'*98}")

    print("\n[INFO] Evaluation complete.")
    print("[HINT] Paper Table 5 — FULL test set:")
    print("       Openface+I3D            : Acc=78.14  AvgAcc=55.74")
    print("       Openface+I3D+Audio      : Acc=78.55  AvgAcc=56.85")
    print("[HINT] Paper Table 5 — SPEECH subset:")
    print("       Openface+I3D            : Acc=71.42  AvgAcc=60.23")
    print("       Openface+I3D+Audio      : Acc=74.60  AvgAcc=63.70")
    print("[NOTE] Mốc đối chiếu — v8 gốc (data_300, FULL): Acc=78.09  AvgAcc=56.56  MacroF1=60.97")
    print("       v9A P1 (visual-only, MAX=100, FULL)    : Acc=77.67  AvgAcc=59.47  MacroF1=62.93")


if __name__ == "__main__":
    main()
