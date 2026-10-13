# =============================================================================
# Transformer Chunk Multimodal Baseline — v8 (Gated Fusion Audio)
#
# Pipeline độc lập:
#   OpenFace raw -> chunk min/max/var -> Transformer Encoder
#   I3D -> MLP
#   BERT (audio_text[:768]) -> bert_proj -> 128d  ┐
#   Acoustics (audio_text[768:]) -> acou_proj -> 128d┘
#   gate = sigmoid(Linear(256→128))
#   x_audio = gate * bert_feat + (1-gate) * acou_feat
#   Fusion -> score regression
#
# Metric chính:
#   Combined = 0.5 * AvgAcc + 0.5 * MacroF1
# =============================================================================

import os
import pickle
import random
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import confusion_matrix
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================
INPUT_ROOT = r"D:\Study\khoa_luan\CODE\processed"
DATA_DIR   = os.path.join(INPUT_ROOT, "data_v4_300")
OUTPUT_DIR = os.path.join(INPUT_ROOT, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PKL = os.path.join(DATA_DIR, "train.pkl")
VAL_PKL   = os.path.join(DATA_DIR, "val.pkl")

# =============================================================================
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CÔNG TẮC THỬ NGHIỆM — sửa ở đây, không cần tìm trong code                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# =============================================================================

# --- CÔNG TẮC 1: dùng audio hay visual-only ---
#   True  → audio_text = BERT 768d + acoustics thật (v8 nguyên bản)
#   False → audio_text = toàn số 0, has_audio = 0 (visual-only, giống v9A P1)
USE_AUDIO = True

# --- CÔNG TẮC 2: nguồn cờ audio (chỉ có tác dụng khi USE_AUDIO=True) ---
#   "has_audio_text" → cờ gốc của v8 (100% mẫu = 1, thực chất vô hiệu)
#   "speech_flag"    → cờ lời nói THẬT khôi phục từ dữ liệu thô (23,6% mẫu = 1)
AUDIO_FLAG_SOURCE = "has_audio_text"

# Tên checkpoint tự sinh theo cấu hình để không ghi đè nhau
_mode_tag = "vis" if not USE_AUDIO else ("aud" if AUDIO_FLAG_SOURCE == "has_audio_text" else "audsp")

BATCH_SIZE = 128

MODEL_PATH    = os.path.join(OUTPUT_DIR, f"model_t_c_m_bacc_v8b_{_mode_tag}_300_10_{BATCH_SIZE}_42_cp1.pth")
EPOCH_LOG_CSV = os.path.join(OUTPUT_DIR, f"train_log_v8b_{_mode_tag}_300_10_{BATCH_SIZE}_42.csv")
EPOCHS = 2000
LR = 2e-4
WEIGHT_DECAY = 1e-3

T_CHUNKS   = 10
HIDDEN_DIM = 128
BERT_DIM   = 768   # chiều BERT trong audio_text vector

N_HEADS  = 4
N_LAYERS = 2
FF_DIM   = 256
DROPOUT  = 0.15

PATIENCE  = 1500
MIN_DELTA = 0.01

CLASS_NAMES = ["HD", "DE", "EG", "HE"]
NUM_CLASSES = 4

THRESHOLDS = [-0.5, 0.0, 0.5]

LABEL_SCORES = {
    0: -1.0,
    1: -0.3,
    2:  0.3,
    3:  1.0,
}

# MoCoRank hyperparameters
MOCORANK_MOMENTUM = 0.999
SCORE_POOL_SIZE   = 2048
MOCORANK_WEIGHT   = 1.0

# Loss weights
MSE_WEIGHT = 1.0

# Resume training từ checkpoint
RESUME = False

# Điều kiện phụ khi save
AVG_ACC_FILTER = False  # nếu True, chỉ save khi avg_acc > AVG_ACC_MIN
AVG_ACC_MIN    = 60.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# =============================================================================
# SEED
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


set_seed(SEED)
print(f"[INFO] Device: {DEVICE}")


# =============================================================================
# BASIC HELPERS
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


def print_class_dist(data, name):
    labels = [int(s["label"]) for s in data]
    counts = np.bincount(labels, minlength=NUM_CLASSES)

    print(f"[INFO] {name} class dist:")
    for i, c in enumerate(counts):
        print(f"  {CLASS_NAMES[i]}: {c}")

    return counts


def chunk_openface(seq, T=T_CHUNKS):
    """
    seq: (frames, D)
    return: (T, 3D)

    Mỗi chunk lấy:
      min
      max
      var
    """
    frames, D = seq.shape

    if frames < T:
        repeats = (T // max(frames, 1)) + 1
        seq = np.tile(seq, (repeats, 1))

    chunks = np.array_split(seq, T, axis=0)
    out = []

    for c in chunks:
        if len(c) == 0:
            out.append(np.zeros(3 * D, dtype=np.float32))
        else:
            out.append(
                np.concatenate(
                    [
                        c.min(axis=0),
                        c.max(axis=0),
                        c.var(axis=0),
                    ]
                ).astype(np.float32)
            )

    return np.stack(out, axis=0)


def compute_norm_stats(train_data):
    """
    Pipeline độc lập nên tự tính norm_stats từ train set.
    Chuẩn hóa I3D và audio_text.
    OpenFace chunk không chuẩn hóa ở đây.
    """
    i3d_list = []
    audio_list = []

    for s in train_data:
        i3d = s["i3d"].astype(np.float32)
        if i3d.shape[0] != 1024:
            i3d = np.zeros(1024, dtype=np.float32)

        if "audio_text" in s:
            audio = s["audio_text"].astype(np.float32)
        else:
            audio = np.zeros(1, dtype=np.float32)

        i3d_list.append(i3d)
        audio_list.append(audio)

    i3d_arr = np.stack(i3d_list, axis=0)
    audio_arr = np.stack(audio_list, axis=0)

    i3d_mean = i3d_arr.mean(axis=0).astype(np.float32)
    i3d_std = i3d_arr.std(axis=0).astype(np.float32)
    i3d_std[i3d_std < 1e-6] = 1.0

    audio_mean = audio_arr.mean(axis=0).astype(np.float32)
    audio_std = audio_arr.std(axis=0).astype(np.float32)
    audio_std[audio_std < 1e-6] = 1.0

    return {
        "i3d_mean": i3d_mean,
        "i3d_std": i3d_std,
        "audio_mean": audio_mean,
        "audio_std": audio_std,
    }


# =============================================================================
# DATASET
# =============================================================================
class CMOSEDataset(Dataset):
    def __init__(self, data, norm_stats=None):
        self.data = data
        self.norm_stats = norm_stats

        if len(data) > 0 and "audio_text" in data[0]:
            self.audio_dim = data[0]["audio_text"].shape[0]
        else:
            self.audio_dim = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]

        openface = chunk_openface(s["openface"], T=T_CHUNKS)

        i3d = s["i3d"].astype(np.float32)
        if i3d.shape[0] != 1024:
            i3d = np.zeros(1024, dtype=np.float32)

        # ── CÔNG TẮC 1: USE_AUDIO ──
        if USE_AUDIO:
            audio_text = (s["audio_text"].astype(np.float32) if "audio_text" in s
                          else np.zeros(self.audio_dim, dtype=np.float32))
            # ── CÔNG TẮC 2: nguồn cờ ──
            if AUDIO_FLAG_SOURCE == "speech_flag":
                has_audio_text = float(s.get("speech_flag", 0))
            else:
                has_audio_text = float(s.get("has_audio_text", 0))
        else:
            # visual-only: audio = 0, cờ = 0 (giống v9A Phase 1)
            audio_text     = np.zeros(self.audio_dim, dtype=np.float32)
            has_audio_text = 0.0

        if self.norm_stats is not None:
            i3d = (i3d - self.norm_stats["i3d_mean"]) / self.norm_stats["i3d_std"]
            audio_text = (
                audio_text - self.norm_stats["audio_mean"]
            ) / self.norm_stats["audio_std"]

        # speech_flag luôn trả về để TÁCH SUBSET lúc eval (không vào model)
        speech_flag = float(s.get("speech_flag", 0))

        return (
            torch.tensor(openface, dtype=torch.float32),
            torch.tensor(i3d, dtype=torch.float32),
            torch.tensor(audio_text, dtype=torch.float32),
            torch.tensor(has_audio_text, dtype=torch.float32),
            torch.tensor(speech_flag, dtype=torch.float32),
            torch.tensor(float(s["score"]), dtype=torch.float32),
            torch.tensor(int(s["label"]), dtype=torch.long),
        )


# =============================================================================
# MODEL
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
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        T=10,
        n_heads=4,
        n_layers=2,
        ff_dim=256,
        dropout=0.15,
    ):
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
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, T, D)
        h = self.input_proj(x)
        h = h + self.pos_embed[:, :h.size(1), :]
        h = self.encoder(h)
        h = self.norm(h)

        # return: (B, T, C)
        return h


class TransformerFusionModel(nn.Module):
    def __init__(
        self,
        openface_dim,
        i3d_dim,
        audio_dim,
        hidden_dim=128,
        T=10,
    ):
        super().__init__()

        self.openface_encoder = OpenFaceTransformer(
            input_dim=openface_dim,
            hidden_dim=hidden_dim,
            T=T,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            ff_dim=FF_DIM,
            dropout=DROPOUT,
        )

        self.i3d_attn = nn.Sequential(
            nn.Linear(i3d_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, T),
        )

        self.i3d_mlp = nn.Sequential(
            nn.Linear(i3d_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Gated fusion: tách BERT và acoustics xử lý riêng
        acou_dim = audio_dim - BERT_DIM  # số chiều acoustics = audio_dim - 768
        self.bert_proj = nn.Sequential(
            nn.Linear(BERT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.acou_proj = nn.Sequential(
            nn.Linear(acou_dim + 1, hidden_dim),  # +1 là has_audio flag
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        self.regressor = NormalizedLinear(hidden_dim)

    def forward(self, openface, i3d, audio_text, has_audio_text, return_embedding=False):
        # OpenFace Transformer.
        # openface_feat: (B, T, C)
        openface_feat = self.openface_encoder(openface)

        # I3D-guided attention.
        attn = F.softmax(self.i3d_attn(i3d), dim=1).unsqueeze(-1)  # (B, T, 1)
        x_openface = (openface_feat * attn).sum(dim=1)             # (B, C)

        x_i3d = self.i3d_mlp(i3d)

        has_audio_text = has_audio_text.unsqueeze(1)

        # Gated fusion: tách BERT và acoustics
        bert_feat  = self.bert_proj(audio_text[:, :BERT_DIM])
        acou_input = torch.cat([audio_text[:, BERT_DIM:], has_audio_text], dim=1)
        acou_feat  = self.acou_proj(acou_input)
        gate       = self.gate(torch.cat([bert_feat, acou_feat], dim=1))
        x_audio    = gate * bert_feat + (1.0 - gate) * acou_feat

        x = torch.cat([x_openface, x_i3d, x_audio], dim=1)
        embedding = self.fusion(x)                                  # (B, hidden_dim) — dùng cho MoCoRank

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
    true = np.asarray(labels, dtype=np.int64)

    pred = score_to_class(scores).numpy()

    cm = confusion_matrix(true, pred, labels=list(range(NUM_CLASSES)))

    overall_acc = (pred == true).mean() * 100.0

    recalls = []
    precisions = []
    f1s = []

    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp

        r = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
        p = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        recalls.append(r)
        precisions.append(p)
        f1s.append(f1)

    avg_acc = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    macro_precision = float(np.mean(precisions))
    combined = 0.5 * avg_acc + 0.5 * macro_f1

    true_group = (true >= 2).astype(np.int64)
    pred_group = (pred >= 2).astype(np.int64)

    group_acc = (true_group == pred_group).mean() * 100.0
    adjacent_acc = (np.abs(pred - true) <= 1).mean() * 100.0

    de_to_eg = int(cm[1, 2])
    eg_to_de = int(cm[2, 1])

    return {
        "combined": combined,
        "overall_acc": overall_acc,
        "avg_acc": avg_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "group_acc": group_acc,
        "adjacent_acc": adjacent_acc,
        "recalls": recalls,
        "precisions": precisions,
        "f1s": f1s,
        "cm": cm,
        "de_to_eg": de_to_eg,
        "eg_to_de": eg_to_de,
        "boundary_errors": de_to_eg + eg_to_de,
    }


def print_metrics(m, prefix):
    print(f"\n[{prefix}]")
    print(f"  Combined Score : {m['combined']:.2f}")
    print(f"  Overall Acc    : {m['overall_acc']:.2f}")
    print(f"  Avg Acc        : {m['avg_acc']:.2f}")
    print(f"  Macro F1       : {m['macro_f1']:.2f}")
    print(f"  Macro Precision: {m['macro_precision']:.2f}")
    print(f"  Group Acc      : {m['group_acc']:.2f}")
    print(f"  Adjacent Acc   : {m['adjacent_acc']:.2f}")

    print("  Per-class Recall / Precision / F1:")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"    {name}: "
            f"R={m['recalls'][i]:6.2f} | "
            f"P={m['precisions'][i]:6.2f} | "
            f"F1={m['f1s'][i]:6.2f}"
        )

    print("  Boundary:")
    print(f"    DE -> EG        : {m['de_to_eg']}")
    print(f"    EG -> DE        : {m['eg_to_de']}")
    print(f"    Boundary Errors : {m['boundary_errors']}")

    print("  Confusion Matrix rows=true, cols=pred")
    print("          HD    DE    EG    HE")
    for i, name in enumerate(CLASS_NAMES):
        row = m["cm"][i]
        print(f"    {name}  {row[0]:4d}  {row[1]:4d}  {row[2]:4d}  {row[3]:4d}")


# =============================================================================
# MOCORANK
# =============================================================================
class MoCoRankTrainer:
    """
    Implements MoCoRank theo paper Section 4.3.

    Score Pool lưu triplets: (label, score, embedding) từ momentum encoder.
    Cấu trúc FIFO queue, size cố định |P| = SCORE_POOL_SIZE.

    Momentum Encoder: cùng kiến trúc model, update EMA mỗi iter:
        momentum_encoder = 0.999 * momentum_encoder + 0.001 * model
    """

    def __init__(self, model, pool_size=SCORE_POOL_SIZE, momentum=MOCORANK_MOMENTUM):
        self.momentum = momentum
        self.pool_size = pool_size

        # Momentum encoder: copy kiến trúc + weight từ model ban đầu
        self.momentum_encoder = copy.deepcopy(model)
        for p in self.momentum_encoder.parameters():
            p.requires_grad_(False)
        self.momentum_encoder.eval()

        # Score Pool — lưu dưới dạng list để FIFO dễ quản lý
        # Mỗi entry: dict {"label": int, "score": float, "embedding": Tensor(C,)}
        self._pool_labels     = []  # list[int]
        self._pool_scores     = []  # list[float]
        self._pool_embeddings = []  # list[Tensor(C,)] on CPU

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_ready(self):
        """Pool đã có đủ dữ liệu để tính loss chưa."""
        return len(self._pool_labels) > 0

    @torch.no_grad()
    def update(self, openface, i3d, audio, has_audio, labels):
        """
        Sau mỗi iter:
          1. Chạy momentum encoder → lấy score + embedding
          2. Đẩy vào queue, xóa phần tử cũ nhất nếu vượt pool_size
          3. EMA update momentum encoder
        """
        # 1. Forward qua momentum encoder (no_grad)
        scores_mom, embs_mom = self.momentum_encoder(
            openface, i3d, audio, has_audio, return_embedding=True
        )
        scores_mom = scores_mom.detach().cpu()   # (B,)
        embs_mom   = embs_mom.detach().cpu()     # (B, C)
        labels_cpu = labels.detach().cpu()       # (B,)

        # 2. Append vào pool
        for i in range(len(labels_cpu)):
            self._pool_labels.append(int(labels_cpu[i].item()))
            self._pool_scores.append(float(scores_mom[i].item()))
            self._pool_embeddings.append(embs_mom[i])           # Tensor(C,)

        # 3. Trim FIFO — giữ pool_size phần tử mới nhất
        if len(self._pool_labels) > self.pool_size:
            overflow = len(self._pool_labels) - self.pool_size
            self._pool_labels     = self._pool_labels[overflow:]
            self._pool_scores     = self._pool_scores[overflow:]
            self._pool_embeddings = self._pool_embeddings[overflow:]

        # 4. EMA update momentum encoder weights
        self._ema_update_momentum_encoder()

    def _ema_update_momentum_encoder(self):
        """
        momentum_encoder = m * momentum_encoder + (1-m) * model
        Theo paper: m = 0.999
        """
        # Lấy model tham chiếu từ momentum encoder (state_dict không có ref đến model)
        # Nên ta sẽ truyền model vào update() — xem phiên bản có model param bên dưới.
        pass  # handled in update_with_model()

    @torch.no_grad()
    def update_with_model(self, model, openface, i3d, audio, has_audio, labels):
        """
        Gộp: chạy momentum encoder + EMA update.
        Đây là entry point chính được gọi trong train_one_epoch.
        """
        # Forward momentum encoder
        scores_mom, embs_mom = self.momentum_encoder(
            openface, i3d, audio, has_audio, return_embedding=True
        )
        scores_mom = scores_mom.detach().cpu()
        embs_mom   = embs_mom.detach().cpu()
        labels_cpu = labels.detach().cpu()

        # Append pool
        for i in range(len(labels_cpu)):
            self._pool_labels.append(int(labels_cpu[i].item()))
            self._pool_scores.append(float(scores_mom[i].item()))
            self._pool_embeddings.append(embs_mom[i])

        if len(self._pool_labels) > self.pool_size:
            overflow = len(self._pool_labels) - self.pool_size
            self._pool_labels     = self._pool_labels[overflow:]
            self._pool_scores     = self._pool_scores[overflow:]
            self._pool_embeddings = self._pool_embeddings[overflow:]

        # EMA update: m_enc = m * m_enc + (1-m) * model
        for p_mom, p_model in zip(
            self.momentum_encoder.parameters(), model.parameters()
        ):
            p_mom.data.mul_(self.momentum).add_(
                p_model.detach().data * (1.0 - self.momentum)
            )

    def compute_mocorank_loss(self, pred_scores, pred_embeddings, batch_labels):
        """
        Multi-Margin Loss theo paper Section 4.3.2.

        Args:
            pred_scores     : Tensor (B,) — scores từ model (on device)
            pred_embeddings : Tensor (B, C) — embeddings từ model (on device)
            batch_labels    : Tensor (B,) long — ground truth labels của batch

        Returns:
            loss: scalar Tensor
        """
        if not self.is_ready():
            return torch.tensor(0.0, device=pred_scores.device)

        B = pred_scores.size(0)
        P = len(self._pool_labels)

        # Chuyển pool sang tensor trên cùng device
        dev = pred_scores.device
        pool_labels_t = torch.tensor(
            self._pool_labels, dtype=torch.long, device=dev
        )                                                            # (P,)
        pool_scores_t = torch.tensor(
            self._pool_scores, dtype=torch.float32, device=dev
        )                                                            # (P,)
        pool_embs_t   = torch.stack(self._pool_embeddings, dim=0).to(dev)  # (P, C)

        # Normalize embeddings để tính cosine similarity
        e1 = F.normalize(pred_embeddings, dim=1)                    # (B, C)
        e2 = F.normalize(pool_embs_t, dim=1)                        # (P, C)

        # cosine similarity: (B, P)
        cos_sim = e1 @ e2.T                                         # (B, P) in [-1, 1]

        # Normalize cosine về [0, 1]: (cos + 1) / 2
        cos_norm = (cos_sim + 1.0) / 2.0                            # (B, P)

        # Margin theo paper:
        #   M1 = 0.5 * cos_norm          (|l1 - l2| = 1)
        #   M2 = 0.5 + 0.5 * cos_norm    (|l1 - l2| = 2)
        #   M3 = 1.0 + 0.5 * cos_norm    (|l1 - l2| = 3)
        # Tổng quát: M_k = (k-1)*0.5 + 0.5 * cos_norm, k = |l1-l2| in {1,2,3}
        #          = 0.5*(k-1) + 0.5*cos_norm

        # Tính label diff: (B, P)
        l1 = batch_labels.unsqueeze(1).float()                      # (B, 1)
        l2 = pool_labels_t.unsqueeze(0).float()                     # (1, P)
        label_diff = l1 - l2                                        # (B, P), signed
        abs_diff   = label_diff.abs().long()                        # (B, P), in {0,1,2,3}

        # Score diff: model(d1) - s2
        s1 = pred_scores.unsqueeze(1)                               # (B, 1)
        s2 = pool_scores_t.unsqueeze(0)                             # (1, P)
        score_diff = s1 - s2                                        # (B, P)

        # Tính margin M cho mỗi cặp
        k = abs_diff.float().clamp(min=1)                           # (B, P)
        margin = 0.5 * (k - 1.0) + 0.5 * cos_norm                  # (B, P)

        # f(l1, d1, l2, s2, e2):
        #   same class   (l1 == l2): L1(model(d1) - s2)  = |score_diff|
        #   l1 > l2:   margin - (model(d1) - s2)   = margin - score_diff
        #   l1 < l2:   margin - (s2 - model(d1))   = margin + score_diff
        same_mask    = (abs_diff == 0)                              # (B, P) bool
        higher_mask  = (label_diff > 0)                             # l1 > l2
        lower_mask   = (label_diff < 0)                             # l1 < l2

        # Tính f cho từng trường hợp rồi blend
        f_same    = score_diff.abs()                                # L1 loss
        f_higher  = margin - score_diff                             # penalty nếu score_diff < margin
        f_lower   = margin + score_diff                             # penalty nếu -score_diff < margin

        f = torch.where(same_mask,  f_same,
            torch.where(higher_mask, f_higher, f_lower))           # (B, P)

        # Loss = mean(max(f, 0))
        loss = F.relu(f).mean()
        return loss



@torch.no_grad()
def evaluate(model, loader, speech_only=None):
    """
    speech_only=None  -> toàn bộ mẫu
    speech_only=True  -> chỉ clip CÓ lời nói (speech_flag=1)
    speech_only=False -> chỉ clip KHÔNG lời nói (speech_flag=0)
    """
    model.eval()

    all_scores = []
    all_labels = []

    for openface, i3d, audio, has_audio, speech_flag, y_score, y_label in tqdm(
        loader,
        desc="  Evaluating",
        leave=False,
    ):
        openface = openface.to(DEVICE)
        i3d = i3d.to(DEVICE)
        audio = audio.to(DEVICE)
        has_audio = has_audio.to(DEVICE)

        scores = model(openface, i3d, audio, has_audio)
        scores = scores.cpu().numpy()
        flags  = speech_flag.numpy()
        labels = y_label.numpy()

        for sc, fl, lb in zip(scores, flags, labels):
            if speech_only is True and fl < 0.5:
                continue
            if speech_only is False and fl >= 0.5:
                continue
            all_scores.append(float(sc))
            all_labels.append(int(lb))

    return compute_metrics(all_scores, all_labels)


def train_one_epoch(model, loader, optimizer, epoch, mocorank_trainer=None):
    model.train()

    total_loss       = 0.0
    total_mse_loss   = 0.0
    total_moco_loss  = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} Train", leave=False)

    for openface, i3d, audio, has_audio, speech_flag, y_score, y_label in pbar:
        openface  = openface.to(DEVICE)
        i3d       = i3d.to(DEVICE)
        audio     = audio.to(DEVICE)
        has_audio = has_audio.to(DEVICE)
        y_score   = y_score.to(DEVICE)
        y_label   = y_label.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        # Forward model — lấy cả score lẫn embedding cho MoCoRank
        if mocorank_trainer is not None:
            pred_score, pred_emb = model(
                openface, i3d, audio, has_audio, return_embedding=True
            )
        else:
            pred_score = model(openface, i3d, audio, has_audio)

        mse_loss = F.mse_loss(pred_score, y_score)

        if mocorank_trainer is not None and mocorank_trainer.is_ready():
            moco_loss = mocorank_trainer.compute_mocorank_loss(
                pred_score, pred_emb, y_label
            )
            loss = MSE_WEIGHT * mse_loss + MOCORANK_WEIGHT * moco_loss
        else:
            moco_loss = torch.tensor(0.0)
            loss = MSE_WEIGHT * mse_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # Sau khi update model, chạy momentum encoder và cập nhật score pool + EMA
        if mocorank_trainer is not None:
            with torch.no_grad():
                mocorank_trainer.update_with_model(
                    model, openface, i3d, audio, has_audio, y_label
                )

        total_loss      += float(loss.item())
        total_mse_loss  += float(mse_loss.item())
        total_moco_loss += float(moco_loss.item())
        n += 1

        pbar.set_postfix({
            "loss": f"{total_loss/n:.4f}",
            "mse":  f"{total_mse_loss/n:.4f}",
            "moco": f"{total_moco_loss/n:.4f}",
        })

    return {
        "loss":      total_loss      / max(n, 1),
        "mse_loss":  total_mse_loss  / max(n, 1),
        "moco_loss": total_moco_loss / max(n, 1),
    }


def make_sampler(data):
    labels = np.asarray([int(s["label"]) for s in data], dtype=np.int64)
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)

    weights_per_class = 1.0 / np.maximum(counts, 1.0)
    weights = weights_per_class[labels]

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def save_model(path, model, optimizer, epoch, metrics, norm_stats, dims):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),

        "val_score": metrics["combined"],
        "val_accuracy": metrics["overall_acc"],
        "val_avg_accuracy": metrics["avg_acc"],
        "val_macro_f1": metrics["macro_f1"],
        "val_macro_precision": metrics["macro_precision"],
        "val_group_acc": metrics["group_acc"],
        "val_adjacent_acc": metrics["adjacent_acc"],
        "val_de_to_eg": metrics["de_to_eg"],
        "val_eg_to_de": metrics["eg_to_de"],
        "val_boundary_errors": metrics["boundary_errors"],

        "norm_stats": norm_stats,

        "config": {
            "model": "Transformer Chunk Multimodal v8b Gated Fusion",
            "use_audio": USE_AUDIO,
            "audio_flag_source": AUDIO_FLAG_SOURCE,
            "openface_dim": dims["openface_dim"],
            "i3d_dim": dims["i3d_dim"],
            "audio_dim": dims["audio_dim"],
            "hidden_dim": HIDDEN_DIM,
            "t_chunks": T_CHUNKS,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "ff_dim": FF_DIM,
            "dropout": DROPOUT,
            "thresholds": THRESHOLDS,
            "class_names": CLASS_NAMES,
            "bert_dim": BERT_DIM,
            "mse_weight": MSE_WEIGHT,
            "mocorank_weight": MOCORANK_WEIGHT,
        },
    }

    torch.save(ckpt, path)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("[INFO] Loading data...")
    train_data = load_pickle(TRAIN_PKL)
    val_data = load_pickle(VAL_PKL)

    print(f"[INFO] Train samples: {len(train_data):,}")
    print(f"[INFO] Val samples  : {len(val_data):,}")

    print_class_dist(train_data, "Train")
    print_class_dist(val_data, "Val")

    print("\n[INFO] Computing norm_stats from train data...")
    norm_stats = compute_norm_stats(train_data)

    train_ds = CMOSEDataset(train_data, norm_stats=norm_stats)
    val_ds = CMOSEDataset(val_data, norm_stats=norm_stats)

    sample = train_ds[0]
    openface_dim = sample[0].shape[-1]
    i3d_dim = sample[1].shape[-1]
    audio_dim = sample[2].shape[-1]

    print("\n[INFO] Feature dims:")
    print(f"  openface_chunk_dim: {openface_dim}")
    print(f"  i3d_dim           : {i3d_dim}")
    print(f"  audio_text_dim    : {audio_dim}")

    dims = {
        "openface_dim": openface_dim,
        "i3d_dim": i3d_dim,
        "audio_dim": audio_dim,
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=make_sampler(train_data),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model = TransformerFusionModel(
        openface_dim=openface_dim,
        i3d_dim=i3d_dim,
        audio_dim=audio_dim,
        hidden_dim=HIDDEN_DIM,
        T=T_CHUNKS,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_score  = -1e9
    best_epoch  = 0
    best_state  = copy.deepcopy(model.state_dict())
    bad_epochs  = 0
    start_epoch = 1

    if RESUME and os.path.exists(MODEL_PATH):
        print(f"\n[INFO] Resuming from: {MODEL_PATH}")
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_score  = ckpt.get("val_accuracy", -1e9)
        best_epoch  = ckpt.get("epoch", 0)
        start_epoch = best_epoch + 1
        best_state  = copy.deepcopy(model.state_dict())
        print(f"[INFO] Resumed epoch={best_epoch}, best_score={best_score:.2f}")
    elif RESUME:
        print(f"[WARN] RESUME=True nhưng không tìm thấy: {MODEL_PATH}")

    print("\n[INFO] Initializing MoCoRankTrainer...")
    print(f"[INFO]   momentum       = {MOCORANK_MOMENTUM}")
    print(f"[INFO]   pool_size      = {SCORE_POOL_SIZE}")
    print(f"[INFO]   mse_weight     = {MSE_WEIGHT}")
    print(f"[INFO]   moco_weight    = {MOCORANK_WEIGHT}")
    print(f"[INFO]   resume         = {RESUME}")
    print(f"[INFO]   avg_acc_filter = {AVG_ACC_FILTER} (min={AVG_ACC_MIN})")
    mocorank_trainer = MoCoRankTrainer(
        model,
        pool_size=SCORE_POOL_SIZE,
        momentum=MOCORANK_MOMENTUM,
    )

    print("\n[INFO] Start training Transformer v8 Gated Fusion...")
    print(f"[INFO] Save path: {MODEL_PATH}")
    mode_desc = ("VISUAL-ONLY (audio = 0)" if not USE_AUDIO
                 else f"AUDIO ({AUDIO_FLAG_SOURCE})")
    print(f"[INFO] Chế độ: {mode_desc}  |  BATCH_SIZE={BATCH_SIZE}")
    print(f"[INFO] Epoch log CSV: {EPOCH_LOG_CSV}")

    csv_new = not os.path.exists(EPOCH_LOG_CSV)
    csv_f   = open(EPOCH_LOG_CSV, "a", encoding="utf-8")
    if csv_new:
        csv_f.write(
            "epoch,loss,mse_loss,moco_loss,"
            "full_acc,full_avg_acc,full_macro_f1,full_combined,"
            "full_HD_R,full_DE_R,full_EG_R,full_HE_R,"
            "sp_acc,sp_avg_acc,sp_macro_f1,sp_combined,"
            "sp_HD_R,sp_DE_R,sp_EG_R,sp_HE_R,"
            "saved,best_epoch\n"
        )
        csv_f.flush()

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_losses = train_one_epoch(
            model, train_loader, optimizer, epoch,
            mocorank_trainer=mocorank_trainer,
        )
        val_metrics = evaluate(model, val_loader, speech_only=None)
        val_speech  = evaluate(model, val_loader, speech_only=True)

        print(f"\nEpoch {epoch:03d}/{EPOCHS}")
        print(
            f"  Train Total Loss : {epoch_losses['loss']:.5f}"
            f"  (MSE={epoch_losses['mse_loss']:.5f}"
            f"  MoCo={epoch_losses['moco_loss']:.5f})"
            f"  PoolSize={len(mocorank_trainer._pool_labels)}"
        )
        print_metrics(val_metrics, "VAL full")
        print_metrics(val_speech,  "VAL speech-only")

        current_score = val_metrics["overall_acc"]
        current_avg   = val_metrics["avg_acc"]

        acc_improved = current_score > best_score + MIN_DELTA
        avg_ok       = (current_avg >= AVG_ACC_MIN) if AVG_ACC_FILTER else True

        if acc_improved and avg_ok:
            best_score = current_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0

            save_model(MODEL_PATH, model, optimizer, epoch, val_metrics, norm_stats, dims)

            print(
                f"[SAVE] New best model saved. "
                f"epoch={epoch}, overall_acc={best_score:.2f}, avg_acc={current_avg:.2f}"
            )
        else:
            bad_epochs += 1
            reason = ""
            if not acc_improved:
                reason += f"acc not improved ({current_score:.2f}<={best_score:.2f}+{MIN_DELTA})"
            if not avg_ok:
                reason += f"  avg_acc too low ({current_avg:.2f}<{AVG_ACC_MIN})"
            print(
                f"[INFO] No save. {reason} | "
                f"bad_epochs={bad_epochs}/{PATIENCE}, "
                f"best_epoch={best_epoch}, best_score={best_score:.2f}"
            )

        # ── ghi CSV mỗi epoch ──
        _saved = 1 if (acc_improved and avg_ok) else 0
        _sp = val_speech if val_speech is not None else {
            "overall_acc": 0, "avg_acc": 0, "macro_f1": 0, "combined": 0,
            "recalls": [0, 0, 0, 0],
        }
        csv_f.write(
            f"{epoch},{epoch_losses['loss']:.5f},{epoch_losses['mse_loss']:.5f},{epoch_losses['moco_loss']:.5f},"
            f"{val_metrics['overall_acc']:.2f},{val_metrics['avg_acc']:.2f},"
            f"{val_metrics['macro_f1']:.2f},{val_metrics['combined']:.2f},"
            f"{val_metrics['recalls'][0]:.2f},{val_metrics['recalls'][1]:.2f},"
            f"{val_metrics['recalls'][2]:.2f},{val_metrics['recalls'][3]:.2f},"
            f"{_sp['overall_acc']:.2f},{_sp['avg_acc']:.2f},"
            f"{_sp['macro_f1']:.2f},{_sp['combined']:.2f},"
            f"{_sp['recalls'][0]:.2f},{_sp['recalls'][1]:.2f},"
            f"{_sp['recalls'][2]:.2f},{_sp['recalls'][3]:.2f},"
            f"{_saved},{best_epoch}\n"
        )
        csv_f.flush()

        if bad_epochs >= PATIENCE:
            print("\n[EARLY STOP] Patience reached.")
            break

    csv_f.close()

    print("\n[INFO] Training finished.")
    print(f"[INFO] Best epoch: {best_epoch}")
    print(f"[INFO] Best overall_acc score: {best_score:.2f}")

    model.load_state_dict(best_state)
    final_metrics = evaluate(model, val_loader, speech_only=None)
    final_speech  = evaluate(model, val_loader, speech_only=True)
    print_metrics(final_metrics, "FINAL BEST VAL — full")
    print_metrics(final_speech,  "FINAL BEST VAL — speech-only")

    print(f"\n[INFO] Chế độ: {mode_desc}")
    print(f"[INFO] Best model path: {MODEL_PATH}")
    print(f"[INFO] Epoch log CSV : {EPOCH_LOG_CSV}")


if __name__ == "__main__":
    main()