import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import ijson
except ImportError:
    raise ImportError("Cần cài ijson: pip install ijson")

# =============================================================================
# STEP 1 v4 — Build CMOSE Dataset + SPEECH METADATA
#
# Khác với v3:
#   - Đọc thêm từ label_results_w_audio_final.json (raw, chưa xử lý):
#       speech_flag        : audio.label (0=no speech, 1=has speech)  ← FLAG THẬT
#       utterance_duration : độ dài speech (chiều L mà 6d cũ thiếu)
#       utterance_length   : số từ trong transcript
#   - Đọc bằng ijson (streaming) để không tràn RAM (file rất nặng vì acoustics)
#   - Clip không tìm thấy trong raw → speech_flag=0, duration=0, length=0
#
# Mỗi sample:
#   {
#       "clip_id": str,
#       "openface": np.ndarray,         # (MAX_LEN, D)
#       "i3d": np.ndarray,              # (1024,)
#       "audio_text": np.ndarray,       # 774d (768 BERT + 6 acoustics) — giữ nguyên từ v3
#       "has_audio_text": int,          # CŨ: 1 nếu có entry trong bert_ac file (gần như luôn 1)
#       "speech_flag": int,             # MỚI: 1 nếu thật sự có speech (từ TalkNet label)
#       "utterance_duration": float,    # MỚI: speech length (giây)
#       "utterance_length": int,        # MỚI: số từ
#       "label": int,                   # 0 HD, 1 DE, 2 EG, 3 HE
#       "score": float,
#       "split": str
#   }
# =============================================================================

# ===== CONFIG =====
CMOSE_ROOT  = r"D:\Study\khoa_luan\Dataset\CMOSE"
OUTPUT_ROOT = r"D:\Study\khoa_luan\CODE"

OPENFACE_DIR = os.path.join(CMOSE_ROOT, "secondfeature")
FINAL_JSON   = os.path.join(CMOSE_ROOT, "final_data_1.json")

# File 6d đã normalize (BERT + acoustics) — giữ nguyên như v3
AUDIO_TEXT_JSON = os.path.join(CMOSE_ROOT, "new_bert_ac_dict.json")

# MỚI: file raw chứa speech metadata thật
RAW_AUDIO_JSON = os.path.join(CMOSE_ROOT, "label_results_w_audio_final.json")

# Đổi tên output để không ghi đè data cũ
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "processed", "data_v4_200")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEN = 200
I3D_DIM = 1024
AUDIO_TEXT_DIM = 0  # tự detect

DROP_COLS = ["frame", "face_id", "timestamp", "confidence", "success"]
CLASS_NAMES = ["HD", "DE", "EG", "HE"]

LABEL_MAP = {
    "Highly Disengage": 0, "Disengage": 1, "Engage": 2, "Highly Engage": 3,
    "Highly Disengaged": 0, "Disengaged": 1, "Engaged": 2, "Highly Engaged": 3,
    "HD": 0, "DE": 1, "EG": 2, "HE": 3,
}


# =============================================================================
# Helper
# =============================================================================
def load_json(path, required=True):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(path)
        print(f"[WARN] Không thấy file: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def label_to_score(label):
    return {0: -1.0, 1: -0.3, 2: 0.3, 3: 1.0}[label]


def normalize_clip_id(x):
    x = str(x).strip()
    for ext in [".csv", ".mp4", ".avi", ".mov", ".npy"]:
        if x.endswith(ext):
            x = x[:-len(ext)]
    return x


def load_openface(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    df = df.fillna(0.0)
    if len(df) == 0 or len(df.columns) == 0:
        return None
    return df.values.astype(np.float32)


def pad_or_truncate(x, max_len=MAX_LEN):
    if len(x) >= max_len:
        return x[:max_len]
    pad = np.zeros((max_len - len(x), x.shape[1]), dtype=np.float32)
    return np.concatenate([x, pad], axis=0)


def flatten_numeric(obj):
    feats = []
    if obj is None:
        return feats
    if isinstance(obj, (int, float, np.integer, np.floating)):
        v = float(obj)
        if np.isfinite(v):
            feats.append(v)
    elif isinstance(obj, list):
        for item in obj:
            feats.extend(flatten_numeric(item))
    elif isinstance(obj, dict):
        for _, value in obj.items():
            feats.extend(flatten_numeric(value))
    return feats


def build_audio_text_lookup(audio_text_json):
    lookup = {}
    if not isinstance(audio_text_json, dict):
        return lookup
    for raw_key, value in audio_text_json.items():
        clip_id = normalize_clip_id(raw_key)
        feats = flatten_numeric(value)
        if len(feats) > 0:
            lookup[clip_id] = np.asarray(feats, dtype=np.float32)
    return lookup


def detect_audio_text_dim(lookup):
    if len(lookup) == 0:
        return 1
    return max(len(v) for v in lookup.values())


def get_audio_text_feature(clip_id, lookup, dim):
    if clip_id not in lookup:
        return np.zeros(dim, dtype=np.float32), 0
    x = lookup[clip_id]
    out = np.zeros(dim, dtype=np.float32)
    n = min(len(x), dim)
    out[:n] = x[:n]
    return out, 1


# =============================================================================
# MỚI: Đọc speech metadata từ file raw bằng ijson (streaming)
# =============================================================================
def build_speech_lookup(raw_json_path):
    """
    Đọc label_results_w_audio_final.json streaming.
    Trả về lookup[clip_id] = {
        "speech_flag": int,            # audio.label (0/1)
        "utterance_duration": float,
        "utterance_length": int,
    }
    Bỏ qua acoustics nặng để không tràn RAM.
    """
    if not os.path.exists(raw_json_path):
        print(f"[WARN] Không thấy file raw: {raw_json_path}")
        print(f"[WARN] speech_flag sẽ = 0 cho toàn bộ.")
        return {}

    lookup = {}
    print(f"[INFO] Đọc speech metadata (streaming) từ: {raw_json_path}")

    with open(raw_json_path, encoding="utf-8") as f:
        parser = ijson.kvitems(f, "")
        for raw_clip_id, clip in tqdm(parser, desc="Reading speech meta"):
            cid   = normalize_clip_id(raw_clip_id)
            audio = clip.get("audio", {}) if isinstance(clip, dict) else {}

            speech_flag = int(audio.get("label", 0) or 0)
            duration    = float(audio.get("utterance_duration", 0) or 0)
            length      = int(audio.get("utterance_length", 0) or 0)

            lookup[cid] = {
                "speech_flag":        speech_flag,
                "utterance_duration": duration,
                "utterance_length":   length,
            }

    n_speech = sum(v["speech_flag"] for v in lookup.values())
    print(f"[INFO] Speech lookup: {len(lookup):,} clips")
    print(f"[INFO]   có speech (flag=1): {n_speech:,} ({n_speech/max(len(lookup),1)*100:.1f}%)")
    return lookup


def get_speech_meta(clip_id, speech_lookup):
    """Trả về (speech_flag, duration, length, found)."""
    if clip_id in speech_lookup:
        m = speech_lookup[clip_id]
        return m["speech_flag"], m["utterance_duration"], m["utterance_length"], 1
    return 0, 0.0, 0, 0


def count_labels(samples):
    counts = [0, 0, 0, 0]
    for s in samples:
        counts[s["label"]] += 1
    return dict(zip(CLASS_NAMES, counts))


# =============================================================================
# Load metadata
# =============================================================================
print("[INFO] Loading final_data_1.json...")
final_data = load_json(FINAL_JSON, required=True)
print(f"[INFO] Clips in final_data_1.json: {len(final_data):,}")

print("[INFO] Loading audio/text feature file (6d BERT+acoustics)...")
audio_text_data = load_json(AUDIO_TEXT_JSON, required=False)
audio_text_lookup = build_audio_text_lookup(audio_text_data)
AUDIO_TEXT_DIM = detect_audio_text_dim(audio_text_lookup)
print(f"[INFO] Clips with audio/text feature: {len(audio_text_lookup):,}")
print(f"[INFO] AUDIO_TEXT_DIM: {AUDIO_TEXT_DIM}")

# MỚI: speech metadata
speech_lookup = build_speech_lookup(RAW_AUDIO_JSON)


# =============================================================================
# Build index từ final_data_1.json
# =============================================================================
index = {}
skip_bad_label = skip_bad_i3d = skip_bad_split = 0

for raw_clip_id, item in final_data.items():
    clip_id   = normalize_clip_id(raw_clip_id)
    split     = item.get("split")
    label_str = item.get("label")
    i3d       = item.get("embeds")

    if split not in ["train", "unlabel", "test"]:
        skip_bad_split += 1
        continue
    if label_str not in LABEL_MAP:
        skip_bad_label += 1
        continue
    if i3d is None or len(i3d) != I3D_DIM:
        skip_bad_i3d += 1
        continue

    label = LABEL_MAP[label_str]
    index[clip_id] = {
        "clip_id": clip_id,
        "split": split,
        "label": label,
        "score": label_to_score(label),
        "i3d": np.asarray(i3d, dtype=np.float32),
        "openface_path": os.path.join(OPENFACE_DIR, clip_id + ".csv"),
    }

print(f"[INFO] Valid clips after metadata filtering: {len(index):,}")
print(f"[INFO] Skip bad split: {skip_bad_split:,}  label: {skip_bad_label:,}  i3d: {skip_bad_i3d:,}")


# =============================================================================
# Split
# =============================================================================
train_ids = [k for k, v in index.items() if v["split"] == "train"]
val_ids   = [k for k, v in index.items() if v["split"] == "unlabel"]
test_ids  = [k for k, v in index.items() if v["split"] == "test"]

assert len(set(train_ids) & set(val_ids))  == 0
assert len(set(train_ids) & set(test_ids)) == 0
assert len(set(val_ids)   & set(test_ids)) == 0

print(f"[INFO] Split: train={len(train_ids):,}  val={len(val_ids):,}  test={len(test_ids):,}")


# =============================================================================
# Build dataset
# =============================================================================
def build_dataset(id_list, name):
    dataset = []
    skip_openface     = 0
    with_audio_text   = 0
    speech_found      = 0
    speech_positive   = 0

    for clip_id in tqdm(id_list, desc=f"Building {name}"):
        meta = index[clip_id]

        openface = load_openface(meta["openface_path"])
        if openface is None:
            skip_openface += 1
            continue
        openface = pad_or_truncate(openface, MAX_LEN)

        audio_text, has_audio_text = get_audio_text_feature(
            clip_id=clip_id, lookup=audio_text_lookup, dim=AUDIO_TEXT_DIM
        )
        with_audio_text += has_audio_text

        # MỚI: speech metadata
        speech_flag, duration, length, found = get_speech_meta(clip_id, speech_lookup)
        speech_found    += found
        speech_positive += speech_flag

        sample = {
            "clip_id": clip_id,
            "openface": openface.astype(np.float32),
            "i3d": meta["i3d"].astype(np.float32),
            "audio_text": audio_text.astype(np.float32),
            "has_audio_text": int(has_audio_text),
            "speech_flag": int(speech_flag),
            "utterance_duration": float(duration),
            "utterance_length": int(length),
            "label": int(meta["label"]),
            "score": float(meta["score"]),
            "split": meta["split"],
        }
        dataset.append(sample)

    print(f"[INFO] {name}:")
    print(f"  samples            : {len(dataset):,}")
    print(f"  label dist         : {count_labels(dataset)}")
    print(f"  with audio/text    : {with_audio_text:,}")
    print(f"  speech meta found  : {speech_found:,}  (not found → flag=0: {len(dataset)-speech_found:,})")
    print(f"  speech positive    : {speech_positive:,} ({speech_positive/max(len(dataset),1)*100:.1f}%)")
    return dataset


train_data = build_dataset(train_ids, "train")
val_data   = build_dataset(val_ids, "val")
test_data  = build_dataset(test_ids, "test")


# =============================================================================
# Save
# =============================================================================
train_path = os.path.join(OUTPUT_DIR, "train.pkl")
val_path   = os.path.join(OUTPUT_DIR, "val.pkl")
test_path  = os.path.join(OUTPUT_DIR, "test.pkl")
meta_path  = os.path.join(OUTPUT_DIR, "dataset_meta.json")

with open(train_path, "wb") as f:
    pickle.dump(train_data, f)
with open(val_path, "wb") as f:
    pickle.dump(val_data, f)
with open(test_path, "wb") as f:
    pickle.dump(test_data, f)


def speech_dist_by_class(samples):
    out = {c: [0, 0] for c in CLASS_NAMES}  # [no_speech, has_speech]
    for s in samples:
        out[CLASS_NAMES[s["label"]]][s["speech_flag"]] += 1
    return out


meta = {
    "MAX_LEN": MAX_LEN,
    "I3D_DIM": I3D_DIM,
    "AUDIO_TEXT_DIM": AUDIO_TEXT_DIM,
    "CLASS_NAMES": CLASS_NAMES,
    "new_fields": ["speech_flag", "utterance_duration", "utterance_length"],
    "splits": {
        "train.pkl": "split == train",
        "val.pkl": "split == unlabel",
        "test.pkl": "split == test",
    },
    "sizes": {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
    "label_dist": {
        "train": count_labels(train_data),
        "val": count_labels(val_data),
        "test": count_labels(test_data),
    },
    "speech_dist_by_class": {
        "train": speech_dist_by_class(train_data),
        "val": speech_dist_by_class(val_data),
        "test": speech_dist_by_class(test_data),
    },
    "paths": {
        "CMOSE_ROOT": CMOSE_ROOT,
        "FINAL_JSON": FINAL_JSON,
        "OPENFACE_DIR": OPENFACE_DIR,
        "AUDIO_TEXT_JSON": AUDIO_TEXT_JSON,
        "RAW_AUDIO_JSON": RAW_AUDIO_JSON,
    },
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n[DONE] Step1 v4 completed.")
print(f"  Saved: {train_path}")
print(f"  Saved: {val_path}")
print(f"  Saved: {test_path}")
print(f"  Saved: {meta_path}")

if len(train_data) > 0:
    s = train_data[0]
    print("\n[INFO] Sample fields:")
    print(f"  openface           : {s['openface'].shape}")
    print(f"  i3d                : {s['i3d'].shape}")
    print(f"  audio_text         : {s['audio_text'].shape}")
    print(f"  speech_flag        : {s['speech_flag']}")
    print(f"  utterance_duration : {s['utterance_duration']}")
    print(f"  utterance_length   : {s['utterance_length']}")
