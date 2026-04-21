# %% [markdown]
# Wczytujemy dane z cache i dzielimy je na zbiory treningowy oraz testowy.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import cast

import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

# %%
# Audio params used across feature extraction and CNN
SAMPLE_RATE   = 16_000
CLIP_DURATION = 10
N_MELS        = 128
N_FFT         = 1024
HOP_LENGTH    = 512
N_MFCC        = 40

def load_audio(path: str, sr: int = SAMPLE_RATE, duration: float = CLIP_DURATION) -> np.ndarray:
    """Load .wav, resample, and fix length."""
    wav, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]
    return wav.astype(np.float32)

def build_group_ids(df: pd.DataFrame) -> np.ndarray:
    """Group IDs to prevent same-source leakage."""
    for col in ("source_file", "machine_id", "id"):
        if col in df.columns:
            return np.asarray(df[col].astype(str).values)

    if "path" not in df.columns:
        raise ValueError("Cannot build group IDs: missing 'path' column.")

    stems = df["path"].astype(str).apply(lambda p: Path(p).stem)
    # Strip label prefix and clip index if present.
    stems = stems.str.replace(r"^(normal|abnormal|anomaly)_", "", regex=True)
    stems = stems.str.replace(r"_\d+$", "", regex=True)
    return np.asarray(stems.values)

# %%
cache = np.load("features_cache.npz", allow_pickle=True)
X_all         = cache["X"]
y_label       = cache["y_label"]
y_machine     = cache["y_machine"]
file_index    = cache["file_index"].tolist()

df_cache = pd.read_csv("file_index.csv")
fan_mask  = df_cache["machine_type"].values == "fan"

df_fan = cast(pd.DataFrame, df_cache.loc[fan_mask, :]).reset_index(drop=True)

X  = X_all[fan_mask]
y  = y_label[fan_mask]

group_ids = build_group_ids(df_fan)

print(f"Fan samples  : {len(X)}")
print(f"Normal       : {(y == 0).sum()}")
print(f"Abnormal     : {(y == 1).sum()}")

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx_all, test_idx = next(splitter.split(X, y, groups=group_ids))

# Validation split inside train (group-aware)
val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_rel, val_rel = next(
    val_splitter.split(X[train_idx_all], y[train_idx_all], groups=group_ids[train_idx_all])
)
train_idx = train_idx_all[train_rel]
val_idx = train_idx_all[val_rel]

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# %% [markdown]
# Trenujemy Random Forest i oceniamy na zbiorze testowym.

# %%
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

y_pred_rf   = rf.predict(X_test)
y_proba_rf  = rf.predict_proba(X_test)[:, 1]

print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf, target_names=["normal", "abnormal"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf),
                       display_labels=["normal", "abnormal"]).plot(ax=ax, colorbar=False)
ax.set_title("Random Forest — Confusion Matrix")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png", dpi=150)
plt.show()

# %% [markdown]
# Najwazniejsze cechy MFCC w Random Forest.

# %%
importances = rf.feature_importances_
n = N_MFCC if "N_MFCC" in globals() else (len(importances) // 2)
labels = [f"MFCC{i+1}_mean" for i in range(n)] + [f"MFCC{i+1}_std" for i in range(n)]

feat_df = pd.DataFrame({"feature": labels, "importance": importances})
feat_df = feat_df.sort_values("importance", ascending=False).head(20)

plt.figure(figsize=(10, 5))
sns.barplot(data=feat_df, x="importance", y="feature", palette="viridis")
plt.title("Top 20 Most Important MFCC Features (Random Forest)")
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=150)
plt.show()

# %% [markdown]
# Spektrogramy melowe dla CNN.

# %%
MEL_CACHE = Path("mel_cache.npz")

if MEL_CACHE.exists():
    print("Loading cached Mel spectrograms...")
    mel_data = np.load(MEL_CACHE)
    X_mel = mel_data["X_mel"]
    y_mel = mel_data["y_mel"]
    print(f"Loaded. Shape: {X_mel.shape}")

else:
    print(f"Extracting Mel spectrograms for {len(df_fan)} fan files...")

    X_mel, y_mel = [], []

    for _, row in tqdm(df_fan.iterrows(), total=len(df_fan)):
        try:
            wav = load_audio(row["path"])
            mel = librosa.feature.melspectrogram(y=wav, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
            mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
            X_mel.append(mel_db)
            y_mel.append(row["label"])
        except Exception as e:
            print(f"  Skipping {row['path']}: {e}")

    X_mel = np.array(X_mel, dtype=np.float32)   # (N, 128, time_frames)
    y_mel = np.array(y_mel, dtype=np.int64)

    np.savez(MEL_CACHE, X_mel=X_mel, y_mel=y_mel)
    print(f"Done. Shape: {X_mel.shape}")
    print(f"Saved to {MEL_CACHE}")

# %% [markdown]
# Normalizacja spektrogramow i DataLoadery.

# %%
# Ensure numpy arrays even if loaded from cache or built as lists
X_mel = np.asarray(X_mel, dtype=np.float32)
y_mel = np.asarray(y_mel, dtype=np.int64)

if X_mel.shape[0] != len(df_fan):
    raise ValueError(
        "Mel cache size does not match fan file count. "
        "Delete mel_cache.npz and regenerate."
    )

# Normalize each spectrogram to [0, 1]
X_mel_norm = (X_mel - X_mel.min()) / (X_mel.max() - X_mel.min() + 1e-8)

# Add channel dim → (N, 1, N_MELS, time_frames)
X_mel_norm = X_mel_norm[:, np.newaxis, :, :]

X_tr, X_val, X_te = X_mel_norm[train_idx], X_mel_norm[val_idx], X_mel_norm[test_idx]
y_tr, y_val, y_te = y_mel[train_idx], y_mel[val_idx], y_mel[test_idx]

class_counts = np.bincount(y_tr, minlength=2)
class_weights = class_counts.sum() / (len(class_counts) * class_counts)

# Convert to tensors
X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_te_t = torch.tensor(X_te, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.long)

sample_weights = class_weights[y_tr]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, sampler=sampler)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
print(f"Input shape per sample: {X_tr_t.shape[1:]}")
print(f"Class weights: {class_weights}")

# %% [markdown]
# Prosta siec CNN do klasyfikacji spektrogramow.

# %%
class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = AudioCNN().to(device)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {total_params:,}")

# %% [markdown]
# Trenujemy CNN i zapisujemy metryki po kazdej epoce.

# %%
EPOCHS    = 40
LR        = 1e-3
PATIENCE  = 6
MIN_DELTA = 1e-4

class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_loss = float("inf")
best_state = None
patience_left = PATIENCE

for epoch in range(1, EPOCHS + 1):

    model.train()
    train_loss, correct, total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(yb)
        correct    += (preds.argmax(1) == yb).sum().item()
        total      += len(yb)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds   = model(xb)
            loss    = criterion(preds, yb)
            val_loss    += loss.item() * len(yb)
            val_correct += (preds.argmax(1) == yb).sum().item()
            val_total   += len(yb)

    scheduler.step(val_loss)

    t_loss = train_loss / total
    t_acc  = correct    / total
    v_loss = val_loss   / val_total
    v_acc  = val_correct / val_total

    history["train_loss"].append(t_loss)
    history["train_acc"].append(t_acc)
    history["val_loss"].append(v_loss)
    history["val_acc"].append(v_acc)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train loss: {t_loss:.4f}  acc: {t_acc:.4f} | "
          f"Val loss: {v_loss:.4f}  acc: {v_acc:.4f}")

    if v_loss < (best_val_loss - MIN_DELTA):
        best_val_loss = v_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_left = PATIENCE
    else:
        patience_left -= 1
        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch:02d} (best val loss: {best_val_loss:.4f}).")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# %% [markdown]
# Rysujemy krzywe uczenia i porownujemy wyniki modeli.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

epochs_range = range(1, len(history["train_loss"]) + 1)

axes[0].plot(epochs_range, history["train_loss"], label="Train")
axes[0].plot(epochs_range, history["val_loss"],   label="Val")
axes[0].set_title("CNN — Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(epochs_range, history["train_acc"], label="Train")
axes[1].plot(epochs_range, history["val_acc"],   label="Val")
axes[1].set_title("CNN — Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("cnn_training_curves.png", dpi=150)
plt.show()

# --- CNN confusion matrix ---
model.eval()
all_preds, all_proba = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb.to(device))
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_proba.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())

print("=== CNN ===")
print(classification_report(y_te, all_preds, target_names=["normal", "abnormal"]))
print(f"ROC-AUC: {roc_auc_score(y_te, all_proba):.4f}")

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_te, all_preds), display_labels=["normal", "abnormal"]).plot(ax=ax, colorbar=False)
ax.set_title("CNN — Confusion Matrix")
plt.tight_layout()
plt.savefig("cnn_confusion_matrix.png", dpi=150)
plt.show()

# --- Side-by-side summary ---
rf_acc  = (y_pred_rf == y_test).mean()
cnn_acc = (np.array(all_preds) == y_te).mean()
rf_auc  = roc_auc_score(y_test, y_proba_rf)
cnn_auc = roc_auc_score(y_te, all_proba)

summary = pd.DataFrame({
    "Model":    ["Random Forest", "CNN"],
    "Accuracy": [f"{rf_acc:.4f}", f"{cnn_acc:.4f}"],
    "ROC-AUC":  [f"{rf_auc:.4f}", f"{cnn_auc:.4f}"],
})
print("\n=== Model Comparison ===")
print(summary.to_string(index=False))


