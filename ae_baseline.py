import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

FEATURES_CACHE = Path("mel_cache.npz")
FILE_INDEX = Path("file_index.csv")

N_MELS = 128
FRAME_STACK = 5
STRIDE = 5
MAX_WINDOWS_PER_FILE = 50

BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-3
PATIENCE = 6
MIN_DELTA = 1e-4
RANDOM_STATE = 2137

RNG = np.random.default_rng(RANDOM_STATE)

def mel_to_windows(
    mel: np.ndarray,
    frame_stack: int = FRAME_STACK,
    stride: int = STRIDE,
    max_windows: int | None = None,
) -> np.ndarray:
    """(n_mels, frames) -> (num_windows, n_mels * frame_stack)."""
    if mel.ndim != 2:
        raise ValueError("mel must be 2D (n_mels, frames)")

    n_mels, frames = mel.shape
    if frames < frame_stack:
        return np.empty((0, n_mels * frame_stack), dtype=np.float32)

    starts = np.arange(0, frames - frame_stack + 1, stride)
    if max_windows is not None and len(starts) > max_windows:
        starts = RNG.choice(starts, size=max_windows, replace=False)

    windows = np.zeros((len(starts), n_mels * frame_stack), dtype=np.float32)
    for i, s in enumerate(starts):
        chunk = mel[:, s : s + frame_stack]
        windows[i] = chunk.reshape(-1)
    return windows


def collect_windows(
    X_source: np.ndarray,
    indices: np.ndarray,
    max_windows_per_file: int | None,
) -> np.ndarray:
    all_windows = []
    for i in indices:
        windows = mel_to_windows(
            X_source[i],
            frame_stack=FRAME_STACK,
            stride=STRIDE,
            max_windows=max_windows_per_file,
        )
        if len(windows) > 0:
            all_windows.append(windows)
    if not all_windows:
        return np.empty((0, N_MELS * FRAME_STACK), dtype=np.float32)
    return np.vstack(all_windows)


class DenseAE(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def score_files(
    X_source: np.ndarray,
    y_source: np.ndarray,
    indices: np.ndarray,
    model: nn.Module,
    scaler: StandardScaler,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    labels = []
    model.eval()

    for i in indices:
        windows = mel_to_windows(
            X_source[i],
            frame_stack=FRAME_STACK,
            stride=STRIDE,
            max_windows=None,
        )
        if len(windows) == 0:
            continue

        windows = scaler.transform(windows)
        xb = torch.tensor(windows, dtype=torch.float32, device=device)
        with torch.no_grad():
            recon = model(xb)
            err = torch.mean((recon - xb) ** 2, dim=1).detach().cpu().numpy()
        scores.append(float(err.mean()))
        labels.append(int(y_source[i]))

    return np.array(scores), np.array(labels)


def safe_split(indices: np.ndarray, y: np.ndarray, test_size: float):
    try:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    except ValueError:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=None,
        )


if not FEATURES_CACHE.exists():
    raise FileNotFoundError("Missing mel_cache.npz. Run flow.py to create it.")
if not FILE_INDEX.exists():
    raise FileNotFoundError("Missing file_index.csv. Run dataset_setup.py first.")

mel_data = np.load(FEATURES_CACHE)
X_mel = mel_data["X_mel"]
y_mel = mel_data["y_mel"]

index_df = pd.read_csv(FILE_INDEX)
fan_mask = index_df["machine_type"].values == "fan"

index_fan = index_df.loc[fan_mask, :].reset_index(drop=True)
X_mel = X_mel[fan_mask]
y_mel = y_mel[fan_mask]

if X_mel.shape[0] != len(index_fan):
    raise ValueError("Mel cache size does not match fan file count.")

print(f"Fan files: {len(X_mel)}")
print(f"Normal: {(y_mel == 0).sum()}, Abnormal: {(y_mel == 1).sum()}")

# %%
# Train and evaluate per machine_id
results = []
machine_ids = sorted(index_fan["machine_id"].unique())

for machine_id in machine_ids:
    file_indices = np.where(index_fan["machine_id"].values == machine_id)[0]
    y_sub = y_mel[file_indices]

    if len(np.unique(y_sub)) < 2:
        print(f"Skipping {machine_id}: only one class present.")
        continue

    print("\n===========================")
    print(f"Machine ID: {machine_id} | Files: {len(file_indices)}")

    idx_train_all, idx_test = safe_split(file_indices, y_sub, test_size=0.2)
    y_train_all = y_mel[idx_train_all]

    idx_train, idx_val = safe_split(idx_train_all, y_train_all, test_size=0.15)

    train_normal_idx = idx_train[y_mel[idx_train] == 0]
    val_normal_idx = idx_val[y_mel[idx_val] == 0]

    X_train_win = collect_windows(X_mel, train_normal_idx, MAX_WINDOWS_PER_FILE)
    X_val_win = collect_windows(X_mel, val_normal_idx, MAX_WINDOWS_PER_FILE)

    if len(X_train_win) == 0:
        raise ValueError("No training windows found. Check mel_cache.npz content.")

    scaler = StandardScaler()
    X_train_win = scaler.fit_transform(X_train_win)
    X_val_win = scaler.transform(X_val_win) if len(X_val_win) else X_val_win

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = N_MELS * FRAME_STACK
    model = DenseAE(input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    train_ds = TensorDataset(torch.tensor(X_train_win, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = None
    if len(X_val_win) > 0:
        val_ds = TensorDataset(torch.tensor(X_val_win, dtype=torch.float32))
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_val = float("inf")
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        total = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            total += len(xb)

        train_loss /= max(total, 1)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            vtotal = 0
            with torch.no_grad():
                for (xb,) in val_loader:
                    xb = xb.to(device)
                    recon = model(xb)
                    loss = criterion(recon, xb)
                    val_loss += loss.item() * len(xb)
                    vtotal += len(xb)
            val_loss /= max(vtotal, 1)
        else:
            val_loss = train_loss

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}"
        )

        if val_loss < (best_val - MIN_DELTA):
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(
                    f"Early stopping at epoch {epoch:02d} "
                    f"(best val loss: {best_val:.6f})."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_scores, val_labels = score_files(X_mel, y_mel, idx_val, model, scaler, device)
    test_scores, test_labels = score_files(X_mel, y_mel, idx_test, model, scaler, device)

    val_auc = None
    if len(val_scores) > 0:
        val_auc = roc_auc_score(val_labels, val_scores)
    test_auc = roc_auc_score(test_labels, test_scores)

    if val_auc is not None:
        print(f"Val ROC-AUC:  {val_auc:.4f}")
    print(f"Test ROC-AUC: {test_auc:.4f}")

    results.append({"machine_id": machine_id, "val_auc": val_auc, "test_auc": test_auc})

results_df = pd.DataFrame(results)
mean_test_auc = results_df["test_auc"].mean()

print("\n=== Per-machine summary ===")
print(results_df.to_string(index=False))
print(f"\nMean Test ROC-AUC: {mean_test_auc:.4f}")
