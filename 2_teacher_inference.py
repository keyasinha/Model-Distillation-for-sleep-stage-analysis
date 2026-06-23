import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
N_CLASSES   = 5
SEQ_LEN     = 20       
FS          = 100     
EPOCH_SAMP  = 3000     

class TinySleepNet(nn.Module):
    """
    PyTorch re-implementation of TinySleepNet.
    """
    def __init__(self, n_classes=5, fs=100, seq_len=20, dropout=0.5):
        super().__init__()
        self.seq_len = seq_len
        lf = fs // 2   # 50 samples
        self.cnn_large = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=lf, stride=lf//2, padding=lf//2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=dropout),
            nn.Conv1d(128, 128, kernel_size=8, padding=4, bias=False),
            nn.Conv1d(128, 128, kernel_size=8, padding=4, bias=False),
            nn.Conv1d(128, 128, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        # Small filter branch (captures fast spindles/K-complexes)
        sf = fs // 4   # 25 samples
        self.cnn_small = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=sf, stride=sf//2, padding=sf//2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=dropout),
            nn.Conv1d(128, 128, kernel_size=6, padding=3, bias=False),
            nn.Conv1d(128, 128, kernel_size=6, padding=3, bias=False),
            nn.Conv1d(128, 128, kernel_size=6, padding=3, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn_drop = nn.Dropout(p=dropout)

        # Compute CNN output dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, EPOCH_SAMP)
            l_out = self.cnn_large(dummy).flatten(1)
            s_out = self.cnn_small(dummy).flatten(1)
            cnn_dim = l_out.shape[1] + s_out.shape[1]

        # LSTM backend 
        self.lstm = nn.LSTM(
            input_size=cnn_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.lstm_drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(128, n_classes)

    def forward_cnn(self, x):
        """x: (B, 1, 3000) → (B, cnn_dim)"""
        l = self.cnn_large(x).flatten(1)
        s = self.cnn_small(x).flatten(1)
        return self.cnn_drop(torch.cat([l, s], dim=1))

    def forward(self, x, h=None):
        """
        x: (B, seq_len, 3000) — batch of epoch sequences
        Returns: logits (B, seq_len, 5), hidden state h
        """
        B, T, L = x.shape
        # CNN over all epochs in parallel
        x_flat = x.reshape(B * T, 1, L)
        feats = self.forward_cnn(x_flat)         # (B*T, cnn_dim)
        feats = feats.reshape(B, T, -1)          # (B, T, cnn_dim)
        # LSTM over sequence
        lstm_out, h_new = self.lstm(feats, h)    # (B, T, 128)
        lstm_out = self.lstm_drop(lstm_out)
        logits = self.fc(lstm_out)               # (B, T, 5)
        return logits, h_new

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class SleepSequenceDataset(Dataset):
    """
    Yields non-overlapping sequences of SEQ_LEN epochs per subject.
    Returns: (X_seq, y_seq) where shapes are (SEQ_LEN, 3000) and (SEQ_LEN,)
    """
    def __init__(self, npz_files, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.sequences = []   

        for f in npz_files:
            d = np.load(f)
            X, y = d['x'], d['y']   
            n_seq = len(X) // seq_len
            for i in range(n_seq):
                s, e = i * seq_len, (i + 1) * seq_len
                self.sequences.append((X[s:e], y[s:e]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, y = self.sequences[idx]
        return torch.FloatTensor(X), torch.LongTensor(y)

def train_teacher(model, train_files, val_files, device,
                  epochs=30, lr=1e-3, save_path="checkpoints/teacher.pt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_ds = SleepSequenceDataset(train_files)
    val_ds   = SleepSequenceDataset(val_files)
    train_dl = DataLoader(train_ds, batch_size=15, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=15, shuffle=False, num_workers=0)

    # Class weights to handle imbalance (N1 is rare)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0
    print(f"\nTraining teacher ({model.count_params():,} params) for {epochs} epochs.")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X_seq, y_seq in tqdm(train_dl, desc=f"Ep {ep}/{epochs}", leave=False):
            X_seq, y_seq = X_seq.to(device), y_seq.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_seq)                    # (B, T, 5)
            loss = F.cross_entropy(logits.reshape(-1, 5), y_seq.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(-1)
            correct += (preds == y_seq).sum().item()
            total   += y_seq.numel()

        scheduler.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_seq, y_seq in val_dl:
                X_seq, y_seq = X_seq.to(device), y_seq.to(device)
                logits, _ = model(X_seq)
                preds = logits.argmax(-1)
                val_correct += (preds == y_seq).sum().item()
                val_total   += y_seq.numel()

        val_acc  = val_correct / val_total
        train_acc = correct / total
        print(f"  Ep {ep:3d} | loss {total_loss/len(train_dl):.4f} | "
              f"train acc {train_acc:.3f} | val acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"           best model saved ({val_acc:.3f})")

    print(f"Teacher training done. Best val acc: {best_val_acc:.3f}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model

def extract_soft_labels(model, npz_files, device, temperature=2.0,
                         out_dir="outputs/soft_labels"):
    """
    Run teacher on every subject, save soft labels (N, 5) as .npy files.
    Uses temperature scaling to soften distributions.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    print(f"\nextracting soft labels (temperature τ={temperature})")
    all_results = {}

    for f in tqdm(npz_files):
        subj = os.path.basename(f).replace(".npz", "")
        d = np.load(f)
        X, y = d['x'], d['y']   # (N, 3000), (N,)
        N = len(X)

        soft_labels = np.zeros((N, N_CLASSES), dtype=np.float32)
        hard_preds  = np.zeros(N, dtype=np.int64)

        # Process in non-overlapping sequences of SEQ_LEN
        with torch.no_grad():
            idx = 0
            while idx < N:
                end = min(idx + SEQ_LEN, N)
                chunk = X[idx:end]
                # Pad if needed
                if len(chunk) < SEQ_LEN:
                    pad = np.zeros((SEQ_LEN - len(chunk), EPOCH_SAMP), dtype=np.float32)
                    chunk = np.concatenate([chunk, pad], axis=0)

                x_t = torch.FloatTensor(chunk).unsqueeze(0).to(device)  # (1, SEQ_LEN, 3000)
                logits, _ = model(x_t)    # (1, SEQ_LEN, 5)
                logits = logits[0]        # (SEQ_LEN, 5)

                # Temperature scaling
                probs = F.softmax(logits / temperature, dim=-1).cpu().numpy()

                actual_len = end - idx
                soft_labels[idx:end] = probs[:actual_len]
                hard_preds[idx:end]  = probs[:actual_len].argmax(axis=1)
                idx = end

        # Save
        out_path = os.path.join(out_dir, f"{subj}_soft.npy")
        np.save(out_path, soft_labels)
        all_results[subj] = {
            'soft_labels': soft_labels,
            'hard_preds':  hard_preds,
            'true_labels': y,
        }

        acc = (hard_preds == y).mean()
        tqdm.write(f"  {subj}: {N} epochs | teacher acc {acc:.3f}")

    print(f"Soft labels saved to {out_dir}/")
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="data/processed")
    parser.add_argument("--ckpt_path",  type=str, default="checkpoints/teacher.pt")
    parser.add_argument("--temperature",type=float, default=2.0)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip teacher training (if checkpoint exists)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not npz_files:
        print(f"No .npz files in {args.data_dir}. Run 1_prepare_data.py first.")
        return

    print(f"Found {len(npz_files)} subjects.")

    # Train/val split by subject (leave-one-out)
    train_files, val_files = train_test_split(npz_files, test_size=0.2, random_state=42)

    model = TinySleepNet().to(device)
    print(f"TinySleepNet parameters: {model.count_params():,}")

    if args.skip_train and os.path.exists(args.ckpt_path):
        print(f"Loading existing checkpoint: {args.ckpt_path}")
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    else:
        model = train_teacher(model, train_files, val_files, device,
                              epochs=args.train_epochs,
                              save_path=args.ckpt_path)

    # Extract soft labels for ALL subjects
    extract_soft_labels(model, npz_files, device,
                        temperature=args.temperature)

if __name__ == "__main__":
    main()