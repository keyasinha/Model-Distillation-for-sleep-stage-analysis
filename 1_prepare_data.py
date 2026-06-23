"""
Step 1: Prepare Sleep-EDF data.
Reads EDF pairs, extracts EEG Fpz-Cz channel, segments into 30s epochs,
maps annotations to AASM integer labels, saves per-subject .npz files.

Label mapping (AASM):
  0 = Wake
  1 = N1
  2 = N2
  3 = N3
  4 = REM
"""

import os
import glob
import argparse
import numpy as np
import mne
from tqdm import tqdm

mne.set_log_level("WARNING")

EPOCH_SEC    = 30
SAMPLE_RATE  = 100   # Sleep-EDF Fpz-Cz is 100 Hz
EPOCH_LEN    = EPOCH_SEC * SAMPLE_RATE   # 3000 samples per epoch
EEG_CHANNEL  = "EEG Fpz-Cz"

# PhysioNet annotation → AASM integer
ANNOTATION_MAP = {
    "Sleep stage W":   0,
    "Sleep stage 1":   1,
    "Sleep stage 2":   2,
    "Sleep stage 3":   3,
    "Sleep stage 4":   3,   # N3 (combines old S3+S4)
    "Sleep stage R":   4,
    "Movement time":   0,   # treat as Wake
}

def process_pair(psg_path, hyp_path, subject_id):
    """Process one PSG + Hypnogram EDF pair into (epochs, labels)."""
    # Load PSG
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    if EEG_CHANNEL not in raw.ch_names:
        raise ValueError(f"Channel '{EEG_CHANNEL}' not found in {psg_path}. "
                         f"Available: {raw.ch_names}")

    raw.pick_channels([EEG_CHANNEL])
    data, times = raw[:]   # shape (1, n_samples)
    data = data[0]          # (n_samples,)
    ann = mne.read_annotations(hyp_path)

    # Build epoch-level label array from annotations
    n_epochs = len(data) // EPOCH_LEN
    labels = np.full(n_epochs, -1, dtype=np.int8)

    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        label = ANNOTATION_MAP.get(desc, -1)
        if label == -1:
            continue
        start_epoch = int(onset // EPOCH_SEC)
        n_ann_epochs = int(duration // EPOCH_SEC)
        for e in range(n_ann_epochs):
            idx = start_epoch + e
            if 0 <= idx < n_epochs:
                labels[idx] = label

    # Keeping only annotated epochs
    valid = labels != -1
    epochs_list = []
    for i in range(n_epochs):
        if valid[i]:
            seg = data[i * EPOCH_LEN : (i + 1) * EPOCH_LEN]
            if len(seg) == EPOCH_LEN:
                epochs_list.append(seg)

    X = np.array(epochs_list, dtype=np.float32)   # (N, 3000)
    y = labels[valid[:len(epochs_list)]]

    # Normalizinggg
    X = (X - X.mean()) / (X.std() + 1e-8)

    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",   type=str, default="data/raw")
    parser.add_argument("--out_dir",   type=str, default="data/processed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    psg_files = sorted(glob.glob(os.path.join(args.raw_dir, "*PSG.edf")))
    if not psg_files:
        print(f"No PSG files found idhar {args.raw_dir}/")  
        return

    print(f"Found {len(psg_files)} PSG files.")

    for psg_path in tqdm(psg_files):
        subj_id = os.path.basename(psg_path)[:6]  # e.g. SC4001
        hyp_path = psg_path.replace("PSG.edf", "").replace("E0", "EC") + "Hypnogram.edf"

        # Try different hypnogram suffix patterns
        for suffix in ["EC", "EH"]:
            candidate = psg_path.replace("E0-PSG.edf", f"{suffix}-Hypnogram.edf")
            if os.path.exists(candidate):
                hyp_path = candidate
                break

        if not os.path.exists(hyp_path):
            print(f" skipped -  No hypnogram for {subj_id}")
            continue

        out_path = os.path.join(args.out_dir, f"{subj_id}.npz")
        if os.path.exists(out_path):
            continue

        try:
            X, y = process_pair(psg_path, hyp_path, subj_id)
            np.savez(out_path, x=X, y=y, subject=subj_id)
            stage_counts = {s: int((y == i).sum()) for i, s in
                            enumerate(["W","N1","N2","N3","REM"])}
            tqdm.write(f"  {subj_id}: {len(y)} epochs | {stage_counts}")
        except Exception as e:
            tqdm.write(f" found error - {subj_id}: {e}")

    print(f"\nfiles saved to {args.out_dir}/")


if __name__ == "__main__":
    main()