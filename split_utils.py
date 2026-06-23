"""
Shared utility: subject-level train/val split.

CRITICAL: All training scripts must use this instead of splitting by pair
index. Splitting by pair index causes severe leakage because consecutive
pairs (x_t, x_{t+1}) and (x_{t+1}, x_{t+2}) share an epoch -- if one lands
in train and the other in val, the val epoch was already seen in training.

This module guarantees that ALL pairs from a given subject/recording go
entirely into train or entirely into val. No epoch from a held-out subject
is ever seen during training, in any role.
"""

import numpy as np


def subject_level_split(subject_ids, val_frac=0.2, seed=42):
    """
    subject_ids : (N,) array of subject identifiers, one per pair
                  (e.g. ['SC4001','SC4001',...,'SC4011',...])
    val_frac    : fraction of SUBJECTS (not pairs) to hold out
    seed        : random seed for reproducibility

    Returns:
        train_idx, val_idx : index arrays into the original pair arrays
        train_subjects, val_subjects : which subject IDs went where (for logging)
    """
    subject_ids = np.array(subject_ids)
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    if n_subjects < 5:
        print(f"[WARNING] Only {n_subjects} unique subjects found. "
              f"Subject-level split with so few subjects gives a very "
              f"coarse-grained val set. Consider using more subjects.")

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_subjects)

    n_val_subjects = max(1, int(n_subjects * val_frac))
    val_subjects   = set(shuffled[:n_val_subjects].tolist())
    train_subjects = set(shuffled[n_val_subjects:].tolist())

    train_idx = np.where(np.isin(subject_ids, list(train_subjects)))[0]
    val_idx   = np.where(np.isin(subject_ids, list(val_subjects)))[0]

    # Sanity check: no subject appears in both splits
    overlap = train_subjects & val_subjects
    assert len(overlap) == 0, f"Subject leakage in split itself: {overlap}"

    print(f"Subject-level split: {len(train_subjects)} train subjects "
          f"({len(train_idx):,} pairs) | "
          f"{len(val_subjects)} val subjects ({len(val_idx):,} pairs)")
    print(f"  Val subjects: {sorted(val_subjects)}")

    return train_idx, val_idx, train_subjects, val_subjects


def verify_no_leakage(x_t, x_t1, train_idx, val_idx, sample_check=5000):
    """
    Post-hoc verification that no epoch in val also appears in train.
    Use this after any split to confirm zero leakage before trusting results.
    Returns leak_fraction (should be 0.0 or very close to it).
    """
    def epoch_hash(arr):
        # arr may be (3000,) for single-channel or (n_ch, 3000) for multi-channel
        flat = arr.flatten()
        return (round(float(flat.sum()), 4), round(float(flat[0]), 4),
                round(float(flat[-1]), 4))

    train_hashes = set()
    for i in train_idx:
        train_hashes.add(epoch_hash(x_t[i]))
        train_hashes.add(epoch_hash(x_t1[i]))

    val_sample = val_idx if len(val_idx) <= sample_check else \
                 np.random.choice(val_idx, sample_check, replace=False)

    leaked = 0
    for i in val_sample:
        if epoch_hash(x_t[i]) in train_hashes or epoch_hash(x_t1[i]) in train_hashes:
            leaked += 1

    leak_frac = leaked / len(val_sample)
    status = "[OK]" if leak_frac < 0.001 else "[WARNING]"
    print(f"{status} Post-split leakage check: {leaked}/{len(val_sample)} "
          f"({leak_frac*100:.2f}%)")
    return leak_frac