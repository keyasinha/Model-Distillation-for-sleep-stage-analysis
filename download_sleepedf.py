"""
Download Sleep-EDF Cassette subset from PhysioNet.
Downloads PSG + Hypnogram EDF pairs for subjects 0-19 (SC subjects, night 1 only).
Total size: ~1.5 GB for all 20 subjects.
Pass --n_subjects to limit (default 10 for quick experiments).
"""

import os
import urllib.request
import argparse

BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"

# Subject IDs for SC (Sleep Cassette) night-1 recordings
SC_SUBJECTS = [
    ("SC4001E0-PSG.edf",    "SC4001EC-Hypnogram.edf"),
    ("SC4011E0-PSG.edf",    "SC4011EH-Hypnogram.edf"),
    ("SC4021E0-PSG.edf",    "SC4021EH-Hypnogram.edf"),
    ("SC4031E0-PSG.edf",    "SC4031EC-Hypnogram.edf"),
    ("SC4041E0-PSG.edf",    "SC4041EC-Hypnogram.edf"),
    ("SC4051E0-PSG.edf",    "SC4051EC-Hypnogram.edf"),
    ("SC4061E0-PSG.edf",    "SC4061EC-Hypnogram.edf"),
    ("SC4071E0-PSG.edf",    "SC4071EC-Hypnogram.edf"),
    ("SC4081E0-PSG.edf",    "SC4081EC-Hypnogram.edf"),
    ("SC4091E0-PSG.edf",    "SC4091EC-Hypnogram.edf"),
    ("SC4101E0-PSG.edf",    "SC4101EC-Hypnogram.edf"),
    ("SC4111E0-PSG.edf",    "SC4111EC-Hypnogram.edf"),
    ("SC4121E0-PSG.edf",    "SC4121EC-Hypnogram.edf"),
    ("SC4131E0-PSG.edf",    "SC4131EC-Hypnogram.edf"),
    ("SC4141E0-PSG.edf",    "SC4141EU-Hypnogram.edf"),
    ("SC4151E0-PSG.edf",    "SC4151EC-Hypnogram.edf"),
    ("SC4161E0-PSG.edf",    "SC4161EC-Hypnogram.edf"),
    ("SC4171E0-PSG.edf",    "SC4171EU-Hypnogram.edf"),
    ("SC4181E0-PSG.edf",    "SC4181EC-Hypnogram.edf"),
    ("SC4191E0-PSG.edf",    "SC4191EP-Hypnogram.edf"),
]

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"  [skip] {os.path.basename(dest)} already exists")
        return
    print(f"  Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_subjects", type=int, default=10,
                        help="Number of subjects to download (max 20). Default 10.")
    parser.add_argument("--out_dir", type=str, default="data/raw")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    subjects = SC_SUBJECTS[:args.n_subjects]

    print(f"Downloading {len(subjects)} subjects to {args.out_dir}/")
    print("(Each subject ~75 MB, total ~{}MB)\n".format(len(subjects) * 75))

    for i, (psg_f, hyp_f) in enumerate(subjects):
        print(f"Subject {i+1}/{len(subjects)}: {psg_f[:6]}")
        download_file(f"{BASE_URL}/{psg_f}", os.path.join(args.out_dir, psg_f))
        download_file(f"{BASE_URL}/{hyp_f}", os.path.join(args.out_dir, hyp_f))

    print(f"\nAll done. Files saved to {args.out_dir}/")
    print("Next: run  python 1_prepare_data.py")

if __name__ == "__main__":
    main()