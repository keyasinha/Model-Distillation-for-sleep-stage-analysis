# Sleep Distillation — Setup & Dataset Guide

## Step 1: Install dependencies

```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn mne pyedflib tqdm
```

## Step 2: Download Sleep-EDF Cassette dataset

The Sleep-EDF Cassette subset has 20 healthy subjects, single-channel EEG (Fpz-Cz),
sampled at 100 Hz. It is free and public from PhysioNet.

Run this to download the first 10 subjects (enough to run experiments):

```bash
python download_sleepedf.py
```

Or manually:
```bash
mkdir -p data/raw
cd data/raw

# Download PSG (signal) and Hypnogram (labels) files
for i in 00 01 02 03 04 05 06 07 08 09; do
  wget -nc https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf
  # Easier: use the downloader script below
done
```

The downloader script (download_sleepedf.py) handles this automatically.

## Step 3: Prepare data

```bash
python 1_prepare_data.py
```

This extracts EEG Fpz-Cz epochs (30s @ 100Hz = 3000 samples each) and saves them
as numpy arrays alongside their AASM labels.

## Output structure after setup:

```
data/
  raw/           <- .edf files from PhysioNet
  processed/     <- .npz files: {'x': (N,3000), 'y': (N,), 'subject': int}
outputs/         <- teacher soft labels, transition matrices
checkpoints/     <- saved model weights
plots/           <- all figures
```

## Step 4: Run the full pipeline

```bash
python 2_teacher_inference.py   # extract TinySleepNet soft labels
python 3_build_transitions.py   # build transition dataset + save graph
python 4_train_student.py       # train student (baseline + proposed)
python 5_evaluate.py            # comparison plots + metrics
python 15_multichannel_experiments #to compare with 4 different kinds of datasets.

```