"""Carve a proper model-selection split out of the CULane training list.

The problem this solves
-----------------------
CULane's `val.txt` covers only ~23 driving sessions. That is a narrow slice of
conditions, so a model that happens to suit those sessions scores high on it,
and a threshold or epoch count chosen on it need not transfer to a test set
drawn from far more sessions. Choosing on `val` has now mis-ranked three
decisions in a row: IoU-valued classification targets, the epoch budget, and
one-to-one versus one-to-many assignment.

The training list covers hundreds of sessions. Holding out whole sessions from
it gives a selection set that is both larger and more diverse than `val`, and
leaves `test` untouched.

Splitting by *session*, not by frame, is the whole point: CULane frames are
consecutive, so a random frame split puts near-duplicates of training images
into the selection set and every model looks good on it.

Usage:
    python tools/clrbezier/make_minival.py /work/dataset/CULane --sequences 60

Writes, next to the existing lists:
    list/train_minus_minival_gt.txt   training list, held-out sessions removed
    list/train_minus_minival_diffs.npz  the frame-difference values for that list
    list/minival.txt                  image paths, for the metric's data_list
    list/minival_sequences.txt        which sessions were held out

Then point the training dataloader at the first and the evaluator at the
second. `val.txt` is left alone so earlier numbers remain comparable.
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

import numpy as np


def sequence_of(image_path):
    """`/driver_23_30frame/05151649_0422.MP4/00030.jpg` -> the session."""
    parts = image_path.strip().lstrip("/").split("/")
    return "/".join(parts[:2]) if len(parts) >= 3 else parts[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root")
    ap.add_argument("--train-list", default="list/train_gt.txt")
    ap.add_argument("--sequences", type=int, default=60,
                    help="sessions to hold out (CULane val has ~23)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="optionally subsample the held-out frames, to keep "
                         "evaluation fast; sessions are kept whole")
    ap.add_argument("--diff-file", default="list/train_diffs.npz",
                    help="frame-difference file of --train-list (CulaneDataset "
                         "indexes it by line number, so it must be subset too)")
    args = ap.parse_args()

    train_path = os.path.join(args.data_root, args.train_list)
    with open(train_path) as handle:
        raw = handle.readlines()  # CulaneDataset.parse_datalist enumerates these
    indexed = [(i, line.rstrip("\n")) for i, line in enumerate(raw) if line.strip()]
    lines = [line for _, line in indexed]

    by_sequence = defaultdict(list)
    for line in lines:
        by_sequence[sequence_of(line.split()[0])].append(line)

    sequences = sorted(by_sequence)
    if args.sequences >= len(sequences):
        raise SystemExit(f"only {len(sequences)} sessions in {args.train_list}")
    random.Random(args.seed).shuffle(sequences)
    held = sorted(sequences[: args.sequences])
    held_set = set(held)

    keep_idx = [i for i, l in indexed if sequence_of(l.split()[0]) not in held_set]
    keep_lines = [l for l in lines if sequence_of(l.split()[0]) not in held_set]
    hold_lines = [l for l in lines if sequence_of(l.split()[0]) in held_set]

    if args.max_frames is not None and len(hold_lines) > args.max_frames:
        stride = len(hold_lines) // args.max_frames + 1
        hold_lines = hold_lines[::stride]

    list_dir = os.path.join(args.data_root, "list")
    out_train = os.path.join(list_dir, "train_minus_minival_gt.txt")
    out_minival = os.path.join(list_dir, "minival.txt")
    out_sequences = os.path.join(list_dir, "minival_sequences.txt")

    with open(out_train, "w") as handle:
        handle.write("\n".join(keep_lines) + "\n")
    # The training loader drops near-duplicate frames with diffs[line_index].
    # A shorter list read against the full diffs array would silently filter
    # the wrong frames, so write the matching subset.
    diff_path = os.path.join(args.data_root, args.diff_file)
    out_diffs = None
    if os.path.exists(diff_path):
        diffs = np.load(diff_path)["data"]
        if len(diffs) != len(raw):
            raise SystemExit(f"{diff_path} has {len(diffs)} entries but {train_path} "
                             f"has {len(raw)} lines; they must correspond")
        out_diffs = os.path.join(list_dir, "train_minus_minival_diffs.npz")
        np.savez(out_diffs, data=diffs[keep_idx])
    with open(out_minival, "w") as handle:
        # The metric's data_list wants image paths only.
        handle.write("\n".join(line.split()[0] for line in hold_lines) + "\n")
    with open(out_sequences, "w") as handle:
        handle.write("\n".join(held) + "\n")

    print(f"{len(sequences)} sessions in {args.train_list}")
    print(f"held out {len(held)} sessions, {len(hold_lines)} frames -> {out_minival}")
    print(f"training on {len(keep_lines)} frames -> {out_train}")
    print(f"\nCULane val.txt covers ~23 sessions; this covers {len(held)}.")
    print("\nIn the config:")
    print(f'  train_dataloader.dataset.data_list = data_root + "/list/train_minus_minival_gt.txt"')
    if out_diffs is not None:
        print(f'  train_dataloader.dataset.diff_file = data_root + "/list/train_minus_minival_diffs.npz"')
    else:
        print(f"  (no {args.diff_file} found: set train_dataloader.dataset.diff_file=None)")
    print(f'  val_dataloader.dataset.data_list   = data_root + "/list/minival.txt"')
    print(f'  val_evaluator.data_list            = data_root + "/list/minival.txt"')
    print("\nRe-run every comparison you selected on val before trusting its ranking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
