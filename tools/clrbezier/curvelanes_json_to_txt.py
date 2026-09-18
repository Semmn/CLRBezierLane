"""Convert native CurveLanes labels to the .lines.txt files the CLRerNet
curvelane code reads (one lane per line: "x0 y0 x1 y1 ...", next to each image).

    <root>/<split>/labels/<name>.lines.json  ->  <root>/<split>/images/<name>.lines.txt

Only needed when the dataset copy has no .lines.txt files yet. Points are
written bottom -> top with duplicate points removed; lanes with fewer than two
points are dropped. Existing files are kept unless --overwrite is given.

    python tools/clrbezier/curvelanes_json_to_txt.py /work/dataset/curvelanes --splits train valid
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm


def convert(json_path, txt_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_lanes = json.load(f).get("Lines", [])
    out = []
    for raw in raw_lanes:
        pts, seen = [], set()
        for p in raw:
            try:
                x, y = float(p["x"]), float(p["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if (x, y) in seen:
                continue
            seen.add((x, y))
            pts.append((x, y))
        pts.sort(key=lambda q: -q[1])
        if len(pts) >= 2:
            out.append(" ".join(f"{x:.5f} {y:.5f}" for x, y in pts))
    txt_path.write_text("\n".join(out) + ("\n" if out else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for split in args.splits:
        labels = sorted((Path(args.root) / split / "labels").rglob("*.lines.json"))
        for jp in tqdm(labels, desc=split):
            rel = jp.relative_to(Path(args.root) / split / "labels")
            txt = Path(args.root) / split / "images" / str(rel).replace(".lines.json", ".lines.txt")
            if txt.exists() and not args.overwrite:
                continue
            txt.parent.mkdir(parents=True, exist_ok=True)
            convert(jp, txt)


if __name__ == "__main__":
    main()
