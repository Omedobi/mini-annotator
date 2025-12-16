import json, argparse, sys
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows


def main(ann_path: Path, labels_path: Path):
    labels = [l.strip() for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = load_jsonl(ann_path)
    if not rows:
        print("No annotations found")
        return 0
    bad = 0
    for i, r in enumerate(rows, 1):
        ip = r.get("image_path")
        ls = r.get("labels", [])
        boxes = r.get("boxes", [])
        if any(l not in labels for l in ls):
            print(f"[{i}] Non-whitelisted image label(s) in {ip}: {set(ls)-set(labels)}"); bad+=1
        for j,b in enumerate(boxes):
            x,y,w,h = b.get("bbox",[0,0,0,0])
            if min(w,h) <= 1:
                print(f"[{i}] Tiny bbox #{j} in {ip}: {w}x{h}"); bad+=1
            if b.get("label") not in labels:
                print(f"[{i}] Non-whitelisted box label in {ip}: {b.get('label')}"); bad+=1
    print(f"Checked {len(rows)} items. Issues: {bad}.")
    return 1 if bad else 0
