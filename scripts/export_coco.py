import json
import argparse
from pathlib import Path
from PIL import Image

def load_jsonl(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows


def export(ann: Path, labels:Path, out: Path):
    cats = [l.strip() for l in labels.read_text(encoding="utf-8").splitlines() if l.strip()]
    cat_map = {n:i+1 for i,n in enumerate(cats)}
    rows = load_jsonl(ann)
    images = {}
    annotations = []
    ann_id = 1
    for r in rows:
        ip = Path(r["image_path"])
        if ip.name not in images:
            try:
                w,h = Image.open(ip).size
            except Exception:
                w=h=None
            images[ip.name] = {
                "id": ip.name,"file_name": ip.name, "width": w, "height": h}
        for b in r.get("Boxes", []):
            x,y,w,h = b["bbox"]
            annotations.append({
                "id": ann_id,
                "image_id": ip.name,
                "category_id": cat_map.get(b["label"], 0),
                "bbox": [float(x),float(y),float(w),float(h)],
                "area": float(w*h),
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1
    
    coco = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": [{"id":i+1,"name":n} for i,n in enumerate(cats)],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "instance.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"COCO saved to {out/'instance.json'}")
    
    
