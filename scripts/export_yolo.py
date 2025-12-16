import json, argparse
from pathlib import Path
from PIL import Image

def load_jsonl(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows

def export (ann:Path, labels:Path, out: Path):
    names = [l.strip() for l in labels.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = load_jsonl(ann)
    out_img = out /"images"
    out_lbl = out /"labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    for r in rows:
        ip = Path(r["image_path"])
        try:
            w,h = Image.open(ip).size
        except Exception:
            continue
        dst = out_img / ip.name
        if not dst.exists():
            try:
                dst.hardlink_to(ip)
            except Exception:
                import shutil; shutil.copy2(ip, dst)
                
        lines = []
        for b in r.get("boxes", []):
            cls = names.index(b.get("label")) if b.get("label") in names else - 1
            if cls < 0: continue
            x,y,ww,hh = b["bbox"]
            
            xc = (x + ww/2) / w; yc = (y + hh/2) / h; nw = ww / w; nh = hh / h
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        (out_lbl / f"{ip.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"YOLO export saved to {out}")

