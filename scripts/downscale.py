from pathlib import Path
from PIL import Image
import argparse

def downscale(src: Path, dst: Path, max_side: int):
    dst.mkdir(parents=True, exist_ok=True)
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    for p in src.rglob("*"):
        if p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = max(w, h)
        if s > max_side:
            scale = max_side / s
            im = im.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.BICUBIC)
        im.save(out)
        
