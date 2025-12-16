from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
dirs = [
    "configs", "tools", "scripts",
    "data/images", "data/cache/embeddings", "data/cache/detections",
    "data/exports/coco", "data/exports/yolo",
    "models", "runs"
]
for d in dirs:
    (ROOT / d).mkdir(parents=True, exist_ok=True)

files = {
    ROOT / "configs/project.yaml": (ROOT / "configs/project.yaml").read_text(encoding="utf-8") if (ROOT/"configs/project.yaml").exists() else """app:
  image_root: data/images
  top_k: 3
  policy: uncertain
classifier:
  model_name: ViT-B-32
  pretrained: openai
  device: auto
detector:
  enabled: true
  weights: rtdetr-l.pt
  conf: 0.25
  iou: 0.6
  max_det: 200
ui:
  max_image_side: 1024
""",
    ROOT / "configs/labels.txt": (ROOT / "configs/labels.txt").read_text(encoding="utf-8") if (ROOT/"configs/labels.txt").exists() else "person\ncar\nbicycle\ncat\ndog\n",
    ROOT / ".gitignore": (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT/".gitignore").exists() else "__pycache__/\n*.pt\ndata/cache/\ndata/exports/*\nruns/\n*.ipynb_checkpoints/\n.env\n",
    ROOT / "README.md": (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT/"README.md").exists() else "# Model-Assisted Annotation\n\nSee Makefile for commands.\n",
}
for p, content in files.items():
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(content, encoding="utf-8")
for f in ["annotations.jsonl", "annotations.csv"]:
    p = ROOT / f
    if not p.exists():
        p.write_text("", encoding="utf-8")
print("Project initialized.")