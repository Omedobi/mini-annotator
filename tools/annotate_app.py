import os
import io
import json
import time
import math
import hashlib
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import torch
import tensorflow
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import streamlit as st
import open_clip 

DEFAULT_LABELS = [
"button",
"cancel icon",
"components",
"componentsb",
"dope", 
"dopeb",
"dropdown",
"hamburger icon",
"input field", 
"loading icon",
"love icon",
"radio",
"search icon",
"share icon",
]

APP_DIR = pathlib.Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"
IMG_EMB_DIR = CACHE_DIR / "embeddings"
DET_EMB_DIR = CACHE_DIR / "detections"
TEXT_EMB_F = CACHE_DIR / "text_labels.pt"
ANNOTATIONS_JSONL = APP_DIR / "annotations.jsonl"
ANNOTATIONS_CSV = APP_DIR / "annotations.csv"
COCO_DIR = APP_DIR / "coco"
os.makedirs(IMG_EMB_DIR, exist_ok=True)
os.makedirs(DET_EMB_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(COCO_DIR, exist_ok=True)



def sha1_path(p: pathlib.Path) -> str:
    return hashlib.sha1(str(p).encode("utf-8")).hexdigest()

def list_images(root: str) -> List[pathlib.Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root_p = pathlib.Path(root)
    if not root_p.exists():
        return []
    return sorted([p for p in root_p.rglob("*") if p.suffix.lower() in exts])

def read_annotations_jsonl(fpath: pathlib.Path) -> List[Dict]:
    if not fpath.exists():
        return []
    with fpath.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_annotation_jsonl(record: Dict, fpath: pathlib.Path) -> None:
    with fpath.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def snapshot_csv(jsonl_path: pathlib.Path, csv_path: pathlib.Path) -> None:
    rows = read_annotations_jsonl(jsonl_path)
    if not rows:
        return
    # flatten boxes count for quick glance
    flat = []
    for r in rows:
        flat.append({
            "image_path": r.get("image_path"),
            "labels": ",".join(r.get("labels", [])),
            "num_boxes": len(r.get("boxes", [])),
            "timestamp": r.get("timestamp"),
            "annotator": r.get("annotator"),
        })
    pd.DataFrame(flat).to_csv(csv_path, index=False)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

def load_image(path: pathlib.Path, max_side: int = 768) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = max(w, h)
    if s > max_side:
        scale = max_side / s
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return img

# Zero-shot Classifier

class ZeroShotLabeler:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device).eval()
        self.text_labels: List[str] = []
        self.text_emb: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def encode_text_labels(self, labels: List[str]) -> torch.Tensor:
        if self.text_labels == labels and self.text_emb is not None:
            return self.text_emb
        prompts = [f"a photo of a {lbl}" for lbl in labels]
        tok = self.tokenizer(prompts).to(self.device)
        txt = self.model.encode_text(tok)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        self.text_labels = labels
        self.text_emb = txt
        return txt

    @torch.inference_mode()
    def encode_image_path(self, path: pathlib.Path) -> torch.Tensor:
        cache_f = IMG_EMB_DIR / f"{sha1_path(path)}.pt"
        if cache_f.exists():
            return torch.load(cache_f, map_location=self.device)
        img = load_image(path, max_side=512)
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        img_emb = self.model.encode_image(img_t)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        torch.save(img_emb.cpu(), cache_f)
        return img_emb.to(self.device)

    @torch.inference_mode()
    def score_image(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> np.ndarray:
        logits = (img_emb @ text_emb.t()).squeeze(0)
        return logits.detach().float().cpu().numpy()

# Detector (Ultralytics RT-DETR / YOLOv8)

class Detector:
    """
    Wrap Ultralyics models; cache outputs. Keeps UI responsive.
    """
    def __init__(self, model_name: str, device: Optional[str], conf: float, iou: float, max_det: int):
        self.device = (None if device == "auto" else device)
        self.model_name = model_name
        try:
            self.model = YOLO(model_name)  # supports 'rtdetr-l.pt', 'rtdetr-x.pt', 'yolov8n.pt'
        except Exception:
            # fallback to CPU + tiny model
            self.model = YOLO("yolov8n.pt")
            self.model_name = "yolov8n.pt"
            self.device = "cpu"
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)

    def infer_cached(self, img_path: pathlib.Path) -> Dict[str, Any]:
        key = sha1_path(img_path) + f"_{self.model_name}_{self.conf}_{self.iou}_{self.max_det}"
        cache_f = DET_EMB_DIR / f"{key}.json"
        if cache_f.exists():
            with cache_f.open("r", encoding="utf-8") as f:
                return json.load(f)
        start = time.time()
        res = self.model.predict(
            source=str(img_path),
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False
        )
        out = self._to_serializable(res[0])
        out["time_ms"] = int((time.time() - start) * 1000)
        with cache_f.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        return out

    def _to_serializable(self, r) -> Dict[str, Any]:
        boxes = r.boxes
        names = r.names
        W, H = int(r.orig_shape[1]), int(r.orig_shape[0])
        dets = []
        if boxes is None or len(boxes) == 0:
            return {"width": W, "height": H, "detections": []}
        xywh = boxes.xywh.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xywh)):
            x, y, w, h = xywh[i].tolist()
            dets.append({
                "bbox": [float(x - w/2), float(y - h/2), float(w), float(h)],  # coco xywh
                "score": float(conf[i]),
                "label": str(names.get(int(cls[i]), str(int(cls[i]))))
            })
        return {"width": W, "height": H, "detections": dets}

# Active Selection

def uncertainty_margin(probs: np.ndarray) -> float:
    s = np.sort(probs)[::-1]
    if len(s) < 2:
        return 1.0
    return 1.0 - float(s[0] - s[1])

def rank_images_by_uncertainty(paths: List[pathlib.Path], model: ZeroShotLabeler, text_emb: torch.Tensor, already_labeled: set, sample_limit: int = 200) -> List[Tuple[pathlib.Path, float]]:
    to_score = [p for p in paths if str(p) not in already_labeled]
    subset = to_score[:sample_limit] if sample_limit and len(to_score) > sample_limit else to_score
    scores = []
    for p in subset:
        img_emb = model.encode_image_path(p)
        logits = model.score_image(img_emb, text_emb)
        probs = softmax(logits)
        scores.append((p, uncertainty_margin(probs)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# Visualization

def draw_boxes(img: Image.Image, boxes: List[Dict], selected_ids: Optional[List[int]] = None) -> Image.Image:
    """
    Light boxes for all proposals; thicker for selected.
    """
    im = img.copy()
    draw = ImageDraw.Draw(im)
    selset = set(selected_ids or [])
    for i, b in enumerate(boxes):
        x, y, w, h = b["bbox"]
        x2, y2 = x + w, y + h
        width = 1 if i not in selset else 3  
        draw.rectangle([x, y, x2, y2], outline=(255, 255, 255), width=width)
        caption = f'#{i} {b.get("label","?")} {b.get("score",0):.2f}'
        tw, th = draw.textlength(caption), 10
        draw.rectangle([x, y - th - 4, x + max(40, tw) + 6, y], fill=(0, 0, 0))
        draw.text((x + 3, y - th - 2), caption, fill=(255, 255, 255))
    return im

# Streamlit App

@dataclass
class AppState:
    image_root: str
    labels: List[str]
    top_k: int
    policy: str
    current_index: int
    queue: List[str]
    session_id: str
    # detection ui
    det_enabled: bool
    det_model_name: str
    det_conf: float
    det_iou: float
    det_max_det: int

def init_state() -> AppState:
    sid = hashlib.sha1(str(time.time()).encode()).hexdigest()[:8]
    return AppState(
        image_root=str(APP_DIR / "images"),
        labels=DEFAULT_LABELS.copy(),
        top_k=3,
        policy="uncertain",
        current_index=0,
        queue=[],
        session_id=sid,
        det_enabled=True,
        det_model_name="rtdetr-l.pt",
        det_conf=0.25,
        det_iou=0.6,
        det_max_det=200,
    )

def ensure_text_cache(model: ZeroShotLabeler, labels: List[str]) -> torch.Tensor:
    if TEXT_EMB_F.exists():
        blob = torch.load(TEXT_EMB_F, map_location=model.device)
        if blob.get("labels") == labels:
            model.text_labels = labels
            model.text_emb = blob["emb"].to(model.device)
            return model.text_emb
    emb = model.encode_text_labels(labels)
    torch.save({"labels": labels, "emb": emb.cpu()}, TEXT_EMB_F)
    return emb

def load_queue(S: AppState, model: ZeroShotLabeler) -> None:
    imgs = list_images(S.image_root)
    ann = read_annotations_jsonl(ANNOTATIONS_JSONL)
    labeled = {r["image_path"] for r in ann}
    text_emb = ensure_text_cache(model, S.labels)
    if S.policy == "uncertain":
        ranked = rank_images_by_uncertainty(imgs, model, text_emb, labeled, sample_limit=2000)
        S.queue = [str(p) for p, _ in ranked] + [str(p) for p in imgs if str(p) not in labeled and str(p) not in {str(q) for q, _ in ranked}]
    elif S.policy == "random":
        rng = np.random.default_rng(0)
        remaining = [str(p) for p in imgs if str(p) not in labeled]
        rng.shuffle(remaining)
        S.queue = remaining
    else:
        S.queue = [str(p) for p in imgs if str(p) not in labeled]
    S.current_index = 0

def propose_labels_for_image(img_path: str, model: ZeroShotLabeler, labels: List[str], top_k: int) -> Tuple[List[str], Dict[str, float]]:
    text_emb = ensure_text_cache(model, labels)
    img_emb = model.encode_image_path(pathlib.Path(img_path))
    logits = model.score_image(img_emb, text_emb)
    probs = softmax(logits)
    idx = np.argsort(probs)[::-1][:top_k]
    proposals = [labels[i] for i in idx]
    conf = {labels[i]: float(probs[i]) for i in idx}
    return proposals, conf

def export_coco(jsonl_path: pathlib.Path, out_dir: pathlib.Path, categories: List[str]) -> pathlib.Path:
    rows = read_annotations_jsonl(jsonl_path)
    images = {}
    annotations = []
    cat_map = {name: i+1 for i, name in enumerate(categories)}
    ann_id = 1
    for r in rows:
        ipath = r.get("image_path")
        if ipath not in images:
            img_p = pathlib.Path(ipath)
            try:
                with Image.open(img_p) as im:
                    w, h = im.size
            except Exception:
                w = h = None
            images[ipath] = {
                "id": sha1_path(img_p)[:12],
                "file_name": str(img_p.name),
                "width": w,
                "height": h,
            }
        for b in r.get("boxes", []):
            bbox = b.get("bbox", [0,0,0,0])
            cat = b.get("label", "object")
            annotations.append({
                "id": ann_id,
                "image_id": images[ipath]["id"],
                "category_id": cat_map.get(cat, 0),
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0,
                "segmentation": [],
                "score": float(b.get("score", 1.0)),
            })
            ann_id += 1
    coco = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": [{"id": i+1, "name": n} for i, n in enumerate(categories)],
    }
    out_f = out_dir / "instances.json"
    with out_f.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    return out_f

def main():
    st.set_page_config(page_title="Model-Assisted Image Annotation + Detection", layout="wide")

    if "app_state" not in st.session_state:
        st.session_state.app_state = init_state()
    S: AppState = st.session_state.app_state

    st.sidebar.header("Settings")
    S.image_root = st.sidebar.text_input("Image folder", S.image_root, help="Folder with images.")
    labels_text = st.sidebar.text_area("Class labels (one per line)", "\n".join(S.labels), height=140)
    S.labels = [l.strip() for l in labels_text.splitlines() if l.strip()]
    S.top_k = st.sidebar.slider("Top-K class suggestions", 1, min(10, max(1, len(S.labels))), value=min(S.top_k, max(1, len(S.labels))))
    S.policy = st.sidebar.selectbox("Next image policy", ["uncertain", "random", "sequential"], index=["uncertain", "random", "sequential"].index(S.policy))

    with st.sidebar.expander("Classifier (CLIP)"):
        model_name = st.selectbox("Backbone", ["ViT-B-32", "ViT-B-16", "RN50"], index=0)
        pretrained = st.selectbox("Weights", ["openai", "laion2b_s34b_b79k"], index=0)
        device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
        chosen_device = None if device == "auto" else device
        if st.button("Load / Reload CLIP", type="primary"):
            st.session_state["model"] = ZeroShotLabeler(model_name=model_name, pretrained=pretrained, device=chosen_device)
            load_queue(S, st.session_state["model"])

    with st.sidebar.expander("Detector (RT-DETR / YOLO)"):
        S.det_enabled = st.checkbox("Enable detection proposals", value=S.det_enabled)
        S.det_model_name = st.selectbox("Weights", ["rtdetr-l.pt", "rtdetr-x.pt", "yolov8n.pt"], index=["rtdetr-l.pt","rtdetr-x.pt","yolov8n.pt"].index(S.det_model_name))
        det_device = st.selectbox("Det device", ["auto", "cpu", "cuda"], index=0, key="detdev")
        S.det_conf = st.slider("Confidence", 0.05, 0.9, value=float(S.det_conf), step=0.05)
        S.det_iou = st.slider("NMS IoU", 0.1, 0.9, value=float(S.det_iou), step=0.05)
        S.det_max_det = st.slider("Max dets", 10, 300, value=int(S.det_max_det), step=10)
        if st.button("Load / Reload Detector"):
            st.session_state["detector"] = Detector(S.det_model_name, det_device, S.det_conf, S.det_iou, S.det_max_det)

    if "model" not in st.session_state:
        st.session_state["model"] = ZeroShotLabeler()
        load_queue(S, st.session_state["model"])
    model: ZeroShotLabeler = st.session_state["model"]

    if S.det_enabled and "detector" not in st.session_state:
        st.session_state["detector"] = Detector(S.det_model_name, "auto", S.det_conf, S.det_iou, S.det_max_det)
    detector: Optional[Detector] = st.session_state.get("detector")

    if st.sidebar.button("Rebuild Queue"):
        load_queue(S, model)

    ann_rows = read_annotations_jsonl(ANNOTATIONS_JSONL)
    labeled_paths = {r["image_path"] for r in ann_rows}
    images = list_images(S.image_root)
    remaining = [p for p in images if str(p) not in labeled_paths]
    total = len(images)
    done = total - len(remaining)
    st.sidebar.markdown(f"**Progress:** {done}/{total} labeled")
    if st.sidebar.button("Export COCO"):
        out_f = export_coco(ANNOTATIONS_JSONL, COCO_DIR, S.labels)
        st.sidebar.success(f"COCO saved → {out_f}")

    col_l, col_r = st.columns([3, 2], gap="large")

    if not S.queue:
        st.info("No images queued. Check folder or click **Rebuild Queue**.")
        return

    S.current_index = max(0, min(S.current_index, len(S.queue) - 1))
    cur_path = S.queue[S.current_index]
    cur_img = load_image(pathlib.Path(cur_path), max_side=1024)

    with col_l:
        st.subheader(f"Image {S.current_index+1} / {len(S.queue)}")
        st.image(cur_img, use_column_width=True)
        with st.expander("Raw path"):
            st.code(cur_path)

        # Detection visualization
        det_result = {"detections": [], "width": cur_img.size[0], "height": cur_img.size[1], "time_ms": None}
        if S.det_enabled and detector:
            with st.spinner("Detecting objects..."):
                det_result = detector.infer_cached(pathlib.Path(cur_path))
        st.caption(f"Detections: {len(det_result.get('detections', []))}  |  model={getattr(detector,'model_name','-')}  |  {det_result.get('time_ms','-')}ms")

        # Selection state per image
        sel_key = f"sel_{sha1_path(pathlib.Path(cur_path))}"
        if sel_key not in st.session_state:
            # preselect confident boxes
            high = [i for i,b in enumerate(det_result.get("detections", [])) if b.get("score",0) >= max(0.5, S.det_conf)]
            st.session_state[sel_key] = high[:20]
        selected_ids = st.session_state[sel_key]

        overlay_all = draw_boxes(cur_img, det_result.get("detections", []), selected_ids=selected_ids)
        st.image(overlay_all, caption="Proposals (bold = selected)", use_column_width=True)

    with col_r:
        st.subheader("Model Class Suggestions")
        with st.spinner("Scoring..."):
            proposals, conf = propose_labels_for_image(cur_path, model, S.labels, S.top_k)
        st.markdown("Top-K:")
        for p in proposals:
            st.write(f"- {p}: {conf[p]:.3f}")
        user_labels = st.multiselect("Image-level labels", options=S.labels, default=proposals)
        extra_labels = st.text_input("Add new labels (comma-separated)", value="")
        note = st.text_area("Notes (optional)", value="", height=60)

        if S.det_enabled:
            st.subheader("Detection Proposals")
            dets = det_result.get("detections", [])
            c1, c2 = st.columns(2)
            if c1.button("Select All"):
                st.session_state[sel_key] = list(range(len(dets)))
                selected_ids = st.session_state[sel_key]
            if c2.button("Clear Selection"):
                st.session_state[sel_key] = []
                selected_ids = []
            # editable table
            for i, d in enumerate(dets):
                with st.expander(f"#{i} {d.get('label','?')} @ {d.get('score',0):.2f}", expanded=(i in selected_ids)):
                    # selection toggle
                    checked = st.checkbox("Accept", value=(i in selected_ids), key=f"acc_{sel_key}_{i}")
                    if checked and i not in selected_ids:
                        st.session_state[sel_key].append(i)
                    if (not checked) and i in selected_ids:
                        st.session_state[sel_key].remove(i)
                    # allow relabel
                    new_label = st.selectbox("Label", options=S.labels + [d.get("label","other")], index=(S.labels + [d.get("label","other")]).index(d.get("label","other")) if d.get("label") in (S.labels + [d.get("label","other")]) else 0, key=f"lbl_{sel_key}_{i}")
                    d["label"] = new_label
                    st.write(f"bbox (xywh): {', '.join([f'{x:.1f}' for x in d['bbox']])}")

        c1, c2, c3, c4 = st.columns(4)
        save_clicked = c1.button("Save & Next", type="primary")
        skip_clicked = c2.button("Skip")
        back_clicked = c3.button("Back")
        refresh_clicked = c4.button("Refresh")

        if refresh_clicked:
            st.experimental_rerun()
        if back_clicked:
            S.current_index = max(0, S.current_index - 1)
            st.experimental_rerun()
        if skip_clicked:
            S.current_index = min(len(S.queue) - 1, S.current_index + 1)
            st.experimental_rerun()

        if save_clicked:
            labels_final = [l.strip() for l in user_labels]
            if extra_labels.strip():
                labels_final.extend([x.strip() for x in extra_labels.split(",") if x.strip()])
            labels_final = sorted(set(labels_final))
            # accepted boxes
            accepted = []
            if S.det_enabled:
                for idx in st.session_state[sel_key]:
                    d = det_result["detections"][idx]
                    accepted.append({
                        "bbox": [float(d["bbox"][0]), float(d["bbox"][1]), float(d["bbox"][2]), float(d["bbox"][3])],
                        "label": str(d.get("label","object")),
                        "score": float(d.get("score", 1.0)),
                    })
            record = {
                "image_path": cur_path,
                "labels": labels_final,
                "proposals": proposals,
                "proposal_conf": conf,
                "boxes": accepted,
                "annotator": f"session-{S.session_id}",
                "timestamp": int(time.time()),
                "detector_meta": {
                    "enabled": S.det_enabled,
                    "model": getattr(detector, "model_name", None),
                    "conf": S.det_conf,
                    "iou": S.det_iou,
                    "max_det": S.det_max_det,
                }
            }
            write_annotation_jsonl(record, ANNOTATIONS_JSONL)
            snapshot_csv(ANNOTATIONS_JSONL, ANNOTATIONS_CSV)
            S.current_index = min(len(S.queue) - 1, S.current_index + 1)
            st.experimental_rerun()

    with st.expander("Recent Annotations"):
        rows = read_annotations_jsonl(ANNOTATIONS_JSONL)
        if rows:
            df = pd.DataFrame(rows[-100:])
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No annotations yet.")

    st.caption("Tip: use RT-DETR for quality; YOLOv8n for speed on CPU. Downscale images if sluggish.")

if __name__ == "__main__":
    main()
