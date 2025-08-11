
from typing import Tuple, List, Dict
from PIL import Image
import numpy as np

from .models import detect_model, blip_processor, blip_model, no_grad

def scene_caption(pil_img: Image.Image) -> str:
    try:
        inputs = blip_processor(images=pil_img, return_tensors="pt")
        with no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=25)
        return blip_processor.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return ""

def detect_objects(img_np: np.ndarray, conf: float = 0.3, max_det: int = 100) -> List[Dict]:
    with no_grad():
        yolo_out = detect_model(img_np, conf=conf, verbose=False, max_det=max_det)[0]
    dets = []
    if not hasattr(yolo_out, "boxes") or yolo_out.boxes is None:
        return dets
    for x1, y1, x2, y2, score, cls_id in yolo_out.boxes.data.tolist():
        if score < conf:
            continue
        dets.append({
            "label": detect_model.names[int(cls_id)],
            "confidence": float(score),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })
    return dets

def summarize_for_api(pil_img: Image.Image, conf: float = 0.3) -> dict:
    """
    Returns ONLY:
      - scene_caption: BLIP caption for the whole image
      - confidence: max YOLO detection confidence in the image (0.0 if none)
    Note: since BLIP doesn't emit confidence, we expose YOLO's strongest signal.
    """
    cap = scene_caption(pil_img)
    img_np = np.array(pil_img)
    dets = detect_objects(img_np, conf=conf)
    max_conf = max((d["confidence"] for d in dets), default=0.0)
    return {"scene_caption": cap, "confidence": float(max_conf)}
