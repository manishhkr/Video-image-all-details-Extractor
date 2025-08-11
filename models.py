
import torch
import whisper
from doctr.models import ocr_predictor
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration


print("[INFO] Loading models...")
ocr_model = ocr_predictor(pretrained=True)
asr_model = whisper.load_model("base")
detect_model = YOLO("yolov8n.pt")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("[INFO] Models loaded.")

no_grad = torch.no_grad
