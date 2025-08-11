from collections import defaultdict
from PIL import Image
import numpy as np
import cv2
from .models import ocr_model, asr_model  

def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds + 0.5)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02}:{seconds:02}"

def extract_text_and_objects(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps * interval))
    frame_id = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            ts = frame_id / fps if fps > 0 else 0.0
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_np = np.array(pil_img)

            # OCR
            ocr_result = ocr_model([img_np])
            text = []
            if ocr_result.pages:
                for block in ocr_result.pages[0].blocks:
                    for line in block.lines:
                        if not line.words:
                            continue
                        avg_conf = sum(w.confidence for w in line.words) / len(line.words)
                        if avg_conf > 0.5:
                            text_line = " ".join(w.value for w in line.words).strip()
                            if text_line:
                                text.append(text_line)

            frames.append({"timestamp": ts, "text": text})

        frame_id += 1

    cap.release()
    return frames

def group_by_segment(frames, segment_length=20):
    segments = defaultdict(lambda: {"visual_text": []})
    for frame in frames:
        sec = int(frame["timestamp"])
        seg_start = (sec // segment_length) * segment_length
        seg_key = (format_timestamp(seg_start), format_timestamp(seg_start + segment_length))
        segments[seg_key]["visual_text"].extend(frame.get("text", []))
    return segments

def transcribe_audio(video_path):
    return asr_model.transcribe(video_path)["segments"]

def combine_segments(asr_segments, visual_segments):
    result = []
    for (start, end), visual_data in visual_segments.items():
        spoken_texts = []
        for seg in asr_segments:
            seg_start = format_timestamp(seg["start"])
            seg_end = format_timestamp(seg["end"])
            if seg_start >= start and seg_end <= end:
                spoken_texts.append(seg["text"].strip())

        visual_text = list({t for t in visual_data["visual_text"] if len(t.strip()) > 1})

        result.append({
            "start": start,
            "end": end,
            "audio_text": " ".join(spoken_texts),
            "visual_text": visual_text
        })
    return result
