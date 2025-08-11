
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import os
import shutil
import tempfile

from app.video_utils import extract_text_and_objects, group_by_segment, transcribe_audio, combine_segments
from app.image_utils import summarize_for_api

app = FastAPI(title="Video+Image Analysis API")

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        video_path = temp_file.name

    try:
        frames = extract_text_and_objects(video_path)
        visual_segments = group_by_segment(frames)
        asr_segments = transcribe_audio(video_path)
        combined = combine_segments(asr_segments, visual_segments)
        return JSONResponse(content=combined)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass


@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...), conf: float = 0.3):
    """
    Returns ONLY:
      {
        "scene_caption": <str>,
        "confidence": <float>   # max YOLO detection confidence in image
      }
    """
    try:
        content = await file.read()
        pil_img = Image.open(BytesIO(content)).convert("RGB")
        out = summarize_for_api(pil_img, conf=conf)
        return JSONResponse(content=out)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
