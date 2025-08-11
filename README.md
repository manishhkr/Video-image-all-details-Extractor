# Video & Image Analysis API

This FastAPI application provides:
- **Video analysis**: OCR (Doctr), object detection (YOLOv8), image captioning (BLIP), and audio transcription (Whisper).
- **Image analysis**: Object detection (YOLOv8) + BLIP scene caption, returns only `scene_caption` and `confidence`.

# Whisper needs ffmpeg for audio extraction.
Windows (Chocolatey):
choco install ffmpeg

# Windows (manual):
Download from https://ffmpeg.org/download.html
Extract the archive
Add the bin folder path to your system PATH environment variable
