from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile

app = FastAPI()
model = WhisperModel("base", compute_type="int8")  # Use "tiny" if you want it lighter

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    segments, _ = model.transcribe(tmp_path)
    text = "".join(segment.text for segment in segments)

    return {"text": text.strip()}

