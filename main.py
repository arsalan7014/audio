from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel
import uvicorn
import tempfile

app = FastAPI()

model = WhisperModel("base", compute_type="int8")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    segments, _ = model.transcribe(tmp_path)
    transcription = "".join([seg.text for seg in segments])

    return {"text": transcription}
