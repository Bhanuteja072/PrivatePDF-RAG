from vosk import Model, KaldiRecognizer
import pyaudio
from pathlib import Path
import json
import pyttsx3
import tempfile

MODEL_PATH = Path(r"C:\Users\Prabha\Downloads\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15")
_model = Model(str(MODEL_PATH))


def transcribe_once(max_seconds: int = 10) -> str:
    rec = KaldiRecognizer(_model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=4096,
    )
    stream.start_stream()
    frames_needed = int(16000 / 4096 * max_seconds)
    text = ""
    try:
        for _ in range(frames_needed):
            data = stream.read(4096, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                break
        if not text:
            result = json.loads(rec.FinalResult())
            text = result.get("text", "")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    return text


def tts_to_bytes(text: str) -> bytes:
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        out_path = tmp.name
    try:
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        try:
            Path(out_path).unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    print(transcribe_once())