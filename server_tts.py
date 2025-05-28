# server_tts.py

from flask import Flask, request, send_file, jsonify
import io
import os
import soundfile as sf
from kokoro_onnx import Kokoro

app = Flask(__name__)

# Adjust these paths to wherever you’ve placed the Kokoro model files
KOKORO_MODEL  = os.path.expanduser("./models/kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.expanduser("./models/voices-v1.0.bin")

# Load once at startup
tts = Kokoro(model_path=KOKORO_MODEL, voices_path=KOKORO_VOICES)

@app.route("/tts", methods=["POST"])
def tts_endpoint():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error":"no text provided"}), 400

    # Generate raw samples + rate
    samples, rate = tts.create(text, voice="am_michael", speed=1.0, lang="en-us")

    # Write WAV into an in-memory buffer
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format="WAV")
    buf.seek(0)

    # Stream that buffer back as audio/wav
    return send_file(
        buf,
        mimetype="audio/wav",
        as_attachment=False,
        download_name="tts.wav"
    )

if __name__ == "__main__":
    # Listen on all interfaces, port 1234
    app.run(host="0.0.0.0", port=1235)
