#!/usr/bin/env python3
"""
Chatterbox TTS Server with GPU acceleration
Based on https://github.com/resemble-ai/chatterbox
"""

import os
import io
import logging
import torch
import torchaudio
from flask import Flask, request, jsonify, send_file
from chatterbox.tts import ChatterboxTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global TTS model instance
tts_model = None

def initialize_tts():
    """Initialize Chatterbox TTS model with GPU support"""
    global tts_model
    try:
        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Chatterbox TTS on device: {device}")
        
        if device == "cuda":
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Initialize the model
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("Chatterbox TTS model loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize TTS model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if tts_model is not None:
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    else:
        return jsonify({"status": "unhealthy", "model_loaded": False}), 503

@app.route('/tts', methods=['POST'])
def generate_tts():
    """Generate TTS audio from text"""
    try:
        if tts_model is None:
            return jsonify({"error": "TTS model not initialized"}), 503
        
        # Parse request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({"error": "Empty text provided"}), 400
        
        # Extract parameters with defaults
        predefined_voice_id = data.get('predefined_voice_id', 'default')
        format_type = data.get('format', 'wav')
        exaggeration = data.get('exaggeration', 0.5)
        cfg_weight = data.get('cfg_weight', 0.5)
        
        # Map voice parameters to Chatterbox parameters
        # Note: Chatterbox uses different parameter names than legacy TTS
        
        logger.info(f"Generating TTS for text: '{text[:50]}...' (length: {len(text)})")
        logger.debug(f"Parameters - voice: {predefined_voice_id}, exaggeration: {exaggeration}, cfg_weight: {cfg_weight}")
        
        # Generate audio using Chatterbox
        # For voice cloning, you would need to provide audio_prompt_path
        if predefined_voice_id != 'default' and os.path.exists(f"/app/voices/{predefined_voice_id}"):
            # Use voice cloning if voice file exists
            audio_prompt_path = f"/app/voices/{predefined_voice_id}"
            wav = tts_model.generate(
                text, 
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
        else:
            # Use default voice
            wav = tts_model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
        
        # Convert to bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav, tts_model.sr, format="wav")
        buffer.seek(0)
        
        audio_bytes = buffer.getvalue()
        logger.info(f"Generated {len(audio_bytes)} bytes of audio")
        
        # Return audio as bytes (compatible with existing TTS service)
        return audio_bytes, 200, {'Content-Type': 'audio/wav'}
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}", exc_info=True)
        return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500

@app.route('/voices', methods=['GET'])
def list_voices():
    """List available voice files"""
    try:
        voices_dir = "/app/voices"
        if os.path.exists(voices_dir):
            voices = [f for f in os.listdir(voices_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
            return jsonify({"voices": voices}), 200
        else:
            return jsonify({"voices": []}), 200
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        return jsonify({"error": "Failed to list voices"}), 500

@app.route('/upload_voice', methods=['POST'])
def upload_voice():
    """Upload a voice file for cloning"""
    try:
        if 'voice_file' not in request.files:
            return jsonify({"error": "No voice file provided"}), 400
        
        file = request.files['voice_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save voice file
        voices_dir = "/app/voices"
        os.makedirs(voices_dir, exist_ok=True)
        
        filename = file.filename
        file_path = os.path.join(voices_dir, filename)
        file.save(file_path)
        
        logger.info(f"Voice file uploaded: {filename}")
        return jsonify({"message": f"Voice file '{filename}' uploaded successfully"}), 200
        
    except Exception as e:
        logger.error(f"Voice upload failed: {e}")
        return jsonify({"error": f"Voice upload failed: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("Starting Chatterbox TTS Server...")
    
    # Initialize TTS model
    if not initialize_tts():
        logger.error("Failed to initialize TTS model. Exiting.")
        exit(1)
    
    # Start Flask server
    app.run(host='0.0.0.0', port=8055, debug=False, threaded=True) 