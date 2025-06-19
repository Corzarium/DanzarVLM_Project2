"""
Script to download the Whisper model.
"""
import os
import whisper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_whisper_model():
    """Download the Whisper base model."""
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Download Whisper base model
        logger.info("Downloading Whisper base model...")
        model = whisper.load_model("base")
        
        # Verify model loaded correctly
        logger.info("Verifying model...")
        test_audio = whisper.load_audio("test.wav") if os.path.exists("test.wav") else None
        if test_audio is not None:
            result = model.transcribe(test_audio)
            logger.info("Model verification successful")
        else:
            logger.info("Model downloaded successfully")
        
    except Exception as e:
        logger.error(f"Error downloading Whisper model: {e}")
        raise

if __name__ == "__main__":
    download_whisper_model() 