#!/usr/bin/env python3
"""
Setup script for DanzarAI Offline Voice Processing
Downloads required models for 100% local operation
"""

import os
import sys
import urllib.request
import zipfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url: str, filename: str) -> bool:
    """Download a file with progress indication."""
    try:
        logger.info(f"📥 Downloading {filename}...")
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                sys.stdout.write(f"\r📊 Progress: {percent}% ({downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filename, progress_hook)
        print()  # New line after progress
        logger.info(f"✅ Downloaded {filename}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {filename}: {e}")
        return False

def extract_zip(zip_path: str, extract_to: str) -> bool:
    """Extract a ZIP file."""
    try:
        logger.info(f"📦 Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"✅ Extracted to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to extract {zip_path}: {e}")
        return False

def setup_vosk_model():
    """Download and setup Vosk STT model."""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Vosk small English model (40MB)
    model_name = "vosk-model-small-en-us-0.15"
    model_url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    model_zip = os.path.join(models_dir, f"{model_name}.zip")
    model_dir = os.path.join(models_dir, model_name)
    
    if os.path.exists(model_dir):
        logger.info(f"✅ Vosk model already exists: {model_dir}")
        return True
    
    logger.info("🎤 Setting up Vosk STT model...")
    
    # Download model
    if not download_file(model_url, model_zip):
        return False
    
    # Extract model
    if not extract_zip(model_zip, models_dir):
        return False
    
    # Clean up ZIP file
    try:
        os.remove(model_zip)
        logger.info("🗑️ Cleaned up ZIP file")
    except:
        pass
    
    logger.info(f"✅ Vosk model ready: {model_dir}")
    return True

def setup_transformers_cache():
    """Pre-download Transformers models to cache."""
    try:
        logger.info("🧠 Setting up local LLM models...")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Download lightweight conversational model
        model_name = "microsoft/DialoGPT-medium"
        logger.info(f"📥 Downloading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        logger.info(f"✅ Local LLM model cached: {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to setup LLM models: {e}")
        return False

def setup_silero_tts():
    """Pre-download Silero TTS models."""
    try:
        logger.info("🔊 Setting up Silero TTS models...")
        
        import torch
        
        # Download Silero TTS model
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='en',
            speaker='v3_en'
        )
        
        logger.info("✅ Silero TTS model cached")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to setup TTS models: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("🚀 Setting up DanzarAI Offline Voice Processing...")
    
    success = True
    
    # Setup Vosk STT
    if not setup_vosk_model():
        success = False
    
    # Setup Transformers LLM
    if not setup_transformers_cache():
        success = False
    
    # Setup Silero TTS
    if not setup_silero_tts():
        success = False
    
    if success:
        logger.info("🎉 All offline models setup successfully!")
        logger.info("🔒 DanzarAI is now ready for 100% offline voice processing")
        logger.info("💡 Use `!offline` command to check status")
        logger.info("🎤 Use `!join` to start offline voice processing")
    else:
        logger.error("❌ Some models failed to setup")
        sys.exit(1)

if __name__ == "__main__":
    main() 