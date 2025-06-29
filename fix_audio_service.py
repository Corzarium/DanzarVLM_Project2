#!/usr/bin/env python3
# Audio Service Fix Script
# Run this to fix common audio issues

import subprocess
import sys

def install_packages():
    packages = [
        'openai-whisper',
        'torch',
        'numpy',
        'pydub',
        'tensorflow',
        'vosk',
        'faster-whisper',
        'discord.py',
        'sounddevice',
        'webrtcvad',
        'librosa',
        'discord-ext-voice-recv'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', package])

def download_vosk_model():
    import os
    import urllib.request
    import zipfile
    
    model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    model_dir = "models"
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    print("Downloading Vosk model...")
    urllib.request.urlretrieve(model_url, "vosk-model.zip")
    
    with zipfile.ZipFile("vosk-model.zip", 'r') as zip_ref:
        zip_ref.extractall(model_dir)
    
    os.remove("vosk-model.zip")
    print("Vosk model downloaded successfully")

if __name__ == "__main__":
    install_packages()
    download_vosk_model()
    print("Audio service fixes completed!")
