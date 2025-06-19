"""
Configuration settings for the voice-enabled Discord bot.
"""
from typing import Dict, Any
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Discord settings
DISCORD_SETTINGS = {
    "TOKEN": os.getenv("DISCORD_TOKEN", ""),
    "GUILD_ID": int(os.getenv("DISCORD_GUILD_ID", "0")),
    "VOICE_CHANNEL_ID": int(os.getenv("DISCORD_VOICE_CHANNEL_ID", "0")),
    "COMMAND_PREFIX": "!",
}

# Voice settings
VOICE_SETTINGS = {
    "VAD_FRAME_DURATION": 30,  # ms
    "VAD_PADDING_DURATION": 300,  # ms
    "VAD_THRESHOLD": 0.5,
    "SAMPLE_RATE": 16000,
    "CHANNELS": 1,
    "CHUNK_SIZE": 1024,
    "SILENCE_THRESHOLD": 0.1,
    "SILENCE_DURATION": 1.0,  # seconds
}

# STT settings
STT_SETTINGS = {
    "MODEL": "whisper",  # Options: whisper, vosk
    "MODEL_PATH": str(DATA_DIR / "models" / "whisper-base"),
    "LANGUAGE": "en",
    "COMPUTE_TYPE": "float16",  # Options: float16, float32
}

# TTS settings
TTS_SETTINGS = {
    "ENGINE": "chatterbox",  # Options: chatterbox, piper
    "VOICE": "en_US-hfc_female-medium",
    "SAMPLE_RATE": 24000,
    "SPEAKER_ID": 0,
}

# LLM settings
LLM_SETTINGS = {
    "MODEL": "qwen2:vl",  # Options: qwen2:vl, gemma:instruct
    "ENDPOINT": "http://ollama:11434/chat/completions",
    "TEMPERATURE": 0.7,
    "MAX_TOKENS": 1024,
    "TOP_P": 0.9,
}

# Memory settings
MEMORY_SETTINGS = {
    "SHORT_TERM_SIZE": 10,  # Number of message pairs to keep in STM
    "RAG_COLLECTION": "voice_bot_memory",
    "RAG_HOST": "qdrant",
    "RAG_PORT": 6333,
    "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
}

# Logging settings
LOGGING_SETTINGS = {
    "LEVEL": "INFO",
    "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "FILE": str(LOGS_DIR / "voice_bot.log"),
}

def get_settings() -> Dict[str, Any]:
    """Get all settings as a dictionary."""
    return {
        "discord": DISCORD_SETTINGS,
        "voice": VOICE_SETTINGS,
        "stt": STT_SETTINGS,
        "tts": TTS_SETTINGS,
        "llm": LLM_SETTINGS,
        "memory": MEMORY_SETTINGS,
        "logging": LOGGING_SETTINGS,
    } 