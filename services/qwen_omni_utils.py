"""
Qwen2.5-Omni Utility Functions
Based on Hugging Face documentation for multimodal processing
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import numpy as np
import soundfile as sf
import resampy

def load_audio_as_array(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load an audio file as a float32 numpy array, resampled to target_sr and mono.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # Convert to mono
    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)
    return audio.astype(np.float32)

def process_mm_info(conversations: List[List[Dict]], use_audio_in_video: bool = False) -> Tuple[List, List, List]:
    """
    Process multimodal information from conversations.
    Based on Hugging Face Qwen2.5-Omni documentation.
    
    Args:
        conversations: List of conversation objects
        use_audio_in_video: Whether to extract audio from videos
        
    Returns:
        Tuple of (audios, images, videos) lists
    """
    audios = []
    images = []
    videos = []
    
    for conversation in conversations:
        for message in conversation:
            if "content" in message and isinstance(message["content"], list):
                for content_item in message["content"]:
                    if isinstance(content_item, dict):
                        content_type = content_item.get("type")
                        if content_type == "audio" and "audio" in content_item:
                            audio_path = content_item["audio"]
                            try:
                                audio_array = load_audio_as_array(audio_path)
                                audios.append(audio_array)
                            except Exception as e:
                                print(f"[Qwen2.5-Omni] Failed to load audio: {audio_path} ({e})")
                        elif content_type == "image" and "image" in content_item:
                            image_path = content_item["image"]
                            if os.path.exists(image_path):
                                images.append(image_path)
                        elif content_type == "video" and "video" in content_item:
                            video_path = content_item["video"]
                            if os.path.exists(video_path):
                                videos.append(video_path)
    return audios, images, videos 