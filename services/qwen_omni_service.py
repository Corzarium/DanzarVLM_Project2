"""
Qwen2.5-Omni Service for DanzarVLM
Handles multimodal processing with text-only output to avoid meta tensor issues
"""

import asyncio
import logging
import os
import tempfile
from typing import Optional, Tuple, Dict, Any

class QwenOmniService:
    """Service for Qwen2.5-Omni multimodal processing"""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Model components
        self.model = None
        self.processor = None
        self.model_loaded = False
        
    async def initialize(self) -> bool:
        """Initialize the Qwen2.5-Omni service"""
        try:
            self.logger.info("🚀 Initializing Qwen2.5-Omni service...")
            
            # Import required modules
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
            from qwen_omni_utils import process_mm_info
            
            self.process_mm_info = process_mm_info
            
            # Load model
            model_name = self.config.get('QWEN_OMNI_MODEL', 'Qwen/Qwen2.5-Omni-7B')
            self.logger.info(f"📦 Loading {model_name}...")
            
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            
            # Disable audio generation to avoid meta tensor issues
            self.model.disable_talker()
            self.logger.info("🔇 Audio generation disabled (using text-only mode)")
            
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
            
            self.model_loaded = True
            self.logger.info("✅ Qwen2.5-Omni service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Qwen2.5-Omni service: {e}")
            return False
    
    def _transcribe_sync(self, audio_path: str, text: str = "What do you hear?") -> str:
        """
        Synchronous transcription using Qwen2.5-Omni
        
        Args:
            audio_path: Path to audio file
            text: Optional text prompt
            
        Returns:
            Transcribed text
        """
        try:
            if not self.model_loaded:
                self.logger.error("Model not loaded")
                return "Error: Model not loaded"
            
            # Create conversation with audio input
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant that can analyze audio content."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": text},
                    ],
                },
            ]
            
            USE_AUDIO_IN_VIDEO = False
            
            # Process inputs
            text_prompt = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            audios, images, videos = self.process_mm_info(
                conversation, 
                use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            
            inputs = self.processor(
                text=text_prompt, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            
            # Generate text response only
            text_ids = self.model.generate(
                **inputs, 
                use_audio_in_video=USE_AUDIO_IN_VIDEO, 
                return_audio=False,
                max_length=512,
                do_sample=True,
                temperature=0.7
            )
            
            # Decode response
            response = self.processor.batch_decode(
                text_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            if response and len(response) > 0:
                # Extract just the assistant's response
                full_response = response[0]
                if "assistant\n" in full_response:
                    assistant_response = full_response.split("assistant\n")[-1].strip()
                    return assistant_response
                else:
                    return full_response.strip()
            else:
                return "No response generated"
                
        except Exception as e:
            self.logger.error(f"❌ Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    async def transcribe_audio(self, audio_path: str, text: str = "Please transcribe this audio.") -> str:
        """
        Asynchronous wrapper for audio transcription
        
        Args:
            audio_path: Path to audio file
            text: Optional prompt text
            
        Returns:
            Transcribed text
        """
        try:
            # Run synchronous transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._transcribe_sync, 
                audio_path, 
                text
            )
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Async transcription error: {e}")
            return f"Error: {str(e)}"
    
    async def analyze_audio(self, audio_path: str, question: str = "What do you hear in this audio?") -> str:
        """
        Analyze audio content with a specific question
        
        Args:
            audio_path: Path to audio file
            question: Question about the audio
            
        Returns:
            Analysis result
        """
        return await self.transcribe_audio(audio_path, question)
    
    def is_available(self) -> bool:
        """Check if the service is available"""
        return self.model_loaded and self.model is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "service": "QwenOmniService",
            "model_loaded": self.model_loaded,
            "available": self.is_available(),
            "mode": "text_only"
        } 