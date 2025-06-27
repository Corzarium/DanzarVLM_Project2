#!/usr/bin/env python3
"""
Azure TTS Service for DanzarVLM
Replaces Chatterbox TTS with Microsoft Azure Text-to-Speech
"""

import asyncio
import aiohttp
import json
import logging
import base64
import time
from typing import Optional, Dict, Any
from urllib.parse import quote
import xml.etree.ElementTree as ET


class AzureTTSService:
    """Microsoft Azure Text-to-Speech Service"""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Azure TTS Configuration
        self.subscription_key = self.config.get('AZURE_TTS_SUBSCRIPTION_KEY')
        self.region = self.config.get('AZURE_TTS_REGION', 'eastus')
        self.voice_name = self.config.get('AZURE_TTS_VOICE', 'en-US-AdamMultilingualNeural') or 'en-US-AdamMultilingualNeural'
        self.speech_rate = self.config.get('AZURE_TTS_SPEECH_RATE', '+0%')
        self.pitch = self.config.get('AZURE_TTS_PITCH', '+0%')
        self.volume = self.config.get('AZURE_TTS_VOLUME', '+0%')
        
        # API endpoints
        self.base_url = f"https://{self.region}.tts.speech.microsoft.com"
        self.synthesis_url = f"{self.base_url}/cognitiveservices/v1"
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        self.logger.info(f"🎤 Azure TTS Service initialized with voice: {self.voice_name}")
    
    async def initialize(self) -> bool:
        """Initialize the Azure TTS service"""
        try:
            if not self.subscription_key:
                self.logger.error("❌ Azure TTS subscription key not configured")
                return False
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'Ocp-Apim-Subscription-Key': self.subscription_key,
                    'Content-Type': 'application/ssml+xml',
                    'X-Microsoft-OutputFormat': 'riff-24khz-16bit-mono-pcm',
                    'User-Agent': 'DanzarVLM-AzureTTS/1.0'
                }
            )
            
            # Test connection
            if await self._test_connection():
                self.logger.info("✅ Azure TTS Service initialized successfully")
                return True
            else:
                self.logger.error("❌ Azure TTS connection test failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Azure TTS Service: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test Azure TTS connection with a simple synthesis request"""
        try:
            test_text = "Azure TTS connection test successful."
            audio_data = await self.synthesize_speech(test_text)
            return audio_data is not None and len(audio_data) > 0
        except Exception as e:
            self.logger.error(f"❌ Azure TTS connection test failed: {e}")
            return False
    
    def _create_ssml(self, text: str) -> str:
        """Create SSML markup for Azure TTS"""
        # Clean text for TTS
        text = self._clean_text_for_tts(text)
        
        # Create SSML with voice configuration
        ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
                xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">
    <voice name="{self.voice_name}">
        <mstts:express-as style="cheerful" styledegree="0.7">
            <prosody rate="{self.speech_rate}" pitch="{self.pitch}" volume="{self.volume}">
                {text}
            </prosody>
        </mstts:express-as>
    </voice>
</speak>"""
        return ssml
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS synthesis"""
        # Remove markdown formatting
        import re
        
        # Remove markdown links
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        text = re.sub(r'`([^`]+)`', r'\1', text)        # Code
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Escape XML characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&apos;')
        
        return text
    
    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Synthesize speech from text using Azure TTS"""
        try:
            if not self.session:
                self.logger.error("❌ Azure TTS session not initialized")
                return None
            
            if not text or not text.strip():
                self.logger.warning("⚠️ Empty text provided for TTS synthesis")
                return None
            
            # Create SSML
            ssml = self._create_ssml(text)
            
            self.logger.debug(f"🎤 Synthesizing speech: {text[:50]}...")
            
            # Make API request
            async with self.session.post(
                self.synthesis_url,
                data=ssml.encode('utf-8'),
                headers={'Content-Type': 'application/ssml+xml'}
            ) as response:
                
                if response.status == 200:
                    audio_data = await response.read()
                    self.logger.debug(f"✅ TTS synthesis successful: {len(audio_data)} bytes")
                    return audio_data
                else:
                    error_text = await response.text()
                    self.logger.error(f"❌ Azure TTS API error {response.status}: {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            self.logger.error("❌ Azure TTS request timed out")
            return None
        except Exception as e:
            self.logger.error(f"❌ Azure TTS synthesis failed: {e}")
            return None
    
    async def synthesize_speech_chunked(self, text: str, max_chunk_length: int = 200) -> list[bytes]:
        """Synthesize speech in chunks to handle long text"""
        try:
            # Split text into sentences
            sentences = self._split_into_sentences(text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= max_chunk_length:
                    current_chunk += sentence + " "
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            # Add remaining chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Synthesize each chunk
            audio_chunks = []
            for chunk in chunks:
                audio_data = await self.synthesize_speech(chunk)
                if audio_data:
                    audio_chunks.append(audio_data)
                else:
                    self.logger.warning(f"⚠️ Failed to synthesize chunk: {chunk[:50]}...")
            
            return audio_chunks
            
        except Exception as e:
            self.logger.error(f"❌ Chunked TTS synthesis failed: {e}")
            return []
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences for chunking"""
        import re
        
        # Split by sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    async def get_available_voices(self) -> list[Dict[str, Any]]:
        """Get list of available Azure TTS voices"""
        try:
            if not self.session:
                return []
            
            voices_url = f"{self.base_url}/cognitiveservices/voices/list"
            
            async with self.session.get(voices_url) as response:
                if response.status == 200:
                    voices_data = await response.json()
                    return voices_data
                else:
                    self.logger.error(f"❌ Failed to get voices: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"❌ Failed to get available voices: {e}")
            return []
    
    async def test_voice(self, voice_name: str = None) -> bool:
        """Test a specific voice"""
        try:
            test_voice = voice_name or self.voice_name
            if not test_voice:
                self.logger.error("❌ No voice name provided for testing")
                return False
                
            test_text = f"Hello, this is a test of the {test_voice} voice."
            
            # Temporarily change voice
            original_voice = self.voice_name
            self.voice_name = test_voice
            
            audio_data = await self.synthesize_speech(test_text)
            
            # Restore original voice
            self.voice_name = original_voice
            
            return audio_data is not None and len(audio_data) > 0
            
        except Exception as e:
            self.logger.error(f"❌ Voice test failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            self.logger.info("🧹 Azure TTS Service cleaned up")
        except Exception as e:
            self.logger.error(f"❌ Error during Azure TTS cleanup: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            'service': 'Azure TTS',
            'voice': self.voice_name,
            'region': self.region,
            'connected': self.session is not None,
            'subscription_key_configured': bool(self.subscription_key)
        }
    
    def generate_audio(self, text: str) -> Optional[bytes]:
        """Generate audio from text (synchronous wrapper for async method)"""
        try:
            # Create a new event loop if one doesn't exist
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, we need to handle this differently
                # For now, return None and let the caller handle async synthesis
                self.logger.warning("⚠️ generate_audio called in async context, use synthesize_speech instead")
                return None
            except RuntimeError:
                # No running loop, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.synthesize_speech(text))
                finally:
                    loop.close()
        except Exception as e:
            self.logger.error(f"❌ generate_audio failed: {e}")
            return None 