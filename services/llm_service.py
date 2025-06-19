# services/llm_service.py

import os
import time
import json
import base64
import random
import re # For Markdown stripping
import cv2
import requests
import queue
import numpy as np
from collections import deque # For in-memory deque
from typing import Optional, Dict, List, Tuple, Any
import logging
import asyncio
import tempfile
import threading
from datetime import datetime, timedelta
# Simple cosine similarity implementation to avoid sklearn dependency
def cosine_similarity_simple(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    import math
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

from utils.text_utils import trim_sentences
from utils.ocr_utils import OCRProcessor
from core.game_profile import GameProfile
from .memory_service import MemoryEntry
# from .rag_service import RAGService  # Temporarily disabled due to dependency issues
from .model_client import ModelClient
from .danzar_factcheck import FactCheckService
from utils.error_logger import ErrorLogger
from collections import defaultdict

logger = logging.getLogger("DanzarVLM.LLMService")

class LLMService:
    def _calculate_next_commentary_delay(self) -> float:
        min_interval = float(self.ctx.global_settings.get("NDI_MIN_RANDOM_COMMENTARY_INTERVAL_S", 30.0))
        max_interval = float(self.ctx.global_settings.get("NDI_MAX_RANDOM_COMMENTARY_INTERVAL_S", 120.0))
        
        if min_interval >= max_interval:
            self.logger.warning(f"[LLMService] Min commentary interval ({min_interval}) was >= max ({max_interval}). Adjusted max to {min_interval + 10.0}.")
            max_interval = min_interval + 10.0 
            
        delay = random.uniform(min_interval, max_interval)
        return delay

    def __init__(self, app_context, audio_service, rag_service=None, model_client=None, default_collection: str = "multimodal_rag_default"):
        self.ctx = app_context
        self.audio_service = audio_service
        self.rag_service = rag_service  # Can be None temporarily
        self.model_client = model_client
        self.default_collection = default_collection
        self.logger = self.ctx.logger

        # Make OCR optional
        try:
            self.ocr_processor = OCRProcessor(app_context)
            if self.ocr_processor.ocr_enabled:
                self.logger.info("[LLMService] OCRProcessor initialized successfully.")
            else:
                self.logger.info("[LLMService] OCR is disabled - running without text detection.")
        except Exception as e:
            self.logger.warning(f"[LLMService] OCR initialization failed (continuing without OCR): {e}")
            self.ocr_processor = None

        self.last_vlm_time = 0.0 
        self.next_vlm_commentary_delay = self._calculate_next_commentary_delay() 
        self.logger.info(f"[LLMService] Initial VLM commentary delay set to: {self.next_vlm_commentary_delay:.2f}s")

        # Initialize memory service
        try:
            from .memory_service import MemoryService
            self.memory_service = MemoryService(app_context)
            self.logger.info("[LLMService] MemoryService initialized successfully.")
        except Exception as e:
            self.logger.error(f"[LLMService] Failed to initialize MemoryService: {e}", exc_info=True)
            self.memory_service = None

        # Initialize fact-checking service
        if self.rag_service:
            self.fact_check = FactCheckService(
                rag_service=self.rag_service,
                model_client=self.model_client
            )
        else:
            self.fact_check = None
            self.logger.warning("[LLMService] RAG service not available, fact-checking disabled")

        # Use Agentic Memory and ReAct Agent from app_context if available
        self.agentic_memory = getattr(app_context, 'agentic_memory_instance', None)
        self.react_agent = getattr(app_context, 'react_agent_instance', None)
        
        if self.agentic_memory and self.react_agent:
            self.logger.info("[LLMService] Using Agentic Memory and ReAct Agent from app_context")
        else:
            self.logger.info("[LLMService] Agentic Memory services not available in app_context")

        self.error_logger = ErrorLogger()

        # Initialize knowledge base cleanup system
        self.cleanup_enabled = self.ctx.global_settings.get("RAG_CLEANUP_ENABLED", True)
        self.cleanup_interval_hours = self.ctx.global_settings.get("RAG_CLEANUP_INTERVAL_HOURS", 24)
        self.cleanup_thread = None
        self.cleanup_running = False
        
        if self.cleanup_enabled and self.rag_service:
            self._start_cleanup_scheduler()
            self.logger.info(f"[LLMService] Knowledge base cleanup enabled (every {self.cleanup_interval_hours}h)")
        else:
            self.logger.info("[LLMService] Knowledge base cleanup disabled")

        self.logger.info("[LLMService] Initialized.")

    def _handle_profile_change_for_memory(self, new_profile: GameProfile):
        """Updates memory settings when profile changes."""
        if self.memory_service:
            self.memory_service._handle_profile_change(new_profile)

    def _strip_markdown_for_tts(self, text: str) -> str:
        """Remove Discord formatting for TTS playback"""
        # Remove Discord code blocks
        text = re.sub(r'```[a-zA-Z]*\n.*?\n```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove Discord formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        text = re.sub(r'__([^_]+)__', r'\1', text)      # Underline
        text = re.sub(r'~~([^~]+)~~', r'\1', text)      # Strikethrough
        
        # Remove URLs in markdown format
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        return text.strip()

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from LLM responses"""
        # Remove think tags and their content, keeping only what comes after
        # Use DOTALL flag to handle multi-line thinking content
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    def _call_llm_api(self, payload: dict, endpoint: str = "chat/completions") -> Optional[dict]:
        base = self.ctx.global_settings.get("LLAMA_API_BASE_URL")
        if not base:
            self.logger.error("[LLMService] LLAMA_API_BASE_URL missing from global_settings.yaml")
            return None
        url = f"{base.rstrip('/')}/{endpoint.lstrip('/')}"
        default_timeouts = {"connect": 10, "read": 180}
        llm_timeouts_config = self.ctx.global_settings.get("LLM_TIMEOUTS", default_timeouts)
        connect_timeout = llm_timeouts_config.get("connect", default_timeouts["connect"])
        read_timeout = llm_timeouts_config.get("read", default_timeouts["read"])
        timeout = (connect_timeout, read_timeout)
        
        try:
            debug_mode = self.ctx.global_settings.get("VLM_DEBUG_MODE", False)
            self.logger.debug(f"[LLMService] Calling LLM API: {url} with payload keys: {list(payload.keys())}")
            
            # Log image information if present for debugging
            if "images" in payload and payload["images"]:
                image_count = len(payload["images"])
                image_sizes = [len(img) for img in payload["images"]]
                self.logger.debug(f"[LLMService] Sending {image_count} image(s) with sizes: {image_sizes} bytes")
            
            # Log detailed messages content in debug mode
            if debug_mode and "messages" in payload:
                for i, msg in enumerate(payload["messages"]):
                    role = msg.get("role", "unknown")
                    
                    if isinstance(msg.get("content"), str):
                        content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                        self.logger.debug(f"[LLMService] Message[{i}] - Role: {role}, Content: {content_preview}")
                    elif isinstance(msg.get("content"), list):
                        self.logger.debug(f"[LLMService] Message[{i}] - Role: {role}, Content: list with {len(msg['content'])} items")
                        for j, content_item in enumerate(msg["content"]):
                            if isinstance(content_item, dict):
                                item_type = content_item.get("type", "unknown")
                                if item_type == "text":
                                    text_preview = content_item["text"][:100] + "..." if len(content_item["text"]) > 100 else content_item["text"]
                                    self.logger.debug(f"[LLMService] Message[{i}].Content[{j}] - Type: {item_type}, Text: {text_preview}")
                                elif item_type == "image" or "image" in item_type:
                                    self.logger.debug(f"[LLMService] Message[{i}].Content[{j}] - Type: {item_type}, Image data included")
            
            # Make the API call
            resp = requests.post(url, json=payload, timeout=timeout)
            
            # Log raw response in debug mode (limit size)
            if debug_mode:
                resp_text = resp.text[:500] + "..." if len(resp.text) > 500 else resp.text
                self.logger.debug(f"[LLMService] Raw API Response: {resp_text}")
                
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            self.logger.error(f"[LLMService] LLM API call timed out to {url}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[LLMService] LLM API request error: {e}", exc_info=True)
            if hasattr(e, 'response') and e.response:
                self.logger.error(f"[LLMService] API Error Response: {e.response.text[:500]}")
            return None
        except Exception as e:
            self.logger.error(f"[LLMService] LLM API general error: {e}", exc_info=True)
            return None

    def generate_vlm_commentary_from_frame(self, frame_bgr_np: np.ndarray):
        profile = self.ctx.active_profile
        gs = self.ctx.global_settings
        now = time.time()

        if now - self.last_vlm_time < self.next_vlm_commentary_delay:
            return

        if self.ctx.is_in_conversation.is_set():
            convo_to = float(gs.get("CONVERSATION_TIMEOUT_S", 45.0))
            if now - self.ctx.last_interaction_time < convo_to:
                self.logger.debug("[LLMService] Pausing VLM: active conversation.")
                return
            else:
                self.logger.info("[LLMService] Conversation timed out, VLM can resume.")
                self.ctx.is_in_conversation.clear()
                self.last_vlm_time = now 
                self.next_vlm_commentary_delay = self._calculate_next_commentary_delay()
                self.logger.debug(f"Reset VLM timer post-convo. Next: ~{self.next_vlm_commentary_delay:.1f}s.")
                return

        self.logger.info("[LLMService] Attempting VLM commentary...")
        
        # --- Image Encoding, OCR, RAG for current context ---
        # Detect if we're using a Qwen model
        model_name = profile.vlm_model.lower()
        is_qwen25vl = "qwen2.5-vl" in model_name.lower()
        
        # Check frame shape and dimensions
        h, w = frame_bgr_np.shape[:2]
        self.logger.info(f"[LLMService] Original image dimensions: {w}x{h}")
        
        # Initialize new dimensions
        new_w, new_h = w, h
        
        # Check if image is 4K or larger
        is_4k = (w >= 3840 or h >= 2160)
        if is_4k:
            self.logger.warning(f"[LLMService] Detected 4K image ({w}x{h}). Will aggressively reduce size.")
        
        # Get configured max image size
        configured_max_size = gs.get("VLM_MAX_IMAGE_SIZE", 512)  # Increased default to 512px
        max_dim = int(configured_max_size)
        self.logger.info(f"[LLMService] Using max image size: {max_dim}px")
        
        # Set JPEG quality - using higher quality settings
        jpeg_quality = 85 if is_4k else 95  # Much higher quality settings
        self.logger.info(f"[LLMService] Using JPEG quality: {jpeg_quality}")
            
        # Resize image if needed
        if w > max_dim or h > max_dim:
            scale = max_dim / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame_bgr_np = cv2.resize(frame_bgr_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)  # Using Lanczos for better quality
            self.logger.info(f"[LLMService] Resized image to {new_w}x{new_h}")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr_np, cv2.COLOR_BGR2RGB)
        
        # Base64 JPEG encoding with high quality settings
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,  # Enable JPEG optimization
            cv2.IMWRITE_JPEG_PROGRESSIVE, 1  # Enable progressive JPEG for better quality
        ]
        _, jpeg_buffer = cv2.imencode('.jpg', frame_rgb, encode_params)
        base64_jpeg = base64.b64encode(jpeg_buffer.tobytes()).decode('utf-8')
        
        # Log the size of the encoded image
        jpeg_size_kb = len(base64_jpeg) / 1024
        self.logger.info(f"[LLMService] Encoded JPEG size: {jpeg_size_kb:.1f}KB")
        
        # If the image is still too large, try to reduce it further but maintain high quality
        max_size_kb = gs.get("VLM_MAX_IMAGE_SIZE_KB", 1000)  # Increased to 1MB
        if jpeg_size_kb > max_size_kb:
            self.logger.warning(f"[LLMService] Image size ({jpeg_size_kb:.1f}KB) exceeds maximum ({max_size_kb}KB). Reducing further.")
            # Reduce dimensions by 25% instead of 50%
            new_w = int(new_w * 0.75)
            new_h = int(new_h * 0.75)
            frame_bgr_np = cv2.resize(frame_bgr_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            frame_rgb = cv2.cvtColor(frame_bgr_np, cv2.COLOR_BGR2RGB)
            # Try encoding again with slightly lower quality but still high
            jpeg_quality = max(80, jpeg_quality - 10)  # Reduce quality but not below 80
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                cv2.IMWRITE_JPEG_PROGRESSIVE, 1
            ]
            _, jpeg_buffer = cv2.imencode('.jpg', frame_rgb, encode_params)
            base64_jpeg = base64.b64encode(jpeg_buffer.tobytes()).decode('utf-8')
            jpeg_size_kb = len(base64_jpeg) / 1024
            self.logger.info(f"[LLMService] After reduction - size: {jpeg_size_kb:.1f}KB, quality: {jpeg_quality}")
        
        # Prepare the message content
        messages = []
        system_prompt = getattr(profile, 'system_prompt_commentary', None)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Create the user message with image
        base_instruction = getattr(profile, 'user_prompt_template_commentary', "Analyze this image from {game_name}.")
        if base_instruction:
            instruction_text = base_instruction.format(
                game_name=profile.game_name,
                ocr_text="OCR disabled" if not getattr(profile, 'ocr_enabled', True) else ""
            )
        else:
            instruction_text = f"Analyze this image from {profile.game_name}."

        # Prepare the API payload
        payload = {
            "model": profile.vlm_model,
            "messages": messages,
            "temperature": float(profile.vlm_temperature),
            "max_tokens": int(profile.vlm_max_tokens),
        }
        
        # Conditionally add images based on VLM_PROVIDER
        vlm_provider = gs.get("VLM_PROVIDER", "default").lower()
        if vlm_provider == "ollama":
            # Ollama expects images as a list of base64 strings in the 'images' field
            payload["images"] = [base64_jpeg]
            self.logger.debug(f"[LLMService] Sending payload formatted for Ollama VLM")
            resp = self.model_client.generate(
                messages=messages,
                temperature=float(profile.vlm_temperature),
                max_tokens=int(profile.vlm_max_tokens),
                model=profile.vlm_model,
                endpoint="chat/completions",
                images=[base64_jpeg]
            )
        else:
            # Default behavior for other providers (e.g., Qwen-VL, etc.)
            # This assumes the image is embedded in the prompt or handled differently
            # For Qwen2.5-VL, the image is embedded in the user message content
            image_content = f"USER: <img>{base64_jpeg}</img>\n\n{instruction_text}\n\nASSISTANT:"
            messages[-1]["content"] = image_content # Update the last user message
            payload["messages"] = messages # Ensure updated messages are in payload
            
            self.logger.debug(f"[LLMService] Sending payload formatted for {vlm_provider} model (non-Ollama)")
            # If using Ollama, resp is already the generated text, not a full response object
            if vlm_provider == "ollama":
                raw_commentary = resp
                if not raw_commentary:
                    self.logger.warning("Empty/invalid VLM response from Ollama.")
                    self.last_vlm_time = now; self.next_vlm_commentary_delay = self._calculate_next_commentary_delay(); return
            else:
                if not resp or "choices" not in resp or not resp["choices"]:
                    self.logger.warning("Empty/invalid VLM response from non-Ollama provider.")
                    self.last_vlm_time = now; self.next_vlm_commentary_delay = self._calculate_next_commentary_delay(); return
                raw_commentary = resp["choices"][0].get("message",{}).get("content","").strip()

        if not resp or "choices" not in resp or not resp["choices"]:
            self.logger.warning("Empty/invalid VLM response.")
            self.last_vlm_time = now; self.next_vlm_commentary_delay = self._calculate_next_commentary_delay(); return
        
        raw_commentary = resp["choices"][0].get("message",{}).get("content","").strip()
        if not raw_commentary:
            self.logger.warning("VLM response content empty.")
            self.last_vlm_time = now; self.next_vlm_commentary_delay = self._calculate_next_commentary_delay(); return

        trimmed_commentary = trim_sentences(raw_commentary, profile.vlm_max_commentary_sentences)
        
        # Clean text for TTS
        text_for_tts_and_discord = self._strip_markdown_for_tts(trimmed_commentary)
        text_for_tts_and_discord = re.sub(r'[*#]', '', text_for_tts_and_discord)  # Remove asterisks and hashes

        if not text_for_tts_and_discord:
            self.logger.info("VLM commentary empty after processing.")
            self.last_vlm_time = now; self.next_vlm_commentary_delay = self._calculate_next_commentary_delay(); return

        # Store in memory service
        if self.memory_service:
            memory_entry = MemoryEntry(
                content=text_for_tts_and_discord,
                source="vlm_commentary",
                timestamp=now,
                metadata={
                    "game": profile.game_name,
                    "ocr_enabled": getattr(profile, 'ocr_enabled', True),
                    "ocr_context": "OCR enabled" if getattr(profile, 'ocr_enabled', True) else "OCR disabled",
                    "type": "commentary"
                }
            )
            self.memory_service.store_memory(memory_entry)

        # Queue for Discord
        discord_msg = f"🎙️ **{profile.game_name} Tip:** {text_for_tts_and_discord}"
        try:
            self.ctx.text_message_queue.put_nowait(discord_msg)
            if self.audio_service and gs.get("ENABLE_TTS_FOR_VLM_COMMENTARY", True):
                tts_audio = self.ctx.tts_service_instance.generate_audio(text_for_tts_and_discord) if self.ctx.tts_service_instance else None
                if tts_audio:
                    # Prevent queue flooding - clear old items if queue is too full
                    max_queue_size = 3  # Allow max 3 TTS items in queue
                    while self.ctx.tts_queue.qsize() >= max_queue_size:
                        try:
                            old_audio = self.ctx.tts_queue.get_nowait()
                            self.ctx.tts_queue.task_done()
                            self.logger.debug(f"[LLMService] Dropped old TTS audio to prevent queue flooding (size: {self.ctx.tts_queue.qsize()})")
                        except queue.Empty:
                            break
                    self.ctx.tts_queue.put_nowait(tts_audio)
        except queue.Full:
            self.logger.warning("Queues full, VLM commentary dropped.")
        
        self.last_vlm_time = now
        self.next_vlm_commentary_delay = self._calculate_next_commentary_delay()
        self.logger.info(f"VLM Commentary: \"{text_for_tts_and_discord}\". Next in ~{self.next_vlm_commentary_delay:.1f}s.")

    def run_vlm_commentary_loop(self):
        # ... (same as your last correct version, ensures latest frame is processed) ...
        self.logger.info("[LLMService] Starting VLM commentary loop.")
        time.sleep(self.ctx.global_settings.get("VLM_LOOP_STARTUP_DELAY_S", 3.0))
        while not self.ctx.shutdown_event.is_set():
            if not self.ctx.ndi_commentary_enabled.is_set(): time.sleep(1); continue
            latest_frame_np = None
            try:
                discarded_count = 0; q_size_before_get = self.ctx.frame_queue.qsize()
                while True:
                    try: temp_frame = self.ctx.frame_queue.get_nowait(); latest_frame_np = temp_frame; self.ctx.frame_queue.task_done()
                    except queue.Empty: break
                if latest_frame_np is not None:
                    if isinstance(latest_frame_np, np.ndarray): self.generate_vlm_commentary_from_frame(latest_frame_np)
                    else: self.logger.warning(f"Non-NumPy frame: {type(latest_frame_np)}")
            except Exception as e: self.logger.error(f"VLM loop error: {e}", exc_info=True)
            time.sleep(self.ctx.global_settings.get("VLM_LOOP_CHECK_INTERVAL_S", 1.0))
        self.logger.info("[LLMService] VLM commentary loop stopped.")

    def _is_conversational(self, text: str) -> bool:
        """Check if text is a basic conversation pattern"""
        conversational_patterns = [
            r'^(hi|hello|hey|greetings|howdy)(\s|$)',
            r'^(how are you|how\'s it going|what\'s up|whats up)(\?|\s|$)',
            r'^(good|nice)\s*(morning|afternoon|evening|night)(\s|$)',
            r'^(bye|goodbye|see you|farewell)(\s|$)',
            r'^(thanks|thank you|ty)(\s|$)'
        ]
        text = text.lower().strip()
        return any(re.match(pattern, text) for pattern in conversational_patterns)
    
    def _has_recent_conversation_context(self, user_name: str) -> bool:
        """Check if the user has recent conversation context that might be relevant"""
        if not self.memory_service:
            return False
        
        try:
            recent_memories = self.memory_service.get_relevant_memories(
                query=f"user:{user_name}",
                top_k=3,
                min_importance=0.1  # Lower threshold for conversation context
            )
            return len(recent_memories) > 0
        except Exception as e:
            self.logger.debug(f"[LLMService] Error checking conversation context: {e}")
            return False

    def _is_search_research_request(self, text: str) -> bool:
        """Check if the user is explicitly requesting a search or research operation"""
        text_lower = text.lower().strip()
        
        # Search triggers
        search_triggers = [
            "search", "search for", "search about", "search up", "search on",
            "can you search", "please search", "search the web",
            "look up", "look for", "find information about",
            "find out about", "google", "bing"
        ]
        
        # Research triggers
        research_triggers = [
            "research", "investigate", "find more information",
            "tell me more about", "learn about", "study",
            "explore", "examine", "analyze"
        ]
        
        # Check for explicit search/research requests
        for trigger in search_triggers + research_triggers:
            if trigger in text_lower:
                return True
        
        # Check for question patterns that suggest web search is needed
        search_question_patterns = [
            "what is the latest", "what's new with", "current status of",
            "recent news about", "updates on", "what happened to",
            "when did", "who invented", "how was", "where can i find"
        ]
        
        for pattern in search_question_patterns:
            if pattern in text_lower:
                return True
        
        return False

    def _is_crawl_website_request(self, text: str) -> bool:
        """Check if the user is requesting to crawl a website"""
        crawl_triggers = [
            "crawl", "scrape", "extract from", "download from",
            "get all pages", "crawl website", "scrape website",
            "fetch all content", "index website", "crawl site",
            "exhaustive crawl", "full crawl", "deep crawl"
        ]
        
        text_lower = text.lower().strip()
        
        # Check for crawl triggers
        for trigger in crawl_triggers:
            if trigger in text_lower:
                return True
        
        # Check if there's a URL and crawl-like language
        import re
        url_pattern = r'(https?://[^\s]+)'
        if re.search(url_pattern, text):
            crawl_patterns = [
                r"\ball\s+(?:pages|content|information)",
                r"\bevery\s+page",
                r"\bentire\s+(?:site|website)",
                r"\bfull\s+(?:site|website)",
                r"\bget\s+everything"
            ]
            for pattern in crawl_patterns:
                if re.search(pattern, text_lower):
                    return True
        
        return False

    async def _handle_search_research_request(self, query: str, user_name: str) -> Optional[str]:
        """Handle search/research requests using web search capabilities"""
        try:
            self.logger.info(f"[LLMService] Processing search/research request: '{query}'")
            
            # Clean the query for web search
            search_query = self._extract_search_query(query)
            self.logger.info(f"[LLMService] Extracted search query: '{search_query}'")
            
            # Check if we have VERY recent search results in RAG (within last 10 minutes)
            existing_results = await self._check_existing_search_results(search_query)
            if existing_results and self._is_search_result_recent(search_query, max_age_minutes=10):
                self.logger.info(f"[LLMService] Found very recent search results for '{search_query}', using cached data")
                response = await self._synthesize_search_response(query, existing_results, user_name)
                if response:
                    return f"Based on recent search data: {response}"
            elif existing_results:
                self.logger.info(f"[LLMService] Found older search results for '{search_query}', performing fresh search")
            
            # Use fact-check service for web search if available
            if self.fact_check:
                search_results = self.fact_check._search_web(search_query, fact_check=True)
                
                if search_results:
                    # Store search results in RAG for future reference
                    await self._store_search_results_in_rag(search_query, search_results, user_name)
                    
                    # Generate a natural response using the search results
                    response = await self._synthesize_search_response(query, search_results, user_name)
                    if response:
                        self.logger.info(f"[LLMService] Search response generated successfully")
                        return response
                else:
                    self.logger.warning(f"[LLMService] No search results found for: '{search_query}'")
            
            # Fallback to AgenticRAG if fact-check service is not available
            if hasattr(self, 'agentic_rag') and self.agentic_rag:
                self.logger.info("[LLMService] Falling back to AgenticRAG for web search")
                response_text, metadata = await self.agentic_rag.smart_retrieve(query, user_name)
                
                if response_text and response_text.strip():
                    # Store AgenticRAG results as well
                    await self._store_search_results_in_rag(search_query, response_text, user_name)
                    return response_text
            
            # Final fallback
            return "I'm unable to search the web right now, but I can try to help with information I already know. What would you like to know?"
            
        except Exception as e:
            self.logger.error(f"[LLMService] Search/research request failed: {e}", exc_info=True)
            return None

    async def _determine_target_collection(self, search_query: str, user_name: str) -> str:
        """Determine the appropriate collection for storing search results based on context"""
        try:
            # First, try to detect the game/topic from the search query itself
            context_prompt = f"""Analyze this search query and determine what game, software, or topic category it relates to.

Search query: "{search_query}"

Common games/topics include:
- EverQuest (everquest)
- World of Warcraft (worldofwarcraft) 
- Rimworld (rimworld)
- Minecraft (minecraft)
- Programming/coding (programming)
- General technology (technology)
- Gaming news (gaming)
- General topics (general)

Rules:
1. If the query clearly mentions a specific game name, use that game's identifier
2. If it's about game mechanics without specifying a game, consider the context
3. If it's about programming, use "programming"
4. If it's general tech/software, use "technology"
5. If unclear, use "general"

Respond with ONLY the category identifier (lowercase, no spaces, use underscores). Examples:
- "What's new in EverQuest?" -> "everquest"
- "Rimworld base building tips" -> "rimworld"
- "Python programming help" -> "programming"
- "Latest gaming news" -> "gaming"
- "Weather forecast" -> "general"

Category identifier:"""

            messages = [
                {"role": "system", "content": "You are a categorization assistant. Respond with only the category identifier."},
                {"role": "user", "content": context_prompt}
            ]
            
            if self.model_client:
                response = await self.model_client.generate(
                    messages=messages,
                    temperature=0.1,  # Very low temperature for consistent categorization
                    max_tokens=20,    # Short response expected
                    model=self.ctx.active_profile.conversational_llm_model
                )
                
                if response:
                    detected_category = response.strip().lower()
                    # Validate the response
                    valid_categories = [
                        "everquest", "worldofwarcraft", "rimworld", "minecraft", 
                        "programming", "technology", "gaming", "general"
                    ]
                    
                    if detected_category in valid_categories:
                        collection_name = f"{detected_category}_search_results"
                        self.logger.info(f"[LLMService] LLM detected category '{detected_category}' for query: '{search_query}'")
                        return collection_name
            
            # Fallback: Try keyword-based detection
            query_lower = search_query.lower()
            
            # Game-specific keywords
            game_keywords = {
                "everquest": ["everquest", "eq", "norrath", "velious", "kunark"],
                "worldofwarcraft": ["wow", "world of warcraft", "azeroth", "horde", "alliance"],
                "rimworld": ["rimworld", "colony", "rimworld mods", "colony sim"],
                "minecraft": ["minecraft", "creeper", "redstone", "villager"],
            }
            
            for game, keywords in game_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    collection_name = f"{game}_search_results"
                    self.logger.info(f"[LLMService] Keyword detection found '{game}' for query: '{search_query}'")
                    return collection_name
            
            # Topic-specific keywords
            if any(word in query_lower for word in ["python", "code", "programming", "javascript", "html", "css"]):
                return "programming_search_results"
            elif any(word in query_lower for word in ["software", "tech", "computer", "hardware"]):
                return "technology_search_results"
            elif any(word in query_lower for word in ["game", "gaming", "esports", "streamer"]):
                return "gaming_search_results"
            
            # Final fallback: use active profile if it seems reasonable
            profile = self.ctx.active_profile
            if profile and profile.game_name:
                profile_collection = f"{profile.game_name.lower()}_search_results"
                self.logger.info(f"[LLMService] Using active profile '{profile.game_name}' as fallback for query: '{search_query}'")
                return profile_collection
            
            # Ultimate fallback
            self.logger.info(f"[LLMService] Using general collection as final fallback for query: '{search_query}'")
            return "general_search_results"
            
        except Exception as e:
            self.logger.error(f"[LLMService] Error determining target collection: {e}", exc_info=True)
            # Safe fallback
            return "general_search_results"

    async def _store_search_results_in_rag(self, search_query: str, search_results: str, user_name: str):
        """Store search results in RAG database for future reference"""
        try:
            # Check if RAG service is available
            if not self.rag_service:
                self.logger.warning("[LLMService] RAG service not available, skipping search result storage")
                return
            
            # Use intelligent collection determination instead of just active profile
            search_collection = await self._determine_target_collection(search_query, user_name)
            
            # Prepare metadata for the search results
            timestamp = time.time()
            metadata = {
                "search_query": search_query,
                "user_name": user_name,
                "timestamp": timestamp,
                "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
                "source": "web_search",
                "search_type": "research" if "research" in search_query.lower() else "search",
                "collection_reason": "llm_determined"  # Track how collection was chosen
            }
            
            # Generate a concise summary for better RAG embeddings
            summary = await self._generate_search_summary(search_query, search_results)
            
            # Create a comprehensive text entry for RAG storage
            rag_text = f"""SEARCH QUERY: {search_query}
SEARCHED BY: {user_name}
DATE: {metadata['date']}
TYPE: {metadata['search_type']}
COLLECTION: {search_collection}

SEARCH RESULTS:
{search_results}

SUMMARY: {summary if summary else f'Web search results for "{search_query}" containing current information and multiple source verification.'}"""
            
            # Store in RAG database
            success = self.rag_service.ingest_text(
                collection=search_collection,
                text=rag_text,
                metadata=metadata
            )
            
            if success:
                self.logger.info(f"[LLMService] Successfully stored search results for '{search_query}' in collection '{search_collection}'")
            else:
                self.logger.warning(f"[LLMService] Failed to store search results for '{search_query}' in RAG")
                
        except Exception as e:
            self.logger.error(f"[LLMService] Error storing search results in RAG: {e}", exc_info=True)

    async def _generate_search_summary(self, search_query: str, search_results: str) -> Optional[str]:
        """Generate a concise summary of search results for better RAG embeddings"""
        try:
            if not self.model_client or len(search_results) < 100:
                return None
                
            summary_prompt = f"""Summarize the following web search results in 2-3 clear, informative sentences that capture the key information.

Search Query: {search_query}

Search Results:
{search_results[:1500]}

Provide a concise summary that:
1. Answers the original query if possible
2. Includes the most important facts and details
3. Is easy to understand and retrieve later

Summary:"""

            messages = [
                {"role": "system", "content": "You are a helpful assistant that creates concise, informative summaries."},
                {"role": "user", "content": summary_prompt}
            ]
            
            summary = await self.model_client.generate(
                messages=messages,
                temperature=0.3,  # Low temperature for consistent summaries
                max_tokens=150,   # Keep summaries concise
                model=self.ctx.active_profile.conversational_llm_model
            )
            
            if summary and len(summary.strip()) > 20:
                self.logger.info(f"[LLMService] Generated summary for search results: {len(summary)} chars")
                return summary.strip()
            
            return None
            
        except Exception as e:
            self.logger.warning(f"[LLMService] Failed to generate search summary: {e}")
            return None

    def _is_search_result_recent(self, search_query: str, max_age_minutes: int = 10) -> bool:
        """Check if search results for a query are recent enough to reuse"""
        try:
            if not self.rag_service:
                return False
            
            # Check timestamp of most recent search result for this query
            current_time = time.time()
            max_age_seconds = max_age_minutes * 60
            
            # This is a simple check - in a real implementation you'd query the RAG database
            # For now, we'll be conservative and always perform fresh searches
            return False
            
        except Exception as e:
            self.logger.error(f"[LLMService] Error checking search result age: {e}")
            return False

    async def _check_existing_search_results(self, search_query: str) -> Optional[str]:
        """Check if we have recent search results for this query in RAG"""
        try:
            if not self.rag_service:
                return None
            
            # Use intelligent collection determination for checking existing results too
            search_collection = await self._determine_target_collection(search_query, "system")
            
            # Check if the collection exists
            if not self.rag_service.collection_exists(search_collection):
                self.logger.debug(f"[LLMService] Search results collection '{search_collection}' does not exist")
                return None
            
            # Search for similar queries in our stored search results
            # Look for results from the last 24 hours that are similar to this query
            search_results = self.rag_service.query(
                collection=search_collection,
                query_text=search_query,
                n_results=3  # Check top 3 most similar searches
            )
            
            if search_results:
                # Check if any result is recent enough (within 24 hours) and similar enough
                current_time = time.time()
                for result in search_results:
                    # Extract the original search results from the stored RAG text
                    if "SEARCH RESULTS:" in result:
                        result_content = result.split("SEARCH RESULTS:")[1].split("SUMMARY:")[0].strip()
                        
                        # Check if this is recent enough (we can make this configurable)
                        # For now, use cached results if they exist and have good similarity
                        if "[Score:" in result:
                            score_str = result.split("[Score:")[1].split("]")[0]
                            try:
                                score = float(score_str)
                                if score > 0.85:  # High similarity threshold for cached results
                                    self.logger.info(f"[LLMService] Found highly similar search result (score: {score:.3f}), using cached data from '{search_collection}'")
                                    return result_content
                            except:
                                pass
                        
                        # Fallback for results without score
                        self.logger.info(f"[LLMService] Found similar search result, using cached data from '{search_collection}'")
                        return result_content
            
            return None
            
        except Exception as e:
            self.logger.error(f"[LLMService] Error checking existing search results: {e}", exc_info=True)
            return None

    def _extract_search_query(self, query: str) -> str:
        """Extract the actual search terms from a user query with intelligent parsing"""
        query_lower = query.lower().strip()
        
        # First, check for natural language patterns that indicate what to search for
        smart_patterns = [
            # "search for X classes in everquest" -> "everquest classes" 
            (r"search.*?for\s+(.*?)\s+(?:in|for|about)\s+(.+)", r"\2 \1"),
            # "look up what classes are available for everquest" -> "everquest classes available"
            (r"look\s+up\s+what\s+(.*?)\s+(?:are\s+)?available\s+(?:for|in)\s+(.+)", r"\2 \1 available"),
            # "what classes are available for everquest" -> "everquest classes available"  
            (r"what\s+(.*?)\s+are\s+available\s+(?:for|in)\s+(.+)", r"\2 \1 available"),
            # "and see what X are available for Y" -> "Y X available" 
            (r"(?:and\s+)?see\s+what\s+(.*?)\s+are\s+available\s+(?:for|in)\s+(.+)", r"\2 \1 available"),
            # "find information about X in Y" -> "Y X information"
            (r"find\s+information\s+about\s+(.*?)\s+(?:in|for)\s+(.+)", r"\2 \1 information"),
            # "tell me about X in Y" -> "Y X"
            (r"tell\s+me\s+about\s+(.*?)\s+(?:in|for)\s+(.+)", r"\2 \1"),
            # Generic "what X for Y" -> "Y X"
            (r"what\s+(.*?)\s+(?:for|in)\s+(.+)", r"\2 \1"),
        ]
        
        # Try smart pattern matching first
        for pattern, replacement in smart_patterns:
            match = re.search(pattern, query_lower)
            if match:
                # Extract the matched groups and format them
                try:
                    smart_query = re.sub(pattern, replacement, query_lower).strip()
                    # Clean up extra spaces
                    smart_query = re.sub(r'\s+', ' ', smart_query)
                    self.logger.info(f"[LLMService] Smart parsing: '{query}' -> '{smart_query}'")
                    return smart_query
                except Exception as e:
                    self.logger.warning(f"[LLMService] Smart parsing failed: {e}, falling back to basic extraction")
                    break
        
        # Fallback to basic prefix removal
        prefixes_to_remove = [
            "search for", "search about", "search up", "search on",
            "can you search", "please search", "search the web for",
            "look up", "look for", "find information about",
            "find out about", "research", "investigate",
            "tell me more about", "learn about", "google",
            "bing", "what is the latest on", "what's new with",
            "and see what", "see what"  # Added to handle "and see what classes..."
        ]
        
        cleaned_query = query
        for prefix in prefixes_to_remove:
            if query_lower.startswith(prefix):
                cleaned_query = query[len(prefix):].strip()
                break
        
        # Additional cleanup for common query patterns
        cleaned_query = re.sub(r'^(and\s+)?', '', cleaned_query, flags=re.IGNORECASE)  # Remove leading "and"
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()  # Normalize spaces
        
        if cleaned_query != query:
            self.logger.info(f"[LLMService] Basic extraction: '{query}' -> '{cleaned_query}'")
        
        return cleaned_query

    async def _synthesize_search_response(self, original_query: str, search_results: str, user_name: str) -> Optional[str]:
        """Synthesize a natural response from search results"""
        try:
            system_prompt = f"""You are a helpful AI assistant responding to a search/research request.

The user asked: "{original_query}"

You have found the following information from web search:
{search_results}

Task: Provide a clear, informative response that directly addresses the user's query using the search results. Be concise but comprehensive. If the search results contain conflicting information, mention this. Always cite that the information comes from web search.

Response style: Natural, conversational, and helpful."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Based on the search results, please answer: {original_query}"}
            ]
            
            # Use model client to generate response
            if self.model_client:
                response = await self.model_client.generate(
                    messages=messages,
                    temperature=0.3,  # Lower temperature for factual responses
                    max_tokens=400,   # Reasonable length for search responses
                    model=self.ctx.active_profile.conversational_llm_model
                )
                
                if response:
                    return response
            
            # Fallback if model client fails
            return f"Based on my web search: {search_results[:500]}..."
            
        except Exception as e:
            self.logger.error(f"[LLMService] Failed to synthesize search response: {e}", exc_info=True)
            return None

    def handle_user_text_query_sync(self, user_text: str, user_name: str = "User"):
        """Synchronous wrapper for handle_user_text_query - runs the async method in a thread"""
        import asyncio
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        try:
            # Try to get the existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, need to run in thread
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.handle_user_text_query(user_text, user_name))
                    return future.result()
            else:
                # No running loop, safe to use asyncio.run
                return asyncio.run(self.handle_user_text_query(user_text, user_name))
        except RuntimeError:
            # Fallback: run in new thread with new event loop
            result = None
            exception = None
            
            def run_async():
                nonlocal result, exception
                try:
                    result = asyncio.run(self.handle_user_text_query(user_text, user_name))
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_async)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result

    async def handle_user_text_query(self, user_text: str, user_name: str = "User"):
        """
        Handle user text query with comprehensive LLM processing including intent classification,
        RAG search, web search, and fallback responses.
        """
        self.logger.info(f"[LLMService] Processing text query from {user_name}: '{user_text[:100]}...'")
        
        try:
            # Check if Qwen2.5-Omni service is available and enabled
            if hasattr(self.ctx, 'qwen_omni_service') and self.ctx.qwen_omni_service:
                try:
                    self.logger.info("[LLMService] Using Qwen2.5-Omni for response generation")
                    response = await self.ctx.qwen_omni_service.generate_response(user_text)
                    
                    if response and len(response.strip()) > 0:
                        # Store response in memory if available
                        if self.memory_service:
                            try:
                                # Use the correct method name for storing interactions
                                if hasattr(self.memory_service, 'store_interaction'):
                                    self.memory_service.store_interaction(user_name, user_text, response)
                                elif hasattr(self.memory_service, 'add_memory'):
                                    self.memory_service.add_memory(user_text, response, user_name)
                            except Exception as e:
                                self.logger.warning(f"[LLMService] Could not store interaction in memory: {e}")
                        
                        return response
                    else:
                        self.logger.warning("[LLMService] Qwen2.5-Omni returned empty response, falling back to standard processing")
                except Exception as e:
                    self.logger.error(f"[LLMService] Qwen2.5-Omni error: {e}, falling back to standard processing")
            
            # Check if tool-aware LLM is enabled
            use_tool_aware = self.ctx.global_settings.get('USE_TOOL_AWARE_LLM', False)
            
            if use_tool_aware and self.model_client:
                self.logger.info("[LLMService] Using tool-aware LLM processing")
                return await self._handle_tool_aware_query(user_text, user_name)
            
            # Standard processing - improved RAG-based response with multiple collections
            if self.rag_service and self.model_client:
                try:
                    self.logger.info("[LLMService] Using standard RAG processing")
                    
                    # Determine which collections to search based on query content
                    collections_to_search = ["danzar_knowledge"]
                    
                    # Add game-specific collections based on keywords
                    text_lower = user_text.lower()
                    if any(keyword in text_lower for keyword in ["everquest", "eq", "norrath", "kunark", "velious"]):
                        collections_to_search.insert(0, "everquest")  # Search everquest first
                    elif any(keyword in text_lower for keyword in ["wow", "world of warcraft", "azeroth"]):
                        collections_to_search.insert(0, "wow")
                    elif any(keyword in text_lower for keyword in ["rimworld", "rim world", "colony"]):
                        collections_to_search.insert(0, "rimworld")
                    
                    docs = None
                    collection_used = None
                    
                    # Try each collection until we find results
                    for collection in collections_to_search:
                        try:
                            docs = self.rag_service.query(
                                collection=collection,
                                query_text=user_text,
                                n_results=3
                            )
                            if docs:
                                collection_used = collection
                                self.logger.info(f"[LLMService] Found {len(docs)} results in collection '{collection}'")
                                break
                        except Exception as e:
                            self.logger.warning(f"[LLMService] Failed to query collection '{collection}': {e}")
                            continue
                    
                    if docs:
                        # Extract text content from dictionary results
                        context_texts = []
                        for doc in docs[:2]:  # Use top 2 results
                            if isinstance(doc, dict):
                                text_content = doc.get("text", "") or doc.get("content", "") or str(doc)
                                context_texts.append(text_content)
                            else:
                                context_texts.append(str(doc))
                        
                        context = "\n".join(context_texts)
                        system_prompt = (
                            f"You are Danzar, an upbeat gaming assistant. Answer based on the context provided from the {collection_used} knowledge base. "
                            "Keep responses conversational and helpful."
                        )
                        
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_text}"}
                        ]
                        
                        response = await self.model_client.generate(
                            messages=messages,
                            temperature=0.7,
                            max_tokens=200,
                            model=self.ctx.active_profile.conversational_llm_model
                        )
                        
                        if response and response.strip():
                            return response.strip()
                    
                    # No RAG results found - try web search if available
                    self.logger.info("[LLMService] No RAG results found, attempting web search")
                    
                    # Check if we have a web search service available
                    web_search_result = None
                    if hasattr(self.ctx, 'fact_check_service') and self.ctx.fact_check_service:
                        try:
                            # Use the existing fact check service for web search
                            import asyncio
                            web_search_result = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.ctx.fact_check_service._search_web(user_text, fact_check=False)
                            )
                            
                            if web_search_result and len(web_search_result.strip()) > 50:
                                self.logger.info(f"[LLMService] Web search found {len(web_search_result)} chars of results")
                                
                                # Store search results in RAG for future learning
                                try:
                                    await self._store_search_results_in_rag(user_text, web_search_result, user_name)
                                    self.logger.info(f"[LLMService] Stored web search results in RAG for future learning")
                                except Exception as e:
                                    self.logger.warning(f"[LLMService] Failed to store search results in RAG: {e}")
                                
                                # Generate response based on web search results
                                web_system_prompt = (
                                    "You are Danzar, an upbeat gaming assistant. Answer the user's question based on the web search results provided. "
                                    "Keep responses conversational and helpful. If the web results don't fully answer the question, say so."
                                )
                                
                                web_messages = [
                                    {"role": "system", "content": web_system_prompt},
                                    {"role": "user", "content": f"Web search results: {web_search_result[:1000]}\n\nQuestion: {user_text}"}
                                ]
                                
                                web_response = await self.model_client.generate(
                                    messages=web_messages,
                                    temperature=0.7,
                                    max_tokens=250,
                                    model=self.ctx.active_profile.conversational_llm_model
                                )
                                
                                if web_response and web_response.strip():
                                    return f"🌐 {web_response.strip()}"
                                    
                        except Exception as e:
                            self.logger.warning(f"[LLMService] Web search failed: {e}")
                    
                    # Final fallback if no RAG results and no web search
                    fallback_messages = [
                        {"role": "system", "content": "You are Danzar, an upbeat gaming assistant. Be helpful and encouraging. Admit when you don't know something and suggest the user could search for more information."},
                        {"role": "user", "content": user_text}
                    ]
                    
                    response = await self.model_client.generate(
                        messages=fallback_messages,
                        temperature=0.7,
                        max_tokens=200,
                        model=self.ctx.active_profile.conversational_llm_model
                    )
                    
                    return response.strip() if response else "I'm here to help! What would you like to know about gaming?"
                    
                except Exception as e:
                    self.logger.error(f"[LLMService] Error in standard processing: {e}")
                    return "I'm having trouble processing that right now. Please try again!"
            
            # Final fallback
            return "I'm processing your request..."
            
        except Exception as e:
            self.logger.error(f"[LLMService] Error in handle_user_text_query: {e}", exc_info=True)
            return "I encountered an error processing your request. Please try again."

    async def _handle_tool_aware_query(self, user_text: str, user_name: str) -> str:
        """
        Handle queries using tool-aware LLM processing.
        The LLM decides when to use RAG, search, or respond directly.
        """
        try:
            self.logger.info(f"[LLMService] Tool-aware processing for: '{user_text[:50]}...'")
            
            # Create a tool-aware system prompt
            system_prompt = """You are Danzar, an upbeat gaming assistant with access to tools. You can:

1. Search your knowledge base for game information
2. Provide direct responses for greetings and simple questions
3. Ask for clarification when needed

For gaming questions, search your knowledge base. For greetings like "hello" or "hi", respond directly without searching.

Respond naturally and conversationally."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
            
            # Check if this is a simple greeting that doesn't need RAG
            greeting_patterns = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
            is_simple_greeting = any(pattern in user_text.lower() for pattern in greeting_patterns) and len(user_text.split()) <= 3
            
            if is_simple_greeting:
                self.logger.info("[LLMService] Detected simple greeting, responding directly")
                direct_response = await self.model_client.generate(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=100,
                    model=self.ctx.active_profile.conversational_llm_model
                )
                return direct_response.strip() if direct_response else "Hello! I'm Danzar, your gaming assistant. How can I help you today?"
            
            # For other queries, use RAG if available
            if self.rag_service:
                try:
                    docs = self.rag_service.query(
                        collection="danzar_knowledge",
                        query_text=user_text,
                        n_results=5
                    )
                    
                    if docs:
                        # Extract text content from dictionary results
                        context_texts = []
                        for doc in docs[:3]:  # Use top 3 results
                            if isinstance(doc, dict):
                                text_content = doc.get("text", "") or doc.get("content", "") or str(doc)
                                context_texts.append(text_content)
                            else:
                                context_texts.append(str(doc))
                        
                        context = "\n".join(context_texts)
                        rag_messages = [
                            {"role": "system", "content": system_prompt + f"\n\nKnowledge base context:\n{context}"},
                            {"role": "user", "content": user_text}
                        ]
                        
                        response = await self.model_client.generate(
                            messages=rag_messages,
                            temperature=0.7,
                            max_tokens=300,
                            model=self.ctx.active_profile.conversational_llm_model
                        )
                        
                        if response and response.strip():
                            return response.strip()
                
                except Exception as e:
                    self.logger.error(f"[LLMService] RAG error in tool-aware processing: {e}")
            
            # Fallback to direct response
            response = await self.model_client.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=200,
                model=self.ctx.active_profile.conversational_llm_model
            )
            
            return response.strip() if response else "I'm here to help with your gaming questions!"
            
        except Exception as e:
            self.logger.error(f"[LLMService] Error in tool-aware processing: {e}")
            return "I'm having trouble processing that right now. Please try again!"

    def _store_and_send_response(self, response_text: str, user_text: str, user_name: str):
        """Persist response + context and dispatch it (text + optional TTS). Returns a value only if caller still needs to send."""
        
        # Strip think tags from response before processing
        original_response = response_text
        response_text = self._strip_think_tags(response_text)
        
        # Log the thinking process separately (terminal only)
        if original_response != response_text:
            think_content = re.search(r'<think>(.*?)</think>', original_response, re.DOTALL | re.IGNORECASE)
            if think_content:
                self.logger.info(f"[LLMService] Model thinking process: {think_content.group(1).strip()}")
        
        # If response is empty after stripping think tags, provide fallback
        if not response_text.strip():
            response_text = "I'm thinking about that... let me get back to you."
            self.logger.warning("[LLMService] Response was empty after stripping think tags, using fallback")
        
        # Store in memory
        if self.memory_service:
            user_memory = MemoryEntry(
                content=f"User ({user_name}): {user_text}",
                source="user_query",
                timestamp=time.time(),
                metadata={"user": user_name, "game": self.ctx.active_profile.game_name, "type": "user_input"}
            )
            self.memory_service.store_memory(user_memory)
            
            bot_memory = MemoryEntry(
                content=f"AI ({self.ctx.global_settings.get('BOT_NAME', 'DanzarVLM')}): {response_text}",
                source="bot_response",
                timestamp=time.time(),
                metadata={
                    "user_query": user_text,
                    "user": user_name,
                    "game": self.ctx.active_profile.game_name,
                    "type": "bot_response",
                    "llm_model": self.ctx.active_profile.conversational_llm_model
                }
            )
            self.memory_service.store_memory(bot_memory)

        # Log response generation (Discord will handle actual sending)
        self.logger.info(f"Response generated (streamed): \"{response_text[:100]}...\"")
        
        # Note: Discord bot will handle text sending and TTS playback
        # No need to send response internally since we're returning it

        self.ctx.last_interaction_time = time.time()
        if self.ctx.global_settings.get("CLEAR_CONVERSATION_FLAG_AFTER_REPLY", True):
            self.ctx.is_in_conversation.clear()

        # Return the response text for Discord bot to handle
        # Discord bot will handle its own text sending and TTS
        return response_text

    def _send_streaming_response(self, response_text: str, user_text: str, user_name: str):
        """Send response using sentence streaming for better TTS processing"""
        streaming_service = self.ctx.streaming_response_instance
        streaming_config = self.ctx.global_settings.get("STREAMING_RESPONSE", {})
        
        # Check if using new sequential processing
        use_sequential_processing = streaming_config.get("use_sequential_processing", True)
        
        if use_sequential_processing:
            # NEW SEQUENTIAL SYSTEM: Let Discord bot handle streaming via sequential processing
            # Don't start streaming here - Discord bot will handle it
            self.logger.debug(f"[LLMService] Using sequential processing - Discord bot will handle streaming")
            return
        
        # OLD PARALLEL SYSTEM: (disabled when sequential processing is enabled)
        # Create callbacks for streaming
        tts_callback = None
        text_callback = None
        
        # Disable old TTS streaming callback - using sequential processing instead  
        # if streaming_config.get("enable_tts_streaming", True) and self.audio_service:
        #     tts_callback = self._create_streaming_tts_callback()
        
        # Disable old text streaming callback - using sequential processing instead
        # if streaming_config.get("enable_text_streaming", False):
        #     text_callback = self._create_streaming_text_callback()
        
        # Start streaming session
        discord_bot_instance = self.ctx.discord_bot_runner_instance
        self.logger.info(f"[LLMService] Discord bot instance debug: {type(discord_bot_instance)} - {discord_bot_instance is not None}")
        
        stream_id = streaming_service.start_response_stream(
            user_text=user_text,
            user_name=user_name,
            tts_callback=tts_callback,
            text_callback=text_callback,
            discord_bot=discord_bot_instance  # Pass Discord bot runner instance
        )
        
        if stream_id:
            # Stream the complete response
            streaming_service.stream_complete_response(stream_id, response_text)
            self.logger.debug(f"[LLMService] Started streaming response with ID: {stream_id}")
        else:
            # Fallback to traditional response
            self._send_traditional_response(response_text)

    def _send_traditional_response(self, response_text: str):
        """Send response using traditional method (all at once)"""
        if self.ctx.tts_service_instance and self.ctx.global_settings.get("ENABLE_TTS_FOR_CHAT_REPLIES", True):
            tts_audio = self.ctx.tts_service_instance.generate_audio(response_text)
            if tts_audio:
                # Prevent queue flooding - clear old items if queue is too full
                max_queue_size = 3  # Allow max 3 TTS items in queue
                while self.ctx.tts_queue.qsize() >= max_queue_size:
                    try:
                        old_audio = self.ctx.tts_queue.get_nowait()
                        self.ctx.tts_queue.task_done()
                        self.logger.debug(f"[LLMService] Dropped old TTS audio to prevent queue flooding (size: {self.ctx.tts_queue.qsize()})")
                    except queue.Empty:
                        break
                self.ctx.tts_queue.put_nowait(tts_audio)

    def _create_streaming_tts_callback(self):
        """Create a callback for streaming TTS processing"""
        def tts_callback(sentence: str):
            # Make TTS processing asynchronous to prevent blocking Discord thread
            import threading
            def async_tts_processing():
                try:
                    if self.ctx.tts_service_instance and self.ctx.global_settings.get("ENABLE_TTS_FOR_CHAT_REPLIES", True):
                        # Strip markdown for TTS
                        clean_sentence = self._strip_markdown_for_tts(sentence)
                        
                        # Add timeout for Chatterbox to prevent hanging
                        streaming_config = self.ctx.global_settings.get("STREAMING_RESPONSE", {})
                        tts_timeout = min(streaming_config.get("sentence_queue_timeout_s", 120), 30)  # Max 30s per sentence
                        
                        self.logger.debug(f"[LLMService] Starting TTS for sentence: '{clean_sentence[:30]}...' (timeout: {tts_timeout}s)")
                        
                        # Use timeout to prevent hanging
                        import signal
                        import time
                        
                        tts_audio = None
                        start_time = time.time()
                        
                        try:
                            # For Chatterbox, we need to be more careful with timeouts
                            tts_audio = self.ctx.tts_service_instance.generate_audio(clean_sentence)
                            generation_time = time.time() - start_time
                            
                            if tts_audio:
                                # Use a separate streaming queue to avoid overwhelming main queue
                                if not hasattr(self.ctx, 'streaming_tts_queue'):
                                    import queue
                                    max_queue_size = streaming_config.get("max_sentence_queue_size", 3)
                                    self.ctx.streaming_tts_queue = queue.Queue(maxsize=max_queue_size)
                                
                                try:
                                    # Try to add to streaming queue first
                                    self.ctx.streaming_tts_queue.put_nowait(tts_audio)
                                    self.logger.info(f"[LLMService] Streamed TTS for sentence in {generation_time:.1f}s: '{clean_sentence[:30]}...'")
                                except queue.Full:
                                    # If streaming queue is full, add to main queue
                                    try:
                                        self.ctx.tts_queue.put_nowait(tts_audio)
                                        self.logger.warning(f"[LLMService] Streaming queue full, used main queue for: '{clean_sentence[:30]}...'")
                                    except queue.Full:
                                        self.logger.warning(f"[LLMService] Both TTS queues full, dropping sentence audio: '{clean_sentence[:30]}...'")
                            else:
                                self.logger.warning(f"[LLMService] No TTS audio generated for sentence: '{clean_sentence[:30]}...'")
                                
                        except Exception as tts_error:
                            generation_time = time.time() - start_time
                            self.logger.error(f"[LLMService] TTS generation failed after {generation_time:.1f}s: {tts_error}")
                            
                            # If Chatterbox fails, we could fallback to fast TTS
                            try:
                                self.logger.info("[LLMService] Attempting fallback to pyttsx3 for this sentence...")
                                from services.tts_service_fast import FastTTSService
                                fast_tts = FastTTSService(self.ctx)
                                fallback_audio = fast_tts.generate_audio(clean_sentence)
                                if fallback_audio:
                                    self.ctx.tts_queue.put_nowait(fallback_audio)
                                    self.logger.info(f"[LLMService] Used pyttsx3 fallback for: '{clean_sentence[:30]}...'")
                            except Exception as fallback_error:
                                self.logger.error(f"[LLMService] Fallback TTS also failed: {fallback_error}")
                                
                except Exception as e:
                    self.logger.error(f"[LLMService] Streaming TTS callback error: {e}")
            
            # Run TTS processing in background thread to avoid blocking Discord
            threading.Thread(target=async_tts_processing, daemon=True, name=f"TTS-{sentence[:20]}").start()
        
        return tts_callback

    def _create_streaming_text_callback(self):
        """Create a callback for streaming text to Discord"""
        def text_callback(sentence: str):
            try:
                # Add sentence to text message queue for Discord
                if hasattr(self.ctx, 'text_message_queue'):
                    self.ctx.text_message_queue.put_nowait(f"💬 {sentence}")
            except Exception as e:
                self.logger.error(f"[LLMService] Streaming text callback error: {e}")
        
        return text_callback

    async def _run_image_visibility_diagnostic(self, base64_image: str, image_formats: dict):
        """Special diagnostic function to test if the model can see the image in any format"""
        self.logger.info("[LLMService] Running deep image visibility diagnostic with all formats")
        profile = self.ctx.active_profile
        gs = self.ctx.global_settings
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug_vlm_frames")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save diagnostic info
        diag_path = os.path.join(debug_dir, f"diagnostic_test_{int(time.time())}.txt")
        with open(diag_path, 'w') as f:
            f.write(f"VLM IMAGE VISIBILITY DIAGNOSTIC\n")
            f.write(f"==============================\n")
            f.write(f"Model: {profile.vlm_model}\n")
            f.write(f"Provider: {gs.get('VLM_PROVIDER', 'unknown')}\n")
            f.write(f"Base64 length: {len(base64_image)} chars\n")
            f.write(f"Available formats: {list(image_formats.keys())}\n")
            f.write(f"==============================\n\n")

        # Run tests for each format
        results = {}
        for format_name, format_content in image_formats.items():
            try:
                response = await self._test_image_format(
                    format_name=format_name,
                    base64_image=base64_image,
                    format_content=format_content,
                    profile=profile,
                    diag_path=diag_path
                )
                if response and "success" in response:
                    self.logger.info(f"[LLMService] Found working format: {format_name}")
                    self.ctx.global_settings["VLM_IMAGE_FORMAT"] = format_name
                    break
            except Exception as e:
                self.logger.error(f"Format test failed for {format_name}: {e}")
                continue

        self.logger.info(f"[LLMService] Diagnostic completed. Results saved to {diag_path}")
        return results

    async def _test_image_format(self, format_name: str, base64_image: str, format_content: str, profile, diag_path: str):
        """Helper method to test a single image format"""
        self.logger.info(f"[LLMService] Testing format: {format_name}")
        
        diagnostic_prompt = "This is an image visibility test. Can you see and describe the image shown?"
        vlm_provider = self.ctx.global_settings.get("VLM_PROVIDER", "default").lower()
        
        messages = [{"role": "user", "content": diagnostic_prompt}]
        images_payload = []

        # Configure the payload based on provider
        if vlm_provider == "ollama":
            images_payload = [base64_image]
        elif "qwen" in profile.vlm_model.lower():
            messages = [{"role": "user", "content": f"USER: <img>{base64_image}</img>\n\n{diagnostic_prompt}\n\nASSISTANT:"}]
        elif "llama" in profile.vlm_model.lower():
            messages = [{"role": "user", "content": f"<img src=\"data:image/jpeg;base64,{base64_image}\">\n\n{diagnostic_prompt}"}]
        else:
            messages = [{"role": "user", "content": f"data:image/jpeg;base64,{base64_image}\n\n{diagnostic_prompt}"}]

        payload = {
            "model": profile.vlm_model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.2,
            "stream": False
        }
        if images_payload:
            payload["images"] = images_payload

        # Log diagnostic info
        with open(diag_path, 'a') as f:
            f.write(f"\nTesting Format: {format_name}\n")
            f.write(f"Payload Summary:\n{json.dumps(payload, indent=2)[:500]}...\n")

        # Make API call
        try:
            response = await self.model_client.generate(
                messages=messages,
                temperature=0.2,
                max_tokens=300,
                model=profile.vlm_model,
                endpoint="chat/completions",
                images=images_payload if images_payload else None
            )
            
            if not response:
                return {"error": "Empty response"}

            # Log response
            with open(diag_path, 'a') as f:
                f.write(f"Response:\n{response[:1000]}\n")
                f.write("="*50 + "\n")

            # Check if model can see image
            if "i cannot see any image" not in response.lower() and "i can't see any image" not in response.lower():
                return {"success": True, "response": response}
            
            return {"success": False, "response": response}

        except Exception as e:
            self.logger.error(f"Error testing format {format_name}: {e}")
            return {"error": str(e)}

    async def get_response(self, user: str, game: str, query: str) -> str:
        """
        1) Run the user's query against RAG.
        2) If no docs returned, fallback immediately.
        3) Otherwise, build a grounded prompt and generate with temp=0.
        """
        self.logger.info(f"[LLMService] Fact-checking query via RAG: '{query}'")
        # Step 1: retrieve top 5 passages
        docs = self.rag_service.query(collection=self.default_collection,
                              query_text=query,
                              n_results=5)

        # Step 2: fallback if nothing to ground on
        if not docs:
            fallback = ("I'm not certain about that. I couldn't find any reference "
                        "in my knowledge base—would you like me to search deeper?")
            self.logger.warning("[LLMService] No RAG hits, returning fallback.")
            return fallback

        # Step 3: build a grounded prompt
        context_block = "\n\n--- Retrieved Context ---\n" + "\n\n".join(docs)
        system_prompt = (
            "You are DanzarAI, a knowledgeable EverQuest assistant. "
            "Answer the user's question **using only** the information in the retrieved context. "
            "If the answer is not contained there, say \"I don't know.\""
        )
        full_prompt = f"{system_prompt}\n\n{context_block}\n\nUser: {query}\nAssistant:"

        # Get max_tokens from profile or use a larger default
        max_tokens = getattr(self.ctx.active_profile, 'conversational_max_tokens', 1024)
        self.logger.info(f"[LLMService] Generating grounded response (temp=0, max_tokens={max_tokens}).")
        
        try:
            if self.model_client:
                answer = await self.model_client.generate(
                    prompt=full_prompt,
                    temperature=0.0,
                    max_tokens=max_tokens
                )
            else:
                # Fallback to using the existing _call_llm_api method
                payload = {
                    "model": self.ctx.active_profile.conversational_llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{context_block}\n\nUser: {query}"}
                    ],
                    "temperature": 0.0,
                    "max_tokens": max_tokens
                }
                resp = self._call_llm_api(payload)
                if resp and "choices" in resp and resp["choices"]:
                    answer = resp["choices"][0].get("message", {}).get("content", "").strip()
                else:
                    raise Exception("Empty or invalid response from LLM API")
        except Exception as e:
            self.logger.error(f"[LLMService] LLM generation failed: {e}", exc_info=True)
            return "Sorry, I ran into an error trying to think that through."

        # Step 4: final sanity check for "I don't know" enforcement
        if answer.strip() == "":
            return "I'm not sure; there's no info in my sources."

        return answer.strip()

    async def process_gaming_message(self, message: str, user_name: str = "User") -> str:
        """Process a gaming message with context and return response"""
        self.logger.info(f"[LLM:DEBUG] === Processing gaming message from {user_name} ===")
        self.logger.info(f"[LLM:DEBUG] Message: '{message}'")
        
        try:
            # Use Smart RAG for contextual response generation
            if self.smart_rag:
                self.logger.info(f"[LLM:DEBUG] Using Smart RAG service for response generation")
                response, metadata = self.smart_rag.smart_generate_response(message, user_name)
                
                self.logger.info(f"[LLM:DEBUG] Smart RAG response metadata: {metadata}")
                self.logger.info(f"[LLM:DEBUG] Smart RAG response length: {len(response)} chars")
                self.logger.info(f"[LLM:DEBUG] Smart RAG response preview: {response[:200]}...")
                
                return response
            else:
                self.logger.warning(f"[LLM:DEBUG] No Smart RAG service available, using direct LLM")
                # Fallback to direct LLM without context
                return await self._generate_direct_response(message)
                
        except Exception as e:
            self.logger.error(f"[LLM:DEBUG] Gaming message processing failed: {e}")
            import traceback
            self.logger.error(f"[LLM:DEBUG] Full traceback: {traceback.format_exc()}")
            return "I'm having trouble processing that request right now."

    async def _generate_direct_response(self, message: str) -> str:
        """Generate a direct response using the LLM without RAG context"""
        try:
            if not self.model_client:
                return "LLM service is not available right now."
            
            messages = [
                {"role": "system", "content": "You are a helpful gaming assistant."},
                {"role": "user", "content": message}
            ]
            
            profile = self.ctx.active_profile
            response = await self.model_client.generate(
                messages=messages,
                temperature=float(profile.conversational_temperature),
                max_tokens=int(profile.conversational_max_tokens),
                model=profile.conversational_llm_model
            )
            
            return response if response else "I'm unable to generate a response right now."
            
        except Exception as e:
            self.logger.error(f"[LLM:DEBUG] Direct response generation failed: {e}")
            return "I'm having trouble processing that request right now."

    async def _handle_crawl_website_request(self, user_text: str, user_name: str) -> str:
        """
        Handle website crawling requests.
        
        Args:
            user_text: The user's crawl request
            user_name: Name of the user making the request
            
        Returns:
            Response about the crawling operation
        """
        try:
            # Extract URL from the request
            import re
            url_pattern = r'(https?://[^\s]+)'
            urls = re.findall(url_pattern, user_text)
            
            if not urls:
                return "I need a valid URL to crawl. Please provide a URL starting with http:// or https://"
            
            target_url = urls[0]
            
            # Check if exhaustive crawler is available
            if not self.ctx.website_crawler_instance:
                return "Website crawler is not available. Please ensure the crawler service is initialized."
            
            # Extract max pages from request if specified
            max_pages = 50  # Default limit
            max_pattern = r'(?:max|limit|up to)\s*(\d+)\s*pages?'
            max_match = re.search(max_pattern, user_text.lower())
            if max_match:
                max_pages = min(int(max_match.group(1)), 200)  # Cap at 200 for safety
            
            self.logger.info(f"[LLMService] Starting exhaustive crawl of {target_url} (max {max_pages} pages)")
            
            # Determine collection name
            from urllib.parse import urlparse
            parsed_url = urlparse(target_url)
            domain = parsed_url.netloc.replace('.', '_').replace('-', '_')
            collection_name = f"crawl_{domain}_{int(time.time())}"
            
            # Perform the crawl
            crawler = self.ctx.website_crawler_instance
            crawl_results = crawler.crawl_website(
                base_url=target_url,
                collection_name=collection_name,
                max_pages=max_pages,
                same_domain_only=True
            )
            
            # Format response
            if "error" in crawl_results:
                return f"❌ Crawl failed: {crawl_results['error']}"
            
            response_parts = [
                f"🕷️ **Website Crawl Complete** for {target_url}",
                f"",
                f"**Results:**",
                f"📄 Pages crawled: {crawl_results['pages_crawled']}",
                f"❌ Pages failed: {crawl_results['pages_failed']}",
                f"💾 Pages stored in RAG: {crawl_results['pages_stored']}",
                f"🔧 Unique content pieces: {crawl_results['unique_content_pieces']}",
                f"⏱️ Duration: {crawl_results['duration_seconds']} seconds",
                f"📊 Crawl speed: {crawl_results['pages_per_second']} pages/sec",
                f"",
                f"**Collection:** `{collection_name}`"
            ]
            
            # Add sample page info if available
            if crawler.crawled_pages:
                sample_page = crawler.crawled_pages[0]
                response_parts.extend([
                    f"",
                    f"**Sample Page:**",
                    f"🔗 URL: {sample_page.url}",
                    f"📝 Title: {sample_page.title}",
                    f"📏 Content: {len(sample_page.content)} characters",
                    f"🔗 Links found: {len(sample_page.links)}"
                ])
            
            # Store crawl summary in memory
            summary_text = f"Website crawl of {target_url} completed. Crawled {crawl_results['pages_crawled']} pages and stored them in collection '{collection_name}'. You can now search this crawled content for information."
            
            crawl_memory = MemoryEntry(
                content=f"User ({user_name}): {user_text}\nAI (DanzarVLM): {summary_text}",
                source="website_crawl",
                timestamp=time.time(),
                metadata={
                    "user": user_name,
                    "crawl_target": target_url,
                    "pages_crawled": crawl_results['pages_crawled'],
                    "collection": collection_name,
                    "type": "crawl_operation"
                },
                importance_score=2.0  # Higher importance for crawl operations
            )
            
            self.memory_service.store_memory(crawl_memory)
            
            return "\n".join(response_parts)
            
        except Exception as e:
            self.logger.error(f"[LLMService] Website crawl failed: {e}", exc_info=True)
            return f"❌ Website crawling failed: {str(e)}"

    async def handle_user_text_query_with_llm_search(self, user_text: str, user_name: str = "User") -> tuple:
        """
        Enhanced query handling that allows the LLM to admit when it doesn't know something
        and formulate its own search queries to find the answer
        """
        try:
            # Step 1: Ask the LLM if it knows the answer
            initial_prompt = f"""You are DanzarAI, an EverQuest gaming assistant. A user asked: "{user_text}"

If you know the answer from your training data, provide a helpful response.

If you don't know the answer or are uncertain, respond with exactly: "I don't know - let me search for that information."

Your response:"""

            initial_response = await self.model_client.generate(
                messages=[{"role": "user", "content": initial_prompt}],
                temperature=0.3,
                max_tokens=200
            )

            # Step 2: Check if LLM admits it doesn't know
            if initial_response and "I don't know" in initial_response:
                self.logger.info(f"[LLMService] LLM admits it doesn't know, proceeding with search")
                
                # Step 3: Ask LLM to formulate search queries
                search_prompt = f"""The user asked: "{user_text}"

You don't know the answer. Generate 3 specific search queries that would help find authoritative information to answer this question.

Focus on:
1. Official sources (wikis, guides, documentation)
2. Simple, direct search terms
3. Different aspects of the question

Format as a numbered list:
1. [search query 1]
2. [search query 2]
3. [search query 3]

Search queries:"""

                search_response = await self.model_client.generate(
                    messages=[{"role": "user", "content": search_prompt}],
                    temperature=0.3,
                    max_tokens=150
                )

                if search_response:
                    # Parse search queries
                    search_queries = self._parse_search_queries(search_response)
                    
                    if search_queries:
                        self.logger.info(f"[LLMService] LLM generated {len(search_queries)} search queries")
                        
                        # Step 4: Perform searches
                        search_results = await self._perform_llm_guided_searches(search_queries)
                        
                        if search_results:
                            # Step 5: Ask LLM to synthesize the search results
                            synthesis_prompt = f"""The user asked: "{user_text}"

I searched for information and found these results:

{search_results}

Based on this information, provide a comprehensive answer to the user's question. If the information is insufficient, say so honestly.

Your response:"""

                            final_response = await self.model_client.generate(
                                messages=[{"role": "user", "content": synthesis_prompt}],
                                temperature=0.3,
                                max_tokens=300
                            )

                            if final_response:
                                metadata = {
                                    "method": "llm_guided_search",
                                    "search_queries": search_queries,
                                    "search_results_found": True,
                                    "llm_admitted_ignorance": True
                                }
                                return final_response, metadata
                
                # Fallback to regular RAG if search fails
                self.logger.info("[LLMService] LLM search failed, falling back to regular RAG")
                return await self.handle_user_text_query(user_text, user_name)
            
            else:
                # LLM thinks it knows the answer
                if initial_response:
                    metadata = {
                        "method": "llm_direct_knowledge",
                        "llm_admitted_ignorance": False
                    }
                    return initial_response, metadata
                else:
                    # Fallback to regular RAG
                    return await self.handle_user_text_query(user_text, user_name)

        except Exception as e:
            self.logger.error(f"[LLMService] Error in LLM-guided search: {e}")
            # Fallback to regular RAG
            return await self.handle_user_text_query(user_text, user_name)

    def _parse_search_queries(self, response: str) -> List[str]:
        """Parse LLM response to extract search queries"""
        try:
            queries = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Look for numbered list items
                if re.match(r'^\d+\.?\s*', line):
                    query = re.sub(r'^\d+\.?\s*', '', line).strip()
                    if query and len(query) > 5:
                        # Remove any quotes or brackets
                        query = query.strip('"\'[]')
                        queries.append(query)
            
            return queries[:3]  # Limit to 3 queries
            
        except Exception as e:
            self.logger.error(f"[LLMService] Error parsing search queries: {e}")
            return []

    async def _perform_llm_guided_searches(self, search_queries: List[str]) -> str:
        """Perform multiple searches guided by LLM and combine results"""
        all_results = []
        
        for query in search_queries:
            self.logger.info(f"[LLMService] Performing guided search: {query}")
            try:
                if self.fact_check:
                    result = await self.fact_check._search_web(query)
                    if result and len(result.strip()) > 50:
                        all_results.append(f"Query: {query}\nResults: {result}\n")
                        self.logger.info(f"[LLMService] Found {len(result)} chars for: {query}")
                    else:
                        self.logger.warning(f"[LLMService] No results for guided search: {query}")
                else:
                    self.logger.warning("[LLMService] No fact_check service available for guided search")
            except Exception as e:
                self.logger.error(f"[LLMService] Error in guided search '{query}': {e}")
        
        if all_results:
            combined_results = "\n".join(all_results)
            self.logger.info(f"[LLMService] Combined {len(all_results)} search results ({len(combined_results)} chars)")
            return combined_results
        else:
            self.logger.warning("[LLMService] No results from any guided searches")
            return ""

    # ========================================
    # AUTOMATIC KNOWLEDGE BASE CLEANUP SYSTEM
    # ========================================
    
    def _start_cleanup_scheduler(self):
        """Start the periodic cleanup scheduler in a background thread"""
        if self.cleanup_running:
            return
            
        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_scheduler_loop, daemon=True)
        self.cleanup_thread.start()
        self.logger.info("[LLMService] Knowledge base cleanup scheduler started")
    
    def _cleanup_scheduler_loop(self):
        """Background thread loop for periodic cleanup"""
        while self.cleanup_running:
            try:
                # Wait for the cleanup interval
                time.sleep(self.cleanup_interval_hours * 3600)  # Convert hours to seconds
                
                if self.cleanup_running and self.rag_service:
                    self.logger.info("[LLMService] Starting scheduled knowledge base cleanup...")
                    asyncio.run(self._perform_knowledge_base_cleanup())
                    
            except Exception as e:
                self.logger.error(f"[LLMService] Error in cleanup scheduler: {e}", exc_info=True)
                # Continue running even if one cleanup fails
                
    async def _perform_knowledge_base_cleanup(self):
        """Main cleanup orchestrator that runs all cleanup operations"""
        try:
            cleanup_start = time.time()
            self.logger.info("[LLMService] 🧹 Starting comprehensive knowledge base cleanup...")
            
            # Get all collections to clean
            collections_to_clean = await self._get_collections_for_cleanup()
            
            total_removed = 0
            total_consolidated = 0
            
            for collection_name in collections_to_clean:
                self.logger.info(f"[LLMService] 🔍 Cleaning collection: {collection_name}")
                
                # Step 1: Remove duplicates and near-duplicates
                duplicates_removed = await self._remove_duplicate_entries(collection_name)
                total_removed += duplicates_removed
                
                # Step 2: Remove outdated entries
                outdated_removed = await self._remove_outdated_entries(collection_name)
                total_removed += outdated_removed
                
                # Step 3: Consolidate similar entries
                consolidated = await self._consolidate_similar_entries(collection_name)
                total_consolidated += consolidated
                
                # Step 4: Remove low-quality entries
                low_quality_removed = await self._remove_low_quality_entries(collection_name)
                total_removed += low_quality_removed
                
                self.logger.info(f"[LLMService] ✅ Collection {collection_name} cleaned")
            
            cleanup_duration = time.time() - cleanup_start
            self.logger.info(
                f"[LLMService] 🎉 Knowledge base cleanup completed in {cleanup_duration:.2f}s. "
                f"Removed: {total_removed}, Consolidated: {total_consolidated}"
            )
            
        except Exception as e:
            self.logger.error(f"[LLMService] Error during knowledge base cleanup: {e}", exc_info=True)
    
    async def _get_collections_for_cleanup(self) -> List[str]:
        """Get list of collections that should be cleaned"""
        try:
            # Get all collections from Qdrant
            if hasattr(self.rag_service, 'client'):
                collections_info = self.rag_service.client.get_collections()
                collection_names = [col.name for col in collections_info.collections]
                
                # Filter out system collections or ones we don't want to clean
                excluded_collections = {'system', 'temp', 'backup'}
                collections_to_clean = [name for name in collection_names if name not in excluded_collections]
                
                self.logger.info(f"[LLMService] Found {len(collections_to_clean)} collections to clean: {collections_to_clean}")
                return collections_to_clean
            else:
                self.logger.warning("[LLMService] RAG service client not available for collection listing")
                return []
                
        except Exception as e:
            self.logger.error(f"[LLMService] Error getting collections for cleanup: {e}")
            return []
    
    async def _remove_duplicate_entries(self, collection_name: str) -> int:
        """Remove duplicate and near-duplicate entries based on embedding similarity"""
        try:
            # Get all points from the collection
            points = await self._get_all_points_from_collection(collection_name)
            if len(points) < 2:
                return 0
            
            duplicates_to_remove = []
            similarity_threshold = 0.95  # Very high similarity threshold for duplicates
            
            # Compare embeddings for similarity
            for i, point1 in enumerate(points):
                if point1['id'] in [dup['id'] for dup in duplicates_to_remove]:
                    continue  # Skip if already marked for removal
                    
                for j, point2 in enumerate(points[i+1:], i+1):
                    if point2['id'] in [dup['id'] for dup in duplicates_to_remove]:
                        continue
                    
                    # Calculate cosine similarity between embeddings
                    similarity = cosine_similarity_simple(
                        point1['vector'], point2['vector']
                    )
                    
                    if similarity > similarity_threshold:
                        # Keep the newer entry (higher timestamp if available)
                        point1_time = self._extract_timestamp_from_payload(point1.get('payload', {}))
                        point2_time = self._extract_timestamp_from_payload(point2.get('payload', {}))
                        
                        if point2_time > point1_time:
                            duplicates_to_remove.append(point1)
                        else:
                            duplicates_to_remove.append(point2)
                        
                        self.logger.debug(f"[LLMService] Found duplicate (similarity: {similarity:.3f})")
                        break
            
            # Remove duplicates
            if duplicates_to_remove:
                ids_to_remove = [dup['id'] for dup in duplicates_to_remove]
                await self._remove_points_from_collection(collection_name, ids_to_remove)
                self.logger.info(f"[LLMService] Removed {len(duplicates_to_remove)} duplicates from {collection_name}")
            
            return len(duplicates_to_remove)
            
        except Exception as e:
            self.logger.error(f"[LLMService] Error removing duplicates from {collection_name}: {e}")
            return 0
    
    async def _remove_outdated_entries(self, collection_name: str) -> int:
        """Remove entries based on intelligent retention policy"""
        try:
            # Get retention policy from config
            retention_policy = self.ctx.global_settings.get("RAG_RETENTION_POLICY", {})
            
            # Collections that should never be cleaned by age
            permanent_collections = retention_policy.get("permanent_collections", ["everquest", "wow", "rimworld", "game_knowledge"])
            if collection_name in permanent_collections:
                self.logger.info(f"[LLMService] Skipping age-based cleanup for permanent collection: {collection_name}")
                return 0
            
            # Get cleanup settings
            cleanup_after_days = retention_policy.get("cleanup_after_days", 7)
            cutoff_time = datetime.now() - timedelta(days=cleanup_after_days)
            
            # Keywords that indicate permanent value
            permanent_keywords = retention_policy.get("permanent_keywords", ["everquest", "game guide", "strategy", "quest", "class guide", "spell", "ability", "lore", "wiki", "tutorial", "how to"])
            cleanup_keywords = retention_policy.get("cleanup_keywords", ["error", "failed", "timeout", "test", "debug", "temporary"])
            high_quality_indicators = retention_policy.get("high_quality_indicators", ["detailed", "comprehensive", "guide", "tutorial", "explanation", "strategy"])
            permanent_min_length = retention_policy.get("permanent_min_length", 100)
            
            points = await self._get_all_points_from_collection(collection_name)
            outdated_points = []
            preserved_count = 0
            
            for point in points:
                timestamp = self._extract_timestamp_from_payload(point.get('payload', {}))
                if not timestamp or timestamp >= cutoff_time:
                    continue  # Not old enough to consider for cleanup
                
                payload = point.get('payload', {})
                text_content = payload.get('text', '').lower()
                
                # Determine if this entry should be preserved forever
                should_preserve = False
                
                # 1. Check if it's long enough to be considered valuable
                if len(text_content) >= permanent_min_length:
                    should_preserve = True
                    
                # 2. Check for permanent keywords
                elif any(keyword.lower() in text_content for keyword in permanent_keywords):
                    should_preserve = True
                    
                # 3. Check for high-quality indicators
                elif any(indicator.lower() in text_content for indicator in high_quality_indicators):
                    should_preserve = True
                    
                # 4. Check if it's from a search that found valuable results
                elif "search results" in text_content and len(text_content) > 200:
                    should_preserve = True
                    
                # 5. Prioritize cleanup of error/debug entries
                elif any(keyword.lower() in text_content for keyword in cleanup_keywords):
                    should_preserve = False  # Explicitly mark for cleanup
                    
                if should_preserve:
                    preserved_count += 1
                    self.logger.debug(f"[LLMService] Preserving valuable entry: {text_content[:50]}...")
                else:
                    outdated_points.append(point)
                    self.logger.debug(f"[LLMService] Marking for cleanup: {text_content[:50]}...")
            
            # Remove outdated points
            if outdated_points:
                ids_to_remove = [point['id'] for point in outdated_points]
                await self._remove_points_from_collection(collection_name, ids_to_remove)
                self.logger.info(f"[LLMService] Removed {len(outdated_points)} low-value entries from {collection_name}")
                self.logger.info(f"[LLMService] Preserved {preserved_count} valuable entries in {collection_name}")
            else:
                self.logger.info(f"[LLMService] No entries marked for cleanup in {collection_name} (preserved {preserved_count} valuable entries)")
            
            return len(outdated_points)
            
        except Exception as e:
            self.logger.error(f"[LLMService] Error removing outdated entries from {collection_name}: {e}")
            return 0
    
    async def _consolidate_similar_entries(self, collection_name: str) -> int:
        """Consolidate similar entries into single, more comprehensive entries"""
        try:
            points = await self._get_all_points_from_collection(collection_name)
            if len(points) < 2:
                return 0
            
            consolidations = 0
            similarity_threshold = 0.85  # Lower threshold for consolidation
            
            # Group similar points
            similar_groups = []
            processed_ids = set()
            
            for i, point1 in enumerate(points):
                if point1['id'] in processed_ids:
                    continue
                
                group = [point1]
                processed_ids.add(point1['id'])
                
                for j, point2 in enumerate(points[i+1:], i+1):
                    if point2['id'] in processed_ids:
                        continue
                    
                    similarity = cosine_similarity_simple(
                        point1['vector'], point2['vector']
                    )
                    
                    if similarity > similarity_threshold:
                        group.append(point2)
                        processed_ids.add(point2['id'])
                
                if len(group) > 1:
                    similar_groups.append(group)
            
            # Consolidate each group
            for group in similar_groups:
                consolidated_point = await self._create_consolidated_entry(group)
                if consolidated_point:
                    # Remove old points
                    old_ids = [point['id'] for point in group]
                    await self._remove_points_from_collection(collection_name, old_ids)
                    
                    # Add consolidated point
                    await self._add_point_to_collection(collection_name, consolidated_point)
                    
                    consolidations += 1
                    self.logger.debug(f"[LLMService] Consolidated {len(group)} similar entries")
            
            if consolidations > 0:
                self.logger.info(f"[LLMService] Consolidated {consolidations} groups of similar entries in {collection_name}")
            
            return consolidations
            
        except Exception as e:
            self.logger.error(f"[LLMService] Error consolidating entries in {collection_name}: {e}")
            return 0
    
    async def _remove_low_quality_entries(self, collection_name: str) -> int:
        """Remove entries that are deemed low quality based on various criteria"""
        try:
            points = await self._get_all_points_from_collection(collection_name)
            low_quality_points = []
            
            for point in points:
                payload = point.get('payload', {})
                text_content = payload.get('text', '')
                
                # Quality criteria
                is_too_short = len(text_content.strip()) < 20
                is_mostly_punctuation = len(re.sub(r'[^\w\s]', '', text_content)) < len(text_content) * 0.5
                is_repetitive = self._is_repetitive_text(text_content)
                has_no_meaningful_content = not re.search(r'[a-zA-Z]{3,}', text_content)
                
                if is_too_short or is_mostly_punctuation or is_repetitive or has_no_meaningful_content:
                    low_quality_points.append(point)
                    self.logger.debug(f"[LLMService] Identified low-quality entry: {text_content[:50]}...")
            
            # Remove low-quality points
            if low_quality_points:
                ids_to_remove = [point['id'] for point in low_quality_points]
                await self._remove_points_from_collection(collection_name, ids_to_remove)
                self.logger.info(f"[LLMService] Removed {len(low_quality_points)} low-quality entries from {collection_name}")
            
            return len(low_quality_points)
            
        except Exception as e:
            self.logger.error(f"[LLMService] Error removing low-quality entries from {collection_name}: {e}")
            return 0
    
    def _is_repetitive_text(self, text: str) -> bool:
        """Check if text contains too much repetition"""
        words = text.lower().split()
        if len(words) < 10:
            return False
        
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        
        # Check if any word appears more than 30% of the time
        max_frequency = max(word_counts.values()) / len(words)
        return max_frequency > 0.3
    
    async def _get_all_points_from_collection(self, collection_name: str) -> List[Dict]:
        """Get all points from a collection"""
        try:
            if hasattr(self.rag_service, 'client'):
                # Use scroll to get all points
                points = []
                offset = None
                
                while True:
                    result = self.rag_service.client.scroll(
                        collection_name=collection_name,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=True
                    )
                    
                    if not result[0]:  # No more points
                        break
                    
                    points.extend([
                        {
                            'id': point.id,
                            'vector': point.vector,
                            'payload': point.payload
                        }
                        for point in result[0]
                    ])
                    
                    offset = result[1]  # Next offset
                
                return points
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"[LLMService] Error getting points from {collection_name}: {e}")
            return []
    
    async def _remove_points_from_collection(self, collection_name: str, point_ids: List[str]):
        """Remove specific points from a collection"""
        try:
            if hasattr(self.rag_service, 'client'):
                self.rag_service.client.delete(
                    collection_name=collection_name,
                    points_selector=point_ids
                )
        except Exception as e:
            self.logger.error(f"[LLMService] Error removing points from {collection_name}: {e}")
    
    async def _add_point_to_collection(self, collection_name: str, point: Dict):
        """Add a point to a collection"""
        try:
            if hasattr(self.rag_service, 'client'):
                self.rag_service.client.upsert(
                    collection_name=collection_name,
                    points=[{
                        'id': point['id'],
                        'vector': point['vector'],
                        'payload': point['payload']
                    }]
                )
        except Exception as e:
            self.logger.error(f"[LLMService] Error adding point to {collection_name}: {e}")
    
    def _extract_timestamp_from_payload(self, payload: Dict) -> Optional[datetime]:
        """Extract timestamp from point payload"""
        try:
            # Try different timestamp formats
            timestamp_fields = ['timestamp', 'date', 'created_at', 'search_date']
            
            for field in timestamp_fields:
                if field in payload:
                    timestamp_str = payload[field]
                    if isinstance(timestamp_str, str):
                        # Try parsing different formats
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S']:
                            try:
                                return datetime.strptime(timestamp_str, fmt)
                            except ValueError:
                                continue
            
            return None
            
        except Exception as e:
            self.logger.debug(f"[LLMService] Error extracting timestamp: {e}")
            return None
    
    async def _create_consolidated_entry(self, similar_points: List[Dict]) -> Optional[Dict]:
        """Create a consolidated entry from similar points"""
        try:
            if not similar_points:
                return None
            
            # Use the most recent point as base
            timestamps = []
            for point in similar_points:
                ts = self._extract_timestamp_from_payload(point.get('payload', {}))
                if ts:
                    timestamps.append((ts, point))
            
            if timestamps:
                # Sort by timestamp and use the most recent
                timestamps.sort(key=lambda x: x[0], reverse=True)
                base_point = timestamps[0][1]
            else:
                base_point = similar_points[0]
            
            # Combine text content from all points
            all_texts = []
            for point in similar_points:
                text = point.get('payload', {}).get('text', '')
                if text and text not in all_texts:
                    all_texts.append(text)
            
            # Create consolidated text
            consolidated_text = ' '.join(all_texts)
            
            # Generate new embedding for consolidated text
            if hasattr(self.rag_service, 'embedding_model'):
                new_vector = self.rag_service.embedding_model.encode([consolidated_text])[0].tolist()
            else:
                # Fallback to averaging existing vectors
                vectors = [point['vector'] for point in similar_points]
                new_vector = np.mean(vectors, axis=0).tolist()
            
            # Create consolidated point
            consolidated_point = {
                'id': f"consolidated_{int(time.time())}_{hash(consolidated_text) % 10000}",
                'vector': new_vector,
                'payload': {
                    **base_point.get('payload', {}),
                    'text': consolidated_text,
                    'consolidated_from': len(similar_points),
                    'consolidated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            return consolidated_point
            
        except Exception as e:
            self.logger.error(f"[LLMService] Error creating consolidated entry: {e}")
            return None
    
    def stop_cleanup_scheduler(self):
        """Stop the cleanup scheduler (useful for shutdown)"""
        self.cleanup_running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        self.logger.info("[LLMService] Knowledge base cleanup scheduler stopped")