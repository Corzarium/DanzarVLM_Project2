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

from utils.text_utils import trim_sentences
from utils.ocr_utils import OCRProcessor
from core.game_profile import GameProfile
from .memory_service import MemoryEntry
# from .rag_service import RAGService  # Temporarily disabled due to dependency issues
from .model_client import ModelClient
from .danzar_factcheck import FactCheckService
from utils.error_logger import ErrorLogger

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
        discord_msg = f"ðŸŽ™ï¸ **{profile.game_name} Tip:** {text_for_tts_and_discord}"
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
                response = self.model_client.generate(
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
            
            # Create a comprehensive text entry for RAG storage
            rag_text = f"""SEARCH QUERY: {search_query}
SEARCHED BY: {user_name}
DATE: {metadata['date']}
TYPE: {metadata['search_type']}
COLLECTION: {search_collection}

SEARCH RESULTS:
{search_results}

SUMMARY: Web search results for "{search_query}" containing current information and multiple source verification."""
            
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
                response = self.model_client.generate(
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
                            self.memory_service.store_interaction(user_name, user_text, response)
                        
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
            
            # Standard processing continues here...
            return "I'm processing your request..."
            
        except Exception as e:
