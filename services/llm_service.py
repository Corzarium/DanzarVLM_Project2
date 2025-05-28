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
from typing import Optional, Dict, List
import logging

from utils.text_utils import trim_sentences
from utils.ocr_utils import OCRProcessor
from core.game_profile import GameProfile
from .memory_service import MemoryEntry
from .rag_service import RAGService
from .model_client import ModelClient
from .danzar_factcheck import FactCheckService

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

    def __init__(self, app_context, audio_service, rag_service, model_client=None, default_collection: str = "multimodal_rag_default"):
        self.ctx = app_context
        self.audio_service = audio_service
        self.rag_service = rag_service
        self.model_client = model_client
        self.default_collection = default_collection
        self.logger = self.ctx.logger

        try:
            self.ocr_processor = OCRProcessor(app_context)
            self.logger.info("[LLMService] OCRProcessor initialized successfully.")
        except Exception as e:
            self.logger.error(f"[LLMService] Failed to initialize OCRProcessor: {e}", exc_info=True)
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
        self.fact_check = FactCheckService(
            rag_service=self.rag_service,
            model_client=self.model_client
        )

        self.logger.info("[LLMService] Initialized.")

    def _handle_profile_change_for_memory(self, new_profile: GameProfile):
        """Updates memory settings when profile changes."""
        if self.memory_service:
            self.memory_service._handle_profile_change(new_profile)

    def _strip_markdown_for_tts(self, text: str) -> str:
        if not text: return ""
        # Remove headings
        text = re.sub(r"^\s*#+\s+", "", text, flags=re.MULTILINE)
        # Remove bold/italics (asterisks and underscores)
        text = text.replace("*", "").replace("_", "")
        # Remove horizontal rules
        text = text.replace("---", "")
        # Remove list item markers (-, +, *) at the beginning of lines
        text = re.sub(r"^\s*[\*\-\+]+\s*", "", text, flags=re.MULTILINE)
        # Normalize excessive newlines/spaces
        text = re.sub(r"\s*\n\s*", "\n", text) # Multiple newlines with spaces to one
        text = re.sub(r" {2,}", " ", text) # Multiple spaces to one
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

        # Use the exact format Qwen2.5-VL expects
        image_content = f"USER: <img>{base64_jpeg}</img>\n\n{instruction_text}\n\nASSISTANT:"
        messages.append({"role": "user", "content": image_content})
        
        # Prepare the API payload
        payload = {
            "model": profile.vlm_model,
            "messages": messages,
            "temperature": float(profile.vlm_temperature),
            "max_tokens": int(profile.vlm_max_tokens),
            "images": [base64_jpeg]  # Always include images field for Qwen2.5-VL
        }
        
        self.logger.debug(f"[LLMService] Sending payload formatted for {model_name} model")
        resp = self._call_llm_api(payload)

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
                tts_audio = self.audio_service.fetch_tts_audio(text_for_tts_and_discord)
                if tts_audio: self.ctx.tts_queue.put_nowait(tts_audio)
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

    def handle_user_text_query(self, user_text: str, user_name: str = "User"):
        profile = self.ctx.active_profile
        gs = self.ctx.global_settings
        self.logger.info(f"[LLMService] User query from '{user_name}': \"{user_text}\"")

        # Store user query in memory
        if self.memory_service:
            user_memory = MemoryEntry(
                content=f"User ({user_name}): {user_text}",
                source="user_query",
                timestamp=time.time(),
                metadata={
                    "user": user_name,
                    "game": profile.game_name,
                    "type": "user_input"
                }
            )
            self.memory_service.store_memory(user_memory)

        # Get fact-checked response
        response_text = self.fact_check.fact_checked_generate(
            prompt=user_text,
            temperature=float(profile.conversational_temperature),
            max_tokens=int(profile.conversational_max_tokens),
        )

        # Store bot response in memory
        if self.memory_service:
            bot_memory = MemoryEntry(
                content=f"AI ({gs.get('BOT_NAME', 'DanzarVLM')}): {response_text}",
                source="bot_response",
                timestamp=time.time(),
                metadata={
                    "user_query": user_text,
                    "user": user_name,
                    "game": profile.game_name,
                    "type": "bot_response",
                    "llm_model": profile.conversational_llm_model
                }
            )
            self.memory_service.store_memory(bot_memory)

        # Output the response
        try:
            self.ctx.text_message_queue.put(f"💬 **{user_name}:** {user_text}\n🤖 **Reply:** {response_text}")
            if self.audio_service and gs.get("ENABLE_TTS_FOR_CHAT_REPLIES", True):
                tts_audio = self.audio_service.fetch_tts_audio(response_text)
                if tts_audio: self.ctx.tts_queue.put(tts_audio)
            self.logger.info(f"Conversational response sent to queue: \"{response_text[:100]}...\"")

        except queue.Full:
            self.logger.warning("[LLMService] Text message queue is full, response not sent.")
        
        self.ctx.last_interaction_time = time.time()
        if gs.get("CLEAR_CONVERSATION_FLAG_AFTER_REPLY", True):
            self.ctx.is_in_conversation.clear()
            self.logger.debug("Cleared is_in_conversation flag after user query.")

    def _run_image_visibility_diagnostic(self, base64_image: str, image_formats: dict):
        """
        Special diagnostic function to test if the model can see the image in any format
        """
        self.logger.info("[LLMService] Running deep image visibility diagnostic with all formats")
        profile = self.ctx.active_profile
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug_vlm_frames")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save diagnostic info
        diag_path = os.path.join(debug_dir, f"diagnostic_test_{int(time.time())}.txt")
        with open(diag_path, 'w') as f:
            f.write(f"VLM IMAGE VISIBILITY DIAGNOSTIC\n")
            f.write(f"==============================\n")
            f.write(f"Model: {profile.vlm_model}\n")
            f.write(f"Provider: {self.ctx.global_settings.get('VLM_PROVIDER', 'unknown')}\n")
            f.write(f"Base64 length: {len(base64_image)} chars\n")
            f.write(f"Available formats: {list(image_formats.keys())}\n")
            f.write(f"==============================\n\n")
        
        # Simple diagnostic prompt that should work with any VLM
        diagnostic_prompt = "This is an image visibility test. Can you see and describe the image shown? What is visible in this image? If you cannot see any image, please explicitly state 'I CANNOT SEE ANY IMAGE'."

        # Add more specialized formats for Qwen models
        is_qwen_model = "qwen" in profile.vlm_model.lower()
        if is_qwen_model:
            self.logger.info("[LLMService] Adding specialized Qwen formats")
            
            # Add very specific Qwen formats
            image_formats.update({
                "qwen-separate": f"{diagnostic_prompt}",  # Will add image separately in payload
                "qwen-msgimg1": f"<image>{base64_image}</image>{diagnostic_prompt}",  # Qwen variant 1
                "qwen-msgimg2": f"{diagnostic_prompt}\n<img>{base64_image}</img>",    # Qwen variant 2  
                "qwen-noprefix": f"<img>{base64_image.split(',')[-1]}</img>\n{diagnostic_prompt}" # No data URI prefix
            })
        
        # For llama.cpp specifically, add the image format recommendations from their docs
        if "llama" in profile.vlm_model.lower() or self.ctx.global_settings.get("VLM_PROVIDER", "").lower() == "llama.cpp":
            self.logger.info("[LLMService] Adding specialized llama.cpp formats")
            
            # Add formats based on llama.cpp specifics
            image_formats.update({
                "llama-cpp-1": f"<img src=\"data:image/jpeg;base64,{base64_image}\">\n\n{diagnostic_prompt}",
                "llama-cpp-2": f"USER: <img src=\"data:image/jpeg;base64,{base64_image}\">\n{diagnostic_prompt}\nASSISTANT:",
                "llama-cpp-3": f"<image>\n{base64_image}\n</image>\n\n{diagnostic_prompt}"
            })
        
        results = {}
        
        # Try each format
        for format_name, format_content in image_formats.items():
            self.logger.info(f"[LLMService] Testing format: {format_name}")
            
            # Use llama.cpp style embedding
            if "llama" in profile.vlm_model.lower() or self.ctx.global_settings.get("VLM_PROVIDER", "").lower() == "llama.cpp":
                # Different payload structures for llama.cpp
                if format_name == "qwen-separate":
                    # For Qwen with separate images field
                    payload = {
                        "model": profile.vlm_model,
                        "messages": [
                            {"role": "user", "content": diagnostic_prompt}
                        ],
                        "images": [base64_image],
                        "max_tokens": 300,
                        "temperature": 0.2,
                        "stream": False
                    }
                else:
                    # Standard approach with image in content
                    # Test with different structure if needed
                    use_alt_payload = format_name.startswith("llama-cpp")
                    
                    if use_alt_payload:
                        # Variation in payload structure
                        payload = {
                            "model": profile.vlm_model,
                            "prompt": format_content,
                            "max_tokens": 300,
                            "temperature": 0.2,
                            "stream": False
                        }
                    else:
                        # Standard payload structure
                        payload = {
                            "model": profile.vlm_model,
                            "messages": [
                                {"role": "user", "content": format_content}
                            ],
                            "max_tokens": 300,
                            "temperature": 0.2,
                            "stream": False
                        }
            else:
                # Use other model style (e.g. OpenAI style)
                if "claude" in profile.vlm_model.lower():
                    # Claude-specific format
                    payload = {
                        "model": profile.vlm_model,
                        "messages": [
                            {"role": "user", "content": [
                                {"type": "text", "text": diagnostic_prompt},
                                {"type": "image", "source": {
                                    "type": "base64", 
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }}
                            ]}
                        ],
                        "max_tokens": 300,
                        "temperature": 0.2,
                        "stream": False
                    }
                elif "gpt" in profile.vlm_model.lower():
                    # GPT-4V format
                    payload = {
                        "model": profile.vlm_model,
                        "messages": [
                            {"role": "user", "content": [
                                {"type": "text", "text": diagnostic_prompt},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }}
                            ]}
                        ],
                        "max_tokens": 300,
                        "temperature": 0.2,
                        "stream": False
                    }
                else:
                    # Generic format
                    payload = {
                        "model": profile.vlm_model,
                        "messages": [
                            {"role": "user", "content": format_content}
                        ],
                        "max_tokens": 300,
                        "temperature": 0.2,
                        "stream": False
                    }
            
            # Try with different API endpoints based on format
            endpoint = "chat/completions"  # default
            
            # For some llama.cpp formats, try both endpoints
            if format_name.startswith("llama-cpp"):
                endpoint = "completion"  # try basic completion endpoint
            
            # Make the API call
            self.logger.info(f"[LLMService] Sending diagnostic payload with format: {format_name} to endpoint {endpoint}")
            
            # Log the full payload for debugging
            self.logger.debug(f"[LLMService] Full payload: {json.dumps(payload, indent=2, default=str)[:1000]}...")
            
            # Save the payload to the diagnostic file
            with open(diag_path, 'a') as f:
                f.write(f"\nFORMAT: {format_name}\n")
                f.write(f"ENDPOINT: {endpoint}\n")
                f.write(f"PAYLOAD Summary:\n")
                
                if "messages" in payload:
                    f.write(f"- messages: {len(payload['messages'])} items\n")
                    if payload['messages'] and "content" in payload['messages'][0]:
                        content = payload['messages'][0]['content']
                        if isinstance(content, str):
                            f.write(f"- content length: {len(content)} chars\n")
                        else:
                            f.write(f"- content: list with {len(content)} items\n")
                elif "prompt" in payload:
                    f.write(f"- prompt length: {len(payload['prompt'])} chars\n")
                
                if "images" in payload:
                    f.write(f"- images field: {len(payload['images'])} items\n")
                    
            # Make the API call
            resp = self._call_llm_api(payload, endpoint=endpoint)
            
            if not resp or "choices" not in resp or not resp["choices"]:
                self.logger.warning(f"[LLMService] Empty/invalid diagnostic response for format: {format_name}")
                results[format_name] = "ERROR: Empty or invalid response"
                
                # Try again with alternate endpoint if using llama.cpp
                if "llama" in profile.vlm_model.lower() and endpoint == "chat/completions":
                    self.logger.info(f"[LLMService] Trying again with 'completion' endpoint")
                    alternate_resp = self._call_llm_api(payload, endpoint="completion")
                    if alternate_resp and "choices" in alternate_resp and alternate_resp["choices"]:
                        resp = alternate_resp
                    else:
                        continue
                else:
                    continue
            
            # Extract content based on response structure
            result = ""
            if "choices" in resp and resp["choices"]:
                if "message" in resp["choices"][0]:
                    result = resp["choices"][0].get("message", {}).get("content", "").strip()
                elif "text" in resp["choices"][0]:
                    # Some llama.cpp endpoints return text directly
                    result = resp["choices"][0].get("text", "").strip()
                    
            if not result:
                self.logger.warning(f"[LLMService] Empty diagnostic response content for format: {format_name}")
                results[format_name] = "ERROR: Empty response content"
                continue
            
            self.logger.info(f"[LLMService] Format {format_name} result: {result[:100]}...")
            results[format_name] = result
            
            # Append to the diagnostic file
            with open(diag_path, 'a') as f:
                f.write(f"\nRESPONSE:\n{result}\n")
                f.write(f"==============================\n")
            
            # Check if the model can see the image
            can_see = "cannot see" not in result.lower() and "no image" not in result.lower()
            if can_see:
                self.logger.info(f"[LLMService] SUCCESS! Format {format_name} appears to work.")
                
                # Update the global settings with the working format
                self.ctx.global_settings["VLM_IMAGE_FORMAT"] = format_name
                self.logger.info(f"[LLMService] Updated image format in global settings to: {format_name}")
                
                # Add this information to the diagnostic file
                with open(diag_path, 'a') as f:
                    f.write(f"\nSUCCESS! Format {format_name} appears to work.\n")
                    f.write(f"Updated global settings to use this format.\n")
                    
                # Only send a single success message to Discord
                try:
                    discord_msg = f"✅ Found working image format. VLM commentary will now work properly."
                    self.ctx.text_message_queue.put_nowait(discord_msg)
                except:
                    pass
                
                # Don't need to test all formats if we found one that works
                break
        
        # Final diagnostic info
        self.logger.info(f"[LLMService] Diagnostic completed. Results saved to {diag_path}")
        
        # Don't send diagnostic summary to Discord anymore
        formats_tested = len(results)
        formats_working = sum(1 for r in results.values() if "cannot see" not in r.lower() and "no image" not in r.lower())
        self.logger.info(f"[LLMService] Tested {formats_tested} formats. Working formats: {formats_working}.")

    def get_response(self, user: str, game: str, query: str) -> str:
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
                answer = self.model_client.generate(
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