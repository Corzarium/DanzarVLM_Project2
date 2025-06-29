#!/usr/bin/env python3
"""
Vision Integration Service for DanzarAI
=======================================

Integrates vision pipeline with DanzarAI's LLM, TTS, and memory services
to provide intelligent real-time commentary on visual content with CLIP video understanding.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json

from vision_pipeline import VisionPipeline, DetectionEvent, CLIPVideoUpdate

@dataclass
class CommentaryEvent:
    """A commentary event triggered by vision detection"""
    event_type: str  # 'detection', 'summary', 'alert', 'clip_insight', 'full_screenshot'
    content: str
    confidence: float
    timestamp: float
    metadata: Dict[str, Any]
    priority: str = 'normal'  # 'low', 'normal', 'high', 'critical'

class VisionIntegrationService:
    """Enhanced vision integration service with conversational commentary and full screenshot context."""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        
        # Vision commentary settings
        self.enable_commentary = app_context.global_settings.get('VISION_COMMENTARY', {}).get('enabled', True)
        self.commentary_frequency = app_context.global_settings.get('VISION_COMMENTARY', {}).get('frequency_seconds', 5.0)
        self.min_confidence = app_context.global_settings.get('VISION_COMMENTARY', {}).get('min_confidence', 0.6)
        self.conversation_mode = app_context.global_settings.get('VISION_COMMENTARY', {}).get('conversation_mode', True)
        
        # STM (Short-Term Memory) settings
        self.stm_enabled = app_context.global_settings.get('ENHANCED_MEMORY', {}).get('stm_enabled', True)
        
        # TEMPORARILY DISABLE CLIP TO RESTORE COMMENTARY
        self.clip_enabled = False  # Changed from True to False
        self.clip_update_interval = 10.0  # Increased interval
        self.last_clip_update = 0
        
        # Commentary state
        self.is_watching = False
        self.pending_commentary = False
        self.last_commentary_time = 0
        self.current_commentary_topic = None
        self.commentary_history = []
        
        # Event processing
        self.recent_detections = []
        self.max_recent_detections = 50  # Maximum number of recent detections to keep
        self.clip_insights = []
        self.pending_events = []
        self.pending_commentary_prompts = []
        self.pending_tts_calls = []
        self.pending_clip_updates = []
        
        # Callbacks
        self.text_callback = None
        self.tts_callback = None
        
        # Event processor
        self.event_processor_task = None
        
        # Game context
        self.game_context = {}
        
        # Vision context key for STM
        self.vision_context_key = "vision_context"
        
        # Screenshot settings
        self.full_screenshot_interval = 60.0  # Take full screenshot every 60 seconds (increased from 30)
        self.last_full_screenshot_time = 0
        
        # Get other services
        self.tts_service = getattr(self.app_context, 'tts_service', None)
        self.memory_service = getattr(self.app_context, 'memory_service', None)
        
        # COORDINATION: Get conversational AI service for synchronization
        self.conversational_ai_service = getattr(self.app_context, 'conversational_ai_service', None)
        if self.conversational_ai_service:
            if self.logger:
                self.logger.info("[VisionIntegration] ✅ Connected to Conversational AI Service for coordination")
        else:
            if self.logger:
                self.logger.warning("[VisionIntegration] ⚠️ No Conversational AI Service available - vision will operate independently")
        
        # Coordination settings
        self.coordination_enabled = True
        self.vision_commentary_cooldown = 3.0  # Seconds between vision commentary to avoid interrupting conversation
        self.last_vision_commentary_time = 0
        
        if self.logger:
            self.logger.info("[VisionIntegration] Vision integration service initialized")
            self.logger.info(f"[VisionIntegration] Commentary enabled: {self.enable_commentary}")
            self.logger.info(f"[VisionIntegration] CLIP TEMPORARILY DISABLED to restore commentary")
            self.logger.info(f"[VisionIntegration] Commentary frequency: {self.commentary_frequency}s")
            self.logger.info(f"[VisionIntegration] Min confidence: {self.min_confidence}")
    
    async def initialize(self) -> bool:
        """Initialize the vision integration service."""
        try:
            if self.logger:
                self.logger.info("[VisionIntegration] Initializing...")
            
            # Try to use existing vision pipeline from app context first
            if hasattr(self.app_context, 'vision_pipeline') and self.app_context.vision_pipeline:
                self.vision_pipeline = self.app_context.vision_pipeline
                if self.logger:
                    self.logger.info("[VisionIntegration] Using existing vision pipeline from app context")
            else:
                # Initialize vision pipeline WITHOUT CLIP callback (CLIP is disabled)
                if self.logger:
                    self.logger.info("[VisionIntegration] Creating new vision pipeline (CLIP disabled)...")
                self.vision_pipeline = VisionPipeline(
                    event_callback=self._handle_vision_event,
                    clip_callback=None,  # CLIP disabled to prevent event loop errors
                    config_path="config/vision_config.yaml"
                )
                
                if not await self.vision_pipeline.initialize():
                    if self.logger:
                        self.logger.error("[VisionIntegration] Failed to initialize vision pipeline")
                    return False
                
                # CRITICAL FIX: Start the vision pipeline immediately
                if self.logger:
                    self.logger.info("[VisionIntegration] 🔥 Starting vision pipeline immediately...")
                self.vision_pipeline.start()
                if self.logger:
                    self.logger.info("[VisionIntegration] ✅ Vision pipeline started successfully")
                
                # Store in app context for other services to use
                self.app_context.vision_pipeline = self.vision_pipeline
            
            # Initialize streaming LLM service
            try:
                from services.real_time_streaming_llm_service import RealTimeStreamingLLMService
                self.streaming_llm = RealTimeStreamingLLMService(
                    app_context=self.app_context,
                    model_client=getattr(self.app_context, 'model_client', None),
                    tts_service=getattr(self.app_context, 'tts_service', None)
                )
                
                if not await self.streaming_llm.initialize():
                    if self.logger:
                        self.logger.error("[VisionIntegration] Failed to initialize streaming LLM")
                    return False
            except ImportError:
                if self.logger:
                    self.logger.warning("[VisionIntegration] RealTimeStreamingLLMService not available")
                self.streaming_llm = None
            
            # Get other services
            self.tts_service = getattr(self.app_context, 'tts_service', None)
            self.memory_service = getattr(self.app_context, 'memory_service', None)
            
            # CRITICAL FIX: Force auto-start the event processor loop immediately
            if self.logger:
                self.logger.info("[VisionIntegration] 🔥 FORCING AUTO-START of event processor loop...")
            
            # Create default callbacks for auto-start
            async def default_text_callback(text: str):
                """Default text callback that logs to console and sends to Discord."""
                if self.logger:
                    self.logger.info(f"[VisionIntegration] 📝 COMMENTARY: {text[:200]}...")
                
                # Also send to Discord text channel if available
                try:
                    bot = getattr(self.app_context, 'bot', None)
                    if bot:
                        text_channel_id = bot.settings.get('DISCORD_TEXT_CHANNEL_ID')
                        if text_channel_id:
                            text_channel = bot.get_channel(int(text_channel_id))
                            if text_channel:
                                await text_channel.send(f"👁️ **Vision Commentary**: {text[:500]}...")
                                if self.logger:
                                    self.logger.info("[VisionIntegration] >>> DIRECT DISCORD MESSAGE SENT SUCCESSFULLY")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"[VisionIntegration] Discord send error: {e}")
            
            async def default_tts_callback(text: str):
                """Default TTS callback that uses TTS service and queues audio for Discord."""
                try:
                    if self.logger:
                        self.logger.info(f"[VisionIntegration] 🔊 TTS: {text[:200]}...")
                    
                    # Use TTS service if available
                    if hasattr(self.app_context, 'tts_service') and self.app_context.tts_service:
                        tts_audio = await self.app_context.tts_service.synthesize_speech(text)
                        if tts_audio:
                            # Queue TTS audio for ordered playback instead of direct playback
                            bot = getattr(self.app_context, 'bot', None)
                            if bot and hasattr(bot, '_queue_tts_audio'):
                                await bot._queue_tts_audio(tts_audio)
                                if self.logger:
                                    self.logger.info("[VisionIntegration] TTS audio queued for Discord playback")
                            else:
                                if self.logger:
                                    self.logger.warning("[VisionIntegration] Bot or TTS queue method not available")
                        else:
                            if self.logger:
                                self.logger.warning("[VisionIntegration] TTS service returned no audio")
                    else:
                        if self.logger:
                            self.logger.warning("[VisionIntegration] No TTS service available")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"[VisionIntegration] TTS callback error: {e}", exc_info=True)
            
            # Force start watching with default callbacks
            self.text_callback = default_text_callback
            self.tts_callback = default_tts_callback
            
            # Set watching flag and start event processor
            self.is_watching = True
            
            # Start the event processor task
            if self.event_processor_task is None or self.event_processor_task.done():
                if self.logger:
                    self.logger.info("[VisionIntegration] 🔥 Starting event processor task...")
                self.event_processor_task = asyncio.create_task(self._event_processor_loop())
                
                # Wait a moment to ensure the task starts
                await asyncio.sleep(0.1)
                
                if self.event_processor_task and not self.event_processor_task.done():
                    if self.logger:
                        self.logger.info("[VisionIntegration] ✅ Event processor task started successfully")
                else:
                    if self.logger:
                        self.logger.error("[VisionIntegration] ❌ Event processor task failed to start")
                    return False
            
            if self.logger:
                self.logger.info("[VisionIntegration] ✅ Initialized successfully with FORCED auto-start commentary")
                self.logger.info("[VisionIntegration] ✅ Vision information will now be passed to VLM automatically")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Initialization error: {e}", exc_info=True)
            return False
    
    async def start_watching(self, text_callback=None, tts_callback=None) -> bool:
        """Start watching for vision events and generating commentary."""
        try:
            if self.is_watching:
                if self.logger:
                    self.logger.info("[VisionIntegration] Already watching")
                return True
            
            if self.logger:
                self.logger.info("[VisionIntegration] Starting vision commentary...")
                self.logger.info(f"[VisionIntegration] Text callback provided: {text_callback is not None}")
                self.logger.info(f"[VisionIntegration] TTS callback provided: {tts_callback is not None}")
                if text_callback:
                    self.logger.info(f"[VisionIntegration] Text callback type: {type(text_callback)}")
                    self.logger.info(f"[VisionIntegration] Text callback is coroutine: {asyncio.iscoroutinefunction(text_callback)}")
            
            # Set callbacks
            self.text_callback = text_callback
            self.tts_callback = tts_callback
            
            # Test callbacks immediately to verify they work
            if self.logger:
                self.logger.info("[VisionIntegration] Testing callbacks immediately...")
            
            if self.text_callback and callable(self.text_callback):
                try:
                    if self.logger:
                        self.logger.info("[VisionIntegration] Testing text callback...")
                    test_message = "👁️ Vision commentary system initialized and ready!"
                    if asyncio.iscoroutinefunction(self.text_callback):
                        await self.text_callback(test_message)
                    else:
                        self.text_callback(test_message)
                    if self.logger:
                        self.logger.info("[VisionIntegration] ✅ Text callback test successful")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"[VisionIntegration] ❌ Text callback test failed: {e}", exc_info=True)
            
            if self.tts_callback and callable(self.tts_callback):
                try:
                    if self.logger:
                        self.logger.info("[VisionIntegration] Testing TTS callback...")
                    test_message = "Vision commentary system initialized."
                    if asyncio.iscoroutinefunction(self.tts_callback):
                        await self.tts_callback(test_message)
                    else:
                        self.tts_callback(test_message)
                    if self.logger:
                        self.logger.info("[VisionIntegration] ✅ TTS callback test successful")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"[VisionIntegration] ❌ TTS callback test failed: {e}", exc_info=True)
            
            # Start the vision pipeline
            if self.vision_pipeline:
                if self.logger:
                    self.logger.info("[VisionIntegration] Starting vision pipeline...")
                self.vision_pipeline.start()
                if self.logger:
                    self.logger.info("[VisionIntegration] Vision pipeline started successfully")
            else:
                if self.logger:
                    self.logger.error("[VisionIntegration] No vision pipeline available")
                return False
            
            # Start event processor - FIXED: Ensure proper task creation
            if self.logger:
                self.logger.info("[VisionIntegration] Starting event processor loop...")
            
            # Cancel any existing task
            if self.event_processor_task and not self.event_processor_task.done():
                self.event_processor_task.cancel()
                try:
                    await self.event_processor_task
                except asyncio.CancelledError:
                    pass
            
            # Create new event processor task
            self.event_processor_task = asyncio.create_task(self._event_processor_loop())
            
            if self.logger:
                self.logger.info(f"[VisionIntegration] Event processor task created: {self.event_processor_task}")
                self.logger.info(f"[VisionIntegration] Event processor task done: {self.event_processor_task.done()}")
            
            # Generate initial commentary
            initial_prompt = "I'm now watching and ready to provide commentary on what I see! I'll be more conversational and wait for your responses."
            await self._generate_commentary(initial_prompt)
            
            self.is_watching = True
            self.enable_commentary = True
            
            if self.logger:
                self.logger.info("[VisionIntegration] Vision commentary started successfully")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Failed to start watching: {e}", exc_info=True)
            return False
    
    async def stop_watching(self) -> bool:
        """Stop watching for vision events."""
        try:
            if not self.is_watching:
                if self.logger:
                    self.logger.info("[VisionIntegration] Not currently watching")
                return True
            
            if self.logger:
                self.logger.info("[VisionIntegration] Stopping vision commentary...")
            
            # Stop the vision pipeline
            if self.vision_pipeline:
                if self.logger:
                    self.logger.info("[VisionIntegration] Stopping vision pipeline...")
                self.vision_pipeline.stop()
                if self.logger:
                    self.logger.info("[VisionIntegration] Vision pipeline stopped successfully")
            
            # Cancel event processor
            if self.event_processor_task and not self.event_processor_task.done():
                self.event_processor_task.cancel()
                try:
                    await self.event_processor_task
                except asyncio.CancelledError:
                    pass
            
            # Generate final commentary
            final_prompt = "I'm stopping my commentary now. Thanks for watching with me!"
            await self._generate_commentary(final_prompt)
            
            self.is_watching = False
            self.enable_commentary = False
            
            if self.logger:
                self.logger.info("[VisionIntegration] Vision commentary stopped successfully")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Failed to stop watching: {e}")
            return False
    
    def _handle_vision_event(self, event: DetectionEvent):
        """Handle incoming vision events and store them in RAG memory."""
        try:
            if self.logger:
                self.logger.debug(f"[VisionIntegration] Handling vision event: {event.object_type} - {event.label} (conf: {event.confidence:.2f})")
            
            # Add to recent detections
            self.recent_detections.append(event)
            
            # Keep only recent detections
            if len(self.recent_detections) > self.max_recent_detections:
                self.recent_detections = self.recent_detections[-self.max_recent_detections:]
            
            # Store vision event in RAG memory for learning
            self._store_vision_event_in_rag(event)
            
            # Update game context
            self.game_context.update({
                'last_event': event,
                'recent_detections': self.recent_detections[-10:],  # Last 10 detections
                'yolo_detections': [d for d in self.recent_detections if d.object_type == 'yolo'],
                'ocr_results': [d.label for d in self.recent_detections if d.object_type == 'ocr' and d.label.strip()],
                'template_matches': [d for d in self.recent_detections if d.object_type == 'template'],
                'clip_insights': [d for d in self.recent_detections if d.object_type == 'clip']
            })
            
            # Store vision context in STM
            if self.stm_enabled and self.memory_service:
                self._store_vision_context_in_stm()
            
            # Check if we should generate commentary for this event
            if self.logger:
                self.logger.debug(f"[VisionIntegration] Checking commentary trigger for {event.object_type}: {event.label} (conf: {event.confidence:.2f})")
            
            if self._should_generate_commentary(event):
                if self.logger:
                    self.logger.info(f"[VisionIntegration] Commentary trigger detected: {event.object_type}: {event.label}")
                
                # Store event for processing by the main event loop
                self.pending_events.append(event)
                if self.logger:
                    self.logger.info(f"[VisionIntegration] Added event to pending_events (total: {len(self.pending_events)})")
            else:
                if self.logger:
                    self.logger.debug(f"[VisionIntegration] Commentary trigger rejected for {event.object_type}: {event.label}")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Vision event handling error: {e}")
    
    def _store_vision_event_in_rag(self, event: DetectionEvent):
        """Store vision event in RAG memory for learning and context retrieval."""
        try:
            # Get RAG service from app context
            rag_service = getattr(self.app_context, 'rag_service_instance', None)
            if not rag_service:
                return
            
            # Create vision event memory content
            memory_content = f"""VISION EVENT DETECTED
Type: {event.object_type}
Label: {event.label}
Confidence: {event.confidence:.3f}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
Game Context: {self.game_context.get('current_game', 'unknown')}
Location: {event.metadata.get('location', 'unknown') if hasattr(event, 'metadata') else 'unknown'}"""
            
            # Determine collection based on event type
            collection_name = f"vision_events_{event.object_type}"
            
            # Store in RAG
            success = rag_service.ingest_text(
                collection=collection_name,
                text=memory_content,
                metadata={
                    'event_type': event.object_type,
                    'label': event.label,
                    'confidence': event.confidence,
                    'timestamp': time.time(),
                    'game_context': self.game_context.get('current_game', 'unknown'),
                    'importance': event.confidence  # Use confidence as importance
                }
            )
            
            # Also store in memory service if available
            if self.memory_service:
                from services.memory_service import MemoryEntry
                memory_entry = MemoryEntry(
                    content=memory_content,
                    source=f"vision_{event.object_type}",
                    timestamp=time.time(),
                    metadata={
                        'event_type': event.object_type,
                        'label': event.label,
                        'confidence': event.confidence,
                        'game_context': self.game_context.get('current_game', 'unknown')
                    },
                    importance_score=event.confidence
                )
                self.memory_service.store_memory(memory_entry)
            
            if success and self.logger:
                self.logger.debug(f"[VisionIntegration] Stored vision event in RAG: {event.object_type} - {event.label}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Error storing vision event in RAG: {e}")
    
    def _store_vision_context_in_stm(self):
        """Store current vision context in short-term memory."""
        try:
            if not self.memory_service or not hasattr(self.memory_service, 'add_to_stm'):
                return
            
            # Create vision context summary
            context_summary = {
                'timestamp': time.time(),
                'total_detections': len(self.recent_detections),
                'recent_objects': [],
                'recent_text': [],
                'game_context': self.game_context
            }
            
            # Add recent high-confidence detections
            for detection in self.recent_detections[-5:]:
                if detection.confidence > 0.7:
                    context_summary['recent_objects'].append({
                        'type': detection.object_type,
                        'label': detection.label,
                        'confidence': detection.confidence
                    })
            
            # Add recent OCR text
            ocr_texts = [d.label for d in self.recent_detections if d.object_type == 'ocr' and d.label.strip()]
            if ocr_texts:
                context_summary['recent_text'] = ocr_texts[-3:]  # Last 3 OCR results
            
            # Store in STM
            self.memory_service.add_to_stm(
                key=self.vision_context_key,
                value=context_summary,
                ttl=60  # Keep for 60 seconds
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Failed to store vision context in STM: {e}")
    
    def _handle_clip_update(self, clip_update: CLIPVideoUpdate):
        """Handle CLIP video understanding updates with monochrome images."""
        # CLIP IS COMPLETELY DISABLED TO RESTORE COMMENTARY
        return
    
    async def _send_clip_to_vlm(self, clip_update: CLIPVideoUpdate):
        """Send CLIP insights to VLM for enhanced understanding."""
        try:
            if not clip_update.visual_descriptions:
                return
            
            # Create CLIP insight prompt
            insight_text = ", ".join(clip_update.visual_descriptions[:3])  # Top 3 insights
            
            prompt = f"""CLIP VIDEO INSIGHT:
{insight_text}

This represents the current visual understanding of the scene. Use this context to enhance your commentary about what's happening in the game."""
            
            # Store in STM for context
            if self.stm_enabled and self.memory_service:
                self.memory_service.add_to_stm(
                    key="clip_insight",
                    value={
                        'insights': clip_update.clip_insights,
                        'visual_descriptions': clip_update.visual_descriptions,
                        'confidence': max(clip_update.confidence_scores.values()) if clip_update.confidence_scores else 0.5,
                        'timestamp': time.time()
                    },
                    ttl=30
                )
            
            if self.logger:
                self.logger.debug(f"[VisionIntegration] CLIP insight sent to VLM: {insight_text}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Failed to send CLIP to VLM: {e}")
    
    async def _event_processor_loop(self):
        """Main event processing loop for commentary generation."""
        try:
            if self.logger:
                self.logger.info("[VisionIntegration] Event processor loop started")
                self.logger.info(f"[VisionIntegration] Event processor running: {self.is_watching}")
                self.logger.info(f"[VisionIntegration] Shutdown event set: {self.app_context.shutdown_event.is_set()}")
            
            loop_count = 0
            while self.is_watching and not self.app_context.shutdown_event.is_set():
                try:
                    loop_count += 1
                    
                    # Log loop status every 10 iterations
                    if loop_count % 10 == 0 and self.logger:
                        self.logger.info(f"[VisionIntegration] Event processor loop iteration {loop_count}")
                        self.logger.info(f"[VisionIntegration] Pending events: {len(self.pending_events)}")
                        self.logger.info(f"[VisionIntegration] Pending prompts: {len(self.pending_commentary_prompts) if hasattr(self, 'pending_commentary_prompts') else 'N/A'}")
                    
                    # Process pending commentary prompts
                    if hasattr(self, 'pending_commentary_prompts') and self.pending_commentary_prompts:
                        pending_prompts = self.pending_commentary_prompts.copy()
                        self.pending_commentary_prompts.clear()
                        
                        if self.logger:
                            self.logger.info(f"[VisionIntegration] 🔥 Processing {len(pending_prompts)} pending commentary prompts")
                        
                        for i, prompt in enumerate(pending_prompts):
                            if self.logger:
                                self.logger.info(f"[VisionIntegration] 🔥 Generating commentary for prompt {i+1}/{len(pending_prompts)}: {prompt[:100]}...")
                            await self._generate_commentary(prompt)
                    else:
                        # Debug: log when no prompts are found
                        if self.logger and loop_count % 20 == 0:  # Log less frequently
                            self.logger.debug(f"[VisionIntegration] No pending commentary prompts to process (count: {len(self.pending_commentary_prompts) if hasattr(self, 'pending_commentary_prompts') else 'N/A'})")
                            if hasattr(self, 'pending_commentary_prompts'):
                                self.logger.debug(f"[VisionIntegration] pending_commentary_prompts exists with {len(self.pending_commentary_prompts)} items")
                            else:
                                self.logger.debug(f"[VisionIntegration] pending_commentary_prompts attribute does not exist")
                    
                    # Check for pending commentary events - FIXED: Better logging and error handling
                    if hasattr(self, 'pending_events') and self.pending_events:
                        pending_events = self.pending_events.copy()
                        self.pending_events.clear()
                        
                        if self.logger:
                            self.logger.info(f"[VisionIntegration] 🔥 Processing {len(pending_events)} pending events")
                        
                        # Process each event
                        for event in pending_events:
                            try:
                                if self.logger:
                                    self.logger.info(f"[VisionIntegration] 🔥 Processing commentary trigger for {event.object_type}: {event.label}")
                                await self._process_commentary_trigger(event)
                            except Exception as event_error:
                                if self.logger:
                                    self.logger.error(f"[VisionIntegration] Error processing event {event.object_type}: {event.label}: {event_error}", exc_info=True)
                    else:
                        # Debug: log when no events are found (less frequently)
                        if self.logger and loop_count % 20 == 0:
                            self.logger.debug(f"[VisionIntegration] No pending events to process (count: {len(self.pending_events) if hasattr(self, 'pending_events') else 'N/A'})")
                    
                    # Check for pending CLIP updates
                    if self.pending_clip_updates:
                        pending_clips = self.pending_clip_updates.copy()
                        self.pending_clip_updates.clear()
                        
                        if self.logger:
                            self.logger.info(f"[VisionIntegration] Processing {len(pending_clips)} pending CLIP updates")
                        
                        # Process each CLIP update
                        for clip_insight in pending_clips:
                            if not clip_insight.get('processed', False):
                                # Create a mock CLIP update for processing
                                class MockCLIPUpdate:
                                    def __init__(self, insight):
                                        self.clip_insights = insight['insights']
                                        self.visual_descriptions = insight['visual_descriptions']
                                        self.confidence_scores = {'overall': insight['confidence']}
                                
                                mock_update = MockCLIPUpdate(clip_insight)
                                await self._send_clip_to_vlm(mock_update)
                                clip_insight['processed'] = True
                    
                    # Process pending TTS calls
                    if hasattr(self, 'pending_tts_calls') and self.pending_tts_calls:
                        pending_tts = self.pending_tts_calls.copy()
                        self.pending_tts_calls.clear()
                        
                        if self.logger:
                            self.logger.info(f"[VisionIntegration] Processing {len(pending_tts)} pending TTS calls")
                        
                        for tts_text in pending_tts:
                            if self.tts_callback and callable(self.tts_callback):
                                try:
                                    if self.logger:
                                        self.logger.info(f"[VisionIntegration] Calling TTS callback with: '{tts_text[:50]}...'")
                                    
                                    # Check voice connection status
                                    if hasattr(self.app_context, 'bot') and self.app_context.bot:
                                        voice_connected = False
                                        for guild in self.app_context.bot.guilds:
                                            if guild.voice_client and guild.voice_client.is_connected():
                                                voice_connected = True
                                                channel_name = getattr(guild.voice_client.channel, 'name', 'Unknown')
                                                if self.logger:
                                                    self.logger.info(f"[VisionIntegration] Voice connected to: {channel_name}")
                                                break
                                        
                                        if not voice_connected:
                                            if self.logger:
                                                self.logger.warning("[VisionIntegration] No voice connection available for TTS playback")
                                    
                                    # Call the TTS callback directly since it's async and now uses queue
                                    await self.tts_callback(tts_text)
                                    if self.logger:
                                        self.logger.info("[VisionIntegration] TTS callback completed successfully")
                                    
                                except Exception as e:
                                    if self.logger:
                                        self.logger.error(f"[VisionIntegration] TTS callback error: {e}")
                    
                    # Wait before next iteration
                    await asyncio.sleep(0.5)
                    
                except asyncio.CancelledError:
                    if self.logger:
                        self.logger.info("[VisionIntegration] Event processor cancelled")
                    break
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"[VisionIntegration] Event processor error: {e}", exc_info=True)
                    await asyncio.sleep(1.0)
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Event processor loop error: {e}", exc_info=True)
        finally:
            if self.logger:
                self.logger.info("[VisionIntegration] Event processor loop ended")
    
    def _should_generate_commentary(self, event: DetectionEvent) -> bool:
        """Determine if commentary should be generated for this event."""
        current_time = time.time()
        
        # Basic frequency check
        if current_time - self.last_commentary_time < self.commentary_frequency:
            return False
        
        # Confidence check
        if event.confidence < self.min_confidence:
            return False
        
        # COORDINATION: Check if conversational AI is in a speaking state
        if self.coordination_enabled and self.conversational_ai_service:
            try:
                # Check if conversational AI is currently speaking
                if hasattr(self.conversational_ai_service, 'conversation_state'):
                    conversation_state = self.conversational_ai_service.conversation_state
                    if conversation_state in ['speaking', 'thinking']:
                        if self.logger:
                            self.logger.debug(f"[VisionIntegration] Skipping commentary - Conversational AI is {conversation_state}")
                        return False
                
                # Check if there's been recent conversation activity
                if hasattr(self.conversational_ai_service, 'last_speech_time'):
                    time_since_speech = current_time - self.conversational_ai_service.last_speech_time
                    if time_since_speech < self.vision_commentary_cooldown:
                        if self.logger:
                            self.logger.debug(f"[VisionIntegration] Skipping commentary - Recent conversation activity ({time_since_speech:.1f}s ago)")
                        return False
                        
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"[VisionIntegration] Error checking conversation state: {e}")
        
        # Check for significant change
        if not self._has_significant_change(event):
            return False
        
        # Update last commentary time
        self.last_commentary_time = current_time
        self.last_vision_commentary_time = current_time
        
        return True
    
    def _has_significant_change(self, event: DetectionEvent) -> bool:
        """Force commentary for every event (debug mode)."""
        if self.logger:
            self.logger.info(f"[VisionIntegration] [DEBUG] Forcing significant change for {event.object_type}: {event.label}")
        return True
    
    def _create_unified_prompt(self, event_data: Dict[str, Any], game_context: str = "unknown") -> Tuple[Optional[str], Optional[str]]:
        """Create a unified prompt for vision commentary using profile-based prompts."""
        try:
            # Capture fresh screenshot for current context
            screenshot_b64 = self._capture_current_screenshot()
            if not screenshot_b64:
                self.logger.warning("[VisionIntegration] Failed to capture screenshot for commentary")
                return None, None
            
            # Get recent vision summary for context
            recent_summary = self.get_recent_vision_summary()
            
            # Get the system prompt from the game profile
            system_prompt = "You are a game commentator. Describe what you see."  # Default fallback
            user_prompt_template = "What is happening in this game scene?"  # Default fallback
            
            try:
                if hasattr(self.app_context, 'active_profile') and self.app_context.active_profile:
                    # Access GameProfile attributes directly
                    system_prompt = getattr(self.app_context.active_profile, 'system_prompt_commentary', system_prompt)
                    user_prompt_template = getattr(self.app_context.active_profile, 'user_prompt_template_commentary', user_prompt_template)
                    self.logger.info(f"[VisionIntegration] Using profile-based prompts for {game_context}")
                else:
                    self.logger.warning("[VisionIntegration] No active profile found, using default prompts")
            except Exception as e:
                self.logger.error(f"[VisionIntegration] Error loading profile prompts: {e}")
            
            # Format the user prompt with event data
            try:
                user_prompt = user_prompt_template.format(event_data=json.dumps(event_data, indent=2))
            except Exception as e:
                self.logger.error(f"[VisionIntegration] Error formatting user prompt: {e}")
                user_prompt = f"What is happening in this game scene? Event: {json.dumps(event_data, indent=2)}"
            
            # Create the unified prompt combining system and user prompts
            prompt = f"""{system_prompt}

CURRENT GAME: {game_context}

RECENT ACTIVITY SUMMARY:
{recent_summary}

{user_prompt}

Here's the current game screenshot:"""

            return prompt, screenshot_b64
            
        except Exception as e:
            self.logger.error(f"[VisionIntegration] Error creating unified prompt: {e}")
            return None, None
    
    def get_recent_vision_summary(self) -> str:
        """Get a summary of recent vision activity for context."""
        try:
            if not self.recent_detections:
                return "No recent activity detected."
            
            # Get last 5 detections
            recent = self.recent_detections[-5:]
            
            summary_parts = []
            for detection in recent:
                time_ago = time.time() - detection.timestamp
                if time_ago < 60:
                    time_str = f"{int(time_ago)}s ago"
                else:
                    time_str = f"{int(time_ago/60)}m ago"
                
                summary_parts.append(f"- {detection.object_type.upper()}: {detection.label} ({time_str})")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"[VisionIntegration] Error getting recent vision summary: {e}")
            return "Recent activity: Unable to retrieve summary."
    
    async def _process_commentary_trigger(self, trigger_event: DetectionEvent):
        """Process a commentary trigger with unified prompt."""
        try:
            if self.logger:
                self.logger.info(f"[VisionIntegration] 🔥 _process_commentary_trigger CALLED for {trigger_event.object_type}: {trigger_event.label}")
            
            self.last_commentary_time = time.time()
            self.pending_commentary = True
            self.current_commentary_topic = f"{trigger_event.object_type}: {trigger_event.label}"
            
            # Try to create unified prompt
            try:
                analysis = self._analyze_recent_detections()
                prompt, screenshot_b64 = self._create_unified_prompt(analysis, self.game_context.get('current_game', 'unknown'))
                
                if not prompt or not screenshot_b64:
                    self.logger.warning("[VisionIntegration] Failed to create unified prompt or capture screenshot")
                    return
                
                if self.logger:
                    self.logger.info(f"[VisionIntegration] 🔥 Created unified prompt: {prompt[:200]}...")
                    self.logger.info(f"[VisionIntegration] 📸 Screenshot captured: {len(screenshot_b64)} chars")
                
                # Create VLM message with image using the correct format for the model client
                # The model client expects the image to be embedded in the text with <image> tags
                vlm_prompt = f"{prompt}\n\n<image>{screenshot_b64}</image>"
                
                messages = [
                    {
                        "role": "user",
                        "content": vlm_prompt
                    }
                ]
                
                # Use the correct model client method (generate, not chat_completion)
                if self.logger:
                    self.logger.info(f"[VisionIntegration] 🔥 Calling model_client.generate with VLM prompt")
                
                response = await self.app_context.model_client.generate(
                    messages=messages,
                    max_tokens=300,
                    temperature=0.7,
                    model="Qwen2.5-VL-7B-Instruct"
                )
                
                if response and response.strip():
                    commentary = response.strip()
                    self.logger.info(f"[VisionIntegration] ✅ Qwen2.5-VL commentary generated: {len(commentary)} chars")
                    self.logger.info(f"[VisionIntegration] 📝 Commentary: {commentary[:100]}...")
                    
                    # Send commentary through callbacks directly
                    await self._send_commentary_direct(commentary, analysis)
                else:
                    self.logger.warning("[VisionIntegration] No response from Qwen2.5-VL model")
                
                # Store the prompt for processing in the event loop (inside the try block where prompt is defined)
                if not hasattr(self, 'pending_commentary_prompts'):
                    self.pending_commentary_prompts = []
                
                self.pending_commentary_prompts.append(prompt)
                
                if self.logger:
                    self.logger.info(f"[VisionIntegration] 🔥 Stored commentary prompt for processing (total prompts: {len(self.pending_commentary_prompts)})")
                    self.logger.info(f"[VisionIntegration] 🔥 Prompt length: {len(prompt)} chars")
                    
            except Exception as e:
                self.logger.error(f"[VisionIntegration] Error in commentary generation: {e}")
                import traceback
                self.logger.error(f"[VisionIntegration] Commentary error traceback: {traceback.format_exc()}")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Commentary processing error: {e}")
                import traceback
                self.logger.error(f"[VisionIntegration] Error traceback: {traceback.format_exc()}")
    
    async def _send_commentary_direct(self, commentary: str, analysis: Dict[str, Any]):
        """Send commentary through text and TTS callbacks directly."""
        try:
            # Send to text callback
            if self.text_callback and callable(self.text_callback):
                try:
                    if asyncio.iscoroutinefunction(self.text_callback):
                        await self.text_callback(commentary)
                    else:
                        self.text_callback(commentary)
                    if self.logger:
                        self.logger.info(f"[VisionIntegration] ✅ Text callback completed")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"[VisionIntegration] Text callback error: {e}")
            
            # Send to TTS callback
            if self.tts_callback and callable(self.tts_callback):
                try:
                    if asyncio.iscoroutinefunction(self.tts_callback):
                        await self.tts_callback(commentary)
                    else:
                        self.tts_callback(commentary)
                    if self.logger:
                        self.logger.info(f"[VisionIntegration] ✅ TTS callback completed")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"[VisionIntegration] TTS callback error: {e}")
                        
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Error sending commentary: {e}")
    
    def _analyze_recent_detections(self) -> Dict[str, Any]:
        """Analyze recent detections to understand current context."""
        if not self.recent_detections:
            return {}
        
        analysis = {
            'total_detections': len(self.recent_detections),
            'object_types': {},
            'ocr_texts': [],
            'high_confidence_objects': [],
            'time_span': 0
        }
        
        if self.recent_detections:
            analysis['time_span'] = (
                self.recent_detections[-1].timestamp - 
                self.recent_detections[0].timestamp
            )
        
        for detection in self.recent_detections:
            # Count object types
            obj_type = detection.object_type
            analysis['object_types'][obj_type] = analysis['object_types'].get(obj_type, 0) + 1
            
            # Collect OCR texts
            if obj_type == 'ocr' and detection.label.strip():
                analysis['ocr_texts'].append(detection.label.strip())
            
            # Collect high confidence objects
            if detection.confidence > 0.8:
                analysis['high_confidence_objects'].append({
                    'type': obj_type,
                    'label': detection.label,
                    'confidence': detection.confidence
                })
        
        return analysis
    
    async def _generate_commentary(self, prompt: str):
        """Generate commentary using the model client directly, with detailed logging."""
        try:
            if self.logger:
                self.logger.info(f"[VisionIntegration] 🔥 _generate_commentary CALLED with prompt length: {len(prompt)}")
                self.logger.info(f"[VisionIntegration] 🔥 Model client available: {hasattr(self.app_context, 'model_client')}")
                if hasattr(self.app_context, 'model_client'):
                    self.logger.info(f"[VisionIntegration] 🔥 Model client type: {type(self.app_context.model_client)}")
            
            if not hasattr(self.app_context, 'model_client') or not self.app_context.model_client:
                if self.logger:
                    self.logger.warning("[VisionIntegration] No model client available for commentary generation")
                return
            
            # Log prompt details
            if self.logger:
                self.logger.info(f"[VisionIntegration] Generating commentary with prompt length: {len(prompt)}")
                self.logger.info(f"[VisionIntegration] Prompt preview: {prompt[:300].replace(chr(10), ' ')} ...")
            
            # Generate response using model client
            # Use profile-based system prompt instead of hardcoded one
            system_prompt = self.app_context.active_profile.system_prompt_commentary
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            if self.logger:
                self.logger.info(f"[VisionIntegration] 🔥 Calling model_client.generate with {len(messages)} messages")
            
            response = await self.app_context.model_client.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=512,
                model="Qwen2.5-VL-7B-Instruct"
            )
            
            # Log model response
            if self.logger:
                if response:
                    self.logger.info(f"[VisionIntegration] Model client response: {response.strip()[:100]}...")
                else:
                    self.logger.warning("[VisionIntegration] Model client returned empty response")
            
            # Process response
            if response and response.strip():
                response = response.strip()
                
                if self.logger:
                    self.logger.info(f"[VisionIntegration] Processing commentary response: '{response.strip()[:50]}...'")
                    self.logger.info(f"[VisionIntegration] Text callback available: {self.text_callback is not None}")
                    self.logger.info(f"[VisionIntegration] TTS callback available: {self.tts_callback is not None}")
                    if self.text_callback:
                        self.logger.info(f"[VisionIntegration] Text callback type: {type(self.text_callback)}")
                        self.logger.info(f"[VisionIntegration] Text callback is coroutine: {asyncio.iscoroutinefunction(self.text_callback)}")
                
                # Always send directly to Discord text channel (like voice chat)
                text_sent = False
                try:
                    bot = getattr(self.app_context, 'bot', None)
                    if bot:
                        # Get text channel ID from bot's settings
                        text_channel_id = bot.settings.get('DISCORD_TEXT_CHANNEL_ID')
                        if text_channel_id:
                            text_channel = bot.get_channel(int(text_channel_id))
                            if text_channel:
                                if self.logger:
                                    self.logger.info(f"[VisionIntegration] >>> SENDING DIRECT DISCORD MESSAGE to {text_channel.name}")
                                await text_channel.send(f"👁️ **Vision Commentary**: {response[:500]}...")
                                if self.logger:
                                    self.logger.info(f"[VisionIntegration] >>> DIRECT DISCORD MESSAGE SENT SUCCESSFULLY")
                                text_sent = True
                            else:
                                if self.logger:
                                    self.logger.warning(f"[VisionIntegration] Configured text channel ID {text_channel_id} not found")
                        else:
                            if self.logger:
                                self.logger.warning("[VisionIntegration] DISCORD_TEXT_CHANNEL_ID not configured in bot settings")
                    else:
                        if self.logger:
                            self.logger.warning("[VisionIntegration] No bot available for direct Discord message")
                except Exception as fallback_e:
                    if self.logger:
                        self.logger.error(f"[VisionIntegration] Direct Discord message failed: {fallback_e}", exc_info=True)
                        self.logger.error(f"[VisionIntegration] Direct send error type: {type(fallback_e)}")
                # Optionally, still call the callback for extensibility
                if self.text_callback and callable(self.text_callback):
                    try:
                        if asyncio.iscoroutinefunction(self.text_callback):
                            await self.text_callback(response)
                        else:
                            self.text_callback(response)
                        if self.logger:
                            self.logger.info(f"[VisionIntegration] >>> TEXT CALLBACK COMPLETED SUCCESSFULLY (after direct send)")
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"[VisionIntegration] Error calling text callback: {e}", exc_info=True)
                
                # Call TTS callback if available
                if self.tts_callback and callable(self.tts_callback):
                    try:
                        if self.logger:
                            self.logger.info(f"[VisionIntegration] Calling TTS callback with: '{response.strip()[:50]}...'")
                        
                        # Check voice connection status
                        if hasattr(self.app_context, 'bot') and self.app_context.bot:
                            voice_connected = False
                            for guild in self.app_context.bot.guilds:
                                if guild.voice_client and guild.voice_client.is_connected():
                                    voice_connected = True
                                    channel_name = getattr(guild.voice_client.channel, 'name', 'Unknown')
                                    if self.logger:
                                        self.logger.info(f"[VisionIntegration] Voice connected to: {channel_name}")
                                    break
                            
                            if not voice_connected:
                                if self.logger:
                                    self.logger.warning("[VisionIntegration] No voice connection available for TTS playback")
                        
                        # Call the TTS callback directly since it's async and now uses queue
                        await self.tts_callback(response)
                        if self.logger:
                            self.logger.info("[VisionIntegration] TTS callback completed successfully")
                            
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"[VisionIntegration] TTS callback error: {e}")
                
                # Reset pending commentary flag
                self.pending_commentary = False
                self.current_commentary_topic = None
                
            else:
                if self.logger:
                    self.logger.warning("[VisionIntegration] No valid response to process")
                # Reset pending commentary flag even if no response
                self.pending_commentary = False
                self.current_commentary_topic = None
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Commentary generation error: {e}", exc_info=True)
            # Reset pending commentary flag on error
            self.pending_commentary = False
            self.current_commentary_topic = None
    
    def update_game_context(self, context: Dict[str, Any]):
        """Update the current game context."""
        self.game_context.update(context)
        if self.logger:
            self.logger.debug(f"[VisionIntegration] Game context updated: {context}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the vision integration service."""
        return {
            'is_watching': self.is_watching,
            'enable_commentary': self.enable_commentary,
            'commentary_frequency': self.commentary_frequency,
            'min_confidence': self.min_confidence,
            'conversation_mode': self.conversation_mode,
            'pending_commentary': self.pending_commentary,
            'current_topic': self.current_commentary_topic,
            'recent_detections_count': len(self.recent_detections),
            'clip_insights_count': len(self.clip_insights),
            'last_commentary_time': self.last_commentary_time,
            'last_full_screenshot_time': self.last_full_screenshot_time,
            'vision_capabilities': {
                'can_capture_screenshots': True,
                'can_analyze_images': True,
                'can_detect_objects': True,
                'can_read_text': True,
                'can_understand_scenes': True,
                'screenshot_source': 'NDI_OBS_STREAM' if hasattr(self, 'vision_pipeline') and self.vision_pipeline and hasattr(self.vision_pipeline, 'ndi_service') else 'SCREEN_CAPTURE'
            }
        }
    
    def get_vision_summary(self) -> str:
        """Get a summary of what the vision system has been seeing recently."""
        try:
            if not self.recent_detections:
                return "I haven't detected anything recently. My vision system is active and watching for game events."
            
            # Group detections by type
            yolo_detections = [d for d in self.recent_detections if d.object_type == 'yolo']
            ocr_detections = [d for d in self.recent_detections if d.object_type == 'ocr']
            template_detections = [d for d in self.recent_detections if d.object_type == 'template']
            
            summary_parts = []
            
            # Add object detections
            if yolo_detections:
                recent_yolo = yolo_detections[-3:]  # Last 3 YOLO detections
                objects = [f"{d.label} (confidence: {d.confidence:.2f})" for d in recent_yolo]
                summary_parts.append(f"Recent objects detected: {', '.join(objects)}")
            
            # Add text detections
            if ocr_detections:
                recent_ocr = ocr_detections[-3:]  # Last 3 OCR detections
                texts = [f"'{d.label}' (confidence: {d.confidence:.2f})" for d in recent_ocr]
                summary_parts.append(f"Recent text detected: {', '.join(texts)}")
            
            # Add template matches
            if template_detections:
                recent_templates = template_detections[-3:]  # Last 3 template matches
                templates = [f"{d.label} (confidence: {d.confidence:.2f})" for d in recent_templates]
                summary_parts.append(f"Recent UI elements: {', '.join(templates)}")
            
            # Add game context
            if self.game_context.get('current_game'):
                summary_parts.append(f"Current game context: {self.game_context['current_game']}")
            
            # Add vision system status
            summary_parts.append(f"Vision system status: {'Active and watching' if self.is_watching else 'Inactive'}")
            summary_parts.append(f"Total detections in memory: {len(self.recent_detections)}")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Error getting vision summary: {e}")
            return "I'm having trouble summarizing my recent vision data."
    
    def get_vision_capabilities_description(self) -> str:
        """Get a human-readable description of vision capabilities optimized for Qwen2.5-VL."""
        capabilities = []
        
        if self.can_see_and_analyze():
            capabilities.append("✅ Real-time screen capture and analysis (Qwen2.5-VL-7B)")
            capabilities.append("✅ Advanced vision-language understanding")
            capabilities.append("✅ Object detection and scene analysis")
            capabilities.append("✅ Text recognition and UI element detection")
            capabilities.append("✅ Gaming context understanding")
            capabilities.append("✅ Screenshot capture on demand")
            capabilities.append("✅ Tool-aware responses")
        else:
            capabilities.append("❌ Vision system not fully operational")
        
        if self.is_watching:
            capabilities.append("✅ Continuously monitoring game activity")
        else:
            capabilities.append("❌ Not currently watching")
        
        if self.enable_commentary:
            capabilities.append("✅ Automatic commentary generation")
        else:
            capabilities.append("❌ Commentary disabled")
        
        # Add Qwen2.5-VL specific capabilities
        capabilities.append("✅ Agentic vision model (can reason about tools)")
        capabilities.append("✅ Real-time image analysis")
        capabilities.append("✅ Context-aware responses")
        
        return "\n".join(capabilities)
    
    def can_see_and_analyze(self) -> bool:
        """Check if the vision system can currently see and analyze images."""
        try:
            # Check if vision pipeline is available and working
            if not hasattr(self, 'vision_pipeline') or not self.vision_pipeline:
                return False
            
            # Check if NDI service is available
            if hasattr(self.vision_pipeline, 'ndi_service') and self.vision_pipeline.ndi_service:
                ndi_service = self.vision_pipeline.ndi_service
                if hasattr(ndi_service, 'is_initialized') and ndi_service.is_initialized:
                    return True
            
            # Check if we can capture screenshots
            if hasattr(self, '_capture_current_screenshot'):
                return True
            
            return False
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Error checking vision capabilities: {e}")
            return False
    
    def get_detailed_vision_report(self) -> Dict[str, Any]:
        """Get a detailed report of vision system activity and capabilities."""
        try:
            # Get recent detections by type
            yolo_detections = [d for d in self.recent_detections if d.object_type == 'yolo']
            ocr_detections = [d for d in self.recent_detections if d.object_type == 'ocr']
            template_detections = [d for d in self.recent_detections if d.object_type == 'template']
            
            # Calculate confidence statistics
            def get_confidence_stats(detections):
                if not detections:
                    return {'count': 0, 'avg_confidence': 0, 'max_confidence': 0}
                confidences = [d.confidence for d in detections]
                return {
                    'count': len(detections),
                    'avg_confidence': sum(confidences) / len(confidences),
                    'max_confidence': max(confidences)
                }
            
            report = {
                'vision_system_status': {
                    'is_active': self.is_watching,
                    'commentary_enabled': self.enable_commentary,
                    'last_activity': time.time() - self.last_commentary_time if self.last_commentary_time > 0 else None,
                    'model_type': 'Qwen2.5-VL-7B-Instruct'
                },
                'capabilities': {
                    'screenshot_capture': True,
                    'object_detection': True,
                    'text_recognition': True,
                    'template_matching': True,
                    'scene_understanding': True,
                    'real_time_analysis': True,
                    'tool_awareness': True,
                    'agentic_behavior': True
                },
                'recent_activity': {
                    'total_detections': len(self.recent_detections),
                    'yolo_detections': get_confidence_stats(yolo_detections),
                    'ocr_detections': get_confidence_stats(ocr_detections),
                    'template_detections': get_confidence_stats(template_detections),
                    'last_detection_time': self.recent_detections[-1].timestamp if self.recent_detections else None
                },
                'game_context': self.game_context,
                'screenshot_info': {
                    'can_capture': True,
                    'preferred_source': 'NDI_OBS_STREAM' if hasattr(self, 'vision_pipeline') and self.vision_pipeline and hasattr(self.vision_pipeline, 'ndi_service') else 'SCREEN_CAPTURE',
                    'last_screenshot': self.last_full_screenshot_time,
                    'model_optimized': 'Qwen2.5-VL-7B-Instruct'
                }
            }
            
            return report
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Error getting detailed vision report: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.logger:
                self.logger.info("[VisionIntegration] Cleaning up...")
            
            # Cancel event processor task
            if self.event_processor_task:
                self.event_processor_task.cancel()
            
            # Clear data structures
            self.recent_detections.clear()
            self.commentary_history.clear()
            self.clip_insights.clear()
            
            if self.logger:
                self.logger.info("[VisionIntegration] Cleanup complete")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Cleanup error: {e}")

    def _capture_current_screenshot(self) -> Optional[str]:
        """Capture a screenshot of the current game screen from OBS NDI stream."""
        try:
            if self.logger:
                self.logger.info("[VisionIntegration] 🎯 Capturing screenshot on demand (agentic mode)")
            
            # Use vision pipeline's on-demand capture if available
            if hasattr(self, 'vision_pipeline') and self.vision_pipeline:
                frame = self.vision_pipeline.capture_frame_on_demand()
                if frame is not None:
                    # Convert frame to base64
                    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if success:
                        image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                        if self.logger:
                            self.logger.info(f"[VisionIntegration] ✅ Screenshot captured on demand: {frame.shape}")
                        return image_base64
                    else:
                        if self.logger:
                            self.logger.error("[VisionIntegration] Failed to encode screenshot")
                else:
                    if self.logger:
                        self.logger.warning("[VisionIntegration] Vision pipeline returned no frame")
            
            # Fallback to NDI service if available
            if hasattr(self.app_context, 'ndi_service') and self.app_context.ndi_service:
                ndi_service = self.app_context.ndi_service
                if ndi_service.is_initialized and ndi_service.last_captured_frame is not None:
                    frame = ndi_service.last_captured_frame.copy()
                    # Convert to BGR if needed
                    if len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
                    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if success:
                        image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                        if self.logger:
                            self.logger.info(f"[VisionIntegration] ✅ Screenshot captured from NDI service: {frame.shape}")
                        return image_base64
            
            # Final fallback to PIL screen capture
            if self.logger:
                self.logger.warning("[VisionIntegration] Falling back to PIL screen capture")
            
            from PIL import ImageGrab
            import io
            
            # Capture full screen
            screenshot = ImageGrab.grab()
            
            # Convert to base64
            buffer = io.BytesIO()
            screenshot.save(buffer, format='JPEG', quality=95)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            if self.logger:
                self.logger.info(f"[VisionIntegration] ✅ Screenshot captured with PIL: {screenshot.size}")
            
            return image_base64
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[VisionIntegration] Screenshot capture error: {e}", exc_info=True)
            return None

# Import the streaming LLM service
try:
    from services.real_time_streaming_llm_service import RealTimeStreamingLLMService
except ImportError:
    RealTimeStreamingLLMService = None 