#!/usr/bin/env python3
"""
DanzarAI Vision Integration
==========================

Demonstrates how to integrate the vision pipeline with DanzarAI's existing services.
"""

import asyncio
import json
import logging
from typing import Dict, Any
from pathlib import Path

# Import DanzarAI services
try:
    from core.app_context import AppContext
    from services.memory_service import MemoryService
    from services.llm_service import LLMService
    from vision_pipeline import VisionPipeline, DetectionEvent
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the DanzarAI project root")
    exit(1)

class DanzarVisionIntegration:
    """
    Integrates vision pipeline with DanzarAI services.
    
    Features:
    - Processes vision events through DanzarAI's memory system
    - Generates commentary based on detected game events
    - Maintains context of visual observations
    - Integrates with existing LLM and memory services
    """
    
    def __init__(self, config_path: str = "config/vision_config.yaml"):
        self.config_path = config_path
        self.app_context = None
        self.memory_service = None
        self.llm_service = None
        self.vision_pipeline = None
        self.logger = self._setup_logging()
        
        # Event processing state
        self.last_events = {}
        self.event_counters = {}
        self.commentary_cooldown = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for vision integration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("DanzarVision")
    
    async def initialize(self) -> bool:
        """Initialize DanzarAI services and vision pipeline"""
        try:
            self.logger.info("Initializing DanzarAI Vision Integration...")
            
            # Initialize DanzarAI app context
            self.app_context = AppContext()
            await self.app_context.initialize()
            
            # Initialize memory service
            self.memory_service = MemoryService(self.app_context)
            await self.memory_service.initialize()
            
            # Initialize LLM service
            self.llm_service = LLMService(self.app_context)
            await self.llm_service.initialize()
            
            # Initialize vision pipeline with event callback
            self.vision_pipeline = VisionPipeline(
                config_path=self.config_path,
                event_callback=self._handle_vision_event
            )
            await self.vision_pipeline.initialize()
            
            self.logger.info("DanzarAI Vision Integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def _handle_vision_event(self, event: DetectionEvent):
        """Handle vision detection events"""
        try:
            # Process event asynchronously
            asyncio.create_task(self._process_vision_event(event))
        except Exception as e:
            self.logger.error(f"Error handling vision event: {e}")
    
    async def _process_vision_event(self, event: DetectionEvent):
        """Process a vision detection event"""
        try:
            # Create event key for tracking
            event_key = f"{event.object_type}_{event.label}"
            
            # Update event counters
            self.event_counters[event_key] = self.event_counters.get(event_key, 0) + 1
            
            # Store in memory
            await self._store_vision_memory(event)
            
            # Check if we should generate commentary
            if await self._should_generate_commentary(event):
                await self._generate_vision_commentary(event)
            
            # Update last event
            self.last_events[event_key] = event
            
        except Exception as e:
            self.logger.error(f"Error processing vision event: {e}")
    
    async def _store_vision_memory(self, event: DetectionEvent):
        """Store vision event in DanzarAI's memory system"""
        try:
            # Create memory entry
            memory_content = f"Vision detection: {event.object_type} - {event.label} (confidence: {event.confidence:.2f})"
            
            # Add metadata
            metadata = {
                'vision_event': True,
                'object_type': event.object_type,
                'label': event.label,
                'confidence': event.confidence,
                'bbox': event.bbox,
                'timestamp': event.timestamp,
                'event_id': event.event_id
            }
            
            # Store in memory
            await self.memory_service.store_memory(
                content=memory_content,
                source="vision_detection",
                metadata=metadata
            )
            
            self.logger.debug(f"Stored vision memory: {event.label}")
            
        except Exception as e:
            self.logger.error(f"Error storing vision memory: {e}")
    
    async def _should_generate_commentary(self, event: DetectionEvent) -> bool:
        """Determine if we should generate commentary for this event"""
        try:
            event_key = f"{event.object_type}_{event.label}"
            current_time = event.timestamp
            
            # Check cooldown
            last_commentary = self.commentary_cooldown.get(event_key, 0)
            cooldown_duration = 30.0  # 30 seconds between commentaries for same event
            
            if current_time - last_commentary < cooldown_duration:
                return False
            
            # Check confidence threshold
            if event.confidence < 0.7:
                return False
            
            # Check if this is a significant event
            significant_events = [
                'boss', 'loot', 'health_bar', 'player', 'enemy',
                'spell_icon', 'cooldown', 'minimap'
            ]
            
            if event.label.lower() in significant_events:
                return True
            
            # Check if this is a new event type
            if event_key not in self.last_events:
                return True
            
            # Check if confidence changed significantly
            last_event = self.last_events[event_key]
            confidence_change = abs(event.confidence - last_event.confidence)
            
            if confidence_change > 0.2:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking commentary conditions: {e}")
            return False
    
    async def _generate_vision_commentary(self, event: DetectionEvent):
        """Generate commentary based on vision event"""
        try:
            event_key = f"{event.object_type}_{event.label}"
            current_time = event.timestamp
            
            # Update cooldown
            self.commentary_cooldown[event_key] = current_time
            
            # Create commentary prompt
            commentary_prompt = self._create_commentary_prompt(event)
            
            # Generate commentary using LLM
            if self.llm_service:
                commentary = await self.llm_service.generate(commentary_prompt)
                
                # Store commentary in memory
                await self.memory_service.store_memory(
                    content=f"Vision commentary: {commentary}",
                    source="vision_commentary",
                    metadata={
                        'trigger_event': event.event_id,
                        'object_type': event.object_type,
                        'label': event.label,
                        'confidence': event.confidence
                    }
                )
                
                self.logger.info(f"Generated commentary: {commentary[:100]}...")
                
                # TODO: Send to TTS service for voice output
                # await self.tts_service.generate_audio(commentary)
            
        except Exception as e:
            self.logger.error(f"Error generating commentary: {e}")
    
    def _create_commentary_prompt(self, event: DetectionEvent) -> str:
        """Create a prompt for generating commentary based on vision event"""
        
        # Base prompt
        base_prompt = """You are Danzar, an upbeat and witty gaming assistant. 
        A vision detection event has occurred in the game. Generate a brief, 
        engaging commentary about what you see. Keep it natural and conversational.
        
        Event details:
        - Type: {object_type}
        - Object: {label}
        - Confidence: {confidence:.2f}
        - Location: {bbox}
        
        Generate a short, witty commentary:"""
        
        # Game-specific prompts
        game_prompts = {
            'boss': "A boss has been detected! This could be exciting or dangerous.",
            'loot': "Some loot has appeared! Time to check what we got.",
            'health_bar': "Health status detected - keeping an eye on your well-being.",
            'player': "Player activity spotted - you're in the action!",
            'enemy': "Enemy detected - stay alert and ready for combat.",
            'spell_icon': "Spell or ability icon detected - magic is afoot!",
            'cooldown': "Cooldown timer spotted - timing is everything.",
            'minimap': "Minimap activity - keeping track of the surroundings."
        }
        
        # Get specific prompt for this event
        specific_prompt = game_prompts.get(event.label.lower(), "")
        
        # Combine prompts
        full_prompt = base_prompt.format(
            object_type=event.object_type,
            label=event.label,
            confidence=event.confidence,
            bbox=event.bbox
        )
        
        if specific_prompt:
            full_prompt += f"\n\nContext: {specific_prompt}"
        
        return full_prompt
    
    async def start(self):
        """Start the vision integration"""
        try:
            self.logger.info("Starting DanzarAI Vision Integration...")
            
            # Start vision pipeline
            self.vision_pipeline.start()
            
            self.logger.info("DanzarAI Vision Integration started")
            
        except Exception as e:
            self.logger.error(f"Error starting vision integration: {e}")
    
    async def stop(self):
        """Stop the vision integration"""
        try:
            self.logger.info("Stopping DanzarAI Vision Integration...")
            
            # Stop vision pipeline
            if self.vision_pipeline:
                self.vision_pipeline.stop()
            
            self.logger.info("DanzarAI Vision Integration stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping vision integration: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            'vision_pipeline_running': self.vision_pipeline.running if self.vision_pipeline else False,
            'memory_service_ready': self.memory_service is not None,
            'llm_service_ready': self.llm_service is not None,
            'event_counters': self.event_counters,
            'last_events_count': len(self.last_events)
        }

# Example usage
async def main():
    """Example usage of DanzarAI Vision Integration"""
    
    # Create integration
    integration = DanzarVisionIntegration()
    
    try:
        # Initialize
        if await integration.initialize():
            print("✅ DanzarAI Vision Integration initialized")
            
            # Start
            await integration.start()
            print("🚀 Vision integration started")
            
            # Run for 60 seconds
            print("⏳ Running for 60 seconds...")
            await asyncio.sleep(60)
            
            # Show status
            status = integration.get_status()
            print(f"📊 Status: {status}")
            
        else:
            print("❌ Failed to initialize vision integration")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping vision integration...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Stop
        await integration.stop()
        print("✅ Vision integration stopped")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 