#!/usr/bin/env python3
"""
Quick Vision Debug
=================

Quick debug to check why vision commentary isn't working.
"""

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def quick_debug():
    """Quick debug of vision system."""
    logger.info("🔍 Quick vision debug starting...")
    
    try:
        # Check if vision integration service exists
        from services.vision_integration_service import VisionIntegrationService
        logger.info("✅ Vision integration service import successful")
        
        # Check if streaming LLM service exists
        from services.real_time_streaming_llm import RealTimeStreamingLLMService
        logger.info("✅ Real-time streaming LLM service import successful")
        
        # Check if vision pipeline exists
        from vision_pipeline import VisionPipeline
        logger.info("✅ Vision pipeline import successful")
        
        # Check if CLIP vision enhancer exists
        from services.clip_vision_enhancer import CLIPVisionEnhancer
        logger.info("✅ CLIP vision enhancer import successful")
        
        logger.info("🎉 All imports successful! Vision system components are available.")
        
        # Check if the issue is in the main app
        logger.info("🔍 Checking main app vision integration...")
        
        # Try to import the main app
        try:
            from DanzarVLM import DanzarVoiceBot
            logger.info("✅ Main app import successful")
        except Exception as e:
            logger.error(f"❌ Main app import failed: {e}")
        
        logger.info("✅ Quick debug completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Quick debug failed: {e}", exc_info=True)

if __name__ == "__main__":
    quick_debug() 