#!/usr/bin/env python3
"""
Diagnose Vision Events
=====================

This script diagnoses why vision events are not being detected or processed.
"""

import asyncio
import logging
import sys
import os
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.app_context import AppContext
from core.config_loader import load_global_settings, load_game_profile
from services.vision_integration_service import VisionIntegrationService, DetectionEvent
from vision_pipeline import VisionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/vision_diagnosis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

async def diagnose_vision_events():
    """Diagnose why vision events are not being detected."""
    
    logger.info("🔍 Starting Vision Events Diagnosis")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        global_settings = load_global_settings()
        if not global_settings:
            logger.error("❌ Failed to load global settings")
            return False
            
        # Load active profile (use generic as default)
        active_profile = load_game_profile("generic", global_settings)
        if not active_profile:
            logger.error("❌ Failed to load active profile")
            return False
        
        # Create app context
        app_context = AppContext(global_settings, active_profile, logger)
        
        logger.info("✅ App context created")
        
        # Test 1: Check if vision pipeline can be created
        logger.info("🧪 Test 1: Creating vision pipeline...")
        try:
            vision_pipeline = VisionPipeline(
                event_callback=lambda event: logger.info(f"🔥 PIPELINE EVENT: {event.object_type} - {event.label}"),
                config_path="config/vision_config.yaml"
            )
            logger.info("✅ Vision pipeline created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create vision pipeline: {e}")
            return False
        
        # Test 2: Check if vision pipeline can be initialized
        logger.info("🧪 Test 2: Initializing vision pipeline...")
        try:
            if await vision_pipeline.initialize():
                logger.info("✅ Vision pipeline initialized successfully")
            else:
                logger.error("❌ Vision pipeline initialization failed")
                return False
        except Exception as e:
            logger.error(f"❌ Vision pipeline initialization error: {e}")
            return False
        
        # Test 3: Check if vision pipeline can be started
        logger.info("🧪 Test 3: Starting vision pipeline...")
        try:
            vision_pipeline.start()
            logger.info("✅ Vision pipeline started successfully")
            
            # Check if it's running
            if hasattr(vision_pipeline, 'running') and vision_pipeline.running:
                logger.info("✅ Vision pipeline is running")
            else:
                logger.warning("⚠️ Vision pipeline running status unclear")
                
        except Exception as e:
            logger.error(f"❌ Failed to start vision pipeline: {e}")
            return False
        
        # Test 4: Check vision pipeline status
        logger.info("🧪 Test 4: Checking vision pipeline status...")
        try:
            status = vision_pipeline.get_status()
            logger.info("📊 Vision pipeline status:")
            for key, value in status.items():
                logger.info(f"   {key}: {value}")
        except Exception as e:
            logger.error(f"❌ Failed to get vision pipeline status: {e}")
        
        # Test 5: Create vision integration service
        logger.info("🧪 Test 5: Creating vision integration service...")
        try:
            vision_service = VisionIntegrationService(app_context)
            logger.info("✅ Vision integration service created")
        except Exception as e:
            logger.error(f"❌ Failed to create vision integration service: {e}")
            return False
        
        # Test 6: Initialize vision integration service
        logger.info("🧪 Test 6: Initializing vision integration service...")
        try:
            if await vision_service.initialize():
                logger.info("✅ Vision integration service initialized successfully")
            else:
                logger.error("❌ Vision integration service initialization failed")
                return False
        except Exception as e:
            logger.error(f"❌ Vision integration service initialization error: {e}")
            return False
        
        # Test 7: Check if vision integration service is watching
        logger.info("🧪 Test 7: Checking vision integration service status...")
        if vision_service.is_watching:
            logger.info("✅ Vision integration service is watching")
        else:
            logger.error("❌ Vision integration service is not watching")
        
        # Test 8: Check if event processor is running
        logger.info("🧪 Test 8: Checking event processor...")
        if hasattr(vision_service, 'event_processor_task') and vision_service.event_processor_task:
            if vision_service.event_processor_task.done():
                logger.error("❌ Event processor task is done (not running)")
            else:
                logger.info("✅ Event processor task is running")
        else:
            logger.error("❌ No event processor task found")
        
        # Test 9: Check pending events
        logger.info("🧪 Test 9: Checking pending events...")
        if hasattr(vision_service, 'pending_events'):
            logger.info(f"📊 Pending events: {len(vision_service.pending_events)}")
            for i, event in enumerate(vision_service.pending_events):
                logger.info(f"   Event {i+1}: {event.object_type} - {event.label}")
        else:
            logger.warning("⚠️ No pending_events attribute found")
        
        # Test 10: Check recent detections
        logger.info("🧪 Test 10: Checking recent detections...")
        if hasattr(vision_service, 'recent_detections'):
            logger.info(f"📊 Recent detections: {len(vision_service.recent_detections)}")
            for i, detection in enumerate(vision_service.recent_detections[-5:]):  # Last 5
                logger.info(f"   Detection {i+1}: {detection.object_type} - {detection.label}")
        else:
            logger.warning("⚠️ No recent_detections attribute found")
        
        # Test 11: Wait for some events
        logger.info("🧪 Test 11: Waiting for vision events (30 seconds)...")
        start_time = time.time()
        event_count = 0
        
        while time.time() - start_time < 30:
            if hasattr(vision_service, 'recent_detections'):
                current_count = len(vision_service.recent_detections)
                if current_count > event_count:
                    logger.info(f"🔥 New events detected! Total: {current_count}")
                    event_count = current_count
            
            await asyncio.sleep(1)
        
        logger.info(f"📊 Final event count: {event_count}")
        
        if event_count == 0:
            logger.warning("⚠️ No vision events detected during the test period")
            logger.info("🔍 Possible issues:")
            logger.info("   - No visual content on screen")
            logger.info("   - Vision pipeline not capturing frames")
            logger.info("   - YOLO/OCR models not detecting anything")
            logger.info("   - Configuration issues")
        else:
            logger.info("✅ Vision events are being detected!")
        
        # Cleanup
        try:
            vision_pipeline.stop()
            await vision_service.stop_watching()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
        
        logger.info("=" * 60)
        logger.info("🎉 Vision Events Diagnosis Completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Diagnosis failed with error: {e}", exc_info=True)
        return False

async def main():
    """Main diagnostic function."""
    logger.info("🚀 Starting Vision Events Diagnosis")
    
    success = await diagnose_vision_events()
    
    if success:
        logger.info("🎉 Diagnosis completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Diagnosis failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 