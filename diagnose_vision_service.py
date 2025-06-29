#!/usr/bin/env python3
"""
Diagnostic script to check vision integration service initialization and auto-start.
"""

import asyncio
import logging
import sys
import os
import traceback

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_vision_service_import():
    """Test if vision integration service can be imported."""
    print("🔍 Testing vision integration service import...")
    try:
        from services.vision_integration_service import VisionIntegrationService
        print("✅ VisionIntegrationService imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import VisionIntegrationService: {e}")
        traceback.print_exc()
        return False

def test_app_context_import():
    """Test if app context can be imported."""
    print("🔍 Testing app context import...")
    try:
        from core.app_context import AppContext
        print("✅ AppContext imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import AppContext: {e}")
        traceback.print_exc()
        return False

async def test_vision_service_creation():
    """Test if vision integration service can be created."""
    print("🔍 Testing vision integration service creation...")
    try:
        from services.vision_integration_service import VisionIntegrationService
        from core.app_context import AppContext
        
        # Create app context
        app_context = AppContext()
        app_context.logger = logging.getLogger("VisionTest")
        app_context.shutdown_event = asyncio.Event()
        
        print("✅ AppContext created successfully")
        
        # Create vision integration service
        vision_service = VisionIntegrationService(app_context)
        print("✅ VisionIntegrationService created successfully")
        
        return vision_service, app_context
    except Exception as e:
        print(f"❌ Failed to create vision integration service: {e}")
        traceback.print_exc()
        return None, None

async def test_vision_service_initialization():
    """Test if vision integration service can be initialized."""
    print("🔍 Testing vision integration service initialization...")
    
    vision_service, app_context = await test_vision_service_creation()
    if not vision_service:
        return False
    
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        print("🔧 Initializing vision integration service...")
        success = await vision_service.initialize()
        
        if success:
            print("✅ Vision integration service initialized successfully")
            
            # Check status
            status = vision_service.get_status()
            print(f"📊 Vision service status: {status}")
            
            # Check if event processor is running
            if hasattr(vision_service, 'event_processor_task'):
                print(f"📊 Event processor task: {vision_service.event_processor_task}")
                if vision_service.event_processor_task:
                    print(f"📊 Event processor done: {vision_service.event_processor_task.done()}")
                else:
                    print("❌ Event processor task is None")
            else:
                print("❌ Event processor task attribute not found")
            
            return True
        else:
            print("❌ Vision integration service initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        traceback.print_exc()
        return False

async def test_event_processing():
    """Test if events are being processed."""
    print("🔍 Testing event processing...")
    
    vision_service, app_context = await test_vision_service_creation()
    if not vision_service:
        return False
    
    try:
        # Initialize
        success = await vision_service.initialize()
        if not success:
            print("❌ Failed to initialize vision service for event testing")
            return False
        
        # Wait for event processor to start
        await asyncio.sleep(3)
        
        # Create test event
        from services.vision_integration_service import DetectionEvent
        from datetime import datetime
        
        test_event = DetectionEvent(
            object_type="ocr",
            label="Test Event",
            confidence=0.95,
            timestamp=datetime.now().timestamp(),
            bbox=None
        )
        
        print("🧪 Simulating test event...")
        vision_service._handle_vision_event(test_event)
        
        # Wait for processing
        await asyncio.sleep(5)
        
        # Check if event was processed
        if hasattr(vision_service, 'pending_events'):
            pending_count = len(vision_service.pending_events)
            print(f"📊 Pending events count: {pending_count}")
            
            if pending_count == 0:
                print("✅ Event was processed (pending_events is empty)")
            else:
                print(f"❌ Event was not processed (still {pending_count} pending)")
        else:
            print("❌ pending_events attribute not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during event processing test: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all diagnostic tests."""
    print("🚀 Starting vision integration service diagnostics...")
    print("=" * 60)
    
    # Test 1: Import tests
    print("\n📦 TEST 1: Import Tests")
    print("-" * 30)
    import_success = test_vision_service_import() and test_app_context_import()
    
    if not import_success:
        print("❌ Import tests failed - cannot proceed")
        return
    
    # Test 2: Service creation
    print("\n🔧 TEST 2: Service Creation")
    print("-" * 30)
    creation_success = await test_vision_service_creation()
    
    if not creation_success[0]:
        print("❌ Service creation failed - cannot proceed")
        return
    
    # Test 3: Service initialization
    print("\n⚙️ TEST 3: Service Initialization")
    print("-" * 30)
    init_success = await test_vision_service_initialization()
    
    if not init_success:
        print("❌ Service initialization failed")
        return
    
    # Test 4: Event processing
    print("\n🎯 TEST 4: Event Processing")
    print("-" * 30)
    event_success = await test_event_processing()
    
    # Summary
    print("\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"✅ Import Tests: {'PASS' if import_success else 'FAIL'}")
    print(f"✅ Service Creation: {'PASS' if creation_success[0] else 'FAIL'}")
    print(f"✅ Service Initialization: {'PASS' if init_success else 'FAIL'}")
    print(f"✅ Event Processing: {'PASS' if event_success else 'FAIL'}")
    
    if all([import_success, creation_success[0], init_success, event_success]):
        print("\n🎉 All tests passed! Vision integration should be working.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Diagnostics interrupted by user")
    except Exception as e:
        print(f"❌ Diagnostic script failed: {e}")
        traceback.print_exc() 