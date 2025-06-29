#!/usr/bin/env python3
"""
Basic Vision Pipeline Starter
============================

Simple script to start the vision pipeline with screen capture.
"""

import asyncio
from vision_pipeline import VisionPipeline, DetectionEvent

def simple_callback(event: DetectionEvent):
    """Simple callback to print detections"""
    print(f"🎯 {event.object_type}: {event.label} (conf: {event.confidence:.2f})")

async def main():
    """Start basic vision pipeline"""
    print("🚀 Starting Basic Vision Pipeline")
    print("=" * 40)
    
    # Create pipeline
    pipeline = VisionPipeline(event_callback=simple_callback)
    
    try:
        # Initialize
        print("⏳ Initializing...")
        if await pipeline.initialize():
            print("✅ Initialized successfully")
            
            # Start pipeline
            print("🎬 Starting capture...")
            pipeline.start()
            print("✅ Pipeline running!")
            print("\n📋 What's happening:")
            print("   • Capturing screen at 10 FPS")
            print("   • Running YOLO object detection")
            print("   • Processing OCR on screen regions")
            print("   • Detecting UI elements and text")
            print("\n🛑 Press Ctrl+C to stop")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        pipeline.stop()
        print("✅ Stopped")

if __name__ == "__main__":
    asyncio.run(main()) 