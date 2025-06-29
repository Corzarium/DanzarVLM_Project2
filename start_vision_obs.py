#!/usr/bin/env python3
"""
OBS NDI Vision Pipeline Starter
==============================

Start vision pipeline with OBS NDI capture.
"""

import asyncio
import yaml
from vision_pipeline import VisionPipeline, DetectionEvent

def obs_callback(event: DetectionEvent):
    """Callback for OBS vision events"""
    print(f"🎯 OBS: {event.object_type} - {event.label} (conf: {event.confidence:.2f})")
    if event.object_type == 'ocr':
        print(f"   📝 Text: {event.label}")
    elif event.object_type == 'yolo':
        print(f"   📍 Location: {event.bbox}")

async def main():
    """Start OBS NDI vision pipeline"""
    print("🚀 Starting OBS NDI Vision Pipeline")
    print("=" * 45)
    
    # Load config
    config_path = "config/vision_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable NDI
    config['capture']['use_ndi'] = True
    config['capture']['ndi_source_name'] = "OBS Studio"  # Change this to your OBS source name
    config['capture']['region'] = "ndi"
    config['capture']['fps'] = 30  # Higher FPS for NDI
    
    # Save NDI config
    ndi_config_path = "config/vision_config_obs.yaml"
    with open(ndi_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📝 Using NDI config: {ndi_config_path}")
    print(f"🎯 Target: {config['capture']['ndi_source_name']}")
    print()
    
    # Create pipeline
    pipeline = VisionPipeline(config_path=ndi_config_path, event_callback=obs_callback)
    
    try:
        # Initialize
        print("⏳ Initializing OBS NDI...")
        if await pipeline.initialize():
            print("✅ OBS NDI initialized successfully")
            
            # Start pipeline
            print("🎬 Starting OBS capture...")
            pipeline.start()
            print("✅ OBS pipeline running!")
            print("\n📋 What's happening:")
            print("   • Capturing from OBS via NDI")
            print("   • Running YOLO object detection")
            print("   • Processing OCR on game regions")
            print("   • Detecting HUD elements and text")
            print("\n🛑 Press Ctrl+C to stop")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        else:
            print("❌ Failed to initialize OBS NDI")
            print("\n🔧 Troubleshooting:")
            print("   1. Make sure OBS is running")
            print("   2. Enable NDI: Tools -> NDI Output Settings")
            print("   3. Set NDI source name in OBS")
            print("   4. Start streaming/recording in OBS")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        pipeline.stop()
        print("✅ Stopped")

if __name__ == "__main__":
    asyncio.run(main()) 