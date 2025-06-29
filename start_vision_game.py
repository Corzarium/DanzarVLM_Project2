#!/usr/bin/env python3
"""
Game-Specific Vision Pipeline Starter
====================================

Start vision pipeline with game-specific profiles.
"""

import asyncio
import sys
from vision_pipeline import VisionPipeline, DetectionEvent

def game_callback(event: DetectionEvent):
    """Callback for game-specific events"""
    game_name = event.metadata.get('game', 'Unknown')
    print(f"🎮 [{game_name}] {event.object_type}: {event.label} (conf: {event.confidence:.2f})")
    
    if event.object_type == 'ocr':
        print(f"   📝 Text: {event.label}")
    elif event.object_type == 'template':
        print(f"   🎯 Template: {event.label} at {event.bbox}")

async def main():
    """Start game-specific vision pipeline"""
    if len(sys.argv) < 2:
        print("Usage: python start_vision_game.py <game>")
        print("Available games: everquest, rimworld, generic")
        return
    
    game = sys.argv[1].lower()
    print(f"🚀 Starting {game.title()} Vision Pipeline")
    print("=" * 50)
    
    # Game-specific config
    config_path = f"config/vision_config.yaml"
    profile_path = f"config/profiles/{game}.yaml"
    
    print(f"📝 Config: {config_path}")
    print(f"🎮 Profile: {profile_path}")
    print()
    
    # Create pipeline with game profile
    pipeline = VisionPipeline(
        config_path=config_path,
        profile_path=profile_path,
        event_callback=game_callback
    )
    
    try:
        # Initialize
        print(f"⏳ Initializing {game.title()} vision...")
        if await pipeline.initialize():
            print(f"✅ {game.title()} vision initialized successfully")
            
            # Start pipeline
            print("🎬 Starting game capture...")
            pipeline.start()
            print(f"✅ {game.title()} pipeline running!")
            print("\n📋 What's happening:")
            print(f"   • Using {game.title()} specific detection regions")
            print("   • Running YOLO object detection")
            print("   • Processing OCR on game UI")
            print("   • Detecting game-specific elements")
            print("\n🛑 Press Ctrl+C to stop")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        else:
            print(f"❌ Failed to initialize {game.title()} vision")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        pipeline.stop()
        print("✅ Stopped")

if __name__ == "__main__":
    asyncio.run(main()) 