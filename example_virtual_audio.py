#!/usr/bin/env python3
"""
Example usage of DanzarVLM Virtual Audio system
"""

import asyncio
import sounddevice as sd

def list_audio_devices():
    """Simple function to list audio devices."""
    print("🎵 Available Audio Input Devices:")
    print("=" * 60)
    
    devices = sd.query_devices()
    virtual_devices = []
    
    for i, device in enumerate(devices):
        try:
            name = str(device.get('name', 'Unknown'))
            max_in = int(device.get('max_input_channels', 0))
            max_out = int(device.get('max_output_channels', 0))
            
            if max_in > 0:  # Only show input devices
                print(f"  {i:2d}: {name}")
                print(f"      Input channels: {max_in}")
                
                # Check for virtual audio keywords
                name_lower = name.lower()
                virtual_keywords = ['cable', 'virtual', 'vb-audio', 'voicemeeter', 
                                  'stereo mix', 'wave out mix', 'what u hear']
                
                if any(keyword in name_lower for keyword in virtual_keywords):
                    virtual_devices.append((i, name))
                    print(f"      ⭐ VIRTUAL AUDIO DEVICE DETECTED")
                
                print()
        except Exception as e:
            print(f"  {i:2d}: Error reading device info: {e}")
    
    if virtual_devices:
        print("🎯 Recommended devices for DanzarVLM:")
        print("=" * 60)
        for device_id, device_name in virtual_devices:
            print(f"  Device {device_id}: {device_name}")
        print("\nTo use with DanzarVLM:")
        print(f"  python DanzarVLM_VirtualAudio.py --device {virtual_devices[0][0]}")
    else:
        print("⚠️  No virtual audio devices found!")
        print("\nTo set up virtual audio:")
        print("1. Install VB-Cable from https://vb-audio.com/Cable/")
        print("2. Or enable Windows Stereo Mix in Sound settings")
        print("3. Run this script again to verify")
    
    return virtual_devices

async def example_usage():
    """Example of how to use the virtual audio system."""
    print("🚀 DanzarVLM Virtual Audio Example")
    print("=" * 60)
    
    # Step 1: List devices
    virtual_devices = list_audio_devices()
    
    if not virtual_devices:
        print("\n❌ No virtual audio devices available.")
        print("Please install VB-Cable and try again.")
        return
    
    # Step 2: Instructions
    print(f"\n📋 Setup Instructions:")
    print("=" * 60)
    print("1. Install VB-Cable if you haven't already")
    print("2. Set your game/application audio output to 'CABLE Input'")
    print("3. In Windows Sound settings → Recording → CABLE Output:")
    print("   - Enable 'Listen to this device'")
    print("   - Set playback device to your speakers")
    print("   - Set level to 100%")
    print()
    
    print("🎮 Usage Examples:")
    print("=" * 60)
    
    # Auto-detect mode
    print("# Auto-detect virtual audio device:")
    print("python DanzarVLM_VirtualAudio.py")
    print()
    
    # Specific device mode
    device_id = virtual_devices[0][0]
    device_name = virtual_devices[0][1]
    print(f"# Use specific device ({device_name}):")
    print(f"python DanzarVLM_VirtualAudio.py --device {device_id}")
    print()
    
    # List devices mode
    print("# List all available devices:")
    print("python DanzarVLM_VirtualAudio.py --list-devices")
    print()
    
    print("🎯 Discord Bot Commands (once DanzarVLM is running):")
    print("=" * 60)
    print("!connect    - Connect bot to your voice channel for TTS output")
    print("!say <text> - Test TTS with custom text")
    print("!disconnect - Disconnect bot from voice channel")
    print()
    
    print("🔄 Complete Workflow:")
    print("=" * 60)
    print("1. Start DanzarVLM:")
    print("   python DanzarVLM_VirtualAudio.py")
    print()
    print("2. Join Discord voice channel and type:")
    print("   !connect")
    print()
    print("3. Set your game audio output to 'CABLE Input'")
    print()
    print("4. Start talking/playing - DanzarAI will respond!")
    print()
    print("5. Stop with Ctrl+C when done")

if __name__ == "__main__":
    asyncio.run(example_usage()) 