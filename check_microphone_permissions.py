#!/usr/bin/env python3
"""
Windows Microphone Permission Checker
Checks if Python has access to your microphone
"""

import sounddevice as sd
import numpy as np
import time
import subprocess
import sys

def check_windows_microphone_permissions():
    """Check Windows microphone permissions"""
    print("🔍 WINDOWS MICROPHONE PERMISSION CHECKER")
    print("=" * 60)
    
    print("📋 Step 1: Checking available audio devices...")
    try:
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device.get('max_input_channels', 0) > 0:
                input_devices.append((i, device))
                is_default = "🎯 DEFAULT" if i == sd.default.device[0] else ""
                print(f"  {i}: {device.get('name', 'Unknown')} (channels: {device.get('max_input_channels', 0)}) {is_default}")
        
        if not input_devices:
            print("❌ No input devices found!")
            return False
            
    except Exception as e:
        print(f"❌ Error querying devices: {e}")
        return False
    
    print(f"\n📋 Step 2: Testing microphone access...")
    
    # Try to access the default microphone
    try:
        print("🎤 Testing default microphone access...")
        
        # Use device 0 (Microsoft Sound Mapper) - your real microphone
        test_device = 0
        sample_rate = 16000
        duration = 2  # seconds
        
        print(f"🎯 Testing device {test_device}: {devices[test_device].get('name', 'Unknown')}")
        print(f"⏱️ Recording for {duration} seconds...")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=1, 
            device=test_device,
            dtype=np.float32
        )
        sd.wait()  # Wait for recording to complete
        
        # Check if we got actual audio data
        max_amplitude = np.max(np.abs(audio_data))
        rms = np.sqrt(np.mean(audio_data**2))
        
        print(f"📊 Audio stats:")
        print(f"   Max amplitude: {max_amplitude:.4f}")
        print(f"   RMS level: {rms:.4f}")
        
        if max_amplitude > 0.001:  # If we have some audio signal
            print("✅ Microphone access successful!")
            print("🔊 Audio data captured successfully")
            return True
        else:
            print("⚠️ Microphone accessed but no audio detected")
            print("💡 This could mean:")
            print("   - Microphone is muted")
            print("   - No sound is being made")
            print("   - Windows privacy settings are blocking access")
            return False
            
    except Exception as e:
        print(f"❌ Microphone access failed: {e}")
        
        # Check if it's a permission error
        if "PaError" in str(e) or "MME error" in str(e):
            print("\n🛠️ WINDOWS PERMISSION ISSUE DETECTED!")
            print("To fix this:")
            print("1. Press Win + I to open Settings")
            print("2. Go to Privacy & Security > Microphone")
            print("3. Make sure 'Microphone access' is ON")
            print("4. Make sure 'Let apps access your microphone' is ON")
            print("5. Make sure 'Let desktop apps access your microphone' is ON")
            print("6. Restart this script")
            
        return False

def check_virtual_audio_setup():
    """Check if VB-Audio is properly configured"""
    print("\n🔍 VIRTUAL AUDIO SETUP CHECKER")
    print("=" * 60)
    
    devices = sd.query_devices()
    vb_audio_devices = []
    
    for i, device in enumerate(devices):
        device_name = device.get('name', '').lower()
        if 'vb-audio' in device_name or 'cable' in device_name:
            vb_audio_devices.append((i, device))
            print(f"  {i}: {device.get('name', 'Unknown')} (channels: {device.get('max_input_channels', 0)})")
    
    if vb_audio_devices:
        print("✅ VB-Audio devices found")
        print("💡 Your system is using virtual audio devices")
        print("   This means audio from YouTube should be routed through VB-Audio")
        return True
    else:
        print("❌ No VB-Audio devices found")
        return False

if __name__ == "__main__":
    print("🎤 MICROPHONE ACCESS DIAGNOSTIC")
    print("=" * 60)
    
    # Check microphone permissions
    mic_ok = check_windows_microphone_permissions()
    
    # Check virtual audio setup
    vb_ok = check_virtual_audio_setup()
    
    print("\n📋 SUMMARY:")
    print("=" * 60)
    if mic_ok:
        print("✅ Microphone access: WORKING")
    else:
        print("❌ Microphone access: FAILED")
        
    if vb_ok:
        print("✅ Virtual audio setup: DETECTED")
    else:
        print("❌ Virtual audio setup: NOT FOUND")
        
    if not mic_ok:
        print("\n🛠️ RECOMMENDED ACTIONS:")
        print("1. Check Windows microphone permissions (Settings > Privacy > Microphone)")
        print("2. Make sure your microphone is not muted")
        print("3. Try running this script as Administrator")
        print("4. Test your microphone in Windows Sound settings") 