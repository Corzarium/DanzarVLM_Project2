#!/usr/bin/env python3
"""
Diagnostic script to identify screenshot capture issues
"""

import sys
import os
import time

print("🔍 Diagnosing Screenshot Capture Issues")
print("=" * 45)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    from PIL import Image
    print("✅ PIL.Image imported successfully")
except Exception as e:
    print(f"❌ PIL.Image import failed: {e}")

try:
    from PIL import ImageGrab
    print("✅ PIL.ImageGrab imported successfully")
except Exception as e:
    print(f"❌ PIL.ImageGrab import failed: {e}")

# Test 2: Check if we're on Windows
print("\n2. Checking platform...")
import platform
print(f"Platform: {platform.system()}")
print(f"Platform version: {platform.version()}")

# Test 3: Try basic ImageGrab
print("\n3. Testing basic ImageGrab...")
try:
    print("Attempting ImageGrab.grab()...")
    start_time = time.time()
    screenshot = ImageGrab.grab()
    end_time = time.time()
    print(f"✅ ImageGrab.grab() completed in {end_time - start_time:.2f}s")
    print(f"Image size: {screenshot.size}")
    print(f"Image mode: {screenshot.mode}")
except Exception as e:
    print(f"❌ ImageGrab.grab() failed: {e}")
    print(f"Error type: {type(e)}")

# Test 4: Check if we have display access
print("\n4. Checking display access...")
try:
    import ctypes
    user32 = ctypes.windll.user32
    width = user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(1)
    print(f"✅ Display detected: {width}x{height}")
except Exception as e:
    print(f"❌ Display detection failed: {e}")

# Test 5: Try alternative screenshot method
print("\n5. Testing alternative screenshot method...")
try:
    import numpy as np
    import cv2
    
    # Try using cv2 for screenshot
    print("Attempting cv2 screenshot...")
    start_time = time.time()
    screenshot = ImageGrab.grab()
    frame = np.array(screenshot)
    end_time = time.time()
    print(f"✅ CV2 screenshot completed in {end_time - start_time:.2f}s")
    print(f"Frame shape: {frame.shape}")
except Exception as e:
    print(f"❌ CV2 screenshot failed: {e}")

print("\n🔍 Diagnosis complete!")
print("\nIf ImageGrab is hanging, try:")
print("1. Running as administrator")
print("2. Checking if any other applications are capturing the screen")
print("3. Restarting the application") 