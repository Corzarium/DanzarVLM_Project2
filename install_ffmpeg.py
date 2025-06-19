#!/usr/bin/env python3
"""
FFmpeg Installation Script for Windows Discord voice functionality
Downloads and installs FFmpeg for Discord.py voice functionality
"""

import subprocess
import sys
import os
import platform
import urllib.request
import zipfile
import shutil
from pathlib import Path

def run_command(cmd, description=""):
    """Run command and handle errors"""
    print(f"🔧 {description}")
    print(f"   Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   ❌ Stderr: {e.stderr}")
        return False

def check_ffmpeg_installed():
    """Check if FFmpeg is already installed and accessible"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is already installed and accessible")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ FFmpeg not found in PATH")
    return False

def install_ffmpeg_windows():
    """Install FFmpeg on Windows"""
    print("🔧 Installing FFmpeg for Windows...")
    
    # Create ffmpeg directory
    ffmpeg_dir = Path("ffmpeg")
    ffmpeg_dir.mkdir(exist_ok=True)
    
    # Download FFmpeg
    ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_file = ffmpeg_dir / "ffmpeg.zip"
    
    print(f"📥 Downloading FFmpeg from {ffmpeg_url}")
    try:
        urllib.request.urlretrieve(ffmpeg_url, zip_file)
        print("✅ FFmpeg downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to download FFmpeg: {e}")
        return False
    
    # Extract FFmpeg
    print("📦 Extracting FFmpeg...")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(ffmpeg_dir)
        
        # Find the extracted directory (it has a version number)
        extracted_dirs = [d for d in ffmpeg_dir.iterdir() if d.is_dir() and d.name.startswith('ffmpeg-')]
        if not extracted_dirs:
            print("❌ Could not find extracted FFmpeg directory")
            return False
        
        extracted_dir = extracted_dirs[0]
        bin_dir = extracted_dir / "bin"
        
        if not bin_dir.exists():
            print("❌ FFmpeg bin directory not found")
            return False
        
        # Copy ffmpeg.exe to project directory
        ffmpeg_exe = bin_dir / "ffmpeg.exe"
        if ffmpeg_exe.exists():
            shutil.copy2(ffmpeg_exe, "ffmpeg.exe")
            print("✅ FFmpeg.exe copied to project directory")
        else:
            print("❌ ffmpeg.exe not found in extracted files")
            return False
        
        # Cleanup
        zip_file.unlink()
        shutil.rmtree(extracted_dir)
        
        print("✅ FFmpeg installation completed")
        return True
        
    except Exception as e:
        print(f"❌ Failed to extract FFmpeg: {e}")
        return False

def install_discord_voice_dependencies():
    """Install Discord voice dependencies"""
    print("🔧 Installing Discord voice dependencies...")
    
    dependencies = [
        "PyNaCl",  # For voice encryption
        "ffmpeg-python"  # Python FFmpeg wrapper
    ]
    
    for dep in dependencies:
        success = run_command(
            [sys.executable, "-m", "pip", "install", dep],
            f"Installing {dep}"
        )
        if not success:
            print(f"⚠️ Failed to install {dep}, but continuing...")
    
    return True

def test_ffmpeg():
    """Test FFmpeg installation"""
    print("🧪 Testing FFmpeg installation...")
    
    # Test local ffmpeg.exe
    if os.path.exists("ffmpeg.exe"):
        try:
            result = subprocess.run(["./ffmpeg.exe", "-version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ Local FFmpeg.exe is working")
                return True
        except Exception as e:
            print(f"❌ Local FFmpeg test failed: {e}")
    
    # Test system FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ System FFmpeg is working")
            return True
    except Exception as e:
        print(f"❌ System FFmpeg test failed: {e}")
    
    print("❌ FFmpeg test failed")
    return False

def main():
    print("=" * 60)
    print("🎵 FFmpeg Installation for Discord Voice")
    print("=" * 60)
    
    # Check if already installed
    if check_ffmpeg_installed():
        print("✅ FFmpeg already available, skipping installation")
    else:
        # Install FFmpeg
        if platform.system() == "Windows":
            if not install_ffmpeg_windows():
                print("❌ FFmpeg installation failed")
                return False
        else:
            print("❌ This script is designed for Windows. Please install FFmpeg manually.")
            return False
    
    # Install Discord voice dependencies
    install_discord_voice_dependencies()
    
    # Test installation
    if test_ffmpeg():
        print("\n🎉 FFmpeg installation successful!")
        print("✅ Discord voice functionality should now work")
        return True
    else:
        print("\n❌ FFmpeg installation verification failed")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Installation failed: {e}")
        sys.exit(1) 