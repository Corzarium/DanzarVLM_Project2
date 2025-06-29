#!/usr/bin/env python3
"""
Installation script for CLIP vision integration.
Installs CLIP and required dependencies for enhanced visual understanding.
"""

import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and log the result."""
    logger.info(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        logger.error(f"   Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    logger.info(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_clip_dependencies():
    """Install CLIP and its dependencies."""
    logger.info("🚀 Installing CLIP Vision Integration")
    logger.info("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install PyTorch first (if not already installed)
    logger.info("📦 Installing PyTorch...")
    if not run_command("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118", "PyTorch installation"):
        logger.warning("⚠️ PyTorch installation failed, trying CPU-only version...")
        if not run_command("pip install torch torchvision", "PyTorch CPU installation"):
            return False
    
    # Install CLIP
    logger.info("📦 Installing CLIP...")
    if not run_command("pip install git+https://github.com/openai/CLIP.git", "CLIP installation"):
        return False
    
    # Install additional dependencies
    dependencies = [
        ("Pillow>=9.0.0", "Pillow"),
        ("numpy>=1.21.0", "NumPy"),
        ("opencv-python>=4.5.0", "OpenCV"),
        ("transformers>=4.20.0", "Transformers")
    ]
    
    for dep, name in dependencies:
        if not run_command(f"pip install {dep}", f"{name} installation"):
            logger.warning(f"⚠️ {name} installation failed, continuing...")
    
    return True

def test_clip_installation():
    """Test if CLIP is working correctly."""
    logger.info("🧪 Testing CLIP installation...")
    
    try:
        import torch
        import clip
        from PIL import Image
        import numpy as np
        
        logger.info("✅ All imports successful")
        
        # Test CLIP model loading
        logger.info("🔧 Testing CLIP model loading...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        logger.info(f"✅ CLIP model loaded successfully on {device}")
        
        # Test basic functionality
        logger.info("🔧 Testing CLIP basic functionality...")
        
        # Create a simple test image
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        test_text = ["a cat", "a dog", "a car"]
        
        # Process image and text
        image_input = preprocess(test_image).unsqueeze(0).to(device)
        text_input = clip.tokenize(test_text).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            
            # Calculate similarity
            similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
        
        logger.info("✅ CLIP similarity calculation successful")
        logger.info(f"   Similarity scores: {similarity.cpu().numpy()}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ CLIP test failed: {e}")
        return False

def create_clip_config():
    """Create CLIP configuration file."""
    config_content = """# CLIP Vision Integration Configuration
clip_integration:
  enabled: true
  model: "ViT-B/32"  # CLIP model variant
  device: "cuda"     # GPU acceleration (auto-detected)
  confidence_threshold: 0.3
  max_concepts: 10
  game_specific_concepts: true

# Game-specific visual concepts
game_concepts:
  everquest:
    - "health bar"
    - "mana bar" 
    - "inventory window"
    - "spell book"
    - "chat window"
    - "character portrait"
    - "experience bar"
    - "compass"
    - "map"
    - "group window"
  
  rimworld:
    - "colonist"
    - "room"
    - "bed"
    - "table"
    - "crop"
    - "animal"
    - "workbench"
    - "power generator"
  
  generic_game:
    - "health bar"
    - "inventory"
    - "menu"
    - "button"
    - "text"
    - "character"
    - "enemy"
    - "weapon"
    - "armor"
    - "potion"
"""
    
    config_path = "config/clip_config.yaml"
    os.makedirs("config", exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            f.write(config_content)
        logger.info(f"✅ Created CLIP configuration: {config_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create config: {e}")
        return False

def main():
    """Main installation function."""
    logger.info("🎯 CLIP Vision Integration Installer")
    logger.info("=" * 50)
    
    # Install dependencies
    if not install_clip_dependencies():
        logger.error("❌ Installation failed")
        return False
    
    # Test installation
    if not test_clip_installation():
        logger.error("❌ CLIP test failed")
        return False
    
    # Create configuration
    create_clip_config()
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 CLIP Vision Integration installed successfully!")
    logger.info("\n📋 Next steps:")
    logger.info("   1. CLIP is now available in your vision-aware conversation system")
    logger.info("   2. The CLIPVisionEnhancer service will automatically initialize")
    logger.info("   3. Your VLM can now understand visual context semantically")
    logger.info("   4. Test with: python test_clip_vision_integration.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 