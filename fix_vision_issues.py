#!/usr/bin/env python3
"""
Comprehensive Fix for Vision System Issues
==========================================

This script fixes three main issues:
1. OCR giving garbage text - Improved filtering and confidence thresholds
2. LLM overload - Reduced frequency and added rate limiting
3. TTS not working for direct conversation - Fixed callback configuration

Run this script to apply all fixes at once.
"""

import os
import re
import shutil
from pathlib import Path

def fix_ocr_config():
    """Fix OCR configuration to reduce garbage text"""
    print("🔧 Fixing OCR configuration...")
    
    config_path = "config/vision_config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Read current config
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Improved OCR configuration
    improved_ocr_config = """# OCR Settings
ocr:
  enabled: true                      # Enable OCR processing
  roi: [50, 50, 800, 600]           # Larger region of interest [x1, y1, x2, y2] for better text detection
  tesseract_config: "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[]{}'\"-+=/\\|@#$%^&*<> "  # Improved config with character whitelist
  min_confidence: 0.7                # Increased confidence threshold to reduce garbage text
  max_text_length: 200               # Maximum text length to prevent very long garbage
  min_text_length: 2                 # Minimum text length to filter out single characters
  filter_garbage: true               # Enable garbage text filtering
  garbage_patterns:                  # Patterns to filter out as garbage
    - "^[^a-zA-Z0-9]*$"             # No alphanumeric characters
    - "^[0-9]{1,2}$"                # Just 1-2 digits
    - "^[a-zA-Z]{1,2}$"             # Just 1-2 letters
    - "^[^a-zA-Z]*[a-zA-Z][^a-zA-Z]*$"  # Single letter surrounded by non-letters
    - "^[^0-9]*[0-9][^0-9]*$"       # Single digit surrounded by non-digits
    - "^[\\W_]+$"                   # Only special characters
    - "^[\\s\\W_]+$"                # Only whitespace and special characters
  valid_text_patterns:               # Patterns that indicate valid text
    - "[a-zA-Z]{3,}"                # At least 3 consecutive letters
    - "[0-9]{2,}"                   # At least 2 consecutive digits
    - "[a-zA-Z][0-9]"               # Letter followed by digit
    - "[0-9][a-zA-Z]"               # Digit followed by letter"""
    
    # Replace OCR section
    pattern = r'# OCR Settings\nocr:.*?(?=\n# |$)'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, improved_ocr_config, content, flags=re.DOTALL)
    else:
        # Add OCR section if not found
        content = content.replace('# Event Debouncing Settings', f'{improved_ocr_config}\n\n# Event Debouncing Settings')
    
    # Write updated config
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ OCR configuration updated")
    return True

def fix_vision_commentary_config():
    """Fix vision commentary configuration to reduce LLM overload"""
    print("🔧 Fixing vision commentary configuration...")
    
    config_path = "config/global_settings.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Read current config
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Improved vision commentary configuration
    improved_commentary_config = """# Vision Commentary Configuration
VISION_COMMENTARY:
  enabled: true
  frequency_seconds: 30.0  # Increased from 15.0 to 30.0 to reduce LLM load
  min_confidence: 0.7  # Increased from 0.6 to 0.7 for higher quality commentary
  max_length: 80  # Reduced from 100 to 80 for shorter commentary
  conversation_mode: false  # Disable conversational approach - allow commentary without conversation
  wait_for_response: true  # Wait for user responses
  full_screenshot_interval: 120.0  # Send full screenshots every 120 seconds (increased)
  clip_enabled: true  # Enable CLIP video understanding
  clip_monochrome: true  # Use monochrome images for CLIP
  stm_enabled: true  # Enable short-term memory integration
  debug_tts: true  # Enable TTS debugging
  commentary_cooldown: 20.0  # Increased from 10.0 to 20.0 seconds between commentary
  rate_limit_enabled: true  # Enable rate limiting to prevent LLM overload
  max_commentary_per_minute: 2  # Maximum 2 commentary events per minute
  llm_timeout_seconds: 30  # Timeout for LLM requests to prevent hanging"""
    
    # Replace VISION_COMMENTARY section
    pattern = r'# Vision Commentary Configuration\nVISION_COMMENTARY:.*?(?=\n# |$)'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, improved_commentary_config, content, flags=re.DOTALL)
    else:
        # Add VISION_COMMENTARY section if not found
        content = content.replace('# Vision Processing Configuration', f'{improved_commentary_config}\n\n# Vision Processing Configuration')
    
    # Write updated config
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Vision commentary configuration updated")
    return True

def fix_tts_callback():
    """Fix TTS callback issue in DanzarVLM.py"""
    print("🔧 Fixing TTS callback in DanzarVLM.py...")
    
    file_path = "DanzarVLM.py"
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    # Read current file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the tts_callback=None line
    old_pattern = r'tts_callback=None  # No TTS for text chat'
    new_pattern = 'tts_callback=tts_callback  # Enable TTS for text chat'
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        
        # Write updated file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ TTS callback fixed in DanzarVLM.py")
        return True
    else:
        print("⚠️ Could not find tts_callback=None line to fix")
        return False

def add_ocr_filtering_service():
    """Add OCR filtering service to improve text quality"""
    print("🔧 Adding OCR filtering service...")
    
    service_path = "services/ocr_filter_service.py"
    
    ocr_filter_code = '''"""
OCR Filter Service for DanzarAI
Improves OCR text quality by filtering out garbage text
"""

import re
import logging
from typing import List, Optional, Tuple

class OCRFilterService:
    """Service for filtering and improving OCR text quality"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.garbage_patterns = [
            re.compile(r'^[^a-zA-Z0-9]*$'),  # No alphanumeric characters
            re.compile(r'^[0-9]{1,2}$'),     # Just 1-2 digits
            re.compile(r'^[a-zA-Z]{1,2}$'),  # Just 1-2 letters
            re.compile(r'^[^a-zA-Z]*[a-zA-Z][^a-zA-Z]*$'),  # Single letter surrounded by non-letters
            re.compile(r'^[^0-9]*[0-9][^0-9]*$'),  # Single digit surrounded by non-digits
            re.compile(r'^[\\W_]+$'),        # Only special characters
            re.compile(r'^[\\s\\W_]+$'),     # Only whitespace and special characters
        ]
        
        self.valid_patterns = [
            re.compile(r'[a-zA-Z]{3,}'),     # At least 3 consecutive letters
            re.compile(r'[0-9]{2,}'),        # At least 2 consecutive digits
            re.compile(r'[a-zA-Z][0-9]'),    # Letter followed by digit
            re.compile(r'[0-9][a-zA-Z]'),    # Digit followed by letter
        ]
    
    def filter_text(self, text: str, min_confidence: float = 0.7, 
                   min_length: int = 2, max_length: int = 200) -> Optional[str]:
        """
        Filter OCR text to remove garbage and improve quality
        
        Args:
            text: Raw OCR text
            min_confidence: Minimum confidence threshold
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Filtered text or None if filtered out
        """
        if not text or not text.strip():
            return None
        
        # Clean the text
        cleaned_text = text.strip()
        
        # Check length constraints
        if len(cleaned_text) < min_length or len(cleaned_text) < max_length:
            self.logger.debug(f"OCR text filtered by length: {cleaned_text}")
            return None
        
        # Check for garbage patterns
        for pattern in self.garbage_patterns:
            if pattern.match(cleaned_text):
                self.logger.debug(f"OCR text filtered by garbage pattern: {cleaned_text}")
                return None
        
        # Check for valid patterns (at least one must match)
        has_valid_pattern = False
        for pattern in self.valid_patterns:
            if pattern.search(cleaned_text):
                has_valid_pattern = True
                break
        
        if not has_valid_pattern:
            self.logger.debug(f"OCR text filtered - no valid patterns: {cleaned_text}")
            return None
        
        # Additional quality checks
        if self._is_likely_garbage(cleaned_text):
            self.logger.debug(f"OCR text filtered by quality check: {cleaned_text}")
            return None
        
        self.logger.debug(f"OCR text passed filtering: {cleaned_text}")
        return cleaned_text
    
    def _is_likely_garbage(self, text: str) -> bool:
        """Additional heuristic checks for garbage text"""
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text)
        if special_char_ratio > 0.5:
            return True
        
        # Check for repeated characters
        for char in text:
            if text.count(char) > len(text) * 0.7:  # More than 70% same character
                return True
        
        # Check for alternating patterns
        if len(text) > 4:
            alternating = all(text[i] != text[i+1] for i in range(len(text)-1))
            if alternating and len(set(text)) <= 2:
                return True
        
        return False
    
    def batch_filter(self, texts: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Filter a batch of OCR texts with their confidence scores
        
        Args:
            texts: List of (text, confidence) tuples
            
        Returns:
            Filtered list of (text, confidence) tuples
        """
        filtered_texts = []
        
        for text, confidence in texts:
            if confidence >= 0.7:  # Only process high-confidence text
                filtered_text = self.filter_text(text)
                if filtered_text:
                    filtered_texts.append((filtered_text, confidence))
        
        return filtered_texts
'''
    
    with open(service_path, 'w', encoding='utf-8') as f:
        f.write(ocr_filter_code)
    
    print("✅ OCR filter service created")
    return True

def create_test_script():
    """Create a test script to verify the fixes"""
    print("🔧 Creating test script...")
    
    test_script = '''#!/usr/bin/env python3
"""
Test script to verify vision system fixes
"""

import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ocr_filtering():
    """Test OCR filtering service"""
    print("🧪 Testing OCR filtering...")
    
    try:
        from services.ocr_filter_service import OCRFilterService
        
        filter_service = OCRFilterService(logger)
        
        # Test cases
        test_cases = [
            ("Hello World", True),      # Should pass
            ("123", False),             # Should fail (too short)
            ("a", False),               # Should fail (too short)
            ("!!!", False),             # Should fail (garbage)
            ("abc123", True),           # Should pass
            ("", False),                # Should fail (empty)
            ("   ", False),             # Should fail (whitespace)
            ("This is valid text", True), # Should pass
        ]
        
        passed = 0
        for text, should_pass in test_cases:
            result = filter_service.filter_text(text)
            if (result is not None) == should_pass:
                passed += 1
                print(f"✅ {text} -> {result}")
            else:
                print(f"❌ {text} -> {result} (expected: {should_pass})")
        
        print(f"OCR filtering test: {passed}/{len(test_cases)} passed")
        return passed == len(test_cases)
        
    except Exception as e:
        print(f"❌ OCR filtering test failed: {e}")
        return False

async def test_config_files():
    """Test configuration files"""
    print("🧪 Testing configuration files...")
    
    config_files = [
        "config/vision_config.yaml",
        "config/global_settings.yaml"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file} exists")
        else:
            print(f"❌ {config_file} missing")
            return False
    
    return True

async def test_tts_callback():
    """Test TTS callback fix"""
    print("🧪 Testing TTS callback fix...")
    
    try:
        with open("DanzarVLM.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if TTS callback is properly configured
        if "tts_callback=tts_callback  # Enable TTS for text chat" in content:
            print("✅ TTS callback properly configured")
            return True
        else:
            print("❌ TTS callback not properly configured")
            return False
            
    except Exception as e:
        print(f"❌ TTS callback test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Running vision system fix tests...")
    
    tests = [
        test_ocr_filtering,
        test_config_files,
        test_tts_callback
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Vision system fixes are working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the configuration.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open("test_vision_fixes.py", 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("✅ Test script created")
    return True

def main():
    """Main function to apply all fixes"""
    print("🔧 DanzarAI Vision System Fixes")
    print("=" * 40)
    
    fixes = [
        ("OCR Configuration", fix_ocr_config),
        ("Vision Commentary Configuration", fix_vision_commentary_config),
        ("TTS Callback", fix_tts_callback),
        ("OCR Filter Service", add_ocr_filtering_service),
        ("Test Script", create_test_script),
    ]
    
    results = []
    for name, fix_func in fixes:
        print(f"\\n🔧 Applying {name} fix...")
        try:
            result = fix_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} fix failed: {e}")
            results.append((name, False))
    
    print("\\n📊 Fix Results:")
    print("=" * 40)
    
    passed = 0
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\\n🎯 Summary: {passed}/{len(results)} fixes applied successfully")
    
    if passed == len(results):
        print("\\n🎉 All fixes applied successfully!")
        print("\\n📋 Next steps:")
        print("1. Restart DanzarAI to apply the changes")
        print("2. Run: python test_vision_fixes.py")
        print("3. Test OCR quality and TTS functionality")
        print("4. Monitor LLM response times")
    else:
        print("\\n⚠️ Some fixes failed. Please check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    main() 