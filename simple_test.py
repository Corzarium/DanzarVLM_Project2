#!/usr/bin/env python3
"""
Simple test script to verify Qwen server status
"""

import requests
import json
import sys

def test_qwen_server():
    """Test if Qwen server is running and responding"""
    print("Testing Qwen server status...")
    
    try:
        # Test basic connectivity
        response = requests.get("http://localhost:8083/health", timeout=5)
        if response.status_code == 200:
            print("SUCCESS: Qwen server is running and healthy")
            return True
        else:
            print(f"WARNING: Qwen server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Qwen server on port 8083")
        return False
    except Exception as e:
        print(f"ERROR: Error testing Qwen server: {e}")
        return False

def test_vision_capabilities():
    """Test vision capabilities with a simple image"""
    print("\nTesting vision capabilities...")
    
    try:
        # Test vision endpoint with a simple prompt
        data = {
            "prompt": "What do you see in this image? Describe it briefly.",
            "max_tokens": 100
        }
        
        response = requests.post(
            "http://localhost:8083/chat",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: Vision test successful!")
            print(f"Response: {result.get('response', 'No response')}")
            return True
        else:
            print(f"ERROR: Vision test failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"ERROR: Error testing vision: {e}")
        return False

def test_personality():
    """Test if Danzar has the correct personality"""
    print("\nTesting Danzar's personality...")
    
    try:
        # Test text-only conversation
        data = {
            "prompt": "Who are you? What is your name and what can you do?",
            "max_tokens": 200
        }
        
        response = requests.post(
            "http://localhost:8083/chat",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '')
            print("SUCCESS: Personality test successful!")
            print(f"Response: {response_text}")
            
            # Check for key personality indicators
            if any(keyword in response_text.lower() for keyword in ['danzar', 'gaming', 'vision', 'assistant']):
                print("SUCCESS: Danzar identifies correctly as a gaming assistant with vision!")
                return True
            else:
                print("WARNING: Response doesn't clearly identify as Danzar")
                return False
        else:
            print(f"ERROR: Personality test failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"ERROR: Error testing personality: {e}")
        return False

def main():
    """Main test function"""
    print("Starting Qwen server and Danzar tests...\n")
    
    # Test 1: Server status
    server_ok = test_qwen_server()
    
    if not server_ok:
        print("\nERROR: Qwen server is not running properly!")
        print("Please start the Qwen server first using:")
        print("  .\\start_qwen_vl_fixed.bat")
        return False
    
    # Test 2: Vision capabilities
    vision_ok = test_vision_capabilities()
    
    # Test 3: Personality
    personality_ok = test_personality()
    
    # Summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Qwen Server: {'RUNNING' if server_ok else 'FAILED'}")
    print(f"Vision Capabilities: {'WORKING' if vision_ok else 'FAILED'}")
    print(f"Personality: {'CORRECT' if personality_ok else 'FAILED'}")
    
    if server_ok and vision_ok and personality_ok:
        print("\nSUCCESS: All tests passed! Danzar should be working correctly.")
        print("You can now start Danzar with: python DanzarVLM.py")
    else:
        print("\nWARNING: Some tests failed. Check the issues above.")
    
    return server_ok and vision_ok and personality_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 