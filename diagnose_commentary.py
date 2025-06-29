#!/usr/bin/env python3
"""
Diagnostic script to check commentary system status
"""
import requests
import json
import time
from datetime import datetime

def check_vlm_server():
    """Check if VLM server is running and responding"""
    try:
        # Check health endpoint
        response = requests.get("http://localhost:8083/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ VLM server health check passed at {datetime.now()}")
            return True
        else:
            print(f"❌ VLM server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ VLM server not accessible: {e}")
        return False

def test_vlm_chat():
    """Test VLM chat completion"""
    try:
        payload = {
            "messages": [
                {"role": "user", "content": "Say hello briefly"}
            ],
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 50
        }
        
        response = requests.post(
            "http://localhost:8083/v1/chat/completions",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✅ VLM chat working: {content}")
            return True
        else:
            print(f"❌ VLM chat failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ VLM chat error: {e}")
        return False

def main():
    print(f"=== Commentary System Diagnostic at {datetime.now()} ===")
    print()
    
    # Check VLM server
    print("1. Checking VLM server...")
    vlm_ok = check_vlm_server()
    
    if vlm_ok:
        print("2. Testing VLM chat...")
        chat_ok = test_vlm_chat()
        
        if chat_ok:
            print("\n✅ VLM server is working correctly!")
            print("\n=== Next Steps ===")
            print("1. Start the main app: python DanzarVLM.py")
            print("2. Use !watch command in Discord")
            print("3. Check for 'VisionIntegration' logs in the terminal")
            print("4. Look for commentary generation messages")
        else:
            print("\n❌ VLM chat is not working")
    else:
        print("\n❌ VLM server is not running")
        print("\n=== To Start VLM Server ===")
        print("1. Open PowerShell in the project directory")
        print("2. Run these commands:")
        print("   cd llama-cpp-cuda")
        print("   $env:CUDA_VISIBLE_DEVICES='1'")
        print("   .\\llama-server.exe --model ..\\models-gguf\\Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --mmproj ..\\models-gguf\\Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf --host 0.0.0.0 --port 8083 --ctx-size 2048 --gpu-layers 30 --threads 4 --temp 0.7 --repeat-penalty 1.1 --n-predict 128 --n-keep 64 --rope-scaling linear --rope-freq-base 10000 --rope-freq-scale 0.5 --mul-mat-q --no-mmap --no-mlock")
        print("\n3. Wait for server to start (you'll see 'server is listening on http://0.0.0.0:8083')")
        print("4. Then start the main app: python DanzarVLM.py")

if __name__ == "__main__":
    main() 