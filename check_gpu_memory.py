#!/usr/bin/env python3
"""
GPU Memory Checker for DanzarAI
Checks available GPU memory and recommends the best GPU for Whisper.
"""

import torch
import subprocess
import sys

def get_gpu_memory_info():
    """Get detailed GPU memory information using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpu_info.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'total_mb': int(parts[2]),
                            'free_mb': int(parts[3]),
                            'used_mb': int(parts[4])
                        })
            return gpu_info
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
    return []

def get_torch_gpu_info():
    """Get GPU information using PyTorch."""
    if not torch.cuda.is_available():
        return []
    
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        total = props.total_memory
        
        gpu_info.append({
            'index': i,
            'name': props.name,
            'total_mb': total // (1024 * 1024),
            'allocated_mb': allocated // (1024 * 1024),
            'reserved_mb': reserved // (1024 * 1024),
            'free_mb': (total - reserved) // (1024 * 1024)
        })
    
    return gpu_info

def main():
    print("🔍 GPU Memory Analysis for DanzarAI")
    print("=" * 50)
    
    # Get GPU info from nvidia-smi
    nvidia_gpus = get_gpu_memory_info()
    print(f"📊 Found {len(nvidia_gpus)} GPUs via nvidia-smi:")
    
    for gpu in nvidia_gpus:
        print(f"  GPU {gpu['index']}: {gpu['name']}")
        print(f"    Memory: {gpu['free_mb']}MB free / {gpu['total_mb']}MB total")
        print(f"    Usage: {gpu['used_mb']}MB used ({gpu['used_mb']/gpu['total_mb']*100:.1f}%)")
    
    print()
    
    # Get GPU info from PyTorch
    torch_gpus = get_torch_gpu_info()
    if torch_gpus:
        print(f"🎯 PyTorch CUDA GPUs ({len(torch_gpus)} found):")
        for gpu in torch_gpus:
            print(f"  GPU {gpu['index']}: {gpu['name']}")
            print(f"    Memory: {gpu['free_mb']}MB free / {gpu['total_mb']}MB total")
            print(f"    PyTorch: {gpu['allocated_mb']}MB allocated, {gpu['reserved_mb']}MB reserved")
    else:
        print("⚠️ No PyTorch CUDA GPUs found")
    
    print()
    
    # Find best GPU for Whisper based on nvidia-smi (more accurate for overall memory)
    best_gpu = None
    max_free_memory = 0
    
    for gpu in nvidia_gpus:
        if gpu['free_mb'] > max_free_memory:
            max_free_memory = gpu['free_mb']
            best_gpu = gpu
    
    if best_gpu:
        print(f"🎯 Recommended GPU for Whisper: GPU {best_gpu['index']} ({best_gpu['name']})")
        print(f"   Free memory: {best_gpu['free_mb']}MB ({best_gpu['free_mb']/1024:.1f}GB)")
        
        if best_gpu['free_mb'] > 2048:  # 2GB
            print("✅ Sufficient memory for Whisper large model")
        elif best_gpu['free_mb'] > 1024:  # 1GB
            print("⚠️ Limited memory - consider using Whisper medium or base model")
        else:
            print("❌ Insufficient memory - recommend using CPU or smaller model")
        
        print()
        print("🔧 Configuration recommendation:")
        print(f"WHISPER_GPU_CONFIG:")
        print(f"  device: cuda:{best_gpu['index']}")
        print(f"  compute_type: float16")
        print(f"  memory_fraction: 0.8")
        
        # Also show the PyTorch device mapping
        print()
        print("📋 GPU Device Mapping:")
        print("   nvidia-smi GPU index -> PyTorch cuda device")
        for nvidia_gpu in nvidia_gpus:
            for torch_gpu in torch_gpus:
                if nvidia_gpu['name'] == torch_gpu['name']:
                    print(f"   GPU {nvidia_gpu['index']} ({nvidia_gpu['name']}) -> cuda:{torch_gpu['index']}")
                    break
    else:
        print("❌ No suitable GPU found - recommend using CPU")
        print()
        print("🔧 Configuration recommendation:")
        print("WHISPER_GPU_CONFIG:")
        print("  device: cpu")
        print("  compute_type: float32")

if __name__ == "__main__":
    main() 