#!/usr/bin/env python3
"""
Force kill any stuck DanzarAI processes
"""

import os
import subprocess
import psutil
import time

def kill_danzar_processes():
    """Kill all DanzarAI related processes"""
    print("🔍 Looking for DanzarAI processes...")
    
    killed_count = 0
    
    # Look for Python processes that might be DanzarAI
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            
            # Check if this is a DanzarAI process
            if any(keyword in cmdline.lower() for keyword in ['danzar', 'danzarvlm', 'vision', 'llm']):
                print(f"🎯 Found DanzarAI process: PID {proc.info['pid']} - {cmdline[:100]}...")
                
                try:
                    proc.terminate()
                    print(f"   ✅ Terminated PID {proc.info['pid']}")
                    killed_count += 1
                except Exception as e:
                    print(f"   ❌ Failed to terminate PID {proc.info['pid']}: {e}")
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    # Also kill any Python processes that might be stuck
    print("\n🔍 Looking for stuck Python processes...")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'danzar' in cmdline.lower() or 'vision' in cmdline.lower():
                    print(f"🎯 Found Python process: PID {proc.info['pid']} - {cmdline[:100]}...")
                    
                    try:
                        proc.terminate()
                        print(f"   ✅ Terminated Python PID {proc.info['pid']}")
                        killed_count += 1
                    except Exception as e:
                        print(f"   ❌ Failed to terminate Python PID {proc.info['pid']}: {e}")
                        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    print(f"\n📊 Summary: Killed {killed_count} processes")
    
    if killed_count > 0:
        print("⏳ Waiting 2 seconds for processes to terminate...")
        time.sleep(2)
        
        # Force kill any remaining processes
        print("🔨 Force killing any remaining processes...")
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if any(keyword in cmdline.lower() for keyword in ['danzar', 'danzarvlm', 'vision', 'llm']):
                    try:
                        proc.kill()
                        print(f"   💀 Force killed PID {proc.info['pid']}")
                    except Exception as e:
                        print(f"   ❌ Failed to force kill PID {proc.info['pid']}: {e}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    
    print("✅ Process cleanup complete!")

if __name__ == "__main__":
    kill_danzar_processes() 