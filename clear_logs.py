#!/usr/bin/env python3
"""
Simple script to clear old logs before starting DanzarAI
"""

import os
import shutil
from datetime import datetime

def clear_logs():
    """Clear old logs and set up fresh logging."""
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Create timestamped backup of old log if it exists
    log_file = 'logs/danzar_voice.log'
    if os.path.exists(log_file):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f'logs/danzar_voice_{timestamp}.log'
        try:
            shutil.move(log_file, backup_file)
            print(f"📁 Backed up previous log to: {backup_file}")
        except Exception as e:
            print(f"⚠️ Could not backup log file: {e}")
    
    # Clean up old backup files (older than 3 days)
    try:
        logs_dir = 'logs'
        current_time = datetime.now()
        cutoff_time = current_time.timestamp() - (3 * 24 * 60 * 60)  # 3 days
        
        cleaned_count = 0
        for filename in os.listdir(logs_dir):
            if filename.startswith('danzar_voice_') and filename.endswith('.log'):
                filepath = os.path.join(logs_dir, filename)
                file_time = os.path.getmtime(filepath)
                
                if file_time < cutoff_time:
                    try:
                        os.remove(filepath)
                        cleaned_count += 1
                    except Exception:
                        pass
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} old log files")
            
    except Exception:
        pass
    
    print("✅ Logs cleared and ready for fresh session!")

if __name__ == "__main__":
    clear_logs() 