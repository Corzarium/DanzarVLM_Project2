#!/usr/bin/env python3
"""
Enhanced Log Rotation Setup for DanzarAI
This script ensures clean log files on startup and provides log rotation functionality.
"""

import os
import shutil
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

def setup_enhanced_logging():
    """Set up enhanced logging with rotation and backup."""
    
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
    
    # Set up rotating file handler (keeps last 5 log files, max 10MB each)
    rotating_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    rotating_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(rotating_handler)
    root_logger.addHandler(console_handler)
    
    # Create main logger
    logger = logging.getLogger("DanzarVLM")
    
    # Suppress some noisy loggers
    logging.getLogger('discord').setLevel(logging.WARNING)
    logging.getLogger('discord.gateway').setLevel(logging.INFO)
    logging.getLogger('discord.voice_client').setLevel(logging.INFO)
    
    # Log startup message
    logger.info("🚀 DanzarAI starting up - Enhanced logging enabled")
    logger.info(f"📅 Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("📁 Log rotation: 10MB max size, keeping last 5 files")
    
    return logger

def cleanup_old_logs(max_age_days=7):
    """Clean up log files older than specified days."""
    try:
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            return
        
        current_time = datetime.now()
        cutoff_time = current_time.timestamp() - (max_age_days * 24 * 60 * 60)
        
        cleaned_count = 0
        for filename in os.listdir(logs_dir):
            if filename.startswith('danzar_voice_') and filename.endswith('.log'):
                filepath = os.path.join(logs_dir, filename)
                file_time = os.path.getmtime(filepath)
                
                if file_time < cutoff_time:
                    try:
                        os.remove(filepath)
                        cleaned_count += 1
                        print(f"🗑️ Removed old log: {filename}")
                    except Exception as e:
                        print(f"⚠️ Could not remove {filename}: {e}")
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} old log files")
        else:
            print("✨ No old log files to clean up")
            
    except Exception as e:
        print(f"⚠️ Error during log cleanup: {e}")

if __name__ == "__main__":
    print("🔧 Setting up enhanced logging for DanzarAI...")
    
    # Clean up old logs (older than 7 days)
    cleanup_old_logs(7)
    
    # Set up enhanced logging
    logger = setup_enhanced_logging()
    
    print("✅ Enhanced logging setup complete!")
    print("📋 Features:")
    print("   - Automatic log rotation (10MB max, 5 backup files)")
    print("   - Timestamped backup of previous session")
    print("   - Automatic cleanup of logs older than 7 days")
    print("   - UTF-8 encoding support")
    print("   - Console and file output") 