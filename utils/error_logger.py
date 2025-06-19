import os
import sys
import time
import traceback
import logging
from datetime import datetime
from pathlib import Path

class ErrorLogger:
    def __init__(self, log_dir="logs/errors"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("DanzarVLM.ErrorLogger")

    def get_latest_error(self):
        """Read the most recent error log"""
        try:
            error_files = list(self.log_dir.glob("error_*.log"))
            if not error_files:
                return None
                
            latest_file = max(error_files, key=lambda p: p.stat().st_mtime)
            with open(latest_file, 'r') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to read latest error: {e}")
            return None

    def get_error_summary(self, max_errors=5):
        """Get summary of recent errors"""
        try:
            error_files = sorted(
                self.log_dir.glob("error_*.log"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )[:max_errors]
            
            summaries = []
            for file in error_files:
                with open(file, 'r') as f:
                    lines = f.readlines()
                    # Extract key info from error file
                    timestamp = next((l for l in lines if "Timestamp:" in l), "").strip()
                    service = next((l for l in lines if "Service:" in l), "").strip()
                    error_type = next((l for l in lines if "Error Type:" in l), "").strip()
                    error_msg = next((l for l in lines if "Error Message:" in l), "").strip()
                    
                    summaries.append({
                        'file': file.name,
                        'timestamp': timestamp.replace("Timestamp: ", ""),
                        'service': service.replace("Service: ", ""),
                        'error_type': error_type.replace("Error Type: ", ""),
                        'error_msg': error_msg.replace("Error Message: ", "")
                    })
            return summaries
            
        except Exception as e:
            self.logger.error(f"Failed to generate error summary: {e}")
            return []

    def log_error(self, error, context=None, service_name=None):
        """Log error with full traceback and context"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"error_{timestamp}.log"
            filepath = self.log_dir / filename

            with open(filepath, 'w') as f:
                # Add section markers for easier parsing
                f.write("=== DanzarVLM Error Report ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                if service_name:
                    f.write(f"Service: {service_name}\n")
                f.write(f"Error Type: {type(error).__name__}\n")
                f.write(f"Error Message: {str(error)}\n\n")

                if context:
                    f.write("=== Context ===\n")
                    for key, value in context.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")

                f.write("=== Traceback ===\n")
                f.write(''.join(traceback.format_exception(type(error), error, error.__traceback__)))
                
                # Add system info
                f.write("\n=== System Info ===\n")
                f.write(f"Python Version: {sys.version}\n")
                f.write(f"Platform: {sys.platform}\n")
                f.write(f"Working Directory: {os.getcwd()}\n")

            self.logger.error(f"Error logged to: {filepath}")
            
            # Print summary to console for immediate feedback
            print(f"\nError Summary:")
            print(f"Type: {type(error).__name__}")
            print(f"Message: {str(error)}")
            print(f"Log: {filepath}\n")
            
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to log error: {e}")
            return None
