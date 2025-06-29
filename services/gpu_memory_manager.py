# services/gpu_memory_manager.py
"""
Enhanced GPU Memory Manager - Optimized memory allocation and cleanup
"""
import torch
import gc
import logging
import time
import threading
from typing import Dict, Optional, Tuple, List
import psutil
import numpy as np
from core.app_context import AppContext

class GPUMemoryManager:
    """
    GPU Memory Manager for DanzarAI
    Coordinates GPU usage between main LLM (4070) and vision processing (4070 Super)
    """
    
    def __init__(self, app_context: AppContext):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # GPU configuration
        self.gpu_config = self.config.get('gpu_memory', {})
        self.main_llm_device = self.gpu_config.get('main_llm_device', 'cuda:0')
        self.vision_device = self.gpu_config.get('vision_device', 'cuda:1')
        
        # Memory limits
        self.main_llm_reservation_gb = self.gpu_config.get('main_llm_memory_reservation_gb', 8.0)
        self.vision_memory_limit_gb = self.gpu_config.get('vision_memory_limit_gb', 2.0)
        self.cpu_fallback_threshold_gb = self.gpu_config.get('cpu_fallback_threshold_gb', 1.0)
        
        # Monitoring
        self.enable_monitoring = self.gpu_config.get('enable_memory_monitoring', True)
        self.monitor_interval = self.gpu_config.get('memory_check_interval', 30)
        self.monitor_thread = None
        self.monitor_running = False
        
        # Memory tracking
        self.gpu_memory_history = {}
        self.last_memory_check = 0
        
        # Device availability
        self.available_devices = self._detect_available_devices()
        
        # Memory thresholds
        self.critical_threshold = 0.95  # 95% usage triggers cleanup
        self.warning_threshold = 0.85   # 85% usage triggers warning
        self.cleanup_threshold = 0.90   # 90% usage triggers aggressive cleanup
        
        # Track memory usage
        self.last_cleanup_time = 0
        self.cleanup_cooldown = 30  # seconds between cleanups
        
        self.logger.info(f"[GPUMemoryManager] Initialized with {len(self.available_devices)} CUDA devices")
        for device_id, device_info in self.available_devices.items():
            self.logger.info(f"[GPUMemoryManager] Device {device_id}: {device_info['name']} ({device_info['memory_gb']:.1f}GB)")
    
    def _detect_available_devices(self) -> Dict[int, Dict]:
        """Detect available CUDA devices and their properties."""
        devices = {}
        
        if not torch.cuda.is_available():
            self.logger.warning("[GPUMemoryManager] CUDA not available")
            return devices
        
        device_count = torch.cuda.device_count()
        self.logger.info(f"[GPUMemoryManager] Found {device_count} CUDA devices")
        
        for device_id in range(device_count):
            try:
                torch.cuda.set_device(device_id)
                props = torch.cuda.get_device_properties(device_id)
                
                devices[device_id] = {
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multi_processor_count': props.multi_processor_count
                }
                
                self.logger.info(f"[GPUMemoryManager] Device {device_id}: {props.name} ({props.total_memory / (1024**3):.1f}GB)")
                
            except Exception as e:
                self.logger.error(f"[GPUMemoryManager] Error detecting device {device_id}: {e}")
        
        return devices
    
    def get_device_memory_info(self, device_id: int) -> Optional[Dict]:
        """Get current memory usage for a specific device."""
        if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
            return None
        
        try:
            torch.cuda.set_device(device_id)
            
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            total = torch.cuda.get_device_properties(device_id).total_memory
            free = total - reserved
            
            return {
                'allocated_gb': allocated / (1024**3),
                'reserved_gb': reserved / (1024**3),
                'total_gb': total / (1024**3),
                'free_gb': free / (1024**3),
                'utilization_percent': (reserved / total) * 100
            }
        except Exception as e:
            self.logger.error(f"[GPUMemoryManager] Error getting memory info for device {device_id}: {e}")
            return None
    
    def get_all_devices_memory_info(self) -> Dict[int, Dict]:
        """Get memory info for all available devices."""
        memory_info = {}
        
        for device_id in self.available_devices.keys():
            info = self.get_device_memory_info(device_id)
            if info:
                memory_info[device_id] = info
        
        return memory_info
    
    def can_use_device_for_vision(self, device_id: int) -> Tuple[bool, str]:
        """
        Check if a device can be used for vision processing.
        Returns (can_use, reason)
        """
        if device_id not in self.available_devices:
            return False, f"Device {device_id} not available"
        
        memory_info = self.get_device_memory_info(device_id)
        if not memory_info:
            return False, f"Could not get memory info for device {device_id}"
        
        device_name = self.available_devices[device_id]['name']
        free_gb = memory_info['free_gb']
        
        # Check if this is the main LLM device
        if device_id == 0:  # cuda:0 - main LLM device
            if free_gb < self.main_llm_reservation_gb:
                return False, f"Device {device_id} ({device_name}) reserved for main LLM, only {free_gb:.1f}GB free (need {self.main_llm_reservation_gb:.1f}GB)"
        
        # Check if device has enough memory for vision processing
        if free_gb < self.cpu_fallback_threshold_gb:
            return False, f"Device {device_id} ({device_name}) has insufficient memory: {free_gb:.1f}GB free (threshold: {self.cpu_fallback_threshold_gb:.1f}GB)"
        
        return True, f"Device {device_id} ({device_name}) available with {free_gb:.1f}GB free"
    
    def get_best_vision_device(self) -> Tuple[Optional[str], str]:
        """
        Get the best available device for vision processing.
        Returns (device_string, reason)
        """
        # Try vision device first (cuda:1)
        if 1 in self.available_devices:
            can_use, reason = self.can_use_device_for_vision(1)
            if can_use:
                return "cuda:1", reason
        
        # Try main LLM device if it has enough free memory
        if 0 in self.available_devices:
            can_use, reason = self.can_use_device_for_vision(0)
            if can_use:
                return "cuda:0", reason
        
        # Try any other available device
        for device_id in self.available_devices.keys():
            if device_id not in [0, 1]:  # Skip already checked devices
                can_use, reason = self.can_use_device_for_vision(device_id)
                if can_use:
                    return f"cuda:{device_id}", reason
        
        # Fallback to CPU
        return "cpu", "No GPU devices available with sufficient memory"
    
    def log_memory_status(self):
        """Log current memory status for all devices."""
        memory_info = self.get_all_devices_memory_info()
        
        self.logger.info("[GPUMemoryManager] Current GPU Memory Status:")
        for device_id, info in memory_info.items():
            device_name = self.available_devices.get(device_id, {}).get('name', 'Unknown')
            self.logger.info(f"  Device {device_id} ({device_name}): "
                           f"{info['allocated_gb']:.1f}GB allocated, "
                           f"{info['reserved_gb']:.1f}GB reserved, "
                           f"{info['free_gb']:.1f}GB free "
                           f"({info['utilization_percent']:.1f}% utilization)")
    
    def start_memory_monitoring(self):
        """Start background memory monitoring."""
        if not self.enable_monitoring:
            return
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.logger.warning("[GPUMemoryManager] Memory monitoring already running")
            return
        
        self.monitor_running = True
        self.monitor_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"[GPUMemoryManager] Memory monitoring started (interval: {self.monitor_interval}s)")
    
    def stop_memory_monitoring(self):
        """Stop background memory monitoring."""
        self.monitor_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("[GPUMemoryManager] Memory monitoring stopped")
    
    def _memory_monitor_loop(self):
        """Background memory monitoring loop."""
        while self.monitor_running and not self.app_context.shutdown_event.is_set():
            try:
                # Log memory status periodically
                self.log_memory_status()
                
                # Check for memory issues
                self._check_memory_alerts()
                
                # Store memory history
                memory_info = self.get_all_devices_memory_info()
                self.gpu_memory_history[time.time()] = memory_info
                
                # Clean old history (keep last hour)
                cutoff_time = time.time() - 3600
                self.gpu_memory_history = {
                    t: data for t, data in self.gpu_memory_history.items() 
                    if t > cutoff_time
                }
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"[GPUMemoryManager] Error in memory monitoring: {e}")
                time.sleep(10)  # Shorter sleep on error
    
    def _check_memory_alerts(self):
        """Check for memory issues and log alerts."""
        memory_info = self.get_all_devices_memory_info()
        
        for device_id, info in memory_info.items():
            device_name = self.available_devices.get(device_id, {}).get('name', 'Unknown')
            
            # Check for high utilization
            if info['utilization_percent'] > 90:
                self.logger.warning(f"[GPUMemoryManager] High GPU utilization on device {device_id} ({device_name}): {info['utilization_percent']:.1f}%")
            
            # Check for low free memory
            if info['free_gb'] < 1.0:
                self.logger.warning(f"[GPUMemoryManager] Low free memory on device {device_id} ({device_name}): {info['free_gb']:.1f}GB")
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory management recommendations."""
        recommendations = []
        memory_info = self.get_all_devices_memory_info()
        
        for device_id, info in memory_info.items():
            device_name = self.available_devices.get(device_id, {}).get('name', 'Unknown')
            
            if info['utilization_percent'] > 95:
                recommendations.append(f"Device {device_id} ({device_name}) is critically full ({info['utilization_percent']:.1f}%)")
            elif info['utilization_percent'] > 85:
                recommendations.append(f"Device {device_id} ({device_name}) is getting full ({info['utilization_percent']:.1f}%)")
            
            if info['free_gb'] < 0.5:
                recommendations.append(f"Device {device_id} ({device_name}) has very low free memory ({info['free_gb']:.1f}GB)")
        
        if not recommendations:
            recommendations.append("All GPU devices have adequate memory")
        
        return recommendations
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_memory_monitoring()
        self.logger.info("[GPUMemoryManager] Cleanup completed")
    
    def get_gpu_memory_status(self) -> Dict[int, Dict[str, float]]:
        """Get detailed memory status for all GPUs"""
        status = {}
        
        if not torch.cuda.is_available():
            return status
            
        for device_id in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(device_id)
                allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(device_id) / 1024**3    # GB
                total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3  # GB
                free = total - reserved
                utilization = reserved / total
                
                status[device_id] = {
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'free_gb': free,
                    'total_gb': total,
                    'utilization': utilization
                }
                
            except Exception as e:
                self.logger.error(f"[GPUMemoryManager] Error getting memory status for GPU {device_id}: {e}")
                
        return status
    
    def needs_cleanup(self, device_id: int) -> bool:
        """Check if GPU needs memory cleanup"""
        status = self.get_gpu_memory_status()
        if device_id not in status:
            return False
            
        utilization = status[device_id]['utilization']
        time_since_cleanup = time.time() - self.last_cleanup_time
        
        return (utilization > self.cleanup_threshold and 
                time_since_cleanup > self.cleanup_cooldown)
    
    def cleanup_gpu_memory(self, device_id: Optional[int] = None):
        """Clean up GPU memory with aggressive garbage collection"""
        if not torch.cuda.is_available():
            return
            
        self.logger.info(f"[GPUMemoryManager] Starting GPU memory cleanup for device {device_id}")
        
        # Set device if specified
        if device_id is not None:
            torch.cuda.set_device(device_id)
        
        # Aggressive garbage collection
        gc.collect()
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        
        # Force memory release
        if device_id is not None:
            torch.cuda.synchronize(device_id)
        else:
            torch.cuda.synchronize()
        
        # Additional cleanup for specific device
        if device_id is not None:
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats(device_id)
            
            # Try to free more memory
            try:
                # Force garbage collection again
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                self.logger.warning(f"[GPUMemoryManager] Error during additional cleanup: {e}")
        
        self.last_cleanup_time = time.time()
        
        # Log cleanup results
        status = self.get_gpu_memory_status()
        if device_id in status:
            info = status[device_id]
            self.logger.info(f"[GPUMemoryManager] Cleanup complete for GPU {device_id}: "
                           f"{info['free_gb']:.1f}GB free ({info['utilization']*100:.1f}% utilization)")
        else:
            self.logger.info(f"[GPUMemoryManager] Cleanup complete for all GPUs")
    
    def find_best_gpu(self, min_memory_gb: float = 2.0) -> Optional[int]:
        """Find the best GPU with sufficient memory"""
        if not torch.cuda.is_available():
            return None
            
        status = self.get_gpu_memory_status()
        best_device = None
        best_score = -1
        
        for device_id, info in status.items():
            free_gb = info['free_gb']
            utilization = info['utilization']
            
            # Check if device has sufficient memory
            if free_gb < min_memory_gb:
                continue
                
            # Score based on free memory and low utilization
            score = free_gb * (1 - utilization)
            
            if score > best_score:
                best_score = score
                best_device = device_id
        
        return best_device
    
    def allocate_gpu_memory(self, required_gb: float = 1.0) -> Optional[int]:
        """Allocate GPU memory and return device ID"""
        if not torch.cuda.is_available():
            return None
            
        # Try to find a GPU with sufficient memory
        device_id = self.find_best_gpu(required_gb + 0.5)  # Add buffer
        
        if device_id is not None:
            self.logger.info(f"[GPUMemoryManager] Allocated GPU {device_id} for {required_gb}GB requirement")
            return device_id
        
        # If no GPU has sufficient memory, try cleanup and retry
        self.logger.warning(f"[GPUMemoryManager] No GPU has {required_gb}GB free, attempting cleanup...")
        
        # Cleanup all GPUs
        for device_id in range(torch.cuda.device_count()):
            if self.needs_cleanup(device_id):
                self.cleanup_gpu_memory(device_id)
        
        # Try again after cleanup
        device_id = self.find_best_gpu(required_gb + 0.5)
        
        if device_id is not None:
            self.logger.info(f"[GPUMemoryManager] Successfully allocated GPU {device_id} after cleanup")
            return device_id
        
        self.logger.error(f"[GPUMemoryManager] Failed to allocate GPU memory for {required_gb}GB requirement")
        return None
    
    def monitor_memory_usage(self):
        """Monitor memory usage and trigger cleanup if needed"""
        status = self.get_gpu_memory_status()
        
        for device_id, info in status.items():
            utilization = info['utilization']
            
            if utilization > self.critical_threshold:
                self.logger.warning(f"[GPUMemoryManager] Critical memory usage on GPU {device_id}: {utilization*100:.1f}%")
                self.cleanup_gpu_memory(device_id)
            elif utilization > self.warning_threshold:
                self.logger.warning(f"[GPUMemoryManager] High memory usage on GPU {device_id}: {utilization*100:.1f}%")
            elif utilization > self.cleanup_threshold:
                if self.needs_cleanup(device_id):
                    self.logger.info(f"[GPUMemoryManager] Triggering cleanup for GPU {device_id}: {utilization*100:.1f}%")
                    self.cleanup_gpu_memory(device_id)
    
    def optimize_for_vision_processing(self):
        """Optimize GPU memory for vision processing"""
        self.logger.info("[GPUMemoryManager] Optimizing GPU memory for vision processing...")
        
        # Cleanup all GPUs
        for device_id in range(torch.cuda.device_count()):
            self.cleanup_gpu_memory(device_id)
        
        # Set memory fraction for vision models
        try:
            # Reduce memory fraction for better sharing
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
            self.logger.info("[GPUMemoryManager] Set memory fraction to 80% for better sharing")
        except Exception as e:
            self.logger.warning(f"[GPUMemoryManager] Could not set memory fraction: {e}")
        
        # Log final status
        self.log_memory_status() 