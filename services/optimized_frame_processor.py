# services/optimized_frame_processor.py
import time
import threading
import queue
import logging
from typing import Optional, Callable, Dict, Any
import numpy as np
import cv2

class OptimizedFrameProcessor:
    """
    Optimized frame processor for handling high-frame-rate NDI streams
    with intelligent frame skipping and queue management.
    """
    
    def __init__(self, app_context, config: Dict[str, Any]):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = config
        
        # Performance settings
        self.target_fps = config.get('fps', 2)
        self.max_queue_size = config.get('max_queue_size', 50)
        self.frame_skip_factor = config.get('frame_skip_factor', 3)
        self.enable_frame_skipping = config.get('enable_frame_skipping', True)
        self.processing_timeout = config.get('processing_timeout', 0.5)
        
        # Frame processing state
        self.frame_count = 0
        self.processed_count = 0
        self.skipped_count = 0
        self.last_processing_time = 0
        self.frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.5
        
        # Queues
        self.input_queue = queue.Queue(maxsize=self.max_queue_size)
        self.output_queue = queue.Queue(maxsize=self.max_queue_size)
        
        # Processing state
        self.running = False
        self.processing_thread = None
        
        # Performance monitoring
        self.performance_stats = {
            'total_frames_received': 0,
            'total_frames_processed': 0,
            'total_frames_skipped': 0,
            'queue_overflow_count': 0,
            'processing_times': [],
            'last_stats_reset': time.time()
        }
        
        self.logger.info(f"[OptimizedFrameProcessor] Initialized with target FPS: {self.target_fps}, "
                        f"frame skip factor: {self.frame_skip_factor}, queue size: {self.max_queue_size}")
    
    def start(self):
        """Start the optimized frame processor."""
        if self.running:
            self.logger.warning("[OptimizedFrameProcessor] Already running")
            return
        
        self.running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="OptimizedFrameProcessor",
            daemon=True
        )
        self.processing_thread.start()
        self.logger.info("[OptimizedFrameProcessor] Started")
    
    def stop(self):
        """Stop the optimized frame processor."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        self.logger.info("[OptimizedFrameProcessor] Stopped")
    
    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Add a frame to the processing queue with intelligent overflow handling.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            bool: True if frame was added successfully, False if dropped
        """
        self.performance_stats['total_frames_received'] += 1
        
        try:
            # Use non-blocking put to avoid blocking the NDI capture loop
            self.input_queue.put_nowait(frame)
            return True
        except queue.Full:
            # Queue is full - implement intelligent frame dropping
            self.performance_stats['queue_overflow_count'] += 1
            
            # Clear old frames and add the new one
            try:
                # Remove oldest frame to make room
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    pass
                
                # Try to add the new frame
                self.input_queue.put_nowait(frame)
                return True
            except queue.Full:
                # Still full after clearing - drop the frame
                self.logger.debug(f"[OptimizedFrameProcessor] Dropped frame due to queue overflow "
                                f"(overflow count: {self.performance_stats['queue_overflow_count']})")
                return False
    
    def get_processed_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get a processed frame from the output queue.
        
        Args:
            timeout: Timeout for getting frame
            
        Returns:
            Optional[np.ndarray]: Processed frame or None if timeout
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _processing_loop(self):
        """Main processing loop with intelligent frame skipping."""
        self.logger.info("[OptimizedFrameProcessor] Processing loop started")
        
        while self.running and not self.app_context.shutdown_event.is_set():
            try:
                # Get frame with timeout
                try:
                    frame = self.input_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                self.frame_count += 1
                current_time = time.time()
                
                # Implement intelligent frame skipping
                should_process = self._should_process_frame(current_time)
                
                if should_process:
                    # Process the frame
                    start_time = time.time()
                    processed_frame = self._process_frame(frame)
                    processing_time = time.time() - start_time
                    
                    if processed_frame is not None:
                        # Add to output queue
                        try:
                            self.output_queue.put_nowait(processed_frame)
                            self.processed_count += 1
                            self.performance_stats['total_frames_processed'] += 1
                            self.performance_stats['processing_times'].append(processing_time)
                            
                            # Keep only last 100 processing times for stats
                            if len(self.performance_stats['processing_times']) > 100:
                                self.performance_stats['processing_times'] = \
                                    self.performance_stats['processing_times'][-100:]
                            
                            self.last_processing_time = current_time
                            
                        except queue.Full:
                            self.logger.debug("[OptimizedFrameProcessor] Output queue full, dropping processed frame")
                else:
                    # Skip this frame
                    self.skipped_count += 1
                    self.performance_stats['total_frames_skipped'] += 1
                
                # Periodic logging
                if self.frame_count % 100 == 0:
                    self._log_performance_stats()
                
            except Exception as e:
                self.logger.error(f"[OptimizedFrameProcessor] Error in processing loop: {e}")
                time.sleep(0.1)
        
        self.logger.info("[OptimizedFrameProcessor] Processing loop stopped")
    
    def _should_process_frame(self, current_time: float) -> bool:
        """
        Determine if a frame should be processed based on timing and frame skipping logic.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            bool: True if frame should be processed
        """
        if not self.enable_frame_skipping:
            return True
        
        # Check if enough time has passed since last processing
        time_since_last = current_time - self.last_processing_time
        if time_since_last < self.frame_interval:
            return False
        
        # Apply frame skip factor
        if self.frame_count % self.frame_skip_factor != 0:
            return False
        
        return True
    
    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame with timeout protection.
        
        Args:
            frame: Input frame
            
        Returns:
            Optional[np.ndarray]: Processed frame or None if processing failed
        """
        try:
            # Resize frame if needed for performance
            if frame.shape[0] > 1080 or frame.shape[1] > 1920:
                # Resize to 1080p for better performance
                height, width = frame.shape[:2]
                scale = min(1080 / height, 1920 / width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Apply basic preprocessing
            # Convert to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Already BGR, convert to RGB for consistency
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            return frame_rgb
            
        except Exception as e:
            self.logger.error(f"[OptimizedFrameProcessor] Error processing frame: {e}")
            return None
    
    def _log_performance_stats(self):
        """Log performance statistics."""
        if not self.performance_stats['processing_times']:
            return
        
        avg_processing_time = np.mean(self.performance_stats['processing_times'])
        max_processing_time = np.max(self.performance_stats['processing_times'])
        
        self.logger.info(f"[OptimizedFrameProcessor] Performance Stats: "
                        f"Received: {self.performance_stats['total_frames_received']}, "
                        f"Processed: {self.performance_stats['total_frames_processed']}, "
                        f"Skipped: {self.performance_stats['total_frames_skipped']}, "
                        f"Overflow: {self.performance_stats['queue_overflow_count']}, "
                        f"Avg Processing: {avg_processing_time:.3f}s, "
                        f"Max Processing: {max_processing_time:.3f}s")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['max_processing_time'] = np.max(stats['processing_times'])
            stats['min_processing_time'] = np.min(stats['processing_times'])
        else:
            stats['avg_processing_time'] = 0
            stats['max_processing_time'] = 0
            stats['min_processing_time'] = 0
        
        stats['current_queue_size'] = self.input_queue.qsize()
        stats['output_queue_size'] = self.output_queue.qsize()
        stats['frame_count'] = self.frame_count
        stats['processed_count'] = self.processed_count
        stats['skipped_count'] = self.skipped_count
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_frames_received': 0,
            'total_frames_processed': 0,
            'total_frames_skipped': 0,
            'queue_overflow_count': 0,
            'processing_times': [],
            'last_stats_reset': time.time()
        }
        self.frame_count = 0
        self.processed_count = 0
        self.skipped_count = 0
        self.logger.info("[OptimizedFrameProcessor] Performance stats reset")
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop()
        
        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
        
        self.logger.info("[OptimizedFrameProcessor] Cleanup completed") 