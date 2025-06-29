# services/ndi_service.py
import numpy as np
import time
import socket
import logging
import queue
from typing import Optional, Tuple
import cv2
import threading

# Import NDI library - use the correct package name
try:
    import NDIlib as ndi
    NDI_AVAILABLE = True
except ImportError:
    try:
        import ndi
        NDI_AVAILABLE = True
    except ImportError:
        NDI_AVAILABLE = False
        ndi = None

# Note: Don't import NDI libraries directly here, will be initialized dynamically
# This allows the services to be instantiated on systems without NDI libraries

class NDIService:
    def __init__(self, app_context):
        self.app_context = app_context
        self.ndi_finder = None
        self.ndi_receiver = None
        self.is_initialized = False
        self.logger = self.app_context.logger
        self.logger.info("[NDIService] Instance created.")
        
        # Store the most recent frame for manual screenshot capture
        self.last_captured_frame = None
        
        # Store the NDI library name used
        self.ndi_lib_loaded = None
        
        # Track NDI sources and current source
        self.available_ndi_sources = []
        self.current_source_idx = 0
        self.next_source_scan_time = 0
        self.last_source_change_time = 0
        
        # Check if NDI is available
        if not NDI_AVAILABLE:
            self.logger.error("[NDIService] NDI library not available. Install with: pip install ndi-python")
            return
        
        # Try to initialize NDI
        self._initialize_ndi()
        self.logger.info("[NDIService] Initialized.")  # Note: initialization might have failed
    
    def _initialize_ndi(self):
        self.logger.info("[NDIService] Trying to initialize NDI...")
        try_libs = ["PyNDI4", "ndi", "NDIlib"]
        success = False
        
        for lib_name in try_libs:
            try:
                self.logger.info(f"[NDIService] Attempting to load NDI library: {lib_name}")
                if lib_name == "PyNDI4":
                    import PyNDI4 as ndi
                    self._init_pyndi4(ndi)
                    success = True
                elif lib_name == "ndi":
                    import ndi
                    self._init_ndi_standard(ndi)
                    success = True
                elif lib_name == "NDIlib":
                    import NDIlib as ndi
                    self._init_ndilib(ndi)
                    success = True
                
                if success:
                    self.ndi_lib_loaded = lib_name
                    self.logger.info(f"[NDIService] Successfully initialized NDI using {lib_name}")
                    break
            except ImportError:
                self.logger.warning(f"[NDIService] Could not import NDI library: {lib_name}")
            except Exception as e:
                self.logger.error(f"[NDIService] Error initializing NDI with {lib_name}: {e}", exc_info=True)
        
        if not success:
            self.logger.error("[NDIService] Failed to initialize NDI using any available library.")
        else:
            self.is_initialized = True

    def initialize_ndi(self) -> bool:
        """Initialize NDI library and connection."""
        if not NDI_AVAILABLE or ndi is None:
            self.logger.error("[NDIService] NDI library not available. Cannot initialize.")
            return False
            
        self.logger.info("[NDIService] Initializing NDI library and connection...")
        gs = self.app_context.global_settings
        try:
            if not ndi.initialize():
                self.logger.critical("[NDIService] NDI C library initialization failed.")
                return False
            self.logger.debug("[NDIService] NDI C library initialized.")

            self.ndi_finder = ndi.find_create_v2()
            if not self.ndi_finder:
                self.logger.critical("[NDIService] Failed to create NDI finder.")
                ndi.destroy()
                return False
            self.logger.debug("[NDIService] NDI finder created.")

            ndi_discovery_timeout_ms = gs.get('NDI_CONNECTION_TIMEOUT_MS', 5000)
            self.logger.info(f"[NDIService] Discovering NDI sources (timeout {ndi_discovery_timeout_ms}ms)...")

            sources_found = False
            if ndi.find_wait_for_sources(self.ndi_finder, ndi_discovery_timeout_ms):
                sources_found = True
            else:
                time.sleep(ndi_discovery_timeout_ms / 1000.0 * 0.5)
                self.logger.warning("[NDIService] find_wait_for_sources timed out or returned false, checking current sources...")

            sources = ndi.find_get_current_sources(self.ndi_finder)
            if not sources:
                self.logger.error("[NDIService] No NDI sources found after discovery period.")
                self._cleanup_finder_and_library()
                return False

            if not sources_found and sources:
                self.logger.info("[NDIService] Found NDI sources via get_current_sources after wait_for_sources issue.")

            self.logger.info("[NDIService] Available NDI sources:")
            target_source_name = gs.get("TARGET_NDI_SOURCE_NAME", None)
            target_source_info = None

            for i, s_info in enumerate(sources):
                if isinstance(s_info.ndi_name, bytes):
                    s_name = s_info.ndi_name.decode('utf-8', errors='ignore')
                else:
                    s_name = str(s_info.ndi_name)

                if isinstance(s_info.url_address, bytes):
                    s_url = s_info.url_address.decode('utf-8', errors='ignore')
                else:
                    s_url = str(s_info.url_address)

                self.logger.info(f"  {i}: {s_name} (URL: {s_url})")
                if target_source_name and target_source_name == s_name:
                    target_source_info = s_info
                    self.logger.info(f"[NDIService] Matched preferred source: {s_name}")
                    # break # If you want to stop after first match, uncomment
                elif not target_source_name and i == 0 and not target_source_info:
                    target_source_info = s_info

            if not target_source_info:
                if target_source_name:
                    self.logger.error(f"[NDIService] Specified NDI source '{target_source_name}' not found among available sources.")
                else:
                    self.logger.error(f"[NDIService] Could not select an NDI source, though sources were listed.")
                self._cleanup_finder_and_library()
                return False

            if isinstance(target_source_info.ndi_name, bytes):
                selected_src_name_str = target_source_info.ndi_name.decode('utf-8', errors='ignore')
            else:
                selected_src_name_str = str(target_source_info.ndi_name)

            self.logger.info(f"[NDIService] Attempting to connect to NDI source: {selected_src_name_str}")

            recv_create_desc = ndi.RecvCreateV3()
            recv_create_desc.source_to_connect_to = target_source_info
            recv_create_desc.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
            recv_create_desc.bandwidth = ndi.RECV_BANDWIDTH_HIGHEST

            self.ndi_receiver = ndi.recv_create_v3(recv_create_desc)
            if not self.ndi_receiver:
                self.logger.error(f"[NDIService] Failed to create NDI receiver for {selected_src_name_str}.")
                self._cleanup_finder_and_library()
                return False

            self.logger.info(f"[NDIService] Successfully connected to NDI source: {selected_src_name_str}")
            self.is_initialized = True
            return True

        except Exception as e:
            self.logger.critical(f"[NDIService] Exception during NDI initialization: {e}", exc_info=True)
            self._cleanup_finder_and_library()
            return False

    def _init_ndilib(self, ndi):
        """Initialize using NDIlib"""
        if not ndi.initialize():
            raise RuntimeError("Failed to initialize NDI")
        
        self.ndi = ndi
        self.find_instance = self.ndi.find_create_v2()
        if not self.find_instance:
            raise RuntimeError("Failed to create NDI finder")
            
        self.recv_instance = None
        self.logger.info("[NDIService] NDIlib initialized successfully")

    def _init_pyndi4(self, ndi):
        """Initialize using PyNDI4"""
        if not ndi.initialize():
            raise RuntimeError("Failed to initialize NDI")
        
        self.ndi = ndi
        self.find_instance = self.ndi.find_create_v2()
        if not self.find_instance:
            raise RuntimeError("Failed to create NDI finder")
            
        self.recv_instance = None
        self.logger.info("[NDIService] PyNDI4 initialized successfully")

    def _init_ndi_standard(self, ndi):
        """Initialize using standard NDI library"""
        if not ndi.initialize():
            raise RuntimeError("Failed to initialize NDI")
        
        self.ndi = ndi
        self.find_instance = self.ndi.find_create_v2()
        if not self.find_instance:
            raise RuntimeError("Failed to create NDI finder")
            
        self.recv_instance = None
        self.logger.info("[NDIService] Standard NDI library initialized successfully")

    def _frame_to_bgr(self, vf: ndi.VideoFrameV2) -> Optional[np.ndarray]:
        try:
            if vf.data is None or vf.xres <= 0 or vf.yres <= 0:
                self.logger.debug(f"[NDIService._frame_to_bgr] Null data or zero dimensions. xres={vf.xres}, yres={vf.yres}")
                return None

            frame_data_len = len(vf.data)
            if frame_data_len == 0:
                self.logger.debug("[NDIService._frame_to_bgr] VideoFrameV2.data has zero length.")
                return None

            data_copy = bytes(vf.data)

            if vf.FourCC == ndi.FOURCC_VIDEO_TYPE_BGRX or vf.FourCC == ndi.FOURCC_VIDEO_TYPE_BGRA:
                bytes_per_pixel = 4
                expected_min_data_size = vf.yres * vf.xres * bytes_per_pixel

                if len(data_copy) < expected_min_data_size:
                    self.logger.warning(f"[NDIService._frame_to_bgr] BGRA/X data too small. Expected {expected_min_data_size}, got {len(data_copy)}. Frame: {vf.xres}x{vf.yres}, Stride: {vf.line_stride_in_bytes}")
                    return None

                if vf.line_stride_in_bytes != vf.xres * bytes_per_pixel and vf.line_stride_in_bytes > 0 :
                     self.logger.debug(f"[NDIService._frame_to_bgr] BGRA/X Stride ({vf.line_stride_in_bytes}) != xres*bpp ({vf.xres*bytes_per_pixel}).")

                arr = np.frombuffer(data_copy, dtype=np.uint8, count=expected_min_data_size)
                image_data_np = arr.reshape((vf.yres, vf.xres, bytes_per_pixel))
                return cv2.cvtColor(image_data_np, cv2.COLOR_BGRA2BGR)

            elif vf.FourCC == ndi.FOURCC_VIDEO_TYPE_UYVY:
                bytes_per_pixel = 2
                expected_min_data_size = vf.yres * vf.xres * bytes_per_pixel

                if len(data_copy) < expected_min_data_size:
                    self.logger.warning(f"[NDIService._frame_to_bgr] UYVY data too small. Expected {expected_min_data_size}, got {len(data_copy)}. Frame: {vf.xres}x{vf.yres}")
                    return None

                arr = np.frombuffer(data_copy, dtype=np.uint8, count=expected_min_data_size)
                image_data_np = arr.reshape((vf.yres, vf.xres, bytes_per_pixel))
                return cv2.cvtColor(image_data_np, cv2.COLOR_YUV2BGR_UYVY)

            else:
                self.logger.warning(f"[NDIService._frame_to_bgr] Unhandled FourCC video type: {vf.FourCC}")
                return None
        except Exception as e:
            self.logger.error(f"[NDIService._frame_to_bgr] Error processing frame: {e}. Details: xres={vf.xres}, yres={vf.yres}, FourCC={vf.FourCC}, stride={vf.line_stride_in_bytes}", exc_info=True)
            return None

    def run_capture_loop(self):
        """Main NDI frame capture loop with frame rate limiting."""
        if not self.is_initialized:
            self.initialize_ndi()
            if not self.is_initialized:
                self.logger.error("[NDIService] Failed to initialize NDI. Capture loop cannot start.")
                return

        self.logger.info("[NDIService] Starting NDI frame capture loop with frame rate limiting...")
        gs = self.app_context.global_settings
        
        # Get frame rate settings
        target_fps = gs.get('vision_capture_fps', 1)  # Default to 1 FPS
        frame_interval = 1.0 / target_fps if target_fps > 0 else 1.0
        receive_timeout_ms = gs.get('NDI_RECEIVE_TIMEOUT_MS', 1000)

        loop_count = 0
        last_frame_time = 0
        last_log_time = time.time()
        frames_captured = 0
        frames_dropped = 0

        self.logger.info(f"[NDIService] Target capture rate: {target_fps} FPS (interval: {frame_interval:.3f}s)")

        while not self.app_context.shutdown_event.is_set():
            # Heartbeat logging
            now_time = time.time()
            if now_time - last_log_time > 60:
                self.logger.info(f"[NDIService] Capture loop alive - Captured: {frames_captured}, Dropped: {frames_dropped}")
                last_log_time = now_time

            if not self.ndi_receiver:
                self.logger.error("[NDIService] NDI receiver is None in capture loop. Attempting re-initialization.")
                time.sleep(5)
                if not self.initialize_ndi():
                    self.logger.critical("[NDIService] Re-initialization failed. Stopping NDI capture loop.")
                    break
                else:
                    self.logger.info("[NDIService] Re-initialized successfully. Continuing capture.")
                    continue

            # Check if enough time has passed for next frame
            time_since_last = now_time - last_frame_time
            if time_since_last < frame_interval:
                # Sleep until next frame time
                sleep_time = frame_interval - time_since_last
                time.sleep(sleep_time)
                continue

            frame_type, video_frame, audio_frame, metadata_frame = ndi.recv_capture_v2(self.ndi_receiver, receive_timeout_ms)
            loop_count += 1

            if frame_type == ndi.FRAME_TYPE_VIDEO:
                if loop_count % 50 == 0:
                    self.logger.debug(f"[NDIService] Video frame received (loop {loop_count}). Timestamp: {video_frame.timestamp if video_frame else 'N/A'}")

                bgr_image = self._frame_to_bgr(video_frame)
                ndi.recv_free_video_v2(self.ndi_receiver, video_frame)

                if bgr_image is not None:
                    # Only process frame if enough time has passed
                    if now_time - last_frame_time >= frame_interval:
                        if self.app_context.ndi_commentary_enabled.is_set():
                            try:
                                self.app_context.frame_queue.put_nowait(bgr_image)
                                frames_captured += 1
                                last_frame_time = now_time
                                
                                if frames_captured % 10 == 0:
                                    self.logger.info(f"[NDIService] Captured {frames_captured} frames at {target_fps} FPS")
                                    
                            except queue.Full:
                                frames_dropped += 1
                                self.logger.warning(f"[NDIService] Frame queue full. Dropping frame. (Captured: {frames_captured}, Dropped: {frames_dropped})")
                                
                                # Clear queue and add latest frame
                                while not self.app_context.frame_queue.empty():
                                    try:
                                        self.app_context.frame_queue.get_nowait()
                                    except queue.Empty:
                                        break
                                try:
                                    self.app_context.frame_queue.put_nowait(bgr_image)
                                    frames_captured += 1
                                    last_frame_time = now_time
                                except queue.Full:
                                    self.logger.error("[NDIService] Frame queue still full after clearing. Dropping frame.")
                    else:
                        # Skip this frame due to rate limiting
                        frames_dropped += 1
                        if frames_dropped % 20 == 0:
                            self.logger.debug(f"[NDIService] Rate limiting: dropped {frames_dropped} frames")

                # Store the last captured frame for manual screenshot capture
                self.last_captured_frame = bgr_image.copy() if bgr_image is not None else None

            elif frame_type == ndi.FRAME_TYPE_AUDIO:
                if audio_frame: ndi.recv_free_audio_v2(self.ndi_receiver, audio_frame)
            elif frame_type == ndi.FRAME_TYPE_METADATA:
                if metadata_frame: ndi.recv_free_metadata(self.ndi_receiver, metadata_frame)
            elif frame_type == ndi.FRAME_TYPE_ERROR:
                self.logger.error("[NDIService] NDI recv_capture_v2 reported FRAME_TYPE_ERROR.")
                time.sleep(1.0)
            elif frame_type == ndi.FRANE_TYPE_STATUS_CHANGE:
                self.logger.info(f"[NDIService] Received NDI FRANE_TYPE_STATUS_CHANGE frame.")
            elif frame_type == ndi.FRAME_TYPE_NONE:
                if loop_count % 500 == 0:
                     self.logger.debug(f"[NDIService] No NDI data received in timeout period (loop {loop_count}).")
            else:
                self.logger.warning(f"[NDIService] Received unhandled NDI frame type: {frame_type}")

            # Small sleep to prevent CPU spinning
            if frame_type != ndi.FRAME_TYPE_VIDEO:
                time.sleep(0.01)

        # Final stats
        self.logger.info(f"[NDIService] NDI capture loop stopped. Final stats - Captured: {frames_captured}, Dropped: {frames_dropped}")
        self.cleanup()

    def _cleanup_finder_and_library(self):
        if not NDI_AVAILABLE or ndi is None:
            return
            
        if self.ndi_finder:
            ndi.find_destroy(self.ndi_finder)
            self.ndi_finder = None
            self.logger.info("[NDIService] NDI finder destroyed.")
        ndi.destroy()
        self.logger.info("[NDIService] NDI library context destroyed.")

    def cleanup(self):
        self.logger.info("[NDIService] Initiating NDI cleanup...")
        if not NDI_AVAILABLE or ndi is None:
            self.logger.info("[NDIService] NDI not available, skipping cleanup.")
            return
            
        if self.ndi_receiver:
            ndi.recv_destroy(self.ndi_receiver)
            self.ndi_receiver = None
            self.logger.info("[NDIService] NDI receiver destroyed.")

        if self.ndi_finder or self.is_initialized: # If finder exists or NDI was successfully initialized at some point
            self._cleanup_finder_and_library()

        self.is_initialized = False # Reset flag
        self.logger.info("[NDIService] NDI cleanup complete.")

# (The if __name__ == "__main__": test block remains the same)
