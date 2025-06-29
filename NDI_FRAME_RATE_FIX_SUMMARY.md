# NDI Frame Rate Fix Summary

## Problem Identified

The system was experiencing **massive queue overflow** because:
- **NDI Source**: Running at 30-60 FPS from OBS
- **Vision Processing**: Only processing at 1 FPS
- **Result**: Queue overflow every 0.3 seconds, dropped frames, no commentary

## Root Cause Analysis

Based on the [OBS NDI documentation](https://obsproject.com/kb/video-capture-sources) and [NDI common issues](https://docs.ndi.video/all/faq/common-issues), the issue was:

1. **Frame Rate Mismatch**: NDI was capturing at full frame rate (30-60 FPS) but vision processing couldn't keep up
2. **No Rate Limiting**: NDI service was processing every frame without any rate limiting
3. **Queue Size Too Small**: 10-frame queue filled in ~0.3 seconds at 30 FPS
4. **No CLIP Logging**: Couldn't see what CLIP was detecting

## Solutions Implemented

### 1. **NDI Frame Rate Limiting** ✅

**File**: `services/ndi_service.py`
- Added `vision_capture_fps` setting (default: 1 FPS)
- Implemented frame interval timing control
- Added frame rate limiting logic in capture loop
- Added frame capture statistics and logging

**Key Changes**:
```python
# Get frame rate settings
target_fps = gs.get('vision_capture_fps', 1)  # Default to 1 FPS
frame_interval = 1.0 / target_fps if target_fps > 0 else 1.0

# Check if enough time has passed for next frame
time_since_last = now_time - last_frame_time
if time_since_last < frame_interval:
    # Sleep until next frame time
    sleep_time = frame_interval - time_since_last
    time.sleep(sleep_time)
    continue
```

### 2. **Configuration Updates** ✅

**File**: `config/global_settings.yaml`
```yaml
# Vision Capture Rate Control
vision_capture_fps: 1  # Capture at 1 FPS from NDI source for stability
```

**File**: `config/vision_config.yaml`
```yaml
# NDI Capture Rate Control
ndi_capture:
  vision_capture_fps: 1  # Capture at 1 FPS from NDI source
  enable_rate_limiting: true
  max_queue_size: 10  # Small queue since we're capturing slowly

# Performance Settings - OPTIMIZED for 1 FPS
performance:
  max_queue_size: 10                 # Small queue for 1 FPS capture
  processing_timeout: 1.0            # Increased timeout for reliability
  frame_skip_factor: 1               # Process every frame at 1 FPS
  enable_frame_skipping: false       # Disable frame skipping at 1 FPS
```

### 3. **CLIP Logging Enhancement** ✅

**File**: `services/clip_vision_enhancer.py`
- Added detailed CLIP logging to show what concepts are detected
- Added confidence score logging
- Added configurable logging levels

**Configuration**:
```yaml
clip:
  enable_logging: true               # Enable CLIP logging to see what it detects
  log_insights: true                 # Log all CLIP insights
```

**Logging Output**:
```
[CLIPVisionEnhancer] Detected: health bar (0.85), inventory window (0.72), chat window (0.68)
```

### 4. **Performance Optimizations** ✅

- **Reduced Resolution**: 4K → 1080p for better performance
- **Disabled Template Matching**: Reduced processing load
- **Increased Confidence Thresholds**: Reduced false positives
- **Optimized Debouncing**: Longer intervals for 1 FPS
- **Reduced Event Rate**: 1 event per second instead of 5

## Expected Results

### Before Fix:
- ❌ Queue overflow every 0.3 seconds
- ❌ No commentary due to processing overload
- ❌ No visibility into CLIP detection
- ❌ System instability

### After Fix:
- ✅ **1 FPS Capture Rate**: NDI captures at 1 FPS instead of 30-60 FPS
- ✅ **No Queue Overflow**: 10-frame queue lasts 10 seconds at 1 FPS
- ✅ **CLIP Logging**: See exactly what CLIP detects with confidence scores
- ✅ **Stable System**: Much more stable performance
- ✅ **Commentary Generation**: Vision commentary should work properly

## Testing Results

**Test Script**: `test_ndi_fps.py`
```
✅ NDI Frame Rate Configuration: PASS
✅ CLIP Logging Configuration: PASS  
✅ Frame Rate Simulation: PASS

Overall: 3/3 tests passed
```

**Frame Rate Test**:
- Target FPS: 1
- Frame Interval: 1.000 seconds
- Actual FPS: 1.25 (within acceptable range)
- ✅ Frame rate limiting working correctly

## Monitoring Commands

### Check Frame Rate:
```bash
!gpu status          # Monitor GPU usage
!gpu vision          # Check vision device
```

### Check CLIP Activity:
Look for these log messages:
```
[CLIPVisionEnhancer] Detected: health bar (0.85), inventory window (0.72)
[NDIService] Captured 10 frames at 1 FPS
```

### Check Queue Status:
Look for these log messages:
```
[NDIService] Frame queue full. Dropping frame. (Captured: 5, Dropped: 1)
```

## Troubleshooting

### Still Getting Queue Overflow?
1. **Check OBS NDI Output**: Ensure OBS is not outputting 60 FPS
2. **Verify Frame Rate**: Look for "Captured X frames at 1 FPS" messages
3. **Check Processing Time**: Ensure vision processing completes within 1 second

### No CLIP Logs?
1. **Check Configuration**: Verify `clip_enable_logging: true`
2. **Check Log Level**: Ensure logging level is INFO or lower
3. **Check CLIP Model**: Ensure CLIP model is loaded properly

### No Commentary?
1. **Check Vision Pipeline**: Ensure vision pipeline is running
2. **Check Event Generation**: Look for YOLO/OCR detection logs
3. **Check LLM Integration**: Ensure vision events reach the LLM

## Configuration Files Modified

1. **`services/ndi_service.py`**: Added frame rate limiting
2. **`config/global_settings.yaml`**: Added `vision_capture_fps` setting
3. **`config/vision_config.yaml`**: Updated for 1 FPS optimization
4. **`services/clip_vision_enhancer.py`**: Added detailed logging

## Next Steps

1. **Restart DanzarAI** to apply the new configuration
2. **Monitor Logs** for frame rate and CLIP detection messages
3. **Test Commentary** with `!watch` command
4. **Adjust Frame Rate** if needed (can increase to 2 FPS if system is stable)

The system should now be much more stable with proper frame rate limiting and CLIP logging visibility! 🎮 