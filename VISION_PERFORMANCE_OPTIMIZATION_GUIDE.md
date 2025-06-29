# Vision Performance Optimization Guide

## Problem Analysis

The "Frame queue full. Clearing queue and adding latest frame" messages indicate a **performance bottleneck** where:

- **NDI Capture**: Running at 30-60 FPS from OBS
- **Vision Processing**: Only processing at 1 FPS
- **Result**: Massive queue overflow and dropped frames

## Root Causes

1. **Frame Rate Mismatch**: NDI captures at 30-60 FPS, vision processes at 1 FPS
2. **Queue Size Too Small**: Only 10 frames in queue (fills in ~0.3 seconds)
3. **Heavy Processing**: YOLO + CLIP + OCR on every frame
4. **No Frame Skipping**: Processing every frame instead of intelligent sampling
5. **High Resolution**: 4K frames require significant processing power

## Optimizations Implemented

### 1. **Queue Management**
```yaml
# Before
max_queue_size: 10

# After  
max_queue_size: 50  # 5x larger queue
```

### 2. **Intelligent Frame Skipping**
```yaml
# New settings
frame_skip_factor: 3               # Process every 3rd frame
enable_frame_skipping: true        # Enable intelligent skipping
fps: 2                            # Increased from 1 to 2 FPS
```

### 3. **Processing Load Reduction**
```yaml
# CLIP Processing
processing_fps: 0.5                # Process CLIP every 2 seconds
max_insights: 5                    # Reduced from 10 to 5

# YOLO Detection
confidence_threshold: 0.6          # Increased from 0.5 to reduce false positives

# Template Matching
enabled: false                     # Disabled to reduce load

# OCR
min_confidence: 0.7                # Increased threshold
```

### 4. **Resolution Optimization**
```yaml
# Before: 4K processing
max_resolution: [3840, 2160]

# After: 1080p processing
max_resolution: [1920, 1080]
```

### 5. **Debouncing Improvements**
```yaml
# Event Debouncing
timeout_ms: 2000                   # Increased from 1000ms to 2000ms
min_confidence_change: 0.15        # Increased from 0.1

# Processing Settings
debounce_interval: 3.0             # Increased from 2.0 seconds
max_events_per_second: 2           # Reduced from 5 to 2
```

### 6. **Logging Optimization**
```yaml
# Before
level: "INFO"                      # Verbose logging

# After
level: "WARNING"                   # Reduced logging overhead
```

## Performance Monitoring

### Discord Commands
```bash
!gpu status          # Monitor GPU memory usage
!gpu vision          # Check best device for vision processing
!gpu monitor         # Toggle memory monitoring
```

### Performance Metrics
- **Frame Queue Size**: Should stay below 80% capacity
- **Processing Time**: Should be < 0.5 seconds per frame
- **GPU Utilization**: Vision should use < 30% of cuda:1
- **Memory Usage**: Monitor for memory leaks

## Expected Performance Improvements

### Before Optimization
- ❌ Queue overflow every 0.3 seconds
- ❌ 1 FPS processing rate
- ❌ 4K resolution processing
- ❌ Heavy processing on every frame
- ❌ Verbose logging overhead

### After Optimization
- ✅ 50-frame queue buffer (25 seconds at 2 FPS)
- ✅ 2 FPS processing rate
- ✅ 1080p resolution processing
- ✅ Intelligent frame skipping
- ✅ Reduced processing load
- ✅ Minimal logging overhead

## Troubleshooting

### Still Getting Queue Overflow?
1. **Check NDI Source FPS**: Ensure OBS is not outputting 60 FPS
2. **Reduce Target FPS**: Lower `fps` to 1 in config
3. **Increase Queue Size**: Set `max_queue_size` to 100
4. **Disable Heavy Features**: Turn off CLIP or YOLO temporarily

### High GPU Usage?
1. **Check GPU Memory**: Use `!gpu status`
2. **Reduce Resolution**: Lower `max_resolution`
3. **Increase Frame Skip**: Set `frame_skip_factor` to 5
4. **Use CPU Fallback**: Let system automatically use CPU

### Slow Processing?
1. **Check Processing Time**: Monitor logs for processing duration
2. **Reduce Model Complexity**: Use smaller YOLO/CLIP models
3. **Optimize ROI**: Reduce OCR region size
4. **Batch Processing**: Enable if multiple frames available

## Configuration Examples

### High Performance (Gaming)
```yaml
fps: 3
frame_skip_factor: 2
max_queue_size: 100
processing_fps: 1.0
max_resolution: [1920, 1080]
```

### Balanced Performance
```yaml
fps: 2
frame_skip_factor: 3
max_queue_size: 50
processing_fps: 0.5
max_resolution: [1920, 1080]
```

### Low Resource Usage
```yaml
fps: 1
frame_skip_factor: 5
max_queue_size: 25
processing_fps: 0.2
max_resolution: [1280, 720]
```

## Best Practices

### 1. **Monitor Performance**
- Use `!gpu status` regularly
- Check logs for processing times
- Monitor queue overflow frequency

### 2. **Adjust Based on Hardware**
- **High-end GPU**: Increase FPS and reduce frame skipping
- **Mid-range GPU**: Use balanced settings
- **Low-end GPU**: Use low resource settings

### 3. **Optimize NDI Source**
- Set OBS output to 30 FPS or lower
- Use "Medium" bandwidth in NDI settings
- Disable audio if not needed

### 4. **Regular Maintenance**
- Restart vision pipeline if performance degrades
- Clear GPU memory if utilization is high
- Monitor for memory leaks

## Expected Results

With these optimizations, you should see:

1. **No More Queue Overflow**: Frame queue stays below 80% capacity
2. **Smooth Processing**: 2 FPS processing rate maintained
3. **Reduced GPU Load**: Vision processing uses < 30% of cuda:1
4. **Better Responsiveness**: Faster reaction to visual changes
5. **Stable Performance**: Consistent processing times

The system now intelligently handles the NDI frame rate mismatch while maintaining responsive vision processing for your gaming commentary system. 