# GPU Memory Management Guide for DanzarAI

## Overview

DanzarAI now includes an intelligent GPU memory management system that coordinates GPU usage between the main LLM (4070) and vision processing (4070 Super). This system ensures optimal performance by preventing VRAM conflicts and automatically selecting the best available GPU for each task.

## Architecture

### GPU Allocation Strategy

- **Main LLM (cuda:0)**: Reserved for the primary language model (Qwen2.5, LlamaCpp, etc.)
- **Vision Processing (cuda:1)**: Used for CLIP, YOLO, and other vision models
- **Fallback to CPU**: When GPU memory is insufficient

### Memory Management Features

- **Real-time monitoring**: Tracks GPU memory usage across all devices
- **Intelligent device selection**: Automatically chooses the best GPU for each task
- **Memory reservation**: Reserves 8GB for main LLM to prevent conflicts
- **Automatic fallback**: Switches to CPU when GPU memory is insufficient

## Configuration

### Vision Configuration (`config/vision_config.yaml`)

```yaml
# GPU Memory Management Settings
gpu_memory:
  # Main LLM GPU (4070) - reserved for LLM operations
  main_llm_device: "cuda:0"
  main_llm_memory_reservation_gb: 8.0  # Reserve 8GB for main LLM
  
  # Vision processing GPU (4070 Super) - for vision models
  vision_device: "cuda:1"
  vision_memory_limit_gb: 2.0  # Limit vision processing to 2GB
  
  # Fallback to CPU if GPU memory is insufficient
  fallback_to_cpu: true
  cpu_fallback_threshold_gb: 1.0  # Fallback if less than 1GB available
  
  # Memory monitoring
  enable_memory_monitoring: true
  memory_check_interval: 30  # seconds

# CLIP Video Understanding Settings
clip:
  enabled: true
  processing_fps: 1
  model_name: "ViT-B/32"
  device: "cuda:1"  # Will be overridden by GPU memory manager
  confidence_threshold: 0.3
  max_insights: 10

# YOLO Object Detection Settings
yolo:
  model_path: "models/yolo-nano.pt"
  confidence_threshold: 0.5
  device: "cuda:1"  # Will be overridden by GPU memory manager
  classes: ["health_bar", "minimap", "boss", "player", "enemy"]

# Performance Settings
performance:
  gpu_memory_fraction: 0.3  # Reduced to leave room for main LLM
```

## Usage

### Discord Commands

#### `!gpu status`
Shows current GPU memory status for all devices:

```
🎯 GPU Memory Status:

Device 0 (RTX 4070):
  📊 Allocated: 6.2GB
  📊 Reserved: 7.1GB
  📊 Free: 4.9GB
  📊 Total: 12.0GB
  📊 Utilization: 59.2%

Device 1 (RTX 4070 Super):
  📊 Allocated: 1.8GB
  📊 Reserved: 2.1GB
  📊 Free: 9.9GB
  📊 Total: 12.0GB
  📊 Utilization: 17.5%

💡 Recommendations:
  • All GPU devices have adequate memory
```

#### `!gpu vision`
Shows the best available device for vision processing:

```
👁️ Best Vision Device:
Device: cuda:1
Reason: Device 1 (RTX 4070 Super) available with 9.9GB free
```

#### `!gpu monitor`
Toggles GPU memory monitoring on/off.

#### `!gpu log`
Logs current GPU memory status to console.

### Programmatic Usage

```python
# Get GPU memory manager from app context
gpu_mm = app_context.gpu_memory_manager

# Get memory info for all devices
memory_info = gpu_mm.get_all_devices_memory_info()

# Get best device for vision processing
device, reason = gpu_mm.get_best_vision_device()

# Check if device can be used for vision
can_use, reason = gpu_mm.can_use_device_for_vision(device_id)

# Get memory recommendations
recommendations = gpu_mm.get_memory_recommendations()

# Log memory status
gpu_mm.log_memory_status()
```

## Service Integration

### Vision Services

All vision services automatically use the GPU memory manager:

- **CLIP Vision Enhancer**: Uses optimal device for CLIP model
- **YOLO Detection**: Uses optimal device for object detection
- **Vision Pipeline**: Coordinates device usage across all vision components

### Automatic Device Selection

The system automatically selects the best device based on:

1. **Memory availability**: Ensures sufficient free memory
2. **Main LLM reservation**: Avoids interfering with cuda:0 when main LLM is active
3. **Performance**: Prefers GPU over CPU when possible
4. **Fallback**: Gracefully falls back to CPU when needed

## Best Practices

### For Main LLM (4070)

- **Reserve 8GB**: Ensures the main LLM has sufficient memory
- **Monitor usage**: Use `!gpu status` to check memory utilization
- **Avoid conflicts**: Don't run heavy vision processing on cuda:0

### For Vision Processing (4070 Super)

- **Limit to 2GB**: Prevents vision models from consuming too much memory
- **Use efficient models**: Prefer smaller models like YOLOv8n and CLIP ViT-B/32
- **Batch processing**: Process multiple frames together when possible

### General Guidelines

- **Monitor regularly**: Use `!gpu monitor` to track memory usage
- **Restart if needed**: Restart the application if memory gets fragmented
- **Check recommendations**: Pay attention to GPU memory recommendations
- **Use fallback**: Don't worry if the system falls back to CPU - it's designed to handle this

## Troubleshooting

### Common Issues

#### "No GPU devices available with sufficient memory"
- **Cause**: All GPUs are full or CUDA not available
- **Solution**: Restart the application or close other GPU-intensive applications

#### "Device 0 reserved for main LLM"
- **Cause**: Main LLM is using most of cuda:0 memory
- **Solution**: This is normal - vision processing will use cuda:1 or CPU

#### High GPU utilization warnings
- **Cause**: GPU memory usage is above 90%
- **Solution**: Monitor with `!gpu status` and consider restarting if needed

### Performance Optimization

1. **Reduce vision FPS**: Lower `fps` in vision config to reduce GPU load
2. **Use smaller models**: Switch to smaller YOLO or CLIP models
3. **Enable CPU fallback**: Let the system automatically use CPU when needed
4. **Monitor memory**: Use `!gpu status` regularly to track usage

## Technical Details

### Memory Monitoring

The GPU memory manager runs a background thread that:

- Checks memory usage every 30 seconds (configurable)
- Logs warnings when utilization is high (>90%)
- Maintains memory usage history
- Provides recommendations for optimization

### Device Selection Algorithm

1. **Check cuda:1 first**: Prefer the dedicated vision GPU
2. **Check cuda:0**: Only if main LLM has sufficient free memory
3. **Check other devices**: Look for any available GPU
4. **Fallback to CPU**: When no GPU has sufficient memory

### Memory Reservation

- **Main LLM**: 8GB reserved on cuda:0
- **Vision processing**: 2GB limit on cuda:1
- **CPU fallback**: Triggered when <1GB available

## Future Enhancements

- **Dynamic memory allocation**: Adjust reservations based on actual usage
- **Model offloading**: Automatically move models between GPU and CPU
- **Memory defragmentation**: Automatically clean up fragmented memory
- **Performance profiling**: Track and optimize memory usage patterns

## Conclusion

The GPU memory management system ensures DanzarAI can efficiently use multiple GPUs without conflicts. The main LLM gets priority on the 4070, while vision processing uses the 4070 Super or falls back to CPU when needed. This provides optimal performance while maintaining system stability. 