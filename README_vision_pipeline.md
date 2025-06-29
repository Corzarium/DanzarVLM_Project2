# DanzarAI Vision Pipeline

Real-time game vision processing with YOLO object detection, template matching, and OCR for gaming commentary and HUD element detection.

## 🎯 Features

- **Real-time Screen Capture**: Fullscreen or window-specific capture at configurable FPS
- **GPU-Accelerated YOLO**: YOLO-Nano object detection on CUDA for HUD elements
- **Template Matching**: Static UI asset detection (skill icons, cooldown overlays)
- **OCR Processing**: Text extraction from specific screen regions
- **Event Debouncing**: Smart filtering to prevent duplicate detections
- **JSON Output**: Structured event data for integration with DanzarAI
- **Async Design**: Non-blocking operation with threaded processing

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements_vision.txt

# Install Tesseract OCR (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR

# Install Tesseract OCR (Linux)
sudo apt-get install tesseract-ocr

# Install Tesseract OCR (macOS)
brew install tesseract
```

### 2. Setup OBS NDI (Optional)

For NDI capture from OBS Studio:

```bash
# Install NDI libraries
pip install PyNDI4

# In OBS Studio:
# 1. Go to Tools -> NDI Output Settings
# 2. Enable "Main Output" 
# 3. Set NDI source name (e.g., "OBS Studio")
# 4. Start streaming/recording

# List available NDI sources
python test_vision_ndi.py list

# Test NDI capture
python test_vision_ndi.py
```

### 3. Download YOLO Model

```bash
# Create models directory
mkdir -p models

# Download YOLOv8n (nano) model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolo-nano.pt

# Or use custom trained model for gaming HUD elements
# Place your custom model at: models/yolo-nano.pt
```

### 4. Setup Template Images

```bash
# Create template directories
mkdir -p assets/templates/everquest
mkdir -p assets/templates/rimworld
mkdir -p assets/templates/generic

# Add template images (PNG format recommended)
# Examples:
# - assets/templates/everquest/health_bar.png
# - assets/templates/everquest/minimap.png
# - assets/templates/everquest/spell_icon.png
```

### 5. Configure Vision Pipeline

Edit `config/vision_config.yaml`:

```yaml
# For EverQuest
capture:
  region: "window"
  window_name: "EverQuest"

# For fullscreen capture
capture:
  region: "fullscreen"
  monitor: 1
```

### 6. Run Vision Pipeline

```bash
# Basic usage
python vision_pipeline.py

# With custom config
python vision_pipeline.py --config config/vision_config.yaml

# Debug mode
python vision_pipeline.py --debug
```

## 📁 Project Structure

```
DanzarAI/
├── vision_pipeline.py              # Main vision pipeline
├── config/
│   └── vision_config.yaml         # Configuration file
├── models/
│   └── yolo-nano.pt               # YOLO model
├── assets/
│   └── templates/                 # Template images
│       ├── everquest/
│       ├── rimworld/
│       └── generic/
├── logs/
│   └── vision_pipeline.log        # Pipeline logs
├── debug_frames/                  # Debug frame output
└── requirements_vision.txt        # Dependencies
```

## ⚙️ Configuration

### Capture Settings

```yaml
capture:
  fps: 10                    # Frames per second
  region: "fullscreen"       # "fullscreen", "window", or "ndi"
  window_name: "EverQuest"   # Window name for capture
  monitor: 1                 # Monitor number
  use_ndi: false             # Enable NDI capture from OBS
  ndi_source_name: "OBS Studio"  # NDI source name
```

### NDI Configuration Examples

**OBS Studio NDI:**
```yaml
capture:
  region: "ndi"
  use_ndi: true
  ndi_source_name: "OBS Studio"
  fps: 30
```

**Screen Capture (Fallback):**
```yaml
capture:
  region: "fullscreen"
  use_ndi: false
  monitor: 1
  fps: 10
```

**Window Capture:**
```yaml
capture:
  region: "window"
  use_ndi: false
  window_name: "EverQuest"
  fps: 10
```

### YOLO Detection

```yaml
yolo:
  model_path: "models/yolo-nano.pt"
  confidence_threshold: 0.5
  device: "cuda:0"           # "cuda:0" or "cpu"
  classes:                   # Detection classes
    - "health_bar"
    - "minimap"
    - "boss"
    - "player"
```

### OCR Regions

```yaml
ocr:
  enabled: true
  roi: [100, 100, 500, 200]  # [x1, y1, x2, y2]
  tesseract_config: "--psm 6"
  min_confidence: 0.6
```

### Event Debouncing

```yaml
debouncing:
  enabled: true
  timeout_ms: 1000           # Debounce timeout
  min_confidence_change: 0.1 # Confidence change threshold
```

## 🎮 Game-Specific Setup

### EverQuest

1. **Window Capture**: Set `window_name: "EverQuest"`
2. **OCR Regions**: Configure chat, loot, and health bar regions
3. **Templates**: Add EverQuest-specific UI elements

```yaml
game_profiles:
  everquest:
    window_name: "EverQuest"
    ocr_regions:
      chat: [50, 400, 600, 500]
      loot: [200, 300, 400, 350]
      health: [50, 50, 200, 80]
```

### RimWorld

1. **Window Capture**: Set `window_name: "RimWorld"`
2. **OCR Regions**: Configure alerts and notifications
3. **Templates**: Add RimWorld UI elements

```yaml
game_profiles:
  rimworld:
    window_name: "RimWorld"
    ocr_regions:
      chat: [100, 500, 700, 600]
      alerts: [50, 50, 400, 150]
```

## 🔧 Integration with DanzarAI

### Event Callback

```python
from vision_pipeline import VisionPipeline, DetectionEvent

def handle_detection(event: DetectionEvent):
    """Handle detection events from vision pipeline"""
    print(f"Detection: {event.object_type} - {event.label}")
    
    # Send to DanzarAI for processing
    if event.object_type == 'ocr' and 'loot' in event.label.lower():
        # Process loot detection
        pass
    elif event.object_type == 'yolo' and event.label == 'boss':
        # Process boss detection
        pass

# Create pipeline with callback
pipeline = VisionPipeline(event_callback=handle_detection)
```

### Async Integration

```python
import asyncio
from vision_pipeline import VisionPipeline

async def main():
    pipeline = VisionPipeline()
    
    # Initialize
    if await pipeline.initialize():
        # Start pipeline
        pipeline.start()
        
        # Run for specified duration
        await asyncio.sleep(300)  # 5 minutes
        
        # Stop pipeline
        pipeline.stop()

# Run async
asyncio.run(main())
```

## 📊 Event Output Format

Each detection event is output as JSON:

```json
{
  "event_id": "uuid-string",
  "timestamp": 1234567890.123,
  "object_type": "yolo",
  "label": "health_bar",
  "confidence": 0.95,
  "bbox": [100, 50, 300, 80],
  "metadata": {
    "class_id": 0,
    "template_size": [200, 30]
  }
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Not Available**
   ```bash
   # Check CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Fallback to CPU
   # Edit config: device: "cpu"
   ```

2. **Tesseract Not Found**
   ```bash
   # Windows: Add to PATH
   # Linux: sudo apt-get install tesseract-ocr
   # macOS: brew install tesseract
   ```

3. **Window Capture Issues**
   ```bash
   # List available windows
   python -c "import mss; print(mss.mss().monitors)"
   
   # Use fullscreen capture as fallback
   # Edit config: region: "fullscreen"
   ```

4. **Performance Issues**
   ```yaml
   # Reduce FPS
   capture:
     fps: 5
   
   # Increase confidence threshold
   yolo:
     confidence_threshold: 0.7
   
   # Disable template matching
   template_matching:
     enabled: false
   ```

### Debug Mode

Enable debug mode for detailed logging:

```bash
python vision_pipeline.py --debug
```

Debug features:
- Save annotated frames to `debug_frames/`
- Detailed console output
- Performance metrics
- Event history

## 📈 Performance Optimization

### GPU Memory Management

```yaml
performance:
  gpu_memory_fraction: 0.8    # Use 80% of GPU memory
  memory_limit_mb: 512        # System memory limit
```

### Processing Optimization

```yaml
# Reduce processing load
capture:
  fps: 5                      # Lower FPS

yolo:
  confidence_threshold: 0.7   # Higher threshold

template_matching:
  max_matches: 2              # Fewer matches

ocr:
  enabled: false              # Disable OCR if not needed
```

## 🔄 Training Custom Models

### YOLO Model Training

1. **Prepare Dataset**: Collect game screenshots with annotations
2. **Train Model**: Use Ultralytics YOLOv8 training
3. **Export Model**: Convert to PT format
4. **Update Config**: Point to custom model

```bash
# Train custom model
yolo train data=game_dataset.yaml model=yolov8n.pt epochs=100

# Export model
yolo export model=runs/train/exp/weights/best.pt format=torchscript
```

### Template Creation

1. **Capture UI Elements**: Screenshot specific UI components
2. **Crop Templates**: Extract clean template images
3. **Organize**: Place in appropriate template directories
4. **Test**: Verify template matching accuracy

## 📝 API Reference

### VisionPipeline Class

```python
class VisionPipeline:
    def __init__(self, config_path: str, event_callback: Callable)
    async def initialize() -> bool
    def start()
    def stop()
    def get_status() -> Dict[str, Any]
```

### DetectionEvent Class

```python
@dataclass
class DetectionEvent:
    event_id: str
    timestamp: float
    object_type: str
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    metadata: Dict[str, Any]
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

This project is part of DanzarAI and follows the same license terms.

## 🆘 Support

For issues and questions:
1. Check troubleshooting section
2. Review debug logs
3. Open GitHub issue with details
4. Include system specs and error messages 