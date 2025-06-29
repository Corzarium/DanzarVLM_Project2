# Vision System Documentation

## 🎯 **System Overview**

The DanzarAI vision system provides real-time gaming commentary by analyzing visual content from OBS streams. It combines multiple AI models (YOLO, OCR, CLIP) with a Vision Language Model (VLM) to generate intelligent commentary about what's happening in games.

## 🏗️ **Architecture Components**

### 1. **OBS Studio** (Source)
- **Purpose**: Captures and streams game content
- **Output**: NDI stream containing game video
- **Configuration**: NDI Output enabled in OBS

### 2. **NDI Service** (`services/ndi_service.py`)
- **Purpose**: Receives OBS NDI stream and captures frames
- **Key Methods**: 
  - `initialize_ndi()`: Connects to OBS NDI stream
  - `run_capture_loop()`: Continuously captures frames
  - `last_captured_frame`: Stores the most recent frame
- **Frame Rate**: Configurable (default: 1 FPS)
- **Output**: BGR numpy arrays

### 3. **Vision Pipeline** (`vision_pipeline.py`)
- **Purpose**: Orchestrates all vision processing
- **Components**:
  - NDI Service integration
  - YOLO object detection
  - OCR text recognition
  - Template matching
  - CLIP video understanding
- **Key Methods**:
  - `initialize()`: Sets up all components
  - `start()`: Begins processing
  - `_capture_loop()`: Captures frames from NDI
  - `_processing_loop()`: Processes frames with AI models

### 4. **Vision Integration Service** (`services/vision_integration_service.py`)
- **Purpose**: Integrates vision with LLM and TTS for commentary
- **Key Features**:
  - Event processing and filtering
  - Screenshot capture for VLM
  - Commentary generation
  - TTS integration
- **Key Methods**:
  - `initialize()`: Sets up vision integration
  - `_capture_current_screenshot()`: Captures OBS frames for VLM
  - `_create_unified_prompt()`: Creates VLM prompts with vision data
  - `_generate_commentary()`: Generates commentary using VLM

### 5. **AI Models**
- **YOLO**: Object detection (people, items, UI elements)
- **OCR**: Text recognition (game text, UI labels)
- **CLIP**: Visual understanding (scene analysis)
- **VLM**: Vision Language Model (commentary generation)

## 🔄 **Data Flow**

### 1. **Frame Capture Flow**
```
OBS Studio → NDI Stream → NDI Service → Vision Pipeline → Vision Integration
```

### 2. **Event Detection Flow**
```
Frame → YOLO Detection → DetectionEvent → Vision Integration → Commentary Trigger
Frame → OCR Recognition → DetectionEvent → Vision Integration → Commentary Trigger
Frame → CLIP Analysis → CLIPVideoUpdate → Vision Integration → Enhanced Context
```

### 3. **Commentary Generation Flow**
```
DetectionEvent → Screenshot Capture → Unified Prompt → VLM → Commentary → TTS
```

### 4. **Screenshot Capture Priority**
```
1. NDI Service (OBS stream) ← PRIMARY
2. Vision Pipeline Screenshot
3. PIL Screen Capture ← FALLBACK ONLY
```

## 📊 **Configuration Files**

### 1. **Vision Config** (`config/vision_config.yaml`)
```yaml
capture:
  use_ndi: true
  region: "ndi"
  fps: 1

yolo:
  model_path: "yolov8n.pt"
  device: "cuda:1"
  confidence_threshold: 0.6

ocr:
  enabled: true
  roi: [0, 0, 1920, 1080]
  confidence_threshold: 0.6

template_matching:
  enabled: false
  templates_dir: "assets/templates"

clip:
  enabled: false  # Temporarily disabled
```

### 2. **Global Settings** (`config/global_settings.yaml`)
```yaml
VISION_COMMENTARY:
  enabled: true
  frequency_seconds: 5.0
  min_confidence: 0.6
  conversation_mode: true

NDI_CONNECTION_TIMEOUT_MS: 5000
NDI_RECEIVE_TIMEOUT_MS: 1000
vision_capture_fps: 1
```

## 🔧 **Key Methods and Their Functions**

### **NDI Service**
```python
# Initialize NDI connection to OBS
ndi_service.initialize_ndi()

# Get latest captured frame
frame = ndi_service.last_captured_frame

# Check if NDI is working
is_working = ndi_service.is_initialized
```

### **Vision Pipeline**
```python
# Initialize all vision components
await vision_pipeline.initialize()

# Start processing
vision_pipeline.start()

# Check status
status = vision_pipeline.get_status()
```

### **Vision Integration Service**
```python
# Initialize with auto-start
await vision_integration.initialize()

# Capture screenshot for VLM
screenshot_b64 = vision_integration._capture_current_screenshot()

# Create VLM prompt with vision data
prompt = vision_integration._create_unified_prompt(event, analysis)

# Generate commentary
commentary = await vision_integration._generate_commentary(prompt)
```

## 🎮 **Event Types and Processing**

### **DetectionEvent Structure**
```python
@dataclass
class DetectionEvent:
    event_id: str
    timestamp: float
    object_type: str  # 'yolo', 'ocr', 'template', 'clip'
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    metadata: Dict[str, Any]
```

### **Event Processing Logic**
```python
# Check if event should trigger commentary
should_comment = vision_integration._should_generate_commentary(event)

# Check for significant changes
is_significant = vision_integration._has_significant_change(event)

# Process commentary trigger
vision_integration._process_commentary_trigger(event)
```

## 📸 **Screenshot Integration**

### **Screenshot Capture Method**
```python
def _capture_current_screenshot(self) -> Optional[str]:
    # Method 1: NDI Service (OBS stream) - PRIMARY
    if hasattr(self, 'vision_pipeline') and self.vision_pipeline.ndi_service:
        frame = self.vision_pipeline.ndi_service.last_captured_frame
        return self._process_frame_for_vlm(frame)
    
    # Method 2: Vision Pipeline Screenshot
    # Method 3: PIL Screen Capture (fallback)
```

### **Frame Processing for VLM**
```python
def _process_frame_for_vlm(self, frame) -> Optional[str]:
    # Resize to optimal VLM size (640x480)
    # Convert to JPEG with 85% quality
    # Encode to base64
    # Return for prompt inclusion
```

## 🎯 **Commentary Generation**

### **Unified Prompt Structure**
```
<|im_start|>system
You are DanzarAI, an intelligent gaming assistant with advanced vision capabilities...

Current Game: {game_name}
<|im_end|>
<|im_start|>user
I'm watching a game and detected some interesting elements. Here's what I found:

**YOLO Object Detections:**
{detected_objects}

**OCR Text Detected:**
{detected_text}

**CLIP Visual Understanding:**
{clip_insights}

**Trigger Event:**
{event_type}: {event_label} (confidence: {confidence})

**Visual Analysis:**
<image>
{screenshot_base64}
</image>

As a gaming commentator with vision capabilities, provide a brief, engaging response about what you see.
<|im_end|>
<|im_start|>assistant
```

### **Commentary Processing**
```python
# Generate commentary with VLM
response = await model_client.generate(messages)

# Process response
if response:
    # Send text to Discord
    await text_callback(response)
    
    # Generate TTS audio
    await tts_callback(response)
```

## 🔍 **Troubleshooting Guide**

### **Common Issues and Solutions**

#### 1. **No Vision Events Detected**
**Symptoms**: No commentary, no events in logs
**Diagnosis**:
```python
# Check if vision pipeline is running
vision_pipeline.get_status()

# Check if NDI service is connected
ndi_service.is_initialized

# Check if OBS NDI output is enabled
# Look for NDI source in OBS
```

**Solutions**:
- Enable NDI Output in OBS
- Check NDI service initialization
- Verify vision pipeline is started

#### 2. **Using PIL Instead of NDI Screenshots**
**Symptoms**: Log shows "Using PIL screen capture"
**Diagnosis**:
```python
# Check NDI service availability
hasattr(vision_pipeline, 'ndi_service')
vision_pipeline.ndi_service.is_initialized
vision_pipeline.ndi_service.last_captured_frame is not None
```

**Solutions**:
- Ensure OBS NDI stream is active
- Check NDI service connection
- Verify vision pipeline initialization

#### 3. **No Commentary Generated**
**Symptoms**: Events detected but no commentary
**Diagnosis**:
```python
# Check if event processor is running
vision_integration.event_processor_task.done()

# Check if model client is available
hasattr(app_context, 'model_client')

# Check commentary settings
vision_integration.enable_commentary
vision_integration.min_confidence
```

**Solutions**:
- Ensure model client is initialized
- Check commentary settings
- Verify event processor is running

#### 4. **TTS Not Working for Vision Commentary**
**Symptoms**: Text sent but no audio
**Diagnosis**:
```python
# Check TTS callback availability
vision_integration.tts_callback is not None

# Check TTS service
hasattr(app_context, 'tts_service')

# Check Discord voice connection
bot.voice_clients
```

**Solutions**:
- Ensure TTS service is initialized
- Check Discord voice connection
- Verify TTS callback is properly set

### **Log Messages to Monitor**

#### **✅ Success Indicators**
```
[NDIService] Successfully connected to NDI source: OBS
[VisionPipeline] YOLO model loaded on cuda:1
[VisionIntegration] 📸 Using NDI service screenshot from vision pipeline
[VisionIntegration] ✅ Screenshot captured successfully: XXXXX chars
[VisionIntegration] TTS audio played through Discord successfully
```

#### **⚠️ Warning Indicators**
```
[NDIService] No NDI sources found after discovery period
[VisionIntegration] 📸 Falling back to PIL screen capture (not OBS stream)
[VisionIntegration] No model client available for commentary generation
[VisionIntegration] TTS service returned no audio
```

#### **❌ Error Indicators**
```
[NDIService] Failed to initialize NDI connection
[VisionIntegration] Event processor task failed to start
[VisionIntegration] Screenshot capture error
[VisionIntegration] Commentary generation error
```

## 🚀 **Performance Optimization**

### **Frame Rate Settings**
- **NDI Capture**: 1 FPS (configurable)
- **YOLO Processing**: Every frame
- **OCR Processing**: Every frame
- **CLIP Processing**: Disabled (performance)
- **Commentary Generation**: Rate limited (5s intervals)

### **Memory Management**
- **Frame Queue**: Limited size to prevent memory bloat
- **Detection History**: Last 50 detections only
- **Screenshot Quality**: 85% JPEG for VLM
- **Image Size**: 640x480 for VLM processing

### **GPU Utilization**
- **YOLO**: CUDA:1 (secondary GPU)
- **VLM**: CUDA:0 (primary GPU)
- **Memory**: Managed by GPU Memory Manager

## 📝 **Maintenance and Updates**

### **Regular Checks**
1. **NDI Connection**: Verify OBS NDI output is working
2. **Model Performance**: Monitor YOLO and OCR accuracy
3. **Commentary Quality**: Review generated commentary
4. **System Resources**: Check GPU memory and CPU usage

### **Configuration Updates**
1. **Game Profiles**: Update for new games
2. **YOLO Models**: Upgrade for better detection
3. **OCR Regions**: Adjust for different UI layouts
4. **Commentary Settings**: Tune frequency and confidence

### **Troubleshooting Workflow**
1. **Check Logs**: Look for error messages
2. **Verify Components**: Test each component individually
3. **Test Configuration**: Validate settings
4. **Restart Services**: Reinitialize if needed
5. **Check Dependencies**: Ensure all services are running

## 🎯 **Success Metrics**

### **System Health Indicators**
- ✅ NDI frames captured continuously
- ✅ YOLO detections with >60% confidence
- ✅ OCR text recognition working
- ✅ Screenshots captured from OBS stream
- ✅ Commentary generated and sent to Discord
- ✅ TTS audio played through Discord

### **Performance Metrics**
- **Frame Capture Rate**: 1 FPS sustained
- **Detection Latency**: <100ms per frame
- **Commentary Generation**: <5s per event
- **TTS Latency**: <2s from generation to audio
- **Memory Usage**: <2GB GPU, <1GB CPU

This documentation provides a complete reference for understanding, maintaining, and troubleshooting the DanzarAI vision system! 🎮👁️📸 