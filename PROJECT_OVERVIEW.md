# DanzarAI Project Overview

**Project Name:** DanzarAI - Multimodal Gaming Commentary Assistant  
**Project Manager:** [Your Name]  
**Compiled By:** AI Assistant  
**Date:** June 28, 2024  
**Executive Sponsor:** [Your Name]  

---

## 🎯 Business Case

DanzarAI addresses the growing demand for intelligent, real-time gaming commentary and interaction. The system provides:

- **Enhanced Gaming Experience:** Real-time AI commentary that understands game context and player actions
- **Accessibility:** Voice and text interaction options for different user preferences
- **Memory & Context:** Persistent memory system that learns from interactions and provides personalized responses
- **Modular Architecture:** Easily swappable components for different AI models and services
- **Discord Integration:** Seamless integration with popular gaming communication platform

**Key Benefits:**
- Reduces cognitive load on players by providing intelligent commentary
- Creates engaging, interactive gaming experiences
- Demonstrates advanced AI integration capabilities
- Provides foundation for future AI-powered gaming applications

---

## 🚀 Reason for the Project

**Problem/Opportunity:** Modern gaming lacks intelligent, context-aware AI companions that can understand both visual game state and player interactions. Current solutions are either too generic or lack the multimodal capabilities needed for truly engaging experiences.

**Solution:** DanzarAI creates a modular, multimodal AI system that combines vision understanding, voice interaction, and persistent memory to provide intelligent gaming commentary and assistance.

**Approach:** The system uses a central LLM/VLM (currently Qwen2.5-VL) as the core intelligence, with specialized modules for vision processing, voice interaction, and memory management. All components communicate through standardized interfaces, allowing for easy model swapping and system evolution.

---

## 🎯 Goals and Objectives

### Primary Goals
1. **Real-time Vision Understanding:** Analyze game frames and provide contextual commentary
2. **Natural Voice Interaction:** Enable voice commands and responses through Discord
3. **Persistent Memory:** Maintain conversation history and game knowledge across sessions
4. **Modular Architecture:** Support easy swapping of AI models and services

### Success Criteria
- **Performance:** Sub-2 second response time for voice interactions
- **Accuracy:** 90%+ transcription accuracy with Whisper Large
- **Reliability:** 99% uptime for core services
- **Scalability:** Support for multiple concurrent users
- **Flexibility:** Easy model swapping without code changes

---

## 📋 Project Scope

### In Scope
- **Core AI System:** LLM/VLM integration with vision and voice capabilities
- **Memory System:** STM (RAM) and LTM (Qdrant) with automatic consolidation
- **Discord Integration:** Bot commands, voice channels, user tracking
- **Vision Pipeline:** NDI/OBS frame capture, OCR, object detection, CLIP analysis
- **Voice Pipeline:** STT (Whisper), TTS (Azure), virtual audio routing
- **Configuration Management:** Centralized settings for all services

### Out of Scope
- **Game-specific integrations:** System is game-agnostic
- **Multi-platform support:** Currently Discord-focused
- **Advanced analytics:** Basic logging only
- **User authentication:** Discord handles user management
- **Commercial deployment:** Development and personal use focus

---

## 🏗️ System Architecture

### Core Components

#### 1. **Main LLM/VLM Module** (Qwen2.5-VL)
- **Endpoint:** `http://localhost:8083/v1/chat/completions`
- **Role:** Central intelligence for all language and vision tasks
- **Swappable:** Easy replacement via configuration
- **Capabilities:** Text and vision understanding, conversation generation

#### 2. **Voice Module**
- **Input:** Transcribed speech from STT
- **Output:** LLM responses to TTS
- **Features:** User tracking, context management, command processing

#### 3. **Vision Module**
- **Input:** NDI/OBS frames
- **Processing:** OCR, YOLO detection, CLIP analysis
- **Output:** Visual context for LLM commentary
- **Rate:** 1 FPS processing for performance

#### 4. **Memory System**
- **STM (Short-Term):** In-RAM buffer of recent interactions
- **LTM (Long-Term):** Qdrant vector database for persistent knowledge
- **Consolidation:** Automatic transfer from STM to LTM
- **Retrieval:** RAG for context-aware responses

#### 5. **STT Service** (Whisper Large)
- **Input:** Discord voice audio
- **Output:** Transcribed text
- **Performance:** Real-time processing

#### 6. **TTS Service** (Azure)
- **Input:** LLM text responses
- **Output:** Synthesized speech to Discord
- **Quality:** Natural-sounding voice synthesis

#### 7. **Discord Integration**
- **Bot Commands:** `!danzar`, `!watch`, `!stopwatch`, etc.
- **Voice Management:** Channel joining, user tracking
- **Audio Routing:** Virtual audio device integration

---

## 🔄 Data Flow

```
Discord User Input (Voice/Text)
    ↓
STT (Whisper) → Voice Module
    ↓
Vision Module ← NDI/OBS Frames
    ↓
Memory System (STM/LTM)
    ↓
Main LLM/VLM (Qwen2.5-VL)
    ↓
TTS (Azure) → Discord Output
```

---

## 📁 File Structure

```
DanzarVLM_Project/
├── DanzarVLM.py                 # Main application entry point
├── config/
│   ├── global_settings.yaml     # Central configuration
│   ├── profiles/                # Game-specific settings
│   └── vision_config.yaml       # Vision processing settings
├── services/
│   ├── qwen_vl_service.py       # Main LLM/VLM client
│   ├── vision_integration_service.py  # Vision processing
│   ├── memory_manager.py        # STM/LTM management
│   ├── ndi_service.py           # Frame capture
│   └── [other service modules]
├── discord_integration/
│   ├── bot_client.py            # Main Discord bot
│   ├── vision_cog.py            # Vision commands
│   └── voice_cog.py             # Voice commands
├── logs/                        # System logs
└── start_*.bat                  # Service startup scripts
```

---

## ⚙️ Configuration

### Key Configuration Files
- **`config/global_settings.yaml`:** Main system configuration
- **`config/vision_config.yaml`:** Vision processing settings
- **`config/profiles/`:** Game-specific configurations

### Important Settings
```yaml
# Main LLM/VLM
QWEN_VL_SERVER:
  endpoint: "http://localhost:8083/v1/chat/completions"
  timeout: 120
  enabled: true

# Memory System
MEMORY_MANAGER:
  stm_max_turns: 50
  ltm_collection: "danzar_knowledge"
  consolidation_interval: 300

# Vision Processing
VISION_PIPELINE:
  capture_fps: 1
  processing_fps: 1
  queue_size: 10
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- NVIDIA GPU (RTX 2080 Ti or better)
- Discord Bot Token
- Azure TTS Subscription
- NDI Tools

### Quick Start
1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Configure Settings:** Update `config/global_settings.yaml`
3. **Start Qwen2.5-VL:** `start_qwen_vl_multi_gpu.bat`
4. **Launch DanzarAI:** `python DanzarVLM.py`
5. **Test Commands:** `!danzar`, `!watch` in Discord

---

## 🔧 Development Guidelines

### Code Standards
- **Type Annotations:** All functions must have type hints
- **Error Handling:** Comprehensive exception handling with logging
- **Documentation:** Docstrings for all public methods
- **Testing:** Pytest for new features
- **Logging:** Structured logging with appropriate levels

### Architecture Principles
- **Modularity:** Each service is independent and swappable
- **Configuration-Driven:** All settings in YAML files
- **Event-Driven:** Asynchronous communication between components
- **Memory-Efficient:** Proper resource management and cleanup

### Adding New Models
1. Create service module in `services/`
2. Add configuration in `global_settings.yaml`
3. Update main app initialization
4. Test with existing interfaces

---

## 📊 Monitoring & Logging

### Log Files
- **`logs/errors/`:** Critical errors for Discord notification
- **`logs/debug/`:** Detailed debugging information
- **Console Output:** Real-time system status

### Key Metrics
- Response times for voice/text interactions
- Vision processing FPS and accuracy
- Memory system performance
- Discord bot uptime and command usage

---

## 🔮 Future Enhancements

### Planned Features
- **Multi-Game Support:** Enhanced game-specific profiles
- **Advanced Memory:** Semantic search and knowledge graphs
- **Performance Optimization:** GPU acceleration for all components
- **User Customization:** Personal AI personality settings
- **Analytics Dashboard:** Web interface for system monitoring

### Potential Integrations
- **Other AI Models:** LLaVA, GPT-4V, Claude Vision
- **Additional Platforms:** Twitch, YouTube, custom web interface
- **Advanced TTS:** Local models, voice cloning
- **Enhanced STT:** Speaker diarization, emotion detection

---

## 🛠️ Troubleshooting

### Common Issues
1. **VLM Timeout:** Check Qwen2.5-VL server status on port 8083
2. **Discord Audio:** Verify virtual audio device configuration
3. **Memory Issues:** Check Qdrant connection and collection status
4. **Vision Pipeline:** Monitor NDI source availability and frame rates

### Debug Commands
- `!status` - System health check
- `!memory` - Memory system status
- `!watch` - Start vision commentary
- `!stopwatch` - Stop vision commentary

---

## 📚 References

### Documentation
- [Discord.py Documentation](https://discordpy.readthedocs.io/)
- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [NDI SDK Documentation](https://docs.ndi.video/)
- [Qdrant Vector Database](https://qdrant.tech/documentation/)

### Related Projects
- [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Whisper](https://github.com/openai/whisper)

---

## 📝 Change Log

### Version 1.0 (Current)
- ✅ Core LLM/VLM integration with Qwen2.5-VL
- ✅ Vision pipeline with OCR, YOLO, and CLIP
- ✅ Voice interaction with Whisper STT and Azure TTS
- ✅ Memory system with STM/LTM
- ✅ Discord bot integration
- ✅ Modular architecture with configuration-driven setup

### Next Version (Planned)
- 🔄 Enhanced memory consolidation
- 🔄 Performance optimizations
- 🔄 Additional AI model support
- 🔄 Web-based monitoring dashboard

---

*This document serves as the primary reference for DanzarAI development. Update this overview when making significant architectural changes or adding new features.*
