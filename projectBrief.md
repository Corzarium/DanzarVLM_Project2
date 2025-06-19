🚀 Project Goal
DanzarVLM is a modular Python-based AI assistant that can:

🔍 Watch gameplay via NDI stream frame capture

🧠 Analyze screenshots with a Visual Language Model (VLM)

📚 Retrieve context using a Qdrant-powered RAG system

💬 Respond through Discord or TTS

🗣️ Activate via wake-word or chat triggers

Use cases include live game commentary, in-game coaching, alerts, and lore referencing.

🧩 System Architecture
🧬 Core Pipeline
mermaid
Copy
Edit
graph TD
    A[Wake Word or Discord Trigger] --> B[NDI Screenshot Capture]
    B --> C[Visual Analysis (VLM)]
    C --> D[RAG Contextual Search (Qdrant)]
    D --> E[LLM Response Generation]
    E --> F{Output Method}
    F -->|TTS| G[Chatterbox Server]
    F -->|Text| H[Discord Message]
📁 Folder Breakdown
Folder	Description
audio_integration/	Wake word listener and microphone input (used for real-time voice triggers).
chroma_rag_db/	Legacy or placeholder RAG code (being phased out in favor of Qdrant).
config/	YAML configuration files to control model behavior, services, and system settings.
discord_integration/	Discord bot logic, command handlers, and async message flows.
ndi_integration/	Captures frames from NDI video streams (OBS/Game client output).
services/	Core logic hub: LLM client, image parser, context dispatcher, and model routing.
tts_integration/	Routes generated responses to Chatterbox TTS server or alternatives.
utils/	Helper scripts used across services for logging, image conversion, or device checks.

🧠 AI/ML Components
Component	Model	Tool
LLM	Qwen2-VL-7B / Gemma / Deepseek	Ollama / LM Studio
VLM	Qwen2-VL / Donut	Torch / Hugging Face
RAG	Qdrant	Local vector search
TTS	Chatterbox	Local Docker-based server

📋 Features & TODOs
✅ Done
Voice-triggered activation

Discord bot text interaction

NDI screenshot pipeline

LLM + TTS integration

Chatterbox web server (running on port 8055)

🛠️ In Progress
Supabase RAG (optional alternative to Qdrant)

Expanded screenshot labeling and interpretation

Auto-alert system for low health/mana

In-game object tracking from screenshots

🤖 Tips for Devs & LLMs
To change behavior or profile, edit config/global_settings.yaml.

Main entry point logic lives in DanzarVLM.py and services/.

Frame analysis begins in ndi_integration/frame_capture.py.

TTS output goes through tts_integration/tts_router.py.

RAG queries are handled via services/context_memory.py (Qdrant-backed).

Discord command routing is in discord_integration/bot_client.py