# core/game_profile.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class GameProfile:
    # — Basic identity
    game_name: str

    # — VLM settings
    vlm_model: str
    system_prompt_commentary: str
    user_prompt_template_commentary: str # e.g., "Current game is {game_name}. Visible text: {ocr_text}. Player actions: {player_actions}. Give advice."
    vlm_max_tokens: int = 200
    vlm_temperature: float = 0.5
    vlm_max_commentary_sentences: int = 2

    # — Vision Analysis Prompts (for screenshot analysis)
    vision_prompts: Dict[str, str] = field(default_factory=lambda: {
        "action": "What is happening in this game screenshot? Describe the main action or activity taking place.",
        "ui": "Describe the UI elements visible in this game screenshot: health bars, menus, buttons, or interface components.",
        "scene": "What type of game environment or location is shown in this screenshot?",
        "character": "Describe any characters, NPCs, or entities visible in this game screenshot.",
        "quick": "Briefly describe what's happening in this gaming screenshot in 1-2 sentences."
    })

    # — Conversational LLM settings (optional)
    conversational_llm_model: Optional[str] = None
    system_prompt_chat: Optional[str]     = None
    conversational_max_tokens: int        = 120
    conversational_temperature: float     = 0.7

    # — RAG settings (optional)
    rag_collection_name: Optional[str] = None
    rag_top_k: int                     = 3

    # — Wake Word overrides (optional)
    oww_model_path_override: Optional[str] = None
    oww_model_name_override: Optional[str] = None

    # — Reference image for UI-map (optional)
    reference_image_path: Optional[str] = None # Keep this if your VLM uses it directly

    # --- OCR Settings ---
    ocr_settings: Dict = field(default_factory=lambda: {
        "engine": "easyocr",
        "easyocr_languages": ['en'],
        "easyocr_gpu": True,
        "default_upscale_factor": 2.0,
        "default_grayscale": True,
        "default_binarize": True,
        "default_binarize_method": "otsu", # 'otsu', 'adaptive', 'simple'
        "default_confidence_threshold": 0.25
    })
    ocr_layout_path: Optional[str] = None # Path to JSON file defining ROIs and their specific settings

    # — Regions of interest (FOR VLM HINTS - distinct from OCR ROIs if needed, but can overlap)
    # This is for telling the VLM "look at this area for this reason"
    # OCR ROIs are for text extraction from specific boxes.
    regions_of_interest: List[Dict[str, str]] = field(default_factory=list)

    # --- Short-Term Memory Settings ---
    memory_deque_size: int = 3  # Number of recent commentary themes to keep in deque
    memory_rag_history_collection_name: Optional[str] = "danzarvlm_conversation_history" # Dedicated RAG collection
    memory_rag_vlm_commentary_lookback_k: int = 2 # How many relevant items to pull from RAG for VLM commentary context
    memory_rag_chat_lookback_k: int = 5         # How many relevant items to pull from RAG for chat context