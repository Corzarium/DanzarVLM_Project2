# core/config_loader.py

import os
import sys
import yaml
from typing import Any, Dict, List, Optional

from .game_profile import GameProfile

# ─── Paths ─────────────────────────────────────────────────────────────────────
CURRENT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
CONFIG_DIR      = os.path.join(PROJECT_ROOT, "config")
GLOBAL_SETTINGS = os.path.join(CONFIG_DIR, "global_settings.yaml")
PROFILES_DIR    = os.path.join(CONFIG_DIR, "profiles")

# ─── Logger helper ──────────────────────────────────────────────────────────────
_logger = None
def _get_logger():
    global _logger
    if _logger is None:
        import logging
        _logger = logging.getLogger("ConfigLoader")
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        handler.setFormatter(fmt)
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
    return _logger

# ─── Public API ─────────────────────────────────────────────────────────────────

def load_global_settings(
    settings_file_path: str = GLOBAL_SETTINGS
) -> Optional[Dict[str, Any]]:
    logger = _get_logger()
    if not os.path.exists(settings_file_path):
        logger.error(f"Global settings not found at {settings_file_path}")
        return None
    try:
        with open(settings_file_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        logger.info(f"Successfully loaded global settings from {settings_file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load global settings: {e}", exc_info=True)
        return None

def list_available_profiles(
    profiles_directory: str = PROFILES_DIR
) -> List[str]:
    logger = _get_logger()
    if not os.path.isdir(profiles_directory):
        logger.warning(f"Profiles directory not found: {profiles_directory}")
        return []
    profiles = [
        os.path.splitext(fn)[0]
        for fn in os.listdir(profiles_directory)
        if fn.endswith((".yaml", ".yml"))
    ]
    logger.info(f"Available profiles: {profiles}")
    return profiles

def load_game_profile(
    profile_name: str,
    global_settings: Optional[Dict[str, Any]] = None,
    profiles_directory: str = PROFILES_DIR
) -> Optional[GameProfile]:
    logger = _get_logger()
    if global_settings is None:
        global_settings = load_global_settings() or {}

    # Locate the YAML file
    profile_path = None
    for ext in ("yaml", "yml"):
        candidate = os.path.join(profiles_directory, f"{profile_name}.{ext}")
        if os.path.exists(candidate):
            profile_path = candidate
            break
    if not profile_path:
        logger.error(f"Profile '{profile_name}' not found in {profiles_directory}")
        return None

    # Parse YAML
    try:
        with open(profile_path, 'r') as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Error parsing profile '{profile_name}': {e}", exc_info=True)
        return None

    # Helper: local → global → literal default
    def gv(key: str, literal_default: Any, gkey: str = None) -> Any:
        if key in data and data[key] is not None:
            return data[key]
        if gkey and gkey in global_settings:
            return global_settings[gkey]
        return literal_default

    # Build GameProfile
    try:
        gp = GameProfile(
            game_name                     = data.get('game_name', profile_name),
            # — VLM settings
            vlm_model                     = gv('vlm_model', None,           'DEFAULT_VLM_MODEL'),
            system_prompt_commentary      = gv('system_prompt_commentary', "", None),
            user_prompt_template_commentary = gv('user_prompt_template_commentary', "", None),
            vlm_max_tokens                = gv('vlm_max_tokens', 150,       'DEFAULT_VLM_MAX_TOKENS'),
            vlm_temperature               = gv('vlm_temperature', 0.5,      'DEFAULT_VLM_TEMPERATURE'),
            vlm_max_commentary_sentences  = gv('vlm_max_commentary_sentences', 2, 'DEFAULT_COMMENTARY_SENTENCES'),

            # — Conversational LLM settings
            conversational_llm_model      = gv('conversational_llm_model', None, 'DEFAULT_CONVERSATIONAL_LLM_MODEL'),
            system_prompt_chat            = gv('system_prompt_chat', "",     None),
            conversational_max_tokens     = gv('conversational_max_tokens', 120, 'DEFAULT_CHAT_MAX_TOKENS'),
            conversational_temperature    = gv('conversational_temperature', 0.7, 'DEFAULT_CHAT_TEMPERATURE'),

            # — RAG settings
            rag_collection_name           = data.get('rag_collection_name'),
            rag_top_k                     = gv('rag_top_k', 3,              'DEFAULT_RAG_TOP_K'),

            # — Wake Word overrides
            oww_model_path_override       = data.get('oww_model_path_override'),
            oww_model_name_override       = data.get('oww_model_name_override'),

            # — Reference image + ROIs
            reference_image_path          = data.get('reference_image_path'),
            regions_of_interest           = data.get('regions_of_interest', []),

            # — Memory settings
            memory_deque_size             = gv('memory_deque_size', 3,      'DEFAULT_MEMORY_DEQUE_SIZE'),
            memory_rag_history_collection_name = gv('memory_rag_history_collection_name', 'danzarvlm_conversation_history', 'DEFAULT_MEMORY_RAG_COLLECTION'),
            memory_rag_vlm_commentary_lookback_k = gv('memory_rag_vlm_commentary_lookback_k', 2, 'DEFAULT_VLM_MEMORY_LOOKBACK_K'),
            memory_rag_chat_lookback_k    = gv('memory_rag_chat_lookback_k', 5, 'DEFAULT_CHAT_MEMORY_LOOKBACK_K'),

            # — OCR Settings
            ocr_settings                  = data.get('ocr_settings', {}),
            ocr_layout_path               = data.get('ocr_layout_path'),
        )
        logger.info(f"Loaded GameProfile for '{gp.game_name}' from {profile_path}")
        return gp
    except Exception as e:
        logger.error(f"Failed to build GameProfile: {e}", exc_info=True)
        return None
