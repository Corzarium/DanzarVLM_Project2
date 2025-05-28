# DanzarVLM.py
import os
import sys
import threading
import time
import queue
# import asyncio # Keep asyncio for discord if DiscordBotRunner uses it internally
import argparse
import logging # For logging levels
from typing import Optional, List, Dict, Callable # Added Callable

from core.config_loader import load_global_settings, load_game_profile, list_available_profiles
from core.game_profile import GameProfile
from utils.general_utils import setup_logger

from services.ndi_service import NDIService
from services.llm_service import LLMService
from services.audio_service import AudioService
from services.rag_service import RAGService
from services.memory_service import MemoryService
from discord_integration.bot_client import DiscordBotRunner
from web.server import run_server  # Add this import
from services.rag_service import RAGService
from services.model_client import ModelClient
from services.llm_service import LLMService

class AppContext:
    def __init__(self, global_settings: dict, active_profile: GameProfile, logger_instance: logging.Logger):
        self.global_settings = global_settings
        self.active_profile = active_profile
        self.logger = logger_instance

        self.ndi_commentary_enabled = threading.Event()
        self.is_in_conversation = threading.Event()
        self.last_interaction_time = time.time()
        self.shutdown_event = threading.Event()

        # --- NEW: Event to track TTS playback state ---
        self.tts_is_playing = threading.Event()
        self.tts_is_playing.clear() # Explicitly start as not playing
        # --- END NEW ---

        frame_q_size = self.global_settings.get("FRAME_QUEUE_MAX_SIZE", 5)
        tts_q_size = self.global_settings.get("TTS_QUEUE_MAX_SIZE", 20)
        text_q_size = self.global_settings.get("TEXT_MESSAGE_QUEUE_MAX_SIZE", 20)

        self.frame_queue = queue.Queue(maxsize=frame_q_size)
        self.tts_queue = queue.Queue(maxsize=tts_q_size)
        self.text_message_queue = queue.Queue(maxsize=text_q_size)

        self.discord_bot_async_loop = None
        self.discord_voice_client = None
        self.discord_bot_instance = None

        self.ndi_service_instance: Optional[NDIService] = None
        self.audio_service_instance: Optional[AudioService] = None
        self.llm_service_instance: Optional[LLMService] = None
        self.rag_service_instance: Optional[RAGService] = None
        self.memory_service_instance: Optional[MemoryService] = None
        self.discord_bot_runner_instance: Optional[DiscordBotRunner] = None
        self.model_client_instance: Optional[ModelClient] = None

        self.active_profile_change_subscribers: List[Callable[[GameProfile], None]] = []

        self.logger.info("[AppContext] Initialized.")

    def update_active_profile(self, new_profile: GameProfile):
        old_profile_name = self.active_profile.game_name if self.active_profile else "N/A"
        self.active_profile = new_profile
        self.logger.info(f"[AppContext] Active profile updated from '{old_profile_name}' to: {new_profile.game_name}")
        for subscriber_callback in self.active_profile_change_subscribers:
            try:
                cb_name = getattr(subscriber_callback, '__name__', str(subscriber_callback))
                if hasattr(subscriber_callback, '__self__'):
                    cb_name = f"{subscriber_callback.__self__.__class__.__name__}.{cb_name}"
                self.logger.debug(f"[AppContext] Notifying subscriber '{cb_name}' of profile change.")
                subscriber_callback(new_profile)
            except Exception as e:
                cb_name_err = getattr(subscriber_callback, '__name__', str(subscriber_callback))
                self.logger.error(f"[AppContext] Error notifying subscriber '{cb_name_err}': {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="DanzarVLM - AI Game Commentary and Interaction Suite")
    parser.add_argument(
        "--profile",
        help="Name of the game profile to load from config/profiles/ (e.g., rimworld). If not set, will use profile from global_settings.yaml or 'generic_game'."
    )
    parser.add_argument(
        "--log-level",
        default=None, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (overrides global_settings.yaml if specified)."
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=5000,
        help="Port for the web interface (default: 5000)"
    )
    args = parser.parse_args()

    try:
        global_cfg = load_global_settings()
        if not global_cfg:
            logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
            logging.critical("FATAL: Failed to load global_settings.yaml. Exiting.")
            sys.exit(1)
    except Exception as e:
        logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.critical(f"FATAL: Error loading global_settings.yaml: {e}", exc_info=True)
        sys.exit(1)

    log_level_to_set = "INFO" 
    if 'LOG_LEVEL' in global_cfg:
        log_level_to_set = global_cfg['LOG_LEVEL'].upper()
    if args.log_level: 
        log_level_to_set = args.log_level.upper()
    
    logger = setup_logger(name="DanzarVLM", level_str=log_level_to_set)
    profile_arg_str = args.profile if args.profile else f"(default from global_settings: {global_cfg.get('DEFAULT_GAME_PROFILE', 'generic_game')})"
    logger.info(f"--- Starting DanzarVLM (Profile Arg: {profile_arg_str}, Effective LogLevel: {log_level_to_set}) ---")

    profile_to_load = args.profile if args.profile else global_cfg.get("DEFAULT_GAME_PROFILE", "generic_game")
    try:
        available_profiles = list_available_profiles()
        logger.info(f"Available game profiles: {available_profiles}")
        active_prof = load_game_profile(profile_to_load, global_cfg)
        if not active_prof:
            logger.warning(f"Profile '{profile_to_load}' not found. Trying 'generic_game'.")
            active_prof = load_game_profile("generic_game", global_cfg)
            if not active_prof:
                logger.critical("Failed to load 'generic_game' profile. Exiting.")
                sys.exit(1)
        logger.info(f"Successfully loaded game profile: {active_prof.game_name}")
    except Exception as e:
        logger.critical(f"Error during game profile loading: {e}", exc_info=True)
        sys.exit(1)

    app_context = AppContext(global_cfg, active_prof, logger)

    logger.info("Initializing services...")
    try:
        app_context.memory_service_instance = MemoryService(app_context)
        app_context.audio_service_instance = AudioService(app_context)
        app_context.audio_service_instance.initialize_audio_systems()
        
        # Initialize RAG service with app_context
        app_context.rag_service_instance = RAGService(app_context)
        
        # Initialize Model client
        app_context.model_client_instance = ModelClient(
            api_base_url=global_cfg.get("LLAMA_API_BASE_URL", ""),
            api_key=global_cfg.get("LLAMA_API_KEY")
        )
        
        # Initialize LLM service with new pattern
        app_context.llm_service_instance = LLMService(
            app_context=app_context,
            audio_service=app_context.audio_service_instance,
            rag_service=app_context.rag_service_instance,
            model_client=app_context.model_client_instance,
            default_collection="multimodal_rag_default"
        )
        
        app_context.ndi_service_instance = NDIService(app_context)
        app_context.discord_bot_runner_instance = DiscordBotRunner(app_context)
        logger.info("All services instantiated.")
    except Exception as e:
        logger.critical(f"FATAL: Error during service initialization: {e}", exc_info=True)
        sys.exit(1)
        
    all_threads: List[threading.Thread] = []
    service_thread_targets = {
        "NDIServiceThread": app_context.ndi_service_instance.run_capture_loop,
        "VLMCommentaryThread": app_context.llm_service_instance.run_vlm_commentary_loop,
        "DiscordBotThread": app_context.discord_bot_runner_instance.run,
        "DiscordPlaybackThread": app_context.discord_bot_runner_instance.run_playback_loop,
    }

    # Start the web interface
    try:
        web_thread = run_server(app_context, port=args.web_port)
        all_threads.append(web_thread)
        logger.info(f"Web interface started on port {args.web_port}")
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}", exc_info=True)
        logger.warning("Continuing without web interface...")

    for name, target_func in service_thread_targets.items():
        if target_func is None: 
            logger.error(f"Target function for thread {name} is None. Skipping.")
            continue
        thread = threading.Thread(target=target_func, name=name, daemon=True)
        all_threads.append(thread)
        thread.start()
        logger.info(f"Started thread: {name}")

    if not all_threads:
        logger.critical("No service threads were started. Exiting.")
        sys.exit(1)
    
    if app_context.global_settings.get("START_WITH_NDI_COMMENTARY_ENABLED", False):
        app_context.ndi_commentary_enabled.set()
        logger.info("NDI commentary started as enabled.")
    else:
        logger.info("NDI commentary started as disabled.")

    logger.info(f"DanzarVLM core setup complete. Profile: {app_context.active_profile.game_name}. Main loop running. Press Ctrl+C to exit.")

    loop_counter = 0
    try:
        while not app_context.shutdown_event.is_set():
            any_critical_thread_dead = False
            critical_thread_names = ["NDIServiceThread", "VLMCommentaryThread", "DiscordBotThread"] 
            for t_obj in all_threads:
                if not t_obj.is_alive() and t_obj.name in critical_thread_names:
                    logger.critical(f"CRITICAL THREAD {t_obj.name} has stopped!")
                    any_critical_thread_dead = True
                    break
            if any_critical_thread_dead:
                logger.error("Critical thread died. Initiating shutdown.")
                app_context.shutdown_event.set()
                break 
            if loop_counter % global_cfg.get("MAIN_LOOP_LOG_INTERVAL_S", 10) == 0:
                 log_msg_parts = [
                     f"Queues - Frame: {app_context.frame_queue.qsize()}/{app_context.frame_queue.maxsize}",
                     f"TextMsg: {app_context.text_message_queue.qsize()}/{app_context.text_message_queue.maxsize}",
                     f"TTSAudio: {app_context.tts_queue.qsize()}/{app_context.tts_queue.maxsize}"
                 ]
                 logger.debug(" | ".join(log_msg_parts))
                 logger.debug(f"NDI Commentary: {'ON' if app_context.ndi_commentary_enabled.is_set() else 'OFF'}, In Convo: {'YES' if app_context.is_in_conversation.is_set() else 'NO'}, TTS Playing: {'YES' if app_context.tts_is_playing.is_set() else 'NO'}")
            loop_counter = (loop_counter + 1) % 3600 
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Initiating graceful shutdown...")
    except Exception as e:
        logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
        logger.info("Initiating emergency shutdown...")
    finally:
        if not app_context.shutdown_event.is_set():
            logger.info("Setting shutdown event from main finally block.")
            app_context.shutdown_event.set()
        logger.info("Shutting down services and joining threads...")
        for t in reversed(all_threads):
            if t.is_alive():
                logger.debug(f"Attempting to join thread {t.name}...")
                t.join(timeout=global_cfg.get("THREAD_JOIN_TIMEOUT_S", 7.0))
                if t.is_alive():
                    logger.warning(f"Thread {t.name} did not shut down cleanly.")
                else:
                    logger.info(f"Thread {t.name} joined.")
        if app_context.audio_service_instance and hasattr(app_context.audio_service_instance, 'cleanup'):
            logger.info("Cleaning up AudioService resources...")
            app_context.audio_service_instance.cleanup()
        if app_context.ndi_service_instance and app_context.ndi_service_instance.is_initialized:
            logger.info("Performing final explicit NDI service cleanup...")
            app_context.ndi_service_instance.cleanup()
        logger.info("--- DanzarVLM Exited ---")

if __name__ == "__main__":
    main()