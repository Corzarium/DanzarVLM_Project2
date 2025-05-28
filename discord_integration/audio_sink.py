# discord_integration/audio_sink.py
import discord
# from discord.sinks import Sink as AudioSink # Try to determine which one is correct
# Try importing AudioSink directly first, if not then from .sinks
try:
    from discord import AudioSink # Common in newer discord.py v2 / Pycord
    from discord.sinks.core import AudioData # For type hinting 'data'
except ImportError:
    try:
        from discord.sinks import Sink as AudioSink # Alternative v2 path
        from discord.sinks.core import AudioData
    except ImportError:
        # Fallback for older discord.py v1.7.x style (less likely if previous errors were v2 related)
        # This would require different write signature handling
        AudioSink = discord.reader.AudioSink # This was the old path causing errors
        AudioData = None # No direct equivalent for data type hint
        print("WARNING: Using older discord.reader.AudioSink. Voice receive might be unstable.")


from typing import TYPE_CHECKING, Optional, Dict, List 

if TYPE_CHECKING:
    from ..DanzarVLM import AppContext 
    from ..services.audio_service import AudioService

class DanzarAudioSink(AudioSink):
    def __init__(self, app_context: 'AppContext', logger):
        super().__init__() 
        self.main_app_context = app_context
        self.audio_service: Optional['AudioService'] = getattr(app_context, 'audio_service_instance', None)
        self.logger = logger
        
        if not self.audio_service:
            self.logger.error("[DanzarAudioSink] CRITICAL: AudioService instance not found!")
        else:
            self.logger.info("[DanzarAudioSink] Instance created, linked.")

    # This method signature handles the case where 'user' might be an int (user_id)
    # OR a discord.User object.
    def write(self, data, user_or_id): # data is discord.sinks.core.AudioData or older 'data'
        actual_user_id: Optional[int] = None
        user_display_name: str = "UnknownUser"

        if isinstance(user_or_id, discord.User):
            if user_or_id.id is None: # Should not happen for a valid User object
                # self.logger.debug("[DanzarAudioSink] Received User object with no ID. Discarding.")
                return
            actual_user_id = user_or_id.id
            user_display_name = user_or_id.display_name
        elif isinstance(user_or_id, int):
            actual_user_id = user_or_id
            user_display_name = f"User ID {actual_user_id}"
            # If it's an int, then 'data' is likely just the PCM bytes for older discord.py versions
        else:
            # self.logger.debug(f"[DanzarAudioSink] Received audio with unknown user/ID type: {type(user_or_id)}. Discarding.")
            return

        if actual_user_id is None: # Should ideally be caught by the 'if not user:' in v2 style
            return

        if not self.audio_service:
            return
        
        pcm_bytes_to_process: Optional[bytes] = None
        if hasattr(data, 'pcm_array') and isinstance(data.pcm_array, memoryview): # For discord.sinks.core.AudioData (v2)
            pcm_bytes_to_process = bytes(data.pcm_array)
        elif isinstance(data, bytes): # Potentially for older discord.reader.AudioData where 'data' was pcm
            pcm_bytes_to_process = data
        else:
            self.logger.warning(f"[DanzarAudioSink] Unrecognized audio data format: {type(data)}. Cannot extract PCM.")
            return

        if not pcm_bytes_to_process:
            self.logger.debug(f"[DanzarAudioSink] No PCM data extracted for user {user_display_name}.")
            return

        # self.logger.debug(f"[DanzarAudioSink] WRITE from {user_display_name} (ID: {actual_user_id}): Data len: {len(pcm_bytes_to_process)}")

        try:
            self.audio_service.process_discord_audio_chunk(pcm_bytes_to_process, actual_user_id)
        except Exception as e:
            self.logger.error(f"[DanzarAudioSink] Error processing audio for user {actual_user_id} ({user_display_name}): {e}", exc_info=True)

    def cleanup(self):
        self.logger.info("[DanzarAudioSink] Cleanup called.")