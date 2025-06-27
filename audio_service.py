        whisper_model = self.ctx.global_settings.get("WHISPER_MODEL_SIZE", "medium")
        whisper_model = "medium"  # Force Whisper model to always use medium 