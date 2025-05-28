from enum import Enum

class AppState(Enum):
    STARTING = 1
    RUNNING = 2
    STOPPING = 3
    ERROR = 4

class CommentaryMode(Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

# Add other enums as needed, e.g., for LLM roles, NDI frame types, etc.\n