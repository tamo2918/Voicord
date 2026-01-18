"""
Configuration settings for the Discord Voice Transcription Bot.
Loads settings from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
RECORDINGS_DIR = BASE_DIR / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

# Discord Bot Configuration
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is required. Please set it in .env file.")

# Bot command prefix
COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "!")

# Whisper Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "12"))
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ja")  # Japanese by default

# Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Recording settings
MAX_RECORDING_DURATION_SECONDS = int(os.getenv("MAX_RECORDING_DURATION_SECONDS", "7200"))  # 2 hours default
AUTO_DELETE_RECORDINGS = os.getenv("AUTO_DELETE_RECORDINGS", "true").lower() == "true"

# Summary settings
SUMMARY_LANGUAGE = os.getenv("SUMMARY_LANGUAGE", "ja")  # Japanese by default

# Long audio processing settings
CHUNK_DURATION_MINUTES = int(os.getenv("CHUNK_DURATION_MINUTES", "5"))  # Split audio into 5-minute chunks
MAX_CONCURRENT_TRANSCRIPTIONS = int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "2"))  # Parallel transcription limit

# LLM context settings
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))  # Max chars before hierarchical summarization

# Progress update settings
PROGRESS_UPDATE_INTERVAL_SECONDS = int(os.getenv("PROGRESS_UPDATE_INTERVAL_SECONDS", "30"))
