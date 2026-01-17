"""
Speech-to-Text transcription using lightning-whisper-mlx.
Optimized for Apple Silicon (M1/M2/M3/M4).
"""

import logging
from pathlib import Path
from typing import Optional

from config import WHISPER_MODEL, WHISPER_BATCH_SIZE, WHISPER_LANGUAGE

logger = logging.getLogger(__name__)

# Global whisper model instance (lazy loaded)
_whisper_model = None


def get_whisper_model():
    """
    Get or initialize the Whisper model.
    Uses lazy loading to avoid loading the model until first use.
    """
    global _whisper_model

    if _whisper_model is None:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
        try:
            from lightning_whisper_mlx import LightningWhisperMLX
            _whisper_model = LightningWhisperMLX(
                model=WHISPER_MODEL,
                batch_size=WHISPER_BATCH_SIZE,
                quant=None  # No quantization for best quality
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    return _whisper_model


def transcribe_audio(audio_path: str | Path, language: Optional[str] = None) -> dict:
    """
    Transcribe an audio file to text.

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.)
        language: Language code (e.g., 'ja' for Japanese, 'en' for English).
                  If None, uses WHISPER_LANGUAGE from config.

    Returns:
        dict with keys:
            - text: Full transcription text
            - segments: List of segments with timestamps (if available)
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    lang = language or WHISPER_LANGUAGE
    logger.info(f"Transcribing: {audio_path.name} (language: {lang})")

    try:
        whisper = get_whisper_model()
        result = whisper.transcribe(
            str(audio_path),
            language=lang
        )

        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        logger.info(f"Transcription complete: {len(text)} characters")

        return {
            "text": text,
            "segments": segments
        }

    except Exception as e:
        logger.error(f"Transcription failed for {audio_path}: {e}")
        raise


def transcribe_multiple(audio_paths: list[str | Path], language: Optional[str] = None) -> list[dict]:
    """
    Transcribe multiple audio files.

    Args:
        audio_paths: List of paths to audio files
        language: Language code for all files

    Returns:
        List of transcription results
    """
    results = []
    for path in audio_paths:
        try:
            result = transcribe_audio(path, language)
            result["file"] = str(path)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to transcribe {path}: {e}")
            results.append({
                "file": str(path),
                "text": "",
                "segments": [],
                "error": str(e)
            })

    return results


def format_transcription_with_timestamps(segments: list) -> str:
    """
    Format transcription segments with timestamps.

    Args:
        segments: List of segment dictionaries with 'start', 'end', 'text' keys

    Returns:
        Formatted string with timestamps
    """
    lines = []
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip()

        # Format timestamps as MM:SS
        start_str = f"{int(start // 60):02d}:{int(start % 60):02d}"
        end_str = f"{int(end // 60):02d}:{int(end % 60):02d}"

        lines.append(f"[{start_str} - {end_str}] {text}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test transcription
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    result = transcribe_audio(audio_file)

    print("\n=== Transcription ===")
    print(result["text"])

    if result["segments"]:
        print("\n=== With Timestamps ===")
        print(format_transcription_with_timestamps(result["segments"]))
