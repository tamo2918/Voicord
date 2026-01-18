"""
Speech-to-Text transcription using lightning-whisper-mlx.
Optimized for Apple Silicon (M1/M2/M3/M4).
Supports long audio files through automatic chunking.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    WHISPER_MODEL,
    WHISPER_BATCH_SIZE,
    WHISPER_LANGUAGE,
    CHUNK_DURATION_MINUTES,
    MAX_CONCURRENT_TRANSCRIPTIONS
)

logger = logging.getLogger(__name__)

# Global whisper model instance (lazy loaded)
_whisper_model = None

# Threshold for chunking (in seconds)
LONG_AUDIO_THRESHOLD_SECONDS = 5 * 60  # 5 minutes


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
    Automatically handles long audio files by chunking.

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.)
        language: Language code (e.g., 'ja' for Japanese, 'en' for English).
                  If None, uses WHISPER_LANGUAGE from config.

    Returns:
        dict with keys:
            - text: Full transcription text
            - segments: List of segments with timestamps (if available)
            - duration_seconds: Audio duration
            - was_chunked: Whether the audio was split into chunks
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    lang = language or WHISPER_LANGUAGE

    # Check audio duration
    from audio_processor import get_audio_info
    info = get_audio_info(audio_path)

    logger.info(
        f"Transcribing: {audio_path.name} "
        f"(duration: {info.duration_formatted}, language: {lang})"
    )

    # For long audio, use chunked transcription
    if info.duration_seconds > LONG_AUDIO_THRESHOLD_SECONDS:
        logger.info(f"Long audio detected ({info.duration_formatted}), using chunked transcription")
        return transcribe_long_audio(audio_path, language=lang)

    # For short audio, transcribe directly
    return _transcribe_single(audio_path, lang)


def _transcribe_single(audio_path: Path, language: str, time_offset: float = 0) -> dict:
    """
    Transcribe a single audio file (internal function).

    Args:
        audio_path: Path to the audio file
        language: Language code
        time_offset: Offset to add to segment timestamps (for chunked processing)

    Returns:
        Transcription result dict
    """
    try:
        whisper = get_whisper_model()
        result = whisper.transcribe(
            str(audio_path),
            language=language
        )

        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        # Adjust timestamps if there's an offset
        if time_offset > 0 and segments:
            for seg in segments:
                seg["start"] = seg.get("start", 0) + time_offset
                seg["end"] = seg.get("end", 0) + time_offset

        logger.info(f"Transcription complete: {len(text)} characters")

        return {
            "text": text,
            "segments": segments,
            "was_chunked": False
        }

    except Exception as e:
        logger.error(f"Transcription failed for {audio_path}: {e}")
        raise


def transcribe_long_audio(
    audio_path: Path,
    language: Optional[str] = None,
    chunk_duration_minutes: Optional[int] = None,
    progress_callback: Optional[callable] = None
) -> dict:
    """
    Transcribe a long audio file by splitting it into chunks.

    Args:
        audio_path: Path to the audio file
        language: Language code
        chunk_duration_minutes: Duration of each chunk in minutes
        progress_callback: Optional callback(current, total, text) for progress updates

    Returns:
        dict with combined transcription results
    """
    from audio_processor import split_audio_by_duration, get_audio_info, cleanup_temp_files

    audio_path = Path(audio_path)
    lang = language or WHISPER_LANGUAGE
    chunk_mins = chunk_duration_minutes or CHUNK_DURATION_MINUTES
    chunk_duration_ms = chunk_mins * 60 * 1000

    info = get_audio_info(audio_path)
    logger.info(f"Starting chunked transcription: {info.duration_formatted}")

    # Create temp directory for chunks
    with tempfile.TemporaryDirectory(prefix="whisper_chunks_") as temp_dir:
        temp_path = Path(temp_dir)

        # Split audio into chunks
        chunk_paths = split_audio_by_duration(
            audio_path,
            temp_path,
            chunk_duration_ms=chunk_duration_ms,
            use_silence_detection=True
        )

        total_chunks = len(chunk_paths)
        logger.info(f"Split into {total_chunks} chunks")

        # Transcribe each chunk
        all_text_parts = []
        all_segments = []
        current_time_offset = 0

        for i, chunk_path in enumerate(chunk_paths):
            logger.info(f"Transcribing chunk {i + 1}/{total_chunks}")

            if progress_callback:
                progress_callback(i + 1, total_chunks, f"Processing chunk {i + 1}/{total_chunks}")

            # Get chunk duration for offset calculation
            chunk_info = get_audio_info(chunk_path)

            # Transcribe chunk
            result = _transcribe_single(chunk_path, lang, time_offset=current_time_offset)

            if result["text"]:
                all_text_parts.append(result["text"])

            if result["segments"]:
                all_segments.extend(result["segments"])

            # Update time offset for next chunk
            current_time_offset += chunk_info.duration_seconds

        # Combine results
        combined_text = " ".join(all_text_parts)

        logger.info(
            f"Chunked transcription complete: {len(combined_text)} characters, "
            f"{len(all_segments)} segments"
        )

        return {
            "text": combined_text,
            "segments": all_segments,
            "duration_seconds": info.duration_seconds,
            "was_chunked": True,
            "chunk_count": total_chunks
        }


def transcribe_multiple(
    audio_paths: list[str | Path],
    language: Optional[str] = None,
    parallel: bool = False
) -> list[dict]:
    """
    Transcribe multiple audio files.

    Args:
        audio_paths: List of paths to audio files
        language: Language code for all files
        parallel: Whether to process files in parallel (careful with memory)

    Returns:
        List of transcription results
    """
    if not parallel or len(audio_paths) <= 1:
        # Sequential processing
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

    # Parallel processing (limited concurrency)
    results = [None] * len(audio_paths)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TRANSCRIPTIONS) as executor:
        future_to_index = {
            executor.submit(transcribe_audio, path, language): i
            for i, path in enumerate(audio_paths)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            path = audio_paths[index]

            try:
                result = future.result()
                result["file"] = str(path)
                results[index] = result
            except Exception as e:
                logger.error(f"Failed to transcribe {path}: {e}")
                results[index] = {
                    "file": str(path),
                    "text": "",
                    "segments": [],
                    "error": str(e)
                }

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

        # Format timestamps as HH:MM:SS for long recordings, MM:SS for short
        if start >= 3600 or end >= 3600:
            start_str = f"{int(start // 3600):02d}:{int((start % 3600) // 60):02d}:{int(start % 60):02d}"
            end_str = f"{int(end // 3600):02d}:{int((end % 3600) // 60):02d}:{int(end % 60):02d}"
        else:
            start_str = f"{int(start // 60):02d}:{int(start % 60):02d}"
            end_str = f"{int(end // 60):02d}:{int(end % 60):02d}"

        lines.append(f"[{start_str} - {end_str}] {text}")

    return "\n".join(lines)


def estimate_transcription_time(audio_duration_seconds: float) -> str:
    """
    Estimate how long transcription will take.

    Args:
        audio_duration_seconds: Duration of audio in seconds

    Returns:
        Human-readable estimate string
    """
    # Rough estimate: large-v3 on M4 Pro processes ~20x realtime
    estimated_seconds = audio_duration_seconds / 20

    if estimated_seconds < 60:
        return f"約{int(estimated_seconds)}秒"
    elif estimated_seconds < 3600:
        return f"約{int(estimated_seconds / 60)}分"
    else:
        hours = int(estimated_seconds / 3600)
        minutes = int((estimated_seconds % 3600) / 60)
        return f"約{hours}時間{minutes}分"


if __name__ == "__main__":
    # Test transcription
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    # Show audio info
    from audio_processor import get_audio_info
    info = get_audio_info(Path(audio_file))
    print(f"\nAudio: {info.duration_formatted}, {info.file_size_mb:.2f}MB")
    print(f"Estimated time: {estimate_transcription_time(info.duration_seconds)}")

    # Transcribe
    result = transcribe_audio(audio_file)

    print("\n=== Transcription ===")
    print(result["text"])

    if result.get("was_chunked"):
        print(f"\n(Processed in {result.get('chunk_count', '?')} chunks)")

    if result["segments"]:
        print("\n=== With Timestamps ===")
        print(format_transcription_with_timestamps(result["segments"]))
