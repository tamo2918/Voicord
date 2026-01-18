"""
Audio processing utilities for handling long audio files.
Provides functions for splitting, analyzing, and managing audio files.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

logger = logging.getLogger(__name__)

# Default chunk settings
DEFAULT_CHUNK_DURATION_MS = 5 * 60 * 1000  # 5 minutes in milliseconds
MAX_CHUNK_DURATION_MS = 10 * 60 * 1000  # 10 minutes max
MIN_CHUNK_DURATION_MS = 30 * 1000  # 30 seconds min

# Silence detection settings
SILENCE_THRESH_DB = -40  # dB threshold for silence
MIN_SILENCE_LEN_MS = 500  # Minimum silence length to consider as break point


@dataclass
class AudioInfo:
    """Information about an audio file."""
    path: Path
    duration_seconds: float
    duration_formatted: str
    sample_rate: int
    channels: int
    file_size_mb: float


def get_audio_info(audio_path: Path) -> AudioInfo:
    """
    Get information about an audio file.

    Args:
        audio_path: Path to the audio file

    Returns:
        AudioInfo object with file details
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = AudioSegment.from_file(str(audio_path))

    duration_seconds = len(audio) / 1000.0
    hours, remainder = divmod(int(duration_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        duration_formatted = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        duration_formatted = f"{minutes}m {seconds}s"
    else:
        duration_formatted = f"{seconds}s"

    file_size_mb = audio_path.stat().st_size / (1024 * 1024)

    return AudioInfo(
        path=audio_path,
        duration_seconds=duration_seconds,
        duration_formatted=duration_formatted,
        sample_rate=audio.frame_rate,
        channels=audio.channels,
        file_size_mb=file_size_mb
    )


def split_audio_by_duration(
    audio_path: Path,
    output_dir: Path,
    chunk_duration_ms: int = DEFAULT_CHUNK_DURATION_MS,
    use_silence_detection: bool = True
) -> list[Path]:
    """
    Split a long audio file into smaller chunks.

    Uses silence detection to find natural break points when possible,
    falling back to fixed-duration splits if needed.

    Args:
        audio_path: Path to the audio file to split
        output_dir: Directory to save chunks
        chunk_duration_ms: Target duration for each chunk in milliseconds
        use_silence_detection: Whether to use silence detection for natural breaks

    Returns:
        List of paths to the chunk files
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading audio file: {audio_path}")
    audio = AudioSegment.from_file(str(audio_path))
    total_duration = len(audio)

    logger.info(f"Audio duration: {total_duration / 1000:.1f}s")

    # If audio is short enough, no need to split
    if total_duration <= chunk_duration_ms:
        logger.info("Audio is short enough, no splitting needed")
        return [audio_path]

    chunk_paths = []
    chunk_index = 0

    if use_silence_detection:
        # Find silence points for natural breaks
        logger.info("Detecting silence points for natural breaks...")
        try:
            nonsilent_ranges = detect_nonsilent(
                audio,
                min_silence_len=MIN_SILENCE_LEN_MS,
                silence_thresh=SILENCE_THRESH_DB
            )

            # Convert to silence ranges (gaps between nonsilent)
            silence_points = []
            for i in range(len(nonsilent_ranges) - 1):
                end_of_speech = nonsilent_ranges[i][1]
                start_of_next = nonsilent_ranges[i + 1][0]
                # Use middle of silence as break point
                silence_points.append((end_of_speech + start_of_next) // 2)

            logger.info(f"Found {len(silence_points)} potential break points")

        except Exception as e:
            logger.warning(f"Silence detection failed: {e}, using fixed splits")
            silence_points = []
    else:
        silence_points = []

    # Split audio into chunks
    current_pos = 0

    while current_pos < total_duration:
        # Target end position
        target_end = min(current_pos + chunk_duration_ms, total_duration)

        if target_end >= total_duration:
            # Last chunk
            chunk_end = total_duration
        elif silence_points:
            # Find the best silence point near our target
            best_point = None
            min_distance = float('inf')

            for point in silence_points:
                if current_pos < point < total_duration:
                    distance = abs(point - target_end)
                    # Accept points within 30% of chunk duration
                    if distance < chunk_duration_ms * 0.3 and distance < min_distance:
                        min_distance = distance
                        best_point = point

            if best_point:
                chunk_end = best_point
                silence_points.remove(best_point)  # Don't reuse
            else:
                chunk_end = target_end
        else:
            chunk_end = target_end

        # Ensure minimum chunk size
        if chunk_end - current_pos < MIN_CHUNK_DURATION_MS and chunk_end < total_duration:
            chunk_end = min(current_pos + MIN_CHUNK_DURATION_MS, total_duration)

        # Extract and save chunk
        chunk = audio[current_pos:chunk_end]
        chunk_filename = f"chunk_{chunk_index:03d}.wav"
        chunk_path = output_dir / chunk_filename

        chunk.export(str(chunk_path), format="wav")
        chunk_paths.append(chunk_path)

        logger.info(
            f"Created chunk {chunk_index}: {current_pos / 1000:.1f}s - {chunk_end / 1000:.1f}s "
            f"({(chunk_end - current_pos) / 1000:.1f}s)"
        )

        current_pos = chunk_end
        chunk_index += 1

    logger.info(f"Split audio into {len(chunk_paths)} chunks")
    return chunk_paths


def split_audio_by_size(
    audio_path: Path,
    output_dir: Path,
    max_size_mb: float = 25.0
) -> list[Path]:
    """
    Split audio file to ensure each chunk is under a size limit.

    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save chunks
        max_size_mb: Maximum size per chunk in MB

    Returns:
        List of paths to chunk files
    """
    info = get_audio_info(audio_path)

    if info.file_size_mb <= max_size_mb:
        return [audio_path]

    # Calculate how many chunks we need
    num_chunks = int(info.file_size_mb / max_size_mb) + 1
    chunk_duration_ms = int((info.duration_seconds * 1000) / num_chunks)

    return split_audio_by_duration(
        audio_path,
        output_dir,
        chunk_duration_ms=chunk_duration_ms,
        use_silence_detection=True
    )


def normalize_audio(audio_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Normalize audio levels for better transcription.

    Args:
        audio_path: Path to input audio
        output_path: Path for output (defaults to overwriting input)

    Returns:
        Path to normalized audio
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path) if output_path else audio_path

    audio = AudioSegment.from_file(str(audio_path))

    # Normalize to -20 dBFS
    target_dbfs = -20.0
    change_in_dbfs = target_dbfs - audio.dBFS
    normalized = audio.apply_gain(change_in_dbfs)

    normalized.export(str(output_path), format="wav")
    logger.info(f"Normalized audio: {audio_path} -> {output_path}")

    return output_path


def convert_to_wav(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Convert any audio format to WAV (16-bit, 16kHz mono - optimal for Whisper).

    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file

    Returns:
        Path to converted WAV file
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix('.wav')

    output_path = Path(output_path)

    audio = AudioSegment.from_file(str(input_path))

    # Convert to Whisper-optimal format: 16kHz mono
    audio = audio.set_frame_rate(16000).set_channels(1)

    audio.export(str(output_path), format="wav")
    logger.info(f"Converted to WAV: {input_path} -> {output_path}")

    return output_path


def merge_audio_files(audio_paths: list[Path], output_path: Path) -> Path:
    """
    Merge multiple audio files into one.

    Args:
        audio_paths: List of audio file paths to merge
        output_path: Path for the merged output file

    Returns:
        Path to merged audio file
    """
    if not audio_paths:
        raise ValueError("No audio files to merge")

    if len(audio_paths) == 1:
        return audio_paths[0]

    output_path = Path(output_path)

    combined = AudioSegment.empty()
    for path in audio_paths:
        audio = AudioSegment.from_file(str(path))
        combined += audio

    combined.export(str(output_path), format="wav")
    logger.info(f"Merged {len(audio_paths)} files into {output_path}")

    return output_path


def cleanup_temp_files(file_paths: list[Path]):
    """
    Clean up temporary files.

    Args:
        file_paths: List of file paths to delete
    """
    for path in file_paths:
        try:
            if path.exists():
                path.unlink()
                logger.debug(f"Deleted temp file: {path}")
        except Exception as e:
            logger.warning(f"Failed to delete {path}: {e}")


if __name__ == "__main__":
    # Test the module
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python audio_processor.py <audio_file>")
        sys.exit(1)

    audio_file = Path(sys.argv[1])
    info = get_audio_info(audio_file)

    print(f"\nAudio Info:")
    print(f"  Path: {info.path}")
    print(f"  Duration: {info.duration_formatted}")
    print(f"  Sample Rate: {info.sample_rate} Hz")
    print(f"  Channels: {info.channels}")
    print(f"  Size: {info.file_size_mb:.2f} MB")
