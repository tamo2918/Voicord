"""
Custom Discord Sink that writes audio directly to disk instead of memory.
Supports long-duration recordings without running out of memory.
"""

import io
import wave
import struct
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import discord
from discord.sinks import Sink, AudioData, Filters, default_filters

logger = logging.getLogger(__name__)

# Discord audio settings
SAMPLE_RATE = 48000  # 48kHz
CHANNELS = 2  # Stereo
SAMPLE_WIDTH = 2  # 16-bit


class DiskAudioData:
    """
    AudioData that writes directly to a WAV file on disk instead of memory.
    This allows for very long recordings without memory issues.
    """

    def __init__(self, file_path: Path, user_id: int):
        self.file_path = file_path
        self.user_id = user_id
        self.finished = False
        self._bytes_written = 0
        self._start_time = datetime.now()

        # Open WAV file for writing
        self._wav_file: Optional[wave.Wave_write] = None
        self._open_wav_file()

        logger.info(f"Started disk recording for user {user_id}: {file_path}")

    def _open_wav_file(self):
        """Open the WAV file with proper settings."""
        self._wav_file = wave.open(str(self.file_path), 'wb')
        self._wav_file.setnchannels(CHANNELS)
        self._wav_file.setsampwidth(SAMPLE_WIDTH)
        self._wav_file.setframerate(SAMPLE_RATE)

    def write(self, data: bytes):
        """Write audio data directly to disk."""
        if self.finished:
            raise RuntimeError("Cannot write to finished AudioData")

        if self._wav_file is None:
            raise RuntimeError("WAV file is not open")

        try:
            self._wav_file.writeframes(data)
            self._bytes_written += len(data)
        except Exception as e:
            logger.error(f"Error writing audio data: {e}")
            raise

    def cleanup(self):
        """Finalize the WAV file."""
        if self.finished:
            return

        self.finished = True

        if self._wav_file:
            try:
                self._wav_file.close()
            except Exception as e:
                logger.error(f"Error closing WAV file: {e}")

        duration = (datetime.now() - self._start_time).total_seconds()
        size_mb = self._bytes_written / (1024 * 1024)
        logger.info(
            f"Finished recording for user {self.user_id}: "
            f"{duration:.1f}s, {size_mb:.2f}MB, file: {self.file_path}"
        )

    @property
    def duration_seconds(self) -> float:
        """Estimate duration based on bytes written."""
        # bytes = samples * channels * sample_width
        # samples = bytes / (channels * sample_width)
        # duration = samples / sample_rate
        samples = self._bytes_written / (CHANNELS * SAMPLE_WIDTH)
        return samples / SAMPLE_RATE

    @property
    def file(self) -> io.BytesIO:
        """
        Compatibility property for Pycord's expected interface.
        Returns the file contents as BytesIO after recording is finished.
        """
        if not self.finished:
            raise RuntimeError("Cannot read file until recording is finished")

        with open(self.file_path, 'rb') as f:
            return io.BytesIO(f.read())


class ChunkedFileSink(Sink):
    """
    A custom sink that writes audio directly to disk files.

    Benefits over default WaveSink:
    - No memory accumulation for long recordings
    - Supports hours of recording without issues
    - Files are immediately available on disk

    Audio is stored per-user in separate WAV files.
    """

    def __init__(
        self,
        output_dir: Path,
        session_id: str,
        *,
        filters=None
    ):
        """
        Initialize the chunked file sink.

        Args:
            output_dir: Directory to save audio files
            session_id: Unique identifier for this recording session
            filters: Audio filters (default: Pycord default filters)
        """
        if filters is None:
            filters = default_filters

        self.filters = filters
        Filters.__init__(self, **self.filters)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id
        self.vc: Optional[discord.VoiceClient] = None
        self.audio_data: dict[int, DiskAudioData] = {}

        logger.info(f"ChunkedFileSink initialized: {output_dir}, session: {session_id}")

    def _get_user_file_path(self, user_id: int) -> Path:
        """Get the file path for a user's audio."""
        return self.output_dir / f"{self.session_id}_user_{user_id}.wav"

    @Filters.container
    def write(self, data: bytes, user_id: int):
        """
        Write audio data to the user's file.
        Called by Pycord for each audio packet.
        """
        if user_id not in self.audio_data:
            # Create new disk audio data for this user
            file_path = self._get_user_file_path(user_id)
            self.audio_data[user_id] = DiskAudioData(file_path, user_id)

        self.audio_data[user_id].write(data)

    def cleanup(self):
        """Clean up all audio data and finalize files."""
        for user_id, audio_data in self.audio_data.items():
            try:
                audio_data.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up audio for user {user_id}: {e}")

    def format_audio(self, audio: DiskAudioData) -> None:
        """
        Format audio - not needed since we write WAV directly.
        This is called by Pycord's cleanup process.
        """
        pass  # WAV files are already properly formatted

    def get_all_audio(self) -> list[tuple[int, Path]]:
        """Get list of (user_id, file_path) for all recorded users."""
        return [
            (user_id, audio_data.file_path)
            for user_id, audio_data in self.audio_data.items()
            if audio_data.file_path.exists()
        ]

    def get_user_audio(self, user_id: int) -> Optional[Path]:
        """Get the file path for a specific user's audio."""
        if user_id in self.audio_data:
            return self.audio_data[user_id].file_path
        return None

    def get_recording_stats(self) -> dict:
        """Get statistics about the current recording."""
        stats = {
            "user_count": len(self.audio_data),
            "users": {}
        }

        for user_id, audio_data in self.audio_data.items():
            stats["users"][user_id] = {
                "duration_seconds": audio_data.duration_seconds,
                "bytes_written": audio_data._bytes_written,
                "file_path": str(audio_data.file_path)
            }

        return stats
