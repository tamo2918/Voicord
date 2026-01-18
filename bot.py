"""
Discord Voice Transcription Bot
Records voice chat, transcribes using Whisper, and summarizes using Ollama.
Supports long recordings (2+ hours) through disk-based recording and chunked processing.
"""

import asyncio
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import discord
from discord.ext import commands

from config import (
    DISCORD_TOKEN,
    COMMAND_PREFIX,
    RECORDINGS_DIR,
    AUTO_DELETE_RECORDINGS,
    PROGRESS_UPDATE_INTERVAL_SECONDS,
)
from chunked_sink import ChunkedFileSink
from transcriber import transcribe_audio, estimate_transcription_time
from summarizer import summarize_conversation, check_ollama_available, estimate_summary_time
from audio_processor import get_audio_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Bot setup with required intents
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)


class RecordingSession:
    """Manages a single recording session with disk-based storage."""

    def __init__(self, guild_id: int, channel: discord.TextChannel, voice_channel: discord.VoiceChannel):
        self.guild_id = guild_id
        self.channel = channel
        self.voice_channel = voice_channel
        self.start_time = datetime.now()
        self.session_id = f"{guild_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = RECORDINGS_DIR / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.sink: Optional[ChunkedFileSink] = None
        self.vc: Optional[discord.VoiceClient] = None
        self.is_processing = False
        self._progress_task: Optional[asyncio.Task] = None

    def get_duration(self) -> tuple[int, int, int]:
        """Get recording duration as (hours, minutes, seconds)."""
        duration = datetime.now() - self.start_time
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return hours, minutes, seconds

    def get_duration_str(self) -> str:
        """Get formatted duration string."""
        hours, minutes, seconds = self.get_duration()
        if hours > 0:
            return f"{hours}時間{minutes}分{seconds}秒"
        elif minutes > 0:
            return f"{minutes}分{seconds}秒"
        else:
            return f"{seconds}秒"

    def cleanup(self):
        """Clean up session files."""
        if AUTO_DELETE_RECORDINGS and self.session_dir.exists():
            try:
                shutil.rmtree(self.session_dir)
                logger.info(f"Cleaned up session directory: {self.session_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup session directory: {e}")


# Store active sessions
active_sessions: dict[int, RecordingSession] = {}


async def send_long_message(channel: discord.TextChannel, content: str, max_length: int = 1900):
    """Send a message, splitting it if it's too long for Discord."""
    if len(content) <= max_length:
        await channel.send(content)
        return

    # Split by lines to avoid breaking mid-sentence
    lines = content.split("\n")
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_length:
            if current_chunk:
                await channel.send(current_chunk)
            current_chunk = line
        else:
            current_chunk = current_chunk + "\n" + line if current_chunk else line

    if current_chunk:
        await channel.send(current_chunk)


async def process_recording(session: RecordingSession):
    """
    Process a completed recording session.
    Transcribes all audio and generates a summary.
    """
    session.is_processing = True
    channel = session.channel

    try:
        await channel.send("🔄 録音を処理中です。しばらくお待ちください...")

        # Get all recorded audio files
        audio_files = session.sink.get_all_audio()

        if not audio_files:
            await channel.send("⚠️ 録音データが見つかりませんでした。")
            return

        # Show recording stats
        stats = session.sink.get_recording_stats()
        total_duration = sum(
            user_stats["duration_seconds"]
            for user_stats in stats["users"].values()
        )

        await channel.send(
            f"📊 **録音統計**\n"
            f"参加者: {stats['user_count']}人\n"
            f"録音時間: {session.get_duration_str()}\n"
            f"処理予定: {len(audio_files)}ファイル"
        )

        # Transcribe each user's audio
        transcriptions: dict[str, str] = {}

        for user_id, audio_path in audio_files:
            # Get user info
            user = bot.get_user(user_id)
            username = user.display_name if user else f"User_{user_id}"

            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                continue

            # Get audio info
            try:
                info = get_audio_info(audio_path)
                est_time = estimate_transcription_time(info.duration_seconds)

                await channel.send(
                    f"📝 **{username}** の音声を文字起こし中...\n"
                    f"   ({info.duration_formatted}, {info.file_size_mb:.1f}MB, 推定{est_time})"
                )
            except Exception as e:
                logger.warning(f"Could not get audio info: {e}")
                await channel.send(f"📝 **{username}** の音声を文字起こし中...")

            # Transcribe
            try:
                result = await asyncio.to_thread(transcribe_audio, audio_path)
                text = result.get("text", "")

                if text:
                    transcriptions[username] = text

                    # Show preview (truncated)
                    preview = text[:300]
                    if len(text) > 300:
                        preview += "..."

                    was_chunked = result.get("was_chunked", False)
                    chunk_info = f" ({result.get('chunk_count', '?')}チャンク処理)" if was_chunked else ""

                    await channel.send(f"✅ **{username}**{chunk_info}:\n>>> {preview}")
                else:
                    await channel.send(f"⚠️ **{username}**: 音声が検出されませんでした")

            except Exception as e:
                logger.error(f"Transcription failed for {username}: {e}")
                await channel.send(f"❌ **{username}**: 文字起こしエラー - {e}")

        # Generate summary if we have transcriptions
        if transcriptions:
            total_chars = sum(len(t) for t in transcriptions.values())
            est_summary_time = estimate_summary_time(total_chars)

            await channel.send(f"📊 要約を生成中... (推定{est_summary_time})")

            try:
                # Progress callback for long summaries
                async def update_progress(stage: str, message: str):
                    try:
                        await channel.send(f"   ↳ {message}")
                    except Exception:
                        pass

                summary = await asyncio.to_thread(
                    summarize_conversation,
                    transcriptions
                )

                # Send summary
                await send_long_message(
                    channel,
                    f"## 📋 会議サマリー\n\n{summary}"
                )

            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                await channel.send(f"⚠️ 要約の生成に失敗しました: {e}")

                # Still provide the raw transcription
                await channel.send("📝 **生の文字起こし結果:**")
                for speaker, text in transcriptions.items():
                    await send_long_message(channel, f"**{speaker}:**\n{text[:2000]}")
        else:
            await channel.send("⚠️ 文字起こしが空のため、要約を生成できませんでした。")

        await channel.send("✅ 処理が完了しました！")

    except Exception as e:
        logger.error(f"Error processing recording: {e}", exc_info=True)
        await channel.send(f"❌ 処理中にエラーが発生しました: {e}")

    finally:
        session.is_processing = False
        session.cleanup()


async def recording_finished_callback(sink: ChunkedFileSink, channel: discord.TextChannel, *args):
    """
    Callback function called when recording stops.
    """
    guild_id = sink.vc.guild.id
    session = active_sessions.get(guild_id)

    if not session:
        logger.error(f"No session found for guild {guild_id}")
        return

    # Finalize the sink
    sink.cleanup()

    # Process the recording
    await process_recording(session)

    # Cleanup session
    if guild_id in active_sessions:
        del active_sessions[guild_id]

    # Disconnect from voice
    if sink.vc and sink.vc.is_connected():
        await sink.vc.disconnect()


# === Bot Events ===

@bot.event
async def on_ready():
    """Called when the bot is ready."""
    logger.info(f"Logged in as {bot.user.name} ({bot.user.id})")
    logger.info(f"Connected to {len(bot.guilds)} guild(s)")

    # Check Ollama availability
    available, message = check_ollama_available()
    if available:
        logger.info(f"✅ {message}")
    else:
        logger.warning(f"⚠️ {message}")

    # Set bot status
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name=f"{COMMAND_PREFIX}help"
        )
    )


@bot.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
    """Handle voice state updates."""
    if member.bot:
        return

    guild_id = member.guild.id

    # Check if we're recording and the voice channel is now empty
    if guild_id in active_sessions:
        session = active_sessions[guild_id]
        if session.vc and session.vc.channel:
            members = [m for m in session.vc.channel.members if not m.bot]
            if len(members) == 0 and not session.is_processing:
                logger.info(f"Voice channel empty, stopping recording for guild {guild_id}")
                if session.vc.recording:
                    session.vc.stop_recording()


# === Bot Commands ===

@bot.command(name="record", aliases=["rec", "録音"])
async def start_recording(ctx: commands.Context):
    """
    Start recording the voice channel.
    Supports long recordings (2+ hours) through disk-based storage.
    """
    # Check if user is in a voice channel
    if not ctx.author.voice:
        await ctx.send("❌ ボイスチャンネルに参加してから実行してください。")
        return

    voice_channel = ctx.author.voice.channel
    guild_id = ctx.guild.id

    # Check if already recording
    if guild_id in active_sessions:
        session = active_sessions[guild_id]
        await ctx.send(
            f"⚠️ すでに録音中です ({session.get_duration_str()})。\n"
            f"`{COMMAND_PREFIX}stop` で停止してください。"
        )
        return

    try:
        # Connect to voice channel
        vc = await voice_channel.connect()

        # Create recording session
        session = RecordingSession(guild_id, ctx.channel, voice_channel)
        session.vc = vc
        active_sessions[guild_id] = session

        # Create disk-based sink
        sink = ChunkedFileSink(
            output_dir=session.session_dir,
            session_id=session.session_id
        )
        session.sink = sink

        # Start recording
        vc.start_recording(
            sink,
            recording_finished_callback,
            ctx.channel
        )

        await ctx.send(
            f"🎙️ **録音を開始しました**\n"
            f"チャンネル: {voice_channel.name}\n"
            f"保存先: ディスク (長時間録音対応)\n"
            f"停止: `{COMMAND_PREFIX}stop`"
        )

        logger.info(f"Started recording in {voice_channel.name} (Guild: {ctx.guild.name})")

    except Exception as e:
        logger.error(f"Failed to start recording: {e}", exc_info=True)
        await ctx.send(f"❌ 録音の開始に失敗しました: {e}")

        # Cleanup on error
        if guild_id in active_sessions:
            active_sessions[guild_id].cleanup()
            del active_sessions[guild_id]


@bot.command(name="stop", aliases=["停止"])
async def stop_recording(ctx: commands.Context):
    """
    Stop recording and process the audio.
    """
    guild_id = ctx.guild.id

    if guild_id not in active_sessions:
        await ctx.send("❌ 現在録音中ではありません。")
        return

    session = active_sessions[guild_id]

    if session.is_processing:
        await ctx.send("⏳ すでに処理中です。しばらくお待ちください。")
        return

    if not session.vc or not session.vc.recording:
        await ctx.send("❌ 録音が開始されていません。")
        return

    duration_str = session.get_duration_str()
    await ctx.send(f"⏹️ 録音を停止しています (録音時間: {duration_str})...")

    # Stop recording (triggers callback)
    session.vc.stop_recording()

    logger.info(f"Stopped recording in guild {ctx.guild.name}")


@bot.command(name="status", aliases=["状態"])
async def recording_status(ctx: commands.Context):
    """
    Check current recording status with detailed stats.
    """
    guild_id = ctx.guild.id

    if guild_id not in active_sessions:
        await ctx.send("📊 現在録音中ではありません。")
        return

    session = active_sessions[guild_id]

    if session.is_processing:
        await ctx.send("⏳ 録音を処理中です...")
        return

    # Get recording stats
    stats = session.sink.get_recording_stats()

    # Build status message
    embed = discord.Embed(
        title="🎙️ 録音ステータス",
        color=discord.Color.green()
    )

    embed.add_field(
        name="チャンネル",
        value=session.voice_channel.name,
        inline=True
    )

    embed.add_field(
        name="録音時間",
        value=session.get_duration_str(),
        inline=True
    )

    embed.add_field(
        name="参加者",
        value=f"{stats['user_count']}人",
        inline=True
    )

    # Per-user stats
    if stats["users"]:
        user_info = []
        for user_id, user_stats in stats["users"].items():
            user = bot.get_user(user_id)
            username = user.display_name if user else f"User_{user_id}"
            duration = user_stats["duration_seconds"]
            size_mb = user_stats["bytes_written"] / (1024 * 1024)
            user_info.append(f"• {username}: {duration:.0f}秒, {size_mb:.1f}MB")

        embed.add_field(
            name="ユーザー別",
            value="\n".join(user_info) or "なし",
            inline=False
        )

    await ctx.send(embed=embed)


@bot.command(name="cancel", aliases=["キャンセル"])
async def cancel_recording(ctx: commands.Context):
    """
    Cancel recording without processing.
    """
    guild_id = ctx.guild.id

    if guild_id not in active_sessions:
        await ctx.send("❌ 現在録音中ではありません。")
        return

    session = active_sessions[guild_id]

    if session.is_processing:
        await ctx.send("⚠️ 処理中のためキャンセルできません。")
        return

    # Stop recording without triggering callback processing
    if session.vc:
        if session.vc.recording:
            # We need to stop recording but skip processing
            session.sink.cleanup()

        if session.vc.is_connected():
            await session.vc.disconnect()

    # Cleanup
    session.cleanup()
    del active_sessions[guild_id]

    await ctx.send("🚫 録音をキャンセルしました。")


@bot.command(name="check", aliases=["チェック"])
async def check_system(ctx: commands.Context):
    """
    Check system status.
    """
    embed = discord.Embed(
        title="🔧 システムステータス",
        color=discord.Color.blue()
    )

    # Check Ollama
    available, message = check_ollama_available()
    ollama_status = "✅ 正常" if available else "❌ エラー"
    embed.add_field(
        name="Ollama (LLM)",
        value=f"{ollama_status}\n{message}",
        inline=False
    )

    # Whisper status
    try:
        from config import WHISPER_MODEL
        embed.add_field(
            name="Whisper (文字起こし)",
            value=f"✅ モデル: {WHISPER_MODEL}",
            inline=False
        )
    except Exception as e:
        embed.add_field(
            name="Whisper (文字起こし)",
            value=f"❌ エラー: {e}",
            inline=False
        )

    # Recording system
    embed.add_field(
        name="録音システム",
        value="✅ ディスクベース録音 (長時間対応)",
        inline=False
    )

    # Bot info
    embed.add_field(
        name="Bot情報",
        value=f"サーバー数: {len(bot.guilds)}\n"
              f"アクティブ録音: {len(active_sessions)}",
        inline=False
    )

    await ctx.send(embed=embed)


@bot.command(name="commands", aliases=["コマンド", "cmds"])
async def show_commands(ctx: commands.Context):
    """
    Show available commands.
    """
    embed = discord.Embed(
        title="📖 コマンド一覧",
        description="Discord音声文字起こし・要約Bot\n**長時間録音対応版**",
        color=discord.Color.green()
    )

    commands_list = [
        (f"`{COMMAND_PREFIX}record`", "録音を開始 (2時間以上対応)"),
        (f"`{COMMAND_PREFIX}stop`", "録音を停止し、文字起こし・要約を生成"),
        (f"`{COMMAND_PREFIX}cancel`", "録音をキャンセル（処理なし）"),
        (f"`{COMMAND_PREFIX}status`", "現在の録音状態を詳細表示"),
        (f"`{COMMAND_PREFIX}check`", "システムの状態を確認"),
        (f"`{COMMAND_PREFIX}commands`", "このヘルプを表示"),
    ]

    for cmd, desc in commands_list:
        embed.add_field(name=cmd, value=desc, inline=False)

    embed.set_footer(text="ボイスチャンネルに参加してから !record を実行してください")

    await ctx.send(embed=embed)


# === Slash Commands ===

@bot.slash_command(name="record", description="ボイスチャンネルの録音を開始 (長時間対応)")
async def slash_record(ctx: discord.ApplicationContext):
    """Slash command version of record."""
    await start_recording(ctx)


@bot.slash_command(name="stop", description="録音を停止し、文字起こし・要約を生成")
async def slash_stop(ctx: discord.ApplicationContext):
    """Slash command version of stop."""
    await stop_recording(ctx)


@bot.slash_command(name="status", description="現在の録音状態を確認")
async def slash_status(ctx: discord.ApplicationContext):
    """Slash command version of status."""
    await recording_status(ctx)


# === Main Entry Point ===

def main():
    """Main entry point for the bot."""
    logger.info("Starting Discord Voice Transcription Bot (Long Recording Edition)...")

    # Validate configuration
    if not DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN is not set. Please configure .env file.")
        sys.exit(1)

    # Check Ollama at startup
    available, message = check_ollama_available()
    if not available:
        logger.warning(f"Ollama is not available: {message}")
        logger.warning("The bot will start, but summarization will fail.")

    # Run the bot
    try:
        bot.run(DISCORD_TOKEN)
    except discord.LoginFailure:
        logger.error("Invalid Discord token. Please check your .env file.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
