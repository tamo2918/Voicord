"""
Discord Voice Transcription Bot
Records voice chat, transcribes using Whisper, and summarizes using Ollama.
"""

import asyncio
import logging
import os
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
    MAX_RECORDING_DURATION_SECONDS,
)
from transcriber import transcribe_audio, format_transcription_with_timestamps
from summarizer import summarize_conversation, check_ollama_available

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

# Store active voice connections and recording state
active_connections: dict[int, discord.VoiceClient] = {}
recording_start_times: dict[int, datetime] = {}


class RecordingSession:
    """Manages a single recording session."""

    def __init__(self, guild_id: int, channel: discord.TextChannel):
        self.guild_id = guild_id
        self.channel = channel
        self.start_time = datetime.now()
        self.session_dir = RECORDINGS_DIR / f"session_{guild_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def get_audio_path(self, user_id: int) -> Path:
        """Get the path for a user's audio file."""
        return self.session_dir / f"user_{user_id}.wav"


# Store recording sessions
recording_sessions: dict[int, RecordingSession] = {}


async def recording_finished_callback(sink: discord.sinks.WaveSink, channel: discord.TextChannel, *args):
    """
    Callback function called when recording stops.
    Processes all recorded audio, transcribes, and summarizes.
    """
    guild_id = sink.vc.guild.id
    session = recording_sessions.get(guild_id)

    if not session:
        logger.error(f"No session found for guild {guild_id}")
        return

    await channel.send("🔄 録音を処理中です。しばらくお待ちください...")

    try:
        # Process each user's audio
        transcriptions: dict[str, str] = {}
        audio_files: list[Path] = []

        for user_id, audio in sink.audio_data.items():
            # Get user info
            user = bot.get_user(user_id)
            username = user.display_name if user else f"User_{user_id}"

            # Save audio file
            audio_path = session.get_audio_path(user_id)
            with open(audio_path, "wb") as f:
                f.write(audio.file.getvalue())

            audio_files.append(audio_path)
            logger.info(f"Saved audio for {username}: {audio_path}")

            # Transcribe
            try:
                await channel.send(f"📝 {username} の音声を文字起こし中...")
                result = await asyncio.to_thread(transcribe_audio, audio_path)
                transcriptions[username] = result["text"]

                if result["text"]:
                    # Send individual transcription (truncated if too long)
                    text_preview = result["text"][:500]
                    if len(result["text"]) > 500:
                        text_preview += "..."
                    await channel.send(f"**{username}**: {text_preview}")

            except Exception as e:
                logger.error(f"Transcription failed for {username}: {e}")
                transcriptions[username] = f"(文字起こしエラー: {e})"

        # Generate summary if we have transcriptions
        if any(t for t in transcriptions.values() if t and not t.startswith("(文字起こしエラー")):
            await channel.send("📊 要約を生成中...")

            try:
                summary = await asyncio.to_thread(summarize_conversation, transcriptions)

                # Send summary (split if too long for Discord)
                await send_long_message(channel, f"## 📋 会議サマリー\n\n{summary}")

            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                await channel.send(f"⚠️ 要約の生成に失敗しました: {e}")
        else:
            await channel.send("⚠️ 文字起こしが空のため、要約を生成できませんでした。")

        # Cleanup audio files if configured
        if AUTO_DELETE_RECORDINGS:
            for audio_path in audio_files:
                try:
                    audio_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete {audio_path}: {e}")

            try:
                session.session_dir.rmdir()
            except Exception:
                pass  # Directory might not be empty

        await channel.send("✅ 処理が完了しました！")

    except Exception as e:
        logger.error(f"Error in recording_finished_callback: {e}")
        await channel.send(f"❌ エラーが発生しました: {e}")

    finally:
        # Cleanup session
        if guild_id in recording_sessions:
            del recording_sessions[guild_id]
        if guild_id in active_connections:
            del active_connections[guild_id]
        if guild_id in recording_start_times:
            del recording_start_times[guild_id]

        # Disconnect from voice
        if sink.vc.is_connected():
            await sink.vc.disconnect()


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
            name=f"{COMMAND_PREFIX}help for commands"
        )
    )


@bot.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
    """Handle voice state updates (e.g., when users leave during recording)."""
    if member.bot:
        return

    guild_id = member.guild.id

    # Check if we're recording and the voice channel is now empty (except bot)
    if guild_id in active_connections:
        vc = active_connections[guild_id]
        if vc.channel:
            # Count non-bot members
            members = [m for m in vc.channel.members if not m.bot]
            if len(members) == 0:
                logger.info(f"Voice channel empty, stopping recording for guild {guild_id}")
                if vc.recording:
                    vc.stop_recording()


# === Bot Commands ===

@bot.command(name="record", aliases=["rec", "録音"])
async def start_recording(ctx: commands.Context):
    """
    Start recording the voice channel.
    Usage: !record
    """
    # Check if user is in a voice channel
    if not ctx.author.voice:
        await ctx.send("❌ ボイスチャンネルに参加してから実行してください。")
        return

    voice_channel = ctx.author.voice.channel
    guild_id = ctx.guild.id

    # Check if already recording
    if guild_id in active_connections:
        await ctx.send("⚠️ すでに録音中です。`!stop` で停止してください。")
        return

    try:
        # Connect to voice channel
        vc = await voice_channel.connect()
        active_connections[guild_id] = vc

        # Create recording session
        session = RecordingSession(guild_id, ctx.channel)
        recording_sessions[guild_id] = session
        recording_start_times[guild_id] = datetime.now()

        # Start recording with WAV sink
        vc.start_recording(
            discord.sinks.WaveSink(),
            recording_finished_callback,
            ctx.channel
        )

        await ctx.send(
            f"🎙️ **録音を開始しました**\n"
            f"チャンネル: {voice_channel.name}\n"
            f"停止するには `{COMMAND_PREFIX}stop` を使用してください。"
        )

        logger.info(f"Started recording in {voice_channel.name} (Guild: {ctx.guild.name})")

    except Exception as e:
        logger.error(f"Failed to start recording: {e}")
        await ctx.send(f"❌ 録音の開始に失敗しました: {e}")

        # Cleanup on error
        if guild_id in active_connections:
            del active_connections[guild_id]
        if guild_id in recording_sessions:
            del recording_sessions[guild_id]


@bot.command(name="stop", aliases=["停止"])
async def stop_recording(ctx: commands.Context):
    """
    Stop recording and process the audio.
    Usage: !stop
    """
    guild_id = ctx.guild.id

    if guild_id not in active_connections:
        await ctx.send("❌ 現在録音中ではありません。")
        return

    vc = active_connections[guild_id]

    if not vc.recording:
        await ctx.send("❌ 録音が開始されていません。")
        return

    # Calculate recording duration
    start_time = recording_start_times.get(guild_id)
    duration_str = ""
    if start_time:
        duration = datetime.now() - start_time
        minutes, seconds = divmod(int(duration.total_seconds()), 60)
        duration_str = f" (録音時間: {minutes}分{seconds}秒)"

    await ctx.send(f"⏹️ 録音を停止しています{duration_str}...")

    # Stop recording (this will trigger the callback)
    vc.stop_recording()

    logger.info(f"Stopped recording in guild {ctx.guild.name}")


@bot.command(name="status", aliases=["状態"])
async def recording_status(ctx: commands.Context):
    """
    Check current recording status.
    Usage: !status
    """
    guild_id = ctx.guild.id

    if guild_id not in active_connections:
        await ctx.send("📊 現在録音中ではありません。")
        return

    vc = active_connections[guild_id]
    start_time = recording_start_times.get(guild_id)

    if start_time:
        duration = datetime.now() - start_time
        minutes, seconds = divmod(int(duration.total_seconds()), 60)

        await ctx.send(
            f"📊 **録音ステータス**\n"
            f"チャンネル: {vc.channel.name}\n"
            f"録音時間: {minutes}分{seconds}秒\n"
            f"参加者: {len([m for m in vc.channel.members if not m.bot])}人"
        )
    else:
        await ctx.send("📊 録音情報を取得できませんでした。")


@bot.command(name="cancel", aliases=["キャンセル"])
async def cancel_recording(ctx: commands.Context):
    """
    Cancel recording without processing.
    Usage: !cancel
    """
    guild_id = ctx.guild.id

    if guild_id not in active_connections:
        await ctx.send("❌ 現在録音中ではありません。")
        return

    vc = active_connections[guild_id]

    # Cleanup without processing
    if vc.recording:
        vc.stop_recording()

    # Manual cleanup
    if guild_id in recording_sessions:
        session = recording_sessions[guild_id]
        # Delete session directory
        try:
            import shutil
            shutil.rmtree(session.session_dir, ignore_errors=True)
        except Exception:
            pass
        del recording_sessions[guild_id]

    if guild_id in active_connections:
        del active_connections[guild_id]

    if guild_id in recording_start_times:
        del recording_start_times[guild_id]

    if vc.is_connected():
        await vc.disconnect()

    await ctx.send("🚫 録音をキャンセルしました。")


@bot.command(name="check", aliases=["チェック"])
async def check_system(ctx: commands.Context):
    """
    Check system status (Ollama, etc.).
    Usage: !check
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

    # Bot info
    embed.add_field(
        name="Bot情報",
        value=f"サーバー数: {len(bot.guilds)}\n"
              f"アクティブ録音: {len(active_connections)}",
        inline=False
    )

    await ctx.send(embed=embed)


@bot.command(name="commands", aliases=["コマンド", "cmds"])
async def show_commands(ctx: commands.Context):
    """
    Show available commands.
    Usage: !commands
    """
    embed = discord.Embed(
        title="📖 コマンド一覧",
        description="Discord音声文字起こし・要約Bot",
        color=discord.Color.green()
    )

    commands_list = [
        (f"`{COMMAND_PREFIX}record`", "録音を開始します"),
        (f"`{COMMAND_PREFIX}stop`", "録音を停止し、文字起こし・要約を生成します"),
        (f"`{COMMAND_PREFIX}cancel`", "録音をキャンセルします（処理なし）"),
        (f"`{COMMAND_PREFIX}status`", "現在の録音状態を確認します"),
        (f"`{COMMAND_PREFIX}check`", "システムの状態を確認します"),
        (f"`{COMMAND_PREFIX}commands`", "このヘルプを表示します"),
    ]

    for cmd, desc in commands_list:
        embed.add_field(name=cmd, value=desc, inline=False)

    embed.set_footer(text="ボイスチャンネルに参加してから !record を実行してください")

    await ctx.send(embed=embed)


# === Slash Commands (optional, for modern Discord experience) ===

@bot.slash_command(name="record", description="ボイスチャンネルの録音を開始します")
async def slash_record(ctx: discord.ApplicationContext):
    """Slash command version of record."""
    # Create a fake context for compatibility
    await start_recording(ctx)


@bot.slash_command(name="stop", description="録音を停止し、文字起こし・要約を生成します")
async def slash_stop(ctx: discord.ApplicationContext):
    """Slash command version of stop."""
    await stop_recording(ctx)


@bot.slash_command(name="status", description="現在の録音状態を確認します")
async def slash_status(ctx: discord.ApplicationContext):
    """Slash command version of status."""
    await recording_status(ctx)


# === Main Entry Point ===

def main():
    """Main entry point for the bot."""
    logger.info("Starting Discord Voice Transcription Bot...")

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
