#!/bin/bash
# Discord Voice Transcription Bot 停止スクリプト

echo "🛑 Discord Voice Bot を停止中..."
pkill -f "python bot.py" && echo "✅ 停止しました" || echo "⚠️  実行中のBotが見つかりません"
