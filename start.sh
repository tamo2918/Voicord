#!/bin/bash
# Discord Voice Transcription Bot 起動スクリプト

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 仮想環境を有効化
source venv/bin/activate

# Ollamaが起動しているか確認
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollamaが起動していません。起動します..."
    open -a Ollama
    sleep 3
fi

echo "🚀 Discord Voice Bot を起動中..."
python bot.py
