# Voicord

Discord音声チャットを録音し、文字起こし・要約を行うボットです。Apple Silicon (M1/M2/M3/M4) に最適化されています。

## 特徴

- **ローカル処理**: すべての処理がローカルで完結。クラウドAPIへの依存なし
- **高速文字起こし**: MLX Whisperによる高速な音声認識
- **自動要約**: Ollamaを使ったLLMによる会議内容の要約
- **話者分離**: ユーザーごとの発言を自動的に分離
- **プライバシー保護**: 音声データが外部に送信されることはありません

## 必要環境

- macOS (Apple Silicon: M1/M2/M3/M4)
- Python 3.11+
- ffmpeg
- Ollama

## セットアップ

### 1. 依存関係のインストール

```bash
# ffmpegのインストール
brew install ffmpeg

# 仮想環境の作成
python3.11 -m venv venv
source venv/bin/activate

# パッケージのインストール
pip install -r requirements.txt
```

### 2. Ollamaのセットアップ

```bash
# Ollamaのインストール（公式サイトから）
# https://ollama.com/download

# モデルのダウンロード
ollama pull llama3.2:8b
```

### 3. Discord Botの作成

1. [Discord Developer Portal](https://discord.com/developers/applications) で新しいApplicationを作成
2. Bot設定でトークンを取得
3. OAuth2 → URL Generator で以下を選択:
   - Scopes: `bot`, `applications.commands`
   - Bot Permissions: `Connect`, `Speak`, `Send Messages`, `Read Message History`
4. 生成されたURLでサーバーに招待

### 4. 環境変数の設定

```bash
cp .env.example .env
# .envファイルを編集してDISCORD_TOKENを設定
```

### 5. 起動

```bash
source venv/bin/activate
python bot.py
```

## 使い方

| コマンド | 説明 |
|---------|------|
| `!record` | ボイスチャンネルに参加して録音を開始 |
| `!stop` | 録音を停止し、文字起こし・要約を実行 |

## 設定

`.env`ファイルで以下の設定が可能です:

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| `DISCORD_TOKEN` | (必須) | Discord Botトークン |
| `COMMAND_PREFIX` | `!` | コマンドのプレフィックス |
| `WHISPER_MODEL` | `large-v3` | Whisperモデル |
| `WHISPER_LANGUAGE` | `ja` | 音声認識の言語 |
| `OLLAMA_MODEL` | `llama3.2:8b` | 要約に使用するLLMモデル |
| `MAX_RECORDING_DURATION_SECONDS` | `3600` | 最大録音時間（秒） |
| `AUTO_DELETE_RECORDINGS` | `true` | 処理後に音声ファイルを自動削除 |

## 推奨スペック

| コンポーネント | メモリ使用量 |
|---------------|-------------|
| Whisper large-v3 | ~3GB |
| Ollama llama3.2:8b | ~5GB |
| システム + Bot | ~2GB |

合計約10GBのメモリがあれば快適に動作します。

## ライセンス

MIT License
