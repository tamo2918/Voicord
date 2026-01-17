"""
Text summarization using Ollama LLM.
Supports local LLM inference for privacy-focused summarization.
"""

import logging
from typing import Optional

import ollama

from config import OLLAMA_MODEL, OLLAMA_HOST, SUMMARY_LANGUAGE

logger = logging.getLogger(__name__)

# System prompts for different languages
SYSTEM_PROMPTS = {
    "ja": """あなたは会議の議事録を作成する優秀なアシスタントです。
与えられた会話の文字起こしを分析し、簡潔で分かりやすい要約を作成してください。
以下の形式で出力してください：

## 概要
（会議全体の簡潔な要約を1-2文で）

## 主な議題
（箇条書きで主な話題を列挙）

## 決定事項
（決まったことがあれば箇条書きで）

## アクションアイテム
（誰が何をするか、タスクがあれば箇条書きで）

## 補足
（その他重要な情報があれば）""",

    "en": """You are an excellent assistant for creating meeting minutes.
Analyze the given conversation transcript and create a clear, concise summary.
Output in the following format:

## Overview
(Brief 1-2 sentence summary of the meeting)

## Main Topics
(Bullet points of main discussion topics)

## Decisions Made
(Bullet points of any decisions made)

## Action Items
(Who does what - list tasks if any)

## Additional Notes
(Any other important information)"""
}


def get_summary_prompt(transcription: str, language: str = "ja") -> str:
    """
    Generate the prompt for summarization.

    Args:
        transcription: The meeting transcription text
        language: Language code ('ja' or 'en')

    Returns:
        Formatted prompt string
    """
    if language == "ja":
        return f"""以下は会議の文字起こしです。これを要約してください。

---
{transcription}
---

上記の会議内容を要約してください。"""
    else:
        return f"""Below is a meeting transcript. Please summarize it.

---
{transcription}
---

Please summarize the meeting content above."""


def summarize_text(
    text: str,
    language: Optional[str] = None,
    model: Optional[str] = None,
    custom_prompt: Optional[str] = None
) -> str:
    """
    Summarize text using Ollama LLM.

    Args:
        text: Text to summarize
        language: Language for summary ('ja' or 'en'). Defaults to config setting.
        model: Ollama model to use. Defaults to config setting.
        custom_prompt: Optional custom system prompt

    Returns:
        Summary text
    """
    if not text or not text.strip():
        return "（文字起こしが空のため、要約できませんでした）"

    lang = language or SUMMARY_LANGUAGE
    llm_model = model or OLLAMA_MODEL
    system_prompt = custom_prompt or SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ja"])

    logger.info(f"Summarizing text ({len(text)} chars) using {llm_model}")

    try:
        # Configure Ollama client
        client = ollama.Client(host=OLLAMA_HOST)

        # Generate summary
        response = client.chat(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": get_summary_prompt(text, lang)}
            ],
            options={
                "temperature": 0.3,  # Lower temperature for more consistent output
                "top_p": 0.9,
            }
        )

        summary = response["message"]["content"]
        logger.info(f"Summary generated: {len(summary)} characters")

        return summary

    except ollama.ResponseError as e:
        logger.error(f"Ollama API error: {e}")
        if "model" in str(e).lower() and "not found" in str(e).lower():
            return f"エラー: モデル '{llm_model}' が見つかりません。`ollama pull {llm_model}` を実行してください。"
        raise
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise


def summarize_conversation(
    speakers: dict[str, str],
    language: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """
    Summarize a multi-speaker conversation.

    Args:
        speakers: Dictionary mapping speaker names/IDs to their transcribed text
        language: Language for summary
        model: Ollama model to use

    Returns:
        Summary text
    """
    # Format conversation with speaker labels
    conversation_parts = []
    for speaker, text in speakers.items():
        if text and text.strip():
            conversation_parts.append(f"【{speaker}】\n{text}")

    if not conversation_parts:
        return "（会話の文字起こしが空のため、要約できませんでした）"

    full_transcription = "\n\n".join(conversation_parts)

    return summarize_text(full_transcription, language, model)


def check_ollama_available() -> tuple[bool, str]:
    """
    Check if Ollama is available and the model is downloaded.

    Returns:
        Tuple of (is_available, message)
    """
    try:
        client = ollama.Client(host=OLLAMA_HOST)

        # Check if we can connect
        models = client.list()

        # Check if our model is available
        model_names = [m["name"] for m in models.get("models", [])]

        if OLLAMA_MODEL in model_names or any(OLLAMA_MODEL.split(":")[0] in m for m in model_names):
            return True, f"Ollama is available with model {OLLAMA_MODEL}"
        else:
            return False, f"Model {OLLAMA_MODEL} not found. Run: ollama pull {OLLAMA_MODEL}"

    except Exception as e:
        return False, f"Cannot connect to Ollama at {OLLAMA_HOST}: {e}"


if __name__ == "__main__":
    # Test summarization
    import sys

    logging.basicConfig(level=logging.INFO)

    # Check Ollama availability
    available, message = check_ollama_available()
    print(f"Ollama status: {message}")

    if not available:
        print("Please ensure Ollama is running and the model is downloaded.")
        sys.exit(1)

    # Test with sample text
    sample_text = """
    田中: 今日のミーティングを始めましょう。まず、新しいプロジェクトの進捗について話し合いたいと思います。
    佐藤: はい、先週からUIの設計を進めていまして、来週には最初のプロトタイプができる予定です。
    田中: いいですね。バックエンドの方はどうですか？
    鈴木: APIの設計は完了しました。来週からデータベースの実装に入ります。
    田中: わかりました。では、来週の金曜日にもう一度進捗確認のミーティングをしましょう。
    佐藤: 了解しました。
    鈴木: はい、それまでに実装を進めておきます。
    """

    print("\n=== Test Summarization ===")
    summary = summarize_text(sample_text, language="ja")
    print(summary)
