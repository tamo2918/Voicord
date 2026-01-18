"""
Text summarization using Ollama LLM.
Supports local LLM inference for privacy-focused summarization.
Handles long transcripts through hierarchical summarization.
"""

import logging
from typing import Optional

import ollama

from config import OLLAMA_MODEL, OLLAMA_HOST, SUMMARY_LANGUAGE, MAX_CONTEXT_CHARS

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

# Prompts for partial summarization (used in hierarchical summarization)
PARTIAL_SUMMARY_PROMPTS = {
    "ja": """以下は長い会議の一部の文字起こしです。
この部分で話された内容の要点を箇条書きで抽出してください。
重要な発言、決定事項、アクションアイテムがあれば含めてください。
簡潔に、しかし重要な情報は漏らさないようにしてください。""",

    "en": """Below is a portion of a longer meeting transcript.
Extract the key points from this section in bullet points.
Include important statements, decisions, and action items if any.
Be concise but don't miss important information."""
}

# Prompts for combining partial summaries
COMBINE_SUMMARY_PROMPTS = {
    "ja": """以下は長い会議の各パートの要約です。
これらを統合して、会議全体の包括的な要約を作成してください。
重複を排除し、全体の流れがわかるようにまとめてください。

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

    "en": """Below are summaries of different parts of a longer meeting.
Combine these into a comprehensive summary of the entire meeting.
Remove duplicates and create a coherent overview.

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


def _call_ollama(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float = 0.3
) -> str:
    """
    Make a call to Ollama API.

    Args:
        prompt: User prompt
        system_prompt: System prompt
        model: Model name
        temperature: Temperature for generation

    Returns:
        Generated text
    """
    client = ollama.Client(host=OLLAMA_HOST)

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": temperature,
            "top_p": 0.9,
        }
    )

    return response["message"]["content"]


def split_text_into_chunks(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks, trying to break at natural boundaries.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current_chunk = ""

    # Split by paragraphs first
    paragraphs = text.split("\n\n")

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If single paragraph is too long, split by sentences
            if len(para) > max_chars:
                sentences = para.replace("。", "。\n").replace(". ", ".\n").split("\n")
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= max_chars:
                        current_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def summarize_text(
    text: str,
    language: Optional[str] = None,
    model: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> str:
    """
    Summarize text using Ollama LLM.
    Automatically handles long texts through hierarchical summarization.

    Args:
        text: Text to summarize
        language: Language for summary ('ja' or 'en'). Defaults to config setting.
        model: Ollama model to use. Defaults to config setting.
        custom_prompt: Optional custom system prompt
        progress_callback: Optional callback(stage, message) for progress updates

    Returns:
        Summary text
    """
    if not text or not text.strip():
        return "（文字起こしが空のため、要約できませんでした）"

    lang = language or SUMMARY_LANGUAGE
    llm_model = model or OLLAMA_MODEL
    system_prompt = custom_prompt or SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ja"])
    max_chars = MAX_CONTEXT_CHARS

    logger.info(f"Summarizing text ({len(text)} chars) using {llm_model}")

    try:
        # Check if text needs hierarchical summarization
        if len(text) <= max_chars:
            # Direct summarization for short texts
            if progress_callback:
                progress_callback("summarizing", "テキストを要約中...")

            return _call_ollama(
                get_summary_prompt(text, lang),
                system_prompt,
                llm_model
            )

        # Hierarchical summarization for long texts
        logger.info(f"Text too long ({len(text)} chars), using hierarchical summarization")

        if progress_callback:
            progress_callback("splitting", "長いテキストを分割中...")

        # Split into chunks
        chunks = split_text_into_chunks(text, max_chars)
        logger.info(f"Split into {len(chunks)} chunks")

        # Summarize each chunk
        partial_summaries = []
        partial_prompt = PARTIAL_SUMMARY_PROMPTS.get(lang, PARTIAL_SUMMARY_PROMPTS["ja"])

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback("partial", f"パート {i + 1}/{len(chunks)} を要約中...")

            logger.info(f"Summarizing chunk {i + 1}/{len(chunks)}")

            chunk_prompt = f"パート {i + 1}/{len(chunks)}:\n\n{chunk}"
            partial_summary = _call_ollama(chunk_prompt, partial_prompt, llm_model)
            partial_summaries.append(f"### パート {i + 1}\n{partial_summary}")

        # Combine partial summaries
        if progress_callback:
            progress_callback("combining", "要約を統合中...")

        combined_partials = "\n\n".join(partial_summaries)

        # If combined partials are still too long, recursively summarize
        if len(combined_partials) > max_chars:
            logger.info("Partial summaries still too long, doing another round")
            return summarize_text(
                combined_partials,
                language=lang,
                model=llm_model,
                progress_callback=progress_callback
            )

        # Final combination
        combine_prompt = COMBINE_SUMMARY_PROMPTS.get(lang, COMBINE_SUMMARY_PROMPTS["ja"])
        final_summary = _call_ollama(combined_partials, combine_prompt, llm_model)

        logger.info(f"Hierarchical summary complete: {len(final_summary)} characters")

        return final_summary

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
    model: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> str:
    """
    Summarize a multi-speaker conversation.

    Args:
        speakers: Dictionary mapping speaker names/IDs to their transcribed text
        language: Language for summary
        model: Ollama model to use
        progress_callback: Optional progress callback

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

    return summarize_text(
        full_transcription,
        language,
        model,
        progress_callback=progress_callback
    )


def summarize_with_timestamps(
    segments: list[dict],
    language: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """
    Summarize transcription segments that include timestamps.

    Args:
        segments: List of segment dicts with 'start', 'end', 'text' keys
        language: Language for summary
        model: Ollama model to use

    Returns:
        Summary text
    """
    # Format segments with timestamps
    formatted_parts = []
    for seg in segments:
        start = seg.get("start", 0)
        text = seg.get("text", "").strip()
        if text:
            # Format timestamp as MM:SS or HH:MM:SS
            if start >= 3600:
                ts = f"{int(start // 3600):02d}:{int((start % 3600) // 60):02d}:{int(start % 60):02d}"
            else:
                ts = f"{int(start // 60):02d}:{int(start % 60):02d}"
            formatted_parts.append(f"[{ts}] {text}")

    if not formatted_parts:
        return "（文字起こしが空のため、要約できませんでした）"

    full_text = "\n".join(formatted_parts)
    return summarize_text(full_text, language, model)


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


def estimate_summary_time(text_length: int) -> str:
    """
    Estimate how long summarization will take.

    Args:
        text_length: Length of text in characters

    Returns:
        Human-readable estimate
    """
    # Rough estimate based on typical LLM speeds
    # ~50 tokens/sec output, ~4 chars/token, summary ~1/10 of input
    estimated_output_chars = text_length / 10
    estimated_tokens = estimated_output_chars / 4
    estimated_seconds = estimated_tokens / 50

    # Add overhead for long texts (hierarchical processing)
    if text_length > MAX_CONTEXT_CHARS:
        num_chunks = (text_length // MAX_CONTEXT_CHARS) + 1
        estimated_seconds *= num_chunks * 1.5

    if estimated_seconds < 60:
        return f"約{max(5, int(estimated_seconds))}秒"
    else:
        return f"約{int(estimated_seconds / 60)}分"


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

    print(f"\nEstimated time: {estimate_summary_time(len(sample_text))}")
    print("\n=== Test Summarization ===")
    summary = summarize_text(sample_text, language="ja")
    print(summary)
