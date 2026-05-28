"""Transcript capture helpers for post-call booking extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import LOG_TRANSCRIPTS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptLine:
    speaker: str
    text: str


def transcript_text(transcript: list[TranscriptLine]) -> str:
    return "\n".join(f"{line.speaker}: {line.text}" for line in transcript if line.text)


def record_transcript_line(transcript: list[TranscriptLine], speaker: str, text: str) -> None:
    clean_text = text.strip()
    if not clean_text:
        return

    if transcript and transcript[-1].speaker == speaker:
        previous = transcript[-1].text
        separator = "" if clean_text[:1] in {".", ",", "!", "?", ";", ":"} else " "
        transcript[-1] = TranscriptLine(speaker, f"{previous}{separator}{clean_text}")
    else:
        transcript.append(TranscriptLine(speaker, clean_text))

    if LOG_TRANSCRIPTS:
        logger.info("Transcript [%s]: %s", speaker, clean_text)
