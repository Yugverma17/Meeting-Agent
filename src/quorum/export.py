"""Rendering a transcript for a human.

The pipeline consumes transcripts as objects; people want them as text, in
whichever shape suits what they are doing. Someone revising wants readable
prose, someone quoting a colleague wants timestamps, and someone cutting a video
wants subtitles. Same data, four presentations.

Filtering matters more than it looks. A two-hour seminar is unreadable in full,
but "everything the speaker said between 40 and 55 minutes" is exactly what you
want when you half-remember something from the middle of it.
"""

from __future__ import annotations

import re
from enum import Enum

from quorum.models import Transcript

_TIME = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})$|^(\d+(?:\.\d+)?)$")


class Style(str, Enum):
    SPEAKERS = "speakers"
    """"Yug Verma (00:15): ..." - the default; readable and attributable."""

    TIMESTAMPED = "timestamped"
    """"[00:15] ..." - no speaker labels, for a single-speaker lecture."""

    PLAIN = "plain"
    """Continuous prose. For pasting somewhere else, or feeding to another tool."""

    MARKDOWN = "markdown"
    """Headed, blockquoted, with speaker changes as paragraph breaks."""

    SRT = "srt"
    """Subtitles. Load alongside a lecture recording to follow along."""


def parse_time(value: str | None) -> float | None:
    """Accept "90", "1:30" or "1:02:03" and return seconds."""
    if value is None or not str(value).strip():
        return None
    match = _TIME.match(str(value).strip())
    if not match:
        raise ValueError(f"Cannot read {value!r} as a time. Use 90, 1:30 or 1:02:03.")
    if match.group(4) is not None:
        return float(match.group(4))
    hours = int(match.group(1) or 0)
    return hours * 3600 + int(match.group(2)) * 60 + int(match.group(3))


def _stamp(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _srt_stamp(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def select(
    transcript: Transcript,
    speaker: str | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    search: str | None = None,
) -> list:
    """The utterances matching the filters, in order.

    `speaker` matches a display name, alias or email local-part, so "yug",
    "Yug Verma" and "Yug" all work.
    """
    chosen = transcript.speakers
    speaker_ids = None
    if speaker:
        matched = [s for s in chosen if s.matches(speaker)]
        if not matched:
            needle = speaker.strip().lower()
            matched = [s for s in chosen if needle in s.display_name.lower()]
        if not matched:
            known = ", ".join(s.display_name for s in chosen)
            raise ValueError(f"No speaker matching {speaker!r}. Present: {known}")
        speaker_ids = {s.id for s in matched}

    needle = search.strip().lower() if search else None
    result = []
    for utterance in transcript.utterances:
        if speaker_ids is not None and utterance.speaker_id not in speaker_ids:
            continue
        if start_s is not None and (utterance.start_s or 0) < start_s:
            continue
        if end_s is not None and (utterance.start_s or 0) > end_s:
            continue
        if needle and needle not in utterance.text.lower():
            continue
        result.append(utterance)
    return result


def render(
    transcript: Transcript,
    style: Style = Style.SPEAKERS,
    speaker: str | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    search: str | None = None,
) -> str:
    utterances = select(transcript, speaker, start_s, end_s, search)
    if not utterances:
        return ""

    names = {s.id: s.display_name for s in transcript.speakers}

    if style is Style.PLAIN:
        return " ".join(u.text.strip() for u in utterances)

    if style is Style.TIMESTAMPED:
        return "\n".join(f"[{_stamp(u.start_s)}] {u.text.strip()}" for u in utterances)

    if style is Style.SRT:
        blocks = []
        for index, utterance in enumerate(utterances, start=1):
            begin = utterance.start_s or 0.0
            # Subtitles need an end time. Whisper does not always give one, so
            # fall back to the next utterance's start, then to a fixed span.
            finish = utterance.end_s
            if finish is None or finish <= begin:
                following = utterances[index] if index < len(utterances) else None
                finish = (following.start_s if following else None) or begin + 4.0
            blocks.append(
                f"{index}\n{_srt_stamp(begin)} --> {_srt_stamp(finish)}\n"
                f"{utterance.text.strip()}\n"
            )
        return "\n".join(blocks)

    if style is Style.MARKDOWN:
        lines = [f"# {transcript.title or 'Transcript'}", ""]
        # People who spoke, not the roster - which carries a placeholder
        # participant on every live recording and so always overcounts by one.
        present = len(transcript.speakers_present)
        lines.append(f"*{transcript.meeting_date.isoformat()}"
                     + (f" · {present} speaker{'s' if present != 1 else ''}*"
                        if present else "*"))
        lines.append("")
        current = None
        for utterance in utterances:
            who = names.get(utterance.speaker_id, "Unknown")
            if who != current:
                lines.append("")
                lines.append(f"**{who}** *({_stamp(utterance.start_s)})*")
                lines.append("")
                current = who
            lines.append(f"> {utterance.text.strip()}")
        return "\n".join(lines).strip()

    return "\n".join(
        f"{names.get(u.speaker_id, 'Unknown')} ({_stamp(u.start_s)}): {u.text.strip()}"
        for u in utterances
    )


def stats(transcript: Transcript) -> dict:
    """Who spoke, how much - useful on its own for a seminar."""
    names = {s.id: s.display_name for s in transcript.speakers}
    words: dict[str, int] = {}
    for utterance in transcript.utterances:
        who = names.get(utterance.speaker_id, "Unknown")
        words[who] = words.get(who, 0) + len(utterance.text.split())

    total = sum(words.values()) or 1
    return {
        "utterances": len(transcript.utterances),
        "words": total,
        "duration_s": transcript.duration_s,
        "by_speaker": {
            who: {"words": count, "share": round(count / total, 3)}
            for who, count in sorted(words.items(), key=lambda kv: -kv[1])
        },
    }
