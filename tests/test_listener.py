"""Tests for jarvis.listener."""

import pytest

from jarvis.listener import _clean_transcript, _BARGEIN_FILLER


class TestCleanTranscript:
    def test_removes_fillers(self):
        assert _clean_transcript("uh can you help me") == "can you help me"

    def test_removes_multiple_fillers(self):
        result = _clean_transcript("um like can you basically help me")
        assert "um" not in result
        assert "like" not in result
        assert "basically" not in result
        assert "help" in result

    def test_deduplicates_words(self):
        assert _clean_transcript("the the code is is broken") == "the code is broken"

    def test_collapses_whitespace(self):
        assert _clean_transcript("hello   world") == "hello world"

    def test_fixes_orphaned_punctuation(self):
        assert _clean_transcript("hello , world") == "hello, world"

    def test_empty_string(self):
        assert _clean_transcript("") == ""

    def test_preserves_clean_text(self):
        text = "Can you refactor the database module?"
        assert _clean_transcript(text) == text

    def test_italian_fillers(self):
        result = _clean_transcript("allora puoi aiutarmi ehm con questo")
        assert "allora" not in result
        assert "ehm" not in result
        assert "aiutarmi" in result


class TestBargeInFiller:
    def test_english_fillers(self):
        for filler in ["uh", "um", "hmm", "okay", "yeah", "right", "sure"]:
            assert _BARGEIN_FILLER.match(filler), f"{filler} should match"

    def test_italian_fillers(self):
        for filler in ["sì", "no", "ok", "va bene", "eh"]:
            assert _BARGEIN_FILLER.match(filler), f"{filler} should match"

    def test_real_speech_not_filler(self):
        for text in [
            "stop that",
            "actually wait",
            "no I meant the other thing",
            "can you help me",
        ]:
            assert not _BARGEIN_FILLER.match(text), f"'{text}' should not match filler"

    def test_filler_with_period(self):
        assert _BARGEIN_FILLER.match("okay.")
        assert _BARGEIN_FILLER.match("hmm.")
