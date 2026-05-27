"""Tests for AdapterFilter — shared E-4 response filtering logic.

Covers: user_blocklist, channel_whitelist/blacklist, DM bypass,
mention_only, ai_decision (keywords + extra_keywords + soul fallback).
"""

from __future__ import annotations

from unittest.mock import patch

from cordbeat.adapters._utils import AdapterFilter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter(
    respond_mode: str = "all",
    channel_whitelist: list[str] | None = None,
    channel_blacklist: list[str] | None = None,
    user_blocklist: list[str] | None = None,
    ai_decision_keywords: list[str] | None = None,
) -> AdapterFilter:
    return AdapterFilter.from_options(
        {
            "respond_mode": respond_mode,
            "channel_whitelist": channel_whitelist or [],
            "channel_blacklist": channel_blacklist or [],
            "user_blocklist": user_blocklist or [],
            "ai_decision_keywords": ai_decision_keywords or [],
        }
    )


# ---------------------------------------------------------------------------
# User blocklist
# ---------------------------------------------------------------------------


class TestUserBlocklist:
    def test_blocked_user_rejected(self) -> None:
        f = _filter(user_blocklist=["bad_user"])
        assert f.should_respond(user_id="bad_user") is False

    def test_non_blocked_user_allowed(self) -> None:
        f = _filter(user_blocklist=["bad_user"])
        assert f.should_respond(user_id="good_user") is True

    def test_empty_blocklist_allows_all(self) -> None:
        f = _filter()
        assert f.should_respond(user_id="anyone") is True

    def test_blocklist_takes_priority_over_dm(self) -> None:
        f = _filter(user_blocklist=["blocked"])
        assert f.should_respond(user_id="blocked", is_dm=True) is False


# ---------------------------------------------------------------------------
# Channel whitelist
# ---------------------------------------------------------------------------


class TestChannelWhitelist:
    def test_non_whitelisted_channel_blocked(self) -> None:
        f = _filter(channel_whitelist=["chan_a"])
        assert f.should_respond(user_id="u", channel_id="chan_b") is False

    def test_whitelisted_channel_allowed(self) -> None:
        f = _filter(channel_whitelist=["chan_a"])
        assert f.should_respond(user_id="u", channel_id="chan_a") is True

    def test_empty_whitelist_allows_all_channels(self) -> None:
        f = _filter()
        assert f.should_respond(user_id="u", channel_id="any_channel") is True

    def test_channel_filter_skipped_for_dm(self) -> None:
        f = _filter(channel_whitelist=["chan_a"])
        assert f.should_respond(user_id="u", channel_id="chan_b", is_dm=True) is True


# ---------------------------------------------------------------------------
# Channel blacklist
# ---------------------------------------------------------------------------


class TestChannelBlacklist:
    def test_blacklisted_channel_blocked(self) -> None:
        f = _filter(channel_blacklist=["spam"])
        assert f.should_respond(user_id="u", channel_id="spam") is False

    def test_non_blacklisted_channel_allowed(self) -> None:
        f = _filter(channel_blacklist=["spam"])
        assert f.should_respond(user_id="u", channel_id="general") is True

    def test_blacklist_skipped_for_dm(self) -> None:
        f = _filter(channel_blacklist=["spam"])
        assert f.should_respond(user_id="u", channel_id="spam", is_dm=True) is True


# ---------------------------------------------------------------------------
# DM bypass
# ---------------------------------------------------------------------------


class TestDMBypass:
    def test_dm_bypasses_respond_mode_mention_only(self) -> None:
        f = _filter(respond_mode="mention_only")
        assert f.should_respond(user_id="u", is_dm=True, is_mentioned=False) is True

    def test_dm_bypasses_respond_mode_ai_decision(self) -> None:
        f = _filter(respond_mode="ai_decision", ai_decision_keywords=["bot"])
        assert f.should_respond(user_id="u", is_dm=True, text="no keyword here") is True

    def test_guild_mention_only_no_mention_rejected(self) -> None:
        f = _filter(respond_mode="mention_only")
        assert f.should_respond(user_id="u", is_dm=False, is_mentioned=False) is False


# ---------------------------------------------------------------------------
# respond_mode: all
# ---------------------------------------------------------------------------


class TestModeAll:
    def test_all_mode_always_responds(self) -> None:
        f = _filter(respond_mode="all")
        assert f.should_respond(user_id="u", is_dm=False, is_mentioned=False) is True

    def test_unknown_mode_treated_as_all(self) -> None:
        f = _filter(respond_mode="unknown_future_mode")
        assert f.should_respond(user_id="u") is True


# ---------------------------------------------------------------------------
# respond_mode: mention_only
# ---------------------------------------------------------------------------


class TestModeMentionOnly:
    def test_mentioned_in_guild_allowed(self) -> None:
        f = _filter(respond_mode="mention_only")
        assert f.should_respond(user_id="u", is_dm=False, is_mentioned=True) is True

    def test_not_mentioned_in_guild_rejected(self) -> None:
        f = _filter(respond_mode="mention_only")
        assert f.should_respond(user_id="u", is_dm=False, is_mentioned=False) is False

    def test_dm_bypasses_mention_requirement(self) -> None:
        f = _filter(respond_mode="mention_only")
        assert f.should_respond(user_id="u", is_dm=True) is True


# ---------------------------------------------------------------------------
# respond_mode: ai_decision
# ---------------------------------------------------------------------------


class TestModeAIDecision:
    def test_keyword_present_in_text_allowed(self) -> None:
        f = _filter(respond_mode="ai_decision", ai_decision_keywords=["cordbeat"])
        assert f.should_respond(user_id="u", text="Hey cordbeat how are you") is True

    def test_keyword_absent_in_text_rejected(self) -> None:
        f = _filter(respond_mode="ai_decision", ai_decision_keywords=["cordbeat"])
        assert f.should_respond(user_id="u", text="Hello everyone") is False

    def test_keyword_match_is_case_insensitive(self) -> None:
        f = _filter(respond_mode="ai_decision", ai_decision_keywords=["CordBeat"])
        assert f.should_respond(user_id="u", text="hey cordbeat!") is True

    def test_no_keywords_no_soul_allows_all(self) -> None:
        """When no keywords configured and soul.yaml unavailable, allow all."""
        f = _filter(respond_mode="ai_decision")
        with patch("cordbeat.adapters._utils._read_soul_keywords", return_value=[]):
            result = f.should_respond(user_id="u", text="hello")
        assert result is True

    def test_extra_keywords_used_as_fallback(self) -> None:
        """extra_keywords fills the gap when ai_keywords list is empty."""
        f = _filter(respond_mode="ai_decision")
        assert (
            f.should_respond(
                user_id="u",
                text="question for MyBot here",
                extra_keywords=["MyBot"],
            )
            is True
        )

    def test_extra_keywords_rejected_when_absent(self) -> None:
        f = _filter(respond_mode="ai_decision")
        assert (
            f.should_respond(
                user_id="u",
                text="just chatting",
                extra_keywords=["MyBot"],
            )
            is False
        )

    def test_extra_keywords_combined_with_config_keywords(self) -> None:
        f = _filter(respond_mode="ai_decision", ai_decision_keywords=["cordbeat"])
        # config keyword not in text but extra_keyword is
        assert (
            f.should_respond(
                user_id="u",
                text="hello BotName",
                extra_keywords=["BotName"],
            )
            is True
        )

    def test_dm_bypasses_ai_decision(self) -> None:
        f = _filter(respond_mode="ai_decision", ai_decision_keywords=["cordbeat"])
        assert f.should_respond(user_id="u", is_dm=True, text="no keyword") is True


# ---------------------------------------------------------------------------
# from_options factory
# ---------------------------------------------------------------------------


class TestFromOptions:
    def test_defaults_when_empty_options(self) -> None:
        f = AdapterFilter.from_options({})
        assert f.respond_mode == "all"
        assert f.channel_whitelist == frozenset()
        assert f.channel_blacklist == frozenset()
        assert f.user_blocklist == frozenset()
        assert f.ai_keywords == []

    def test_options_converted_to_strings(self) -> None:
        f = AdapterFilter.from_options(
            {
                "user_blocklist": [123456, 789012],
                "channel_whitelist": [111, 222],
            }
        )
        assert "123456" in f.user_blocklist
        assert "111" in f.channel_whitelist

    def test_respond_mode_stored(self) -> None:
        f = AdapterFilter.from_options({"respond_mode": "mention_only"})
        assert f.respond_mode == "mention_only"
