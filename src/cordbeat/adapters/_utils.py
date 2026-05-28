"""Shared utilities for platform adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cordbeat.ai.backend import AIBackend

logger = logging.getLogger(__name__)


# ── ai_decision_llm: global judge backend ─────────────────────────────
#
# When ``config.ai_decision`` is configured, the adapter runner instantiates
# a small/fast LLM backend (e.g. gemma2:2b) and registers it here.  Adapters
# in ``respond_mode: ai_decision_llm`` consult this backend to decide whether
# a public-channel message warrants a reply.  Default ``None`` → adapters
# fall back to the keyword-only ``ai_decision`` behaviour.

_judge_backend: AIBackend | None = None


def set_judge_backend(backend: AIBackend | None) -> None:
    """Register (or clear) the global ai_decision_llm judge backend."""
    global _judge_backend
    _judge_backend = backend


def get_judge_backend() -> AIBackend | None:
    """Return the currently registered ai_decision_llm judge backend, if any."""
    return _judge_backend


def _read_soul_keywords() -> list[str]:
    """Return the AI name from soul.yaml as a single-element keyword list.

    Reads ``~/.cordbeat/soul/soul.yaml`` (or the path configured via
    ``CORDBEAT_HOME``) to find ``identity.name``.  Returns an empty list if
    the file is missing or cannot be parsed.
    """
    try:
        import yaml

        from cordbeat.config import cordbeat_home

        soul_yaml = cordbeat_home() / "soul" / "soul.yaml"
        if not soul_yaml.is_file():
            return []
        with soul_yaml.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        name: str = data.get("identity", {}).get("name", "")
        return [name] if name else []
    except Exception:  # noqa: BLE001
        logger.debug("Could not read soul name for keyword filter", exc_info=True)
        return []


@dataclass
class AdapterFilter:
    """E-4 response filtering logic, shared across all platform adapters.

    Build from adapter ``options`` dict via :meth:`from_options`, then call
    :meth:`should_respond` in each message handler::

        self._filter = AdapterFilter.from_options(config.options)
        ...
        if not self._filter.should_respond(user_id=uid, channel_id=ch,
                                            is_dm=is_dm, is_mentioned=mentioned,
                                            text=content):
            return
    """

    respond_mode: str = "all"
    channel_whitelist: frozenset[str] = field(default_factory=frozenset)
    channel_blacklist: frozenset[str] = field(default_factory=frozenset)
    user_blocklist: frozenset[str] = field(default_factory=frozenset)
    ai_keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> AdapterFilter:
        """Create an :class:`AdapterFilter` from adapter ``options`` dict."""
        return cls(
            respond_mode=str(options.get("respond_mode", "all")),
            channel_whitelist=frozenset(
                str(c) for c in options.get("channel_whitelist", [])
            ),
            channel_blacklist=frozenset(
                str(c) for c in options.get("channel_blacklist", [])
            ),
            user_blocklist=frozenset(str(u) for u in options.get("user_blocklist", [])),
            ai_keywords=[str(k) for k in options.get("ai_decision_keywords", [])],
        )

    def should_respond(
        self,
        *,
        user_id: str,
        channel_id: str = "",
        is_dm: bool = False,
        is_mentioned: bool = False,
        text: str = "",
        extra_keywords: list[str] | None = None,
    ) -> bool:
        """Return *True* if the adapter should process and forward this message.

        Args:
            user_id: Platform-specific user identifier.
            channel_id: Platform-specific channel/chat identifier.  Empty
                string means "not applicable" (e.g. Signal 1-to-1).
            is_dm: *True* when the message is a direct/private message.
                DMs bypass channel filters and ``respond_mode`` restrictions.
            is_mentioned: *True* when the bot was explicitly @mentioned.
                Used for ``mention_only`` mode.
            text: Raw message text.  Used for ``ai_decision`` keyword matching.
            extra_keywords: Optional adapter-specific extra keywords appended to
                the configured ``ai_decision_keywords`` list (e.g. Discord bot
                display name or Telegram ``@username``).
        """
        # 1. User blocklist always checked first.
        if user_id in self.user_blocklist:
            return False

        # 2. Channel filters — skipped for DMs.
        if not is_dm and channel_id:
            if channel_id in self.channel_blacklist:
                return False
            if self.channel_whitelist and channel_id not in self.channel_whitelist:
                return False

        # 3. DMs always bypass respond_mode.
        if is_dm:
            return True

        # 4. Mode-specific filtering.
        if self.respond_mode == "mention_only":
            return is_mentioned

        if self.respond_mode == "ai_decision":
            keywords: list[str] = list(self.ai_keywords) or list(_read_soul_keywords())
            if extra_keywords:
                keywords.extend(k for k in extra_keywords if k)
            if not keywords:
                return True  # no filter available → allow
            text_lower = text.lower()
            return any(kw.lower() in text_lower for kw in keywords)

        if self.respond_mode == "ai_decision_llm":
            # Sync pre-filter: when no judge backend is registered, degrade
            # gracefully to keyword matching.  When one IS registered the
            # final yes/no call happens in :meth:`should_respond_async`,
            # which adapters await *after* this sync check passes.
            if get_judge_backend() is None:
                keywords = list(self.ai_keywords) or list(_read_soul_keywords())
                if extra_keywords:
                    keywords.extend(k for k in extra_keywords if k)
                if not keywords:
                    return True
                text_lower = text.lower()
                return any(kw.lower() in text_lower for kw in keywords)
            return True  # defer to async judge

        return True  # "all" or unrecognised mode

    async def should_respond_async(
        self,
        *,
        user_id: str,
        channel_id: str = "",
        is_dm: bool = False,
        is_mentioned: bool = False,
        text: str = "",
        extra_keywords: list[str] | None = None,
    ) -> bool:
        """Async variant of :meth:`should_respond` with LLM judgement support.

        For modes other than ``ai_decision_llm`` (or when no judge backend
        is registered), this is equivalent to :meth:`should_respond`.  When
        ``respond_mode == "ai_decision_llm"`` *and* a judge backend has
        been registered via :func:`set_judge_backend`, a short yes/no LLM
        prompt is issued and the message is forwarded only when the model
        replies affirmatively.
        """
        if not self.should_respond(
            user_id=user_id,
            channel_id=channel_id,
            is_dm=is_dm,
            is_mentioned=is_mentioned,
            text=text,
            extra_keywords=extra_keywords,
        ):
            return False
        if self.respond_mode != "ai_decision_llm" or is_dm:
            return True
        backend = get_judge_backend()
        if backend is None:
            return True  # already keyword-judged above
        soul_name = ""
        soul_keys = _read_soul_keywords()
        if soul_keys:
            soul_name = soul_keys[0]
        prompt = (
            "You are deciding whether an AI assistant"
            + (f" named {soul_name}" if soul_name else "")
            + " should reply to a public-channel message.\n"
            "Reply with exactly one word: yes or no.\n"
            "Answer 'yes' only if the message clearly addresses the "
            "assistant, asks a question the assistant could answer, or "
            "invites it into the conversation.\n"
            "Answer 'no' for unrelated chatter between other users.\n\n"
            f"Message: {text}\n"
            "Answer:"
        )
        try:
            reply = await backend.generate(prompt, system="", max_tokens=8,
                                            temperature=0.0)
        except Exception:  # noqa: BLE001
            logger.warning("ai_decision_llm backend failed; allowing message",
                           exc_info=True)
            return True
        first = reply.strip().lower().split()[:1]
        return bool(first and first[0].startswith("y"))
