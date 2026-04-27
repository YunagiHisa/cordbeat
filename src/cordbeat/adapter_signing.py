"""Webhook signature verification helpers for platform adapters.

This module collects the small HMAC-based verification routines used by
the various third-party platform adapters. They are exposed as public
helpers so that:

* downstream code (e.g., custom webhook handlers, integration tests) can
  reuse the exact same verification logic, and
* the verification surface is centralized and individually unit-tested.

All helpers return ``False`` (rather than raising) on missing/malformed
inputs so callers can simply reject the request when the function
returns falsy. Comparisons use :func:`hmac.compare_digest` to avoid
timing-side-channel leaks.
"""

from __future__ import annotations

import hashlib
import hmac
import time

__all__ = [
    "verify_line_signature",
    "verify_slack_signature",
]

# Default replay window for Slack: per Slack docs, reject if the request
# timestamp is more than 5 minutes off the local clock.
SLACK_REPLAY_WINDOW_SECONDS = 5 * 60


def verify_line_signature(
    channel_secret: str, body: bytes, signature_header: str | None
) -> bool:
    """Verify a LINE ``X-Line-Signature`` webhook header.

    LINE signs the raw request body with HMAC-SHA256 keyed by the
    channel secret, and base64-encodes the digest. The header value is
    the resulting base64 string verbatim (no prefix).

    Returns ``False`` for any missing/malformed input.
    """
    import base64

    if not channel_secret or not signature_header:
        return False
    expected = base64.b64encode(
        hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    ).decode("ascii")
    return hmac.compare_digest(signature_header, expected)


def verify_slack_signature(
    signing_secret: str,
    body: bytes,
    timestamp_header: str | None,
    signature_header: str | None,
    *,
    now: float | None = None,
    replay_window_seconds: int = SLACK_REPLAY_WINDOW_SECONDS,
) -> bool:
    """Verify a Slack ``X-Slack-Signature`` webhook header.

    Per `Slack's docs <https://api.slack.com/authentication/verifying-requests-from-slack>`_,
    the signature is computed as::

        v0=HMAC_SHA256(signing_secret, "v0:" + timestamp + ":" + body)

    where the result is the lowercase hex digest. Additionally, requests
    older than ``replay_window_seconds`` (default 5 minutes per Slack's
    recommendation) are rejected to defend against replay attacks.

    ``now`` is injectable for tests; defaults to :func:`time.time`.

    Returns ``False`` for any missing/malformed input or stale timestamp.
    """
    if not signing_secret or not timestamp_header or not signature_header:
        return False
    if not signature_header.startswith("v0="):
        return False
    try:
        ts_value = int(timestamp_header)
    except (TypeError, ValueError):
        return False

    current = time.time() if now is None else now
    if abs(current - ts_value) > replay_window_seconds:
        return False

    base = f"v0:{timestamp_header}:".encode() + body
    digest = hmac.new(signing_secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    expected = f"v0={digest}"
    return hmac.compare_digest(signature_header, expected)
