"""Tests for platform adapter webhook signature verification helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import time

from cordbeat.adapter_signing import (
    verify_line_signature,
    verify_slack_signature,
)
from cordbeat.whatsapp_adapter import verify_whatsapp_signature


def _sign(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


class TestVerifyWhatsAppSignature:
    def test_valid_signature(self) -> None:
        secret = "s3cret"
        body = b'{"entry": []}'
        header = _sign(secret, body)
        assert verify_whatsapp_signature(secret, body, header) is True

    def test_tampered_body_rejected(self) -> None:
        secret = "s3cret"
        header = _sign(secret, b"original")
        assert verify_whatsapp_signature(secret, b"tampered", header) is False

    def test_wrong_secret_rejected(self) -> None:
        body = b'{"entry": []}'
        header = _sign("right", body)
        assert verify_whatsapp_signature("wrong", body, header) is False

    def test_missing_prefix_rejected(self) -> None:
        secret = "s3cret"
        body = b""
        bad = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert verify_whatsapp_signature(secret, body, bad) is False

    def test_missing_header_rejected(self) -> None:
        assert verify_whatsapp_signature("s3cret", b"x", None) is False
        assert verify_whatsapp_signature("s3cret", b"x", "") is False

    def test_empty_secret_rejected(self) -> None:
        body = b"x"
        header = _sign("s3cret", body)
        assert verify_whatsapp_signature("", body, header) is False

    def test_binary_body(self) -> None:
        secret = "k"
        body = bytes(range(256))
        assert verify_whatsapp_signature(secret, body, _sign(secret, body)) is True


def _line_sign(secret: str, body: bytes) -> str:
    return base64.b64encode(
        hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    ).decode("ascii")


def _slack_sign(secret: str, ts: str, body: bytes) -> str:
    base = f"v0:{ts}:".encode() + body
    digest = hmac.new(secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    return f"v0={digest}"


class TestVerifyLineSignature:
    def test_valid(self) -> None:
        secret = "ch_secret"
        body = b'{"events":[]}'
        assert verify_line_signature(secret, body, _line_sign(secret, body)) is True

    def test_tampered_body_rejected(self) -> None:
        secret = "s"
        sig = _line_sign(secret, b"original")
        assert verify_line_signature(secret, b"tampered", sig) is False

    def test_wrong_secret_rejected(self) -> None:
        body = b"x"
        sig = _line_sign("right", body)
        assert verify_line_signature("wrong", body, sig) is False

    def test_missing_inputs(self) -> None:
        assert verify_line_signature("", b"x", "abc") is False
        assert verify_line_signature("s", b"x", None) is False
        assert verify_line_signature("s", b"x", "") is False


class TestVerifySlackSignature:
    def test_valid(self) -> None:
        secret = "sgn"
        ts = "1700000000"
        body = b'{"type":"event_callback"}'
        sig = _slack_sign(secret, ts, body)
        assert verify_slack_signature(secret, body, ts, sig, now=float(ts)) is True

    def test_replay_too_old_rejected(self) -> None:
        secret = "sgn"
        ts = "1700000000"
        body = b""
        sig = _slack_sign(secret, ts, body)
        # 10 minutes later — outside default 5 min window
        future = float(ts) + 10 * 60
        assert verify_slack_signature(secret, body, ts, sig, now=future) is False

    def test_replay_future_skew_rejected(self) -> None:
        secret = "sgn"
        ts = "1700000000"
        body = b""
        sig = _slack_sign(secret, ts, body)
        past = float(ts) - 10 * 60
        assert verify_slack_signature(secret, body, ts, sig, now=past) is False

    def test_tampered_body_rejected(self) -> None:
        secret = "sgn"
        ts = "1700000000"
        sig = _slack_sign(secret, ts, b"original")
        assert (
            verify_slack_signature(secret, b"tampered", ts, sig, now=float(ts)) is False
        )

    def test_missing_prefix_rejected(self) -> None:
        secret = "sgn"
        ts = "1700000000"
        body = b""
        bad = hmac.new(
            secret.encode(), f"v0:{ts}:".encode() + body, hashlib.sha256
        ).hexdigest()
        # Missing "v0=" prefix
        assert verify_slack_signature(secret, body, ts, bad, now=float(ts)) is False

    def test_bad_timestamp_rejected(self) -> None:
        assert (
            verify_slack_signature("sgn", b"", "not-a-number", "v0=abc", now=0.0)
            is False
        )

    def test_missing_inputs(self) -> None:
        assert verify_slack_signature("", b"", "1", "v0=abc") is False
        assert verify_slack_signature("sgn", b"", None, "v0=abc") is False
        assert verify_slack_signature("sgn", b"", "1", None) is False

    def test_uses_real_clock_when_now_omitted(self) -> None:
        secret = "sgn"
        ts = str(int(time.time()))
        body = b"x"
        sig = _slack_sign(secret, ts, body)
        assert verify_slack_signature(secret, body, ts, sig) is True
