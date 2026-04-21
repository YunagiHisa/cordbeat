"""Tests for platform adapter webhook signature verification helpers."""

from __future__ import annotations

import hashlib
import hmac

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
