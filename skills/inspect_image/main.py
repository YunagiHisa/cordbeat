"""inspect_image skill - securely fetch and normalize a public image."""

from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import socket
from typing import Any
from urllib.parse import urlsplit

import httpx
from PIL import Image

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_BLOCKED_HOSTS = frozenset(
    {"localhost", "localhost.localdomain", "metadata", "metadata.google.internal"}
)
_MAX_RAW_BYTES = 5_000_000
_MAX_DIMENSION = 1600
_MAX_NORMALIZED_BYTES = 600_000


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_and_check(host: str) -> tuple[str | None, list[tuple[str, int]]]:
    if host.lower() in _BLOCKED_HOSTS:
        return ("Requests to local/internal addresses are not allowed", [])
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None:
        if _is_blocked_ip(literal):
            return ("Requests to local/internal addresses are not allowed", [])
        return (None, [])
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return (f"Could not resolve host: {host!r}", [])
    resolved: list[tuple[str, int]] = []
    for family, _socktype, _proto, _canon, sockaddr in infos:
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])
        except ValueError:
            return (f"Host {host!r} resolved to an invalid address", [])
        if _is_blocked_ip(ip):
            return ("Requests to local/internal addresses are not allowed", [])
        resolved.append((addr, family))
    return (None, resolved)


async def execute(*, url: str, **_kwargs: Any) -> dict[str, Any]:
    """Fetch, validate, resize, and return an image as base64 JPEG."""

    if not isinstance(url, str) or not url.strip():
        return {"error": "url is required"}
    url = url.strip()
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES or not parts.hostname:
        return {"error": "A valid http:// or https:// image URL is required"}
    block_reason, resolved = _resolve_and_check(parts.hostname)
    if block_reason:
        return {"error": block_reason}

    request_url = url
    extensions: dict[str, Any] = {}
    headers = {
        "User-Agent": "CordBeat/1.0 (+https://github.com/YunagiHisa/cordbeat)",
        "Accept": "image/*",
    }
    if resolved:
        first_ip, family = resolved[0]
        port = parts.port or (443 if scheme == "https" else 80)
        headers["Host"] = parts.hostname
        host = f"[{first_ip}]" if family == socket.AF_INET6 else first_ip
        request_url = f"{scheme}://{host}:{port}{parts.path or '/'}"
        if parts.query:
            request_url += f"?{parts.query}"
        if scheme == "https":
            extensions["sni_hostname"] = parts.hostname

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
            response = await client.get(
                request_url,
                headers=headers,
                extensions=extensions,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return {"error": f"Image request failed: {exc}"}

    raw = response.content
    if len(raw) > _MAX_RAW_BYTES:
        return {"error": "Image is too large"}
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        image.thumbnail((_MAX_DIMENSION, _MAX_DIMENSION))
        if image.mode != "RGB":
            image = image.convert("RGB")
        normalized = b""
        quality = 85
        while True:
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            normalized = output.getvalue()
            if len(normalized) <= _MAX_NORMALIZED_BYTES:
                break
            if max(image.size) <= 320:
                return {"error": "Image could not be reduced below the output limit"}
            image.thumbnail(
                (max(320, int(image.width * 0.75)), max(320, int(image.height * 0.75)))
            )
            quality = max(60, quality - 5)
    except Exception as exc:
        return {"error": f"Unsupported or invalid image: {exc}"}

    return {
        "url": url,
        "mime_type": "image/jpeg",
        "width": image.width,
        "height": image.height,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "image_base64": base64.b64encode(normalized).decode("ascii"),
        "result": "Image fetched and attached for visual inspection.",
    }
