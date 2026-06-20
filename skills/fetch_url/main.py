"""fetch_url skill — download a URL and return readable text.

This is a *safe* skill (no user confirmation) intended for the AI to use
whenever it needs to actually read the contents of a page — for example
to summarize an article whose URL the user just pasted, or to follow a
link returned by ``web_search``. It deliberately does NOT accept search
queries: for search use the ``web_search`` skill.

SSRF hardening mirrors ``api_call``:

1. Scheme must be ``http`` or ``https``.
2. Hostname is resolved via ``socket.getaddrinfo`` (IPv4 + IPv6).
3. Any resolved address that is private / loopback / link-local /
   multicast / reserved or the cloud metadata endpoint is rejected.
4. Redirects are not followed automatically; the resolved IP is pinned
   into the request URL and the original hostname is preserved via the
   ``Host`` header to defeat DNS rebinding.

These pre-flight checks are intentionally duplicated with ``api_call`` (and
deliberately not shared via a ``cordbeat`` import): skills run self-contained
inside the subprocess sandbox, which restricts ``sys.path`` and forbids
importing the host package. The subprocess runner additionally enforces SSRF
centrally at ``connect()`` time (see ``_install_ssrf_guard``), so this block
is defense-in-depth and provides clear, early error messages — it is no
longer the only safeguard.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

_ALLOWED_SCHEMES = frozenset({"http", "https"})

_BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata",
        "metadata.google.internal",
    }
)

_BLOCKED_IPS = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),
        ipaddress.ip_address("fd00:ec2::254"),
    }
)

_DEFAULT_MAX_LENGTH = 8000
_HARD_MAX_LENGTH = 64_000
_MAX_RAW_BYTES = 2_000_000  # 2 MB ceiling on the HTTP response body
_MAX_IMAGE_CANDIDATES = 6


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip in _BLOCKED_IPS:
        return True
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
            if family == socket.AF_INET6:
                ip: ipaddress.IPv4Address | ipaddress.IPv6Address = (
                    ipaddress.IPv6Address(addr.split("%")[0])
                )
            else:
                ip = ipaddress.IPv4Address(addr)
        except ValueError:
            return (f"Host {host!r} resolved to an unparseable address: {addr}", [])
        if _is_blocked_ip(ip):
            return ("Requests to local/internal addresses are not allowed", [])
        resolved.append((addr, family))
    return (None, resolved)


_SCRIPT_STYLE_PATTERN = r"<(script|style|noscript|template)\b[^>]*>.*?</\1>"
_TAG_PATTERN = r"<[^>]+>"
_WHITESPACE_PATTERN = r"[ \t]+"
_BLANKLINES_PATTERN = r"\n\s*\n+"

_ENTITY_MAP = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
    "&apos;": "'",
    "&nbsp;": " ",
}


def _html_to_text(html: str) -> str:
    """Strip HTML tags and collapse whitespace into readable text."""
    text = re.sub(_SCRIPT_STYLE_PATTERN, "", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(_TAG_PATTERN, "", text)
    for entity, char in _ENTITY_MAP.items():
        text = text.replace(entity, char)
    text = re.sub(
        r"&#(\d+);",
        lambda m: chr(int(m.group(1))) if int(m.group(1)) < 0x110000 else "",
        text,
    )
    text = re.sub(
        r"&#x([0-9a-fA-F]+);",
        lambda m: chr(int(m.group(1), 16)) if int(m.group(1), 16) < 0x110000 else "",
        text,
    )
    text = re.sub(_WHITESPACE_PATTERN, " ", text)
    text = re.sub(_BLANKLINES_PATTERN, "\n\n", text)
    return text.strip()


def _extract_attr(tag: str, attr: str) -> str:
    match = re.search(
        rf"\b{re.escape(attr)}\s*=\s*(['\"])(.*?)\1",
        tag,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(2).strip() if match else ""


def _extract_image_candidates(html: str, base_url: str) -> list[dict[str, str]]:
    """Return a bounded list of useful-looking image references."""

    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in re.finditer(r"<img\b[^>]*>", html, flags=re.IGNORECASE | re.DOTALL):
        tag = match.group(0)
        src = _extract_attr(tag, "src") or _extract_attr(tag, "data-src")
        if not src or src.startswith(("data:", "javascript:")):
            continue
        resolved = urljoin(base_url, src)
        if not resolved.startswith(("http://", "https://")) or resolved in seen:
            continue
        alt = re.sub(r"\s+", " ", _extract_attr(tag, "alt"))[:300]
        width = _extract_attr(tag, "width")
        height = _extract_attr(tag, "height")
        if width.isdigit() and height.isdigit():
            if int(width) <= 32 and int(height) <= 32:
                continue
        seen.add(resolved)
        candidates.append({"url": resolved, "alt": alt})
        if len(candidates) >= _MAX_IMAGE_CANDIDATES:
            break
    return candidates


async def execute(
    *,
    url: str,
    max_length: int = _DEFAULT_MAX_LENGTH,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Fetch *url* and return its body as readable text."""
    if not isinstance(url, str) or not url.strip():
        return {"error": "url is required"}
    url = url.strip()

    try:
        cap = int(max_length)
    except (TypeError, ValueError):
        cap = _DEFAULT_MAX_LENGTH
    if cap <= 0:
        cap = _DEFAULT_MAX_LENGTH
    cap = min(cap, _HARD_MAX_LENGTH)

    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        return {"error": f"Unsupported URL scheme: {parts.scheme!r}"}
    if not parts.hostname:
        return {"error": f"Invalid URL: {url}"}

    block_reason, resolved = _resolve_and_check(parts.hostname)
    if block_reason is not None:
        return {"error": block_reason}

    request_url = url
    extensions: dict[str, Any] = {}
    headers: dict[str, str] = {
        "User-Agent": "CordBeat/1.0 (+https://github.com/YunagiHisa/cordbeat)",
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "text/plain;q=0.8,*/*;q=0.5"
        ),
        "Accept-Language": "ja,en;q=0.8",
    }
    if resolved:
        first_ip, family = resolved[0]
        port = parts.port or (443 if scheme == "https" else 80)
        if parts.hostname not in (first_ip, f"[{first_ip}]"):
            headers["Host"] = parts.hostname
            if family == socket.AF_INET6:
                netloc = f"[{first_ip}]:{port}"
            else:
                netloc = f"{first_ip}:{port}"
            request_url = f"{scheme}://{netloc}{parts.path or '/'}"
            if parts.query:
                request_url += f"?{parts.query}"
            if scheme == "https":
                extensions["sni_hostname"] = parts.hostname

    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=False,
        ) as client:
            resp = await client.get(
                request_url,
                headers=headers,
                extensions=extensions,
            )
    except httpx.HTTPError as exc:
        return {"error": f"Request failed: {exc}"}

    content_type = resp.headers.get("content-type", "").lower()
    raw = resp.content[:_MAX_RAW_BYTES]
    try:
        encoding = resp.encoding or "utf-8"
        body_text = raw.decode(encoding, errors="replace")
    except (LookupError, UnicodeDecodeError):
        body_text = raw.decode("utf-8", errors="replace")

    image_candidates: list[dict[str, str]] = []
    if "html" in content_type or body_text.lstrip().startswith("<"):
        image_candidates = _extract_image_candidates(body_text, url)
        text = _html_to_text(body_text)
    else:
        text = body_text

    truncated = False
    if len(text) > cap:
        text = text[:cap]
        truncated = True

    return {
        "url": url,
        "status_code": resp.status_code,
        "content_type": content_type or "unknown",
        "text": text,
        "length": len(text),
        "truncated": truncated,
        "images": image_candidates,
    }
