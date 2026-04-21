"""api_call skill — make HTTP requests to external APIs.

SSRF hardening
--------------

Before sending any request we:

1. Require the scheme to be ``http`` or ``https``.
2. Resolve the hostname via ``socket.getaddrinfo`` (covers IPv4 + IPv6).
3. Reject if *any* resolved address is private, loopback, link-local,
   multicast, reserved, or the cloud metadata endpoint
   (``169.254.169.254`` / ``fd00:ec2::254``).
4. Disable redirect-following; the skill caller must explicitly re-issue
   the request for any redirect, which will go through the same checks.

This is intentionally conservative; cloud VM deployments in particular
must *not* be able to reach their metadata service.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urlsplit

import httpx

_ALLOWED_METHODS = frozenset({"GET", "POST"})
_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Additional explicit host blocklist on top of the IP-range checks below.
# Useful for names we can't always detect via getaddrinfo (e.g. on
# development machines where /etc/hosts may be customised).
_BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata",
        "metadata.google.internal",
    }
)

# Cloud provider metadata IPs — rejected regardless of DNS resolution.
_BLOCKED_IPS = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),  # AWS / Azure / GCP
        ipaddress.ip_address("fd00:ec2::254"),  # AWS IMDSv2 over IPv6
    }
)

_MAX_RESPONSE_BYTES = 32_768  # 32 KB cap on response body


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the address is in a range we refuse to contact."""
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
    """Resolve *host* and ensure no returned IP is internal.

    Returns a tuple of (error_message, resolved_addresses).
    error_message is None if safe; resolved_addresses are (ip_str, family)
    pairs that can be used to pin the connection.
    """
    if host.lower() in _BLOCKED_HOSTS:
        return ("Requests to local/internal addresses are not allowed", [])

    # If the host is a literal IP, validate directly.
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None:
        if _is_blocked_ip(literal):
            return ("Requests to local/internal addresses are not allowed", [])
        return (None, [])

    # Hostname: resolve both IPv4 and IPv6 and check every candidate.
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


async def execute(
    *,
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 15,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Send an HTTP request to an external API and return the response."""
    method = method.upper()
    if method not in _ALLOWED_METHODS:
        return {"error": f"Unsupported method: {method}. Use GET or POST."}

    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        return {"error": f"Unsupported URL scheme: {parts.scheme!r}"}

    if not parts.hostname:
        return {"error": f"Invalid URL: {url}"}

    block_reason, resolved = _resolve_and_check(parts.hostname)
    if block_reason is not None:
        return {"error": block_reason}

    req_headers = {"User-Agent": "CordBeat/1.0"}
    if headers:
        req_headers.update(headers)

    # DNS-rebinding mitigation: connect directly to the address we already
    # verified in ``_resolve_and_check``. We rewrite the URL to the resolved
    # IP and preserve the original hostname via the Host header so virtual
    # hosting (and, for HTTPS, SNI-less servers) still work. This is the
    # single source of truth for pinning — we intentionally do *not* pass
    # a custom httpx transport: ``local_address`` only binds the source
    # address, it does not pin the destination.
    if resolved:
        first_ip, family = resolved[0]
        port = parts.port or (443 if scheme == "https" else 80)
        if parts.hostname not in (first_ip, f"[{first_ip}]"):
            req_headers.setdefault("Host", parts.hostname)
            if family == socket.AF_INET6:
                netloc = f"[{first_ip}]:{port}"
            else:
                netloc = f"{first_ip}:{port}"
            url = f"{scheme}://{netloc}{parts.path}"
            if parts.query:
                url += f"?{parts.query}"

    try:
        async with httpx.AsyncClient(
            timeout=float(timeout),
            follow_redirects=False,
        ) as client:
            if method == "POST":
                resp = await client.post(url, json=body, headers=req_headers)
            else:
                resp = await client.get(url, headers=req_headers)
    except httpx.HTTPError as exc:
        return {"error": f"Request failed: {exc}"}

    raw_body = resp.text[:_MAX_RESPONSE_BYTES]

    response_data: Any
    try:
        response_data = resp.json()
    except (ValueError, TypeError):
        response_data = raw_body

    return {
        "status_code": resp.status_code,
        "headers": dict(resp.headers),
        "body": response_data,
    }
