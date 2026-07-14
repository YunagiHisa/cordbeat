"""web_search skill — search the web and return results."""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import Any

import httpx

_SEARCH_HOST = "lite.duckduckgo.com"


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the address is in a range we refuse to contact."""
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _preflight_blocked(host: str) -> str | None:
    """SSRF pre-flight: refuse if *host* resolves to an internal address.

    The runner's connect-time SSRF guard remains the enforcement point; this
    mirrors the api_call/fetch_url pre-flights so a poisoned resolution of the
    fixed search domain fails with a clear error. Resolution failures are not
    treated as blocking — the subsequent request fails on its own.
    """
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return None
    for *_unused, sockaddr in infos:
        try:
            ip = ipaddress.ip_address(str(sockaddr[0]).split("%")[0])
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            return f"Host {host!r} resolved to an internal address; refusing"
    return None


async def execute(
    *,
    query: str,
    max_results: int = 5,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Search the web using DuckDuckGo Lite and return results."""
    block_reason = _preflight_blocked(_SEARCH_HOST)
    if block_reason is not None:
        return {"error": block_reason, "results": []}

    url = f"https://{_SEARCH_HOST}/lite/"
    headers = {"User-Agent": "CordBeat/1.0"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, data={"q": query}, headers=headers)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        return {"error": f"Search request failed: {exc}", "results": []}

    results = _parse_results(resp.text, max_results)
    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


def _parse_results(html: str, max_results: int) -> list[dict[str, str]]:
    """Extract search results from DuckDuckGo Lite HTML response."""
    results: list[dict[str, str]] = []
    lines = html.split("\n")

    i = 0
    while i < len(lines) and len(results) < max_results:
        line = lines[i].strip()
        if "<a" in line and "result-link" in line:
            href = _extract_attr(line, "href")
            title = _strip_tags(line)
            snippet = _extract_snippet(lines, i + 1)
            if href and title:
                results.append({"title": title, "url": href, "snippet": snippet})
        i += 1

    return results


def _extract_attr(tag: str, attr: str) -> str:
    """Extract an attribute value from an HTML tag string."""
    match = re.search(rf"\b{re.escape(attr)}\s*=\s*(['\"])(.*?)\1", tag)
    return match.group(2) if match else ""


def _extract_snippet(lines: list[str], start: int) -> str:
    """Extract a possibly multiline result snippet following a result link."""
    for index in range(start, min(start + 20, len(lines))):
        if "result-snippet" not in lines[index]:
            continue
        snippet_lines = [lines[index]]
        while "</td>" not in snippet_lines[-1] and index + 1 < len(lines):
            index += 1
            snippet_lines.append(lines[index])
        return _strip_tags(" ".join(snippet_lines))
    return ""


def _strip_tags(html: str) -> str:
    """Remove HTML tags and decode basic entities."""
    text = re.sub(r"<[^>]+>", "", html)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    return text.strip()
