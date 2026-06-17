"""web_search skill — search the web and return results."""

from __future__ import annotations

import re
from typing import Any

import httpx


async def execute(
    *,
    query: str,
    max_results: int = 5,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Search the web using DuckDuckGo Lite and return results."""
    url = "https://lite.duckduckgo.com/lite/"
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
