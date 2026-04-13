"""web_search skill — search the web and return results."""

from __future__ import annotations

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
        # Result links have class="result-link"
        if 'class="result-link"' in line:
            href = _extract_attr(line, "href")
            title = _strip_tags(line)
            # Snippet follows in a <td class="result-snippet"> element
            snippet = ""
            for j in range(i + 1, min(i + 10, len(lines))):
                if "result-snippet" in lines[j]:
                    snippet = _strip_tags(lines[j])
                    break
            if href and title:
                results.append({"title": title, "url": href, "snippet": snippet})
        i += 1

    return results


def _extract_attr(tag: str, attr: str) -> str:
    """Extract an attribute value from an HTML tag string."""
    key = f'{attr}="'
    start = tag.find(key)
    if start == -1:
        return ""
    start += len(key)
    end = tag.find('"', start)
    return tag[start:end] if end != -1 else ""


def _strip_tags(html: str) -> str:
    """Remove HTML tags and decode basic entities."""
    import re

    text = re.sub(r"<[^>]+>", "", html)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    return text.strip()
