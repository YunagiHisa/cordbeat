"""api_call skill — make HTTP requests to external APIs."""

from __future__ import annotations

from typing import Any

import httpx

_ALLOWED_METHODS = {"GET", "POST"}

# Block requests to private/internal networks
_BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

_MAX_RESPONSE_BYTES = 32_768  # 32 KB cap on response body


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

    # Validate URL — block internal/private addresses
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        return {"error": f"Invalid URL: {url}"}

    if parsed.host in _BLOCKED_HOSTS:
        return {"error": "Requests to local/internal addresses are not allowed"}

    req_headers = {"User-Agent": "CordBeat/1.0"}
    if headers:
        req_headers.update(headers)

    try:
        async with httpx.AsyncClient(timeout=float(timeout)) as client:
            if method == "POST":
                resp = await client.post(url, json=body, headers=req_headers)
            else:
                resp = await client.get(url, headers=req_headers)
    except httpx.HTTPError as exc:
        return {"error": f"Request failed: {exc}"}

    # Truncate large responses
    raw_body = resp.text[:_MAX_RESPONSE_BYTES]

    # Try to parse as JSON
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
