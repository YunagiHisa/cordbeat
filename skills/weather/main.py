"""weather skill — fetch weather information for a location."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

import httpx

_WEATHER_HOST = "wttr.in"


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
    fixed weather domain fails with a clear error. Resolution failures are not
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
    location: str,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Fetch current weather using wttr.in (no API key required)."""
    block_reason = _preflight_blocked(_WEATHER_HOST)
    if block_reason is not None:
        return {"error": block_reason}

    url = f"https://{_WEATHER_HOST}/{_safe_location(location)}"
    params = {"format": "j1"}
    headers = {"User-Agent": "CordBeat/1.0"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        return {"error": f"Weather request failed: {exc}"}
    except (ValueError, KeyError):
        return {"error": "Failed to parse weather response"}

    return _format_weather(location, data)


def _safe_location(location: str) -> str:
    """Sanitise location string for URL path segment."""
    import re

    return re.sub(r"[^\w\s,.-]", "", location).strip()


def _format_weather(location: str, data: dict[str, Any]) -> dict[str, Any]:
    """Extract relevant fields from wttr.in JSON response."""
    try:
        current = data["current_condition"][0]
        result: dict[str, Any] = {
            "location": location,
            "temperature_c": current.get("temp_C", ""),
            "feels_like_c": current.get("FeelsLikeC", ""),
            "humidity": current.get("humidity", ""),
            "description": current.get("weatherDesc", [{}])[0].get("value", ""),
            "wind_speed_kmh": current.get("windspeedKmph", ""),
            "wind_direction": current.get("winddir16Point", ""),
        }

        forecasts = []
        for day in data.get("weather", [])[:3]:
            forecasts.append(
                {
                    "date": day.get("date", ""),
                    "max_c": day.get("maxtempC", ""),
                    "min_c": day.get("mintempC", ""),
                    "description": day.get("hourly", [{}])[4]
                    .get("weatherDesc", [{}])[0]
                    .get("value", ""),
                }
            )
        result["forecast"] = forecasts
        return result
    except (KeyError, IndexError):
        return {"location": location, "error": "Unexpected response format"}
