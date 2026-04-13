"""weather skill — fetch weather information for a location."""

from __future__ import annotations

from typing import Any

import httpx


async def execute(
    *,
    location: str,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Fetch current weather using wttr.in (no API key required)."""
    url = f"https://wttr.in/{_safe_location(location)}"
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
