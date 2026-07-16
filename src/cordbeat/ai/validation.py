"""AI output validation layer — 3-layer defense."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from cordbeat.exceptions import OutputValidationError
from cordbeat.models import ValidationError, ValidationResult

from .backend import AIBackend

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


# ── Validators ────────────────────────────────────────────────────────


def validate_heartbeat_decision(data: dict[str, Any]) -> ValidationResult:
    """Validate HEARTBEAT decision JSON from AI."""
    errors: list[ValidationError] = []
    valid_actions = {
        "message",
        "skill",
        "propose_improvement",
        "propose_trait_change",
        "propose_skill",
        "none",
    }

    action = data.get("action")
    if action not in valid_actions:
        errors.append(
            ValidationError(
                field="action",
                message=f"must be one of {valid_actions}, got '{action}'",
                value=action,
            )
        )

    content = data.get("content", "")
    requires_content = (
        "message",
        "propose_improvement",
        "propose_trait_change",
    )
    if action in requires_content and not content:
        errors.append(
            ValidationError(
                field="content",
                message=(
                    "content is required when action is message, "
                    "propose_improvement, or propose_trait_change"
                ),
            )
        )

    target_adapter_id = data.get("target_adapter_id")
    if target_adapter_id is not None and not isinstance(target_adapter_id, str):
        errors.append(
            ValidationError(
                field="target_adapter_id",
                message="must be a string when provided",
                value=target_adapter_id,
            )
        )

    minutes = data.get("next_heartbeat_minutes")
    if minutes is not None and not _is_clampable_interval(minutes):
        errors.append(
            ValidationError(
                field="next_heartbeat_minutes",
                message=_INTERVAL_MESSAGE,
                value=minutes,
            )
        )

    return ValidationResult(valid=len(errors) == 0, errors=errors)


# The heartbeat loop clamps intervals to the configured min/max bounds (as the
# decision prompt promises), so validation only rejects values that cannot be
# clamped meaningfully. Rejecting large values here forced pointless retries
# whenever the model reasonably asked for a multi-day pause.
_INTERVAL_MESSAGE = (
    "must be a finite number of minutes >= 1 "
    "(values above the configured maximum are clamped)"
)


def _is_clampable_interval(minutes: Any) -> bool:
    if isinstance(minutes, bool) or not isinstance(minutes, int | float):
        return False
    return math.isfinite(float(minutes)) and minutes >= 1


def validate_heartbeat_triage(data: dict[str, Any]) -> ValidationResult:
    """Validate HEARTBEAT Layer 1 triage JSON from AI.

    Expected format::

        {
            "users": [
                {"user_id": "...", "reason": "..."},
                ...
            ],
            "next_heartbeat_minutes": 60
        }
    """
    errors: list[ValidationError] = []

    users = data.get("users")
    if not isinstance(users, list):
        errors.append(
            ValidationError(
                field="users",
                message="must be a list",
                value=users,
            )
        )
    else:
        for i, entry in enumerate(users):
            if not isinstance(entry, dict):
                errors.append(
                    ValidationError(
                        field=f"users[{i}]",
                        message="must be a dict",
                        value=entry,
                    )
                )
                continue
            if not entry.get("user_id"):
                errors.append(
                    ValidationError(
                        field=f"users[{i}].user_id",
                        message="user_id is required",
                    )
                )

    minutes = data.get("next_heartbeat_minutes")
    if minutes is not None and not _is_clampable_interval(minutes):
        errors.append(
            ValidationError(
                field="next_heartbeat_minutes",
                message=_INTERVAL_MESSAGE,
                value=minutes,
            )
        )

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_user_summary_update(data: dict[str, Any]) -> ValidationResult:
    """Validate AI-generated user summary updates."""
    errors: list[ValidationError] = []

    score = data.get("attention_score")
    if score is not None:
        if not isinstance(score, int | float) or score < 0 or score > 1:
            errors.append(
                ValidationError(
                    field="attention_score",
                    message="must be between 0 and 1",
                    value=score,
                )
            )

    topic = data.get("last_topic", "")
    if not isinstance(topic, str):
        errors.append(
            ValidationError(
                field="last_topic",
                message="must be a string",
                value=topic,
            )
        )
    elif len(topic) > 50:
        errors.append(
            ValidationError(
                field="last_topic",
                message=f"too long ({len(topic)} chars, max 50)",
                value=topic,
            )
        )

    tone = data.get("emotional_tone", "")
    if not isinstance(tone, str):
        errors.append(
            ValidationError(
                field="emotional_tone",
                message="must be a string",
                value=tone,
            )
        )
    elif len(tone) > 50:
        errors.append(
            ValidationError(
                field="emotional_tone",
                message=f"too long ({len(tone)} chars, max 50)",
                value=tone,
            )
        )

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_skill_selection(
    data: dict[str, Any],
    available_skills: set[str],
) -> ValidationResult:
    """Validate AI's skill selection."""
    errors: list[ValidationError] = []

    skill_name = data.get("skill_name")
    if skill_name and skill_name not in available_skills:
        errors.append(
            ValidationError(
                field="skill_name",
                message=f"unknown skill '{skill_name}'",
                value=skill_name,
            )
        )

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_soul_update(data: dict[str, Any]) -> ValidationResult:
    """Validate AI-proposed SOUL file updates."""
    errors: list[ValidationError] = []

    # AI must not touch immutable_rules
    if "immutable_rules" in data:
        errors.append(
            ValidationError(
                field="immutable_rules",
                message="immutable_rules cannot be modified",
            )
        )

    emotion = data.get("current_emotion", {})
    if emotion and not isinstance(emotion, dict):
        errors.append(
            ValidationError(
                field="current_emotion",
                message="must be an object",
                value=emotion,
            )
        )
    elif emotion:
        intensity = emotion.get("primary_intensity")
        if intensity is not None and (
            not isinstance(intensity, int | float)
            or intensity < 0
            or intensity > 1
        ):
            errors.append(
                ValidationError(
                    field="current_emotion.primary_intensity",
                    message="must be between 0 and 1",
                    value=intensity,
                )
            )

    return ValidationResult(valid=len(errors) == 0, errors=errors)


# ── Retry wrapper ─────────────────────────────────────────────────────


async def validated_ai_json(
    backend: AIBackend,
    prompt: str,
    system: str,
    validator: Any,
    fallback: dict[str, Any] | None = None,
    **validator_kwargs: Any,
) -> dict[str, Any]:
    """Generate AI JSON output with validation and retry.

    1st attempt → validate → if NG, include errors in retry prompt
    Max 2 retries. On final failure, return fallback or raise.
    """
    last_errors = ""
    for attempt in range(1 + MAX_RETRIES):
        retry_prompt = prompt
        if last_errors:
            retry_prompt += (
                "\n\nYour previous output failed validation:\n"
                f"{last_errors}\nPlease fix and regenerate."
            )

        try:
            data = await backend.generate_json(retry_prompt, system=system)
        except (json.JSONDecodeError, KeyError) as exc:
            last_errors = f"JSON parse error: {exc}"
            logger.warning(
                "AI JSON parse failed (attempt %d): %s",
                attempt + 1,
                exc,
            )
            continue

        try:
            result: ValidationResult = validator(data, **validator_kwargs)
        except (TypeError, AttributeError, KeyError) as exc:
            last_errors = f"Validation failed (malformed types): {exc}"
            logger.warning(
                "AI output validator crashed (attempt %d): %s",
                attempt + 1,
                exc,
            )
            continue
        if result.valid:
            return data

        last_errors = result.error_summary
        logger.warning(
            "AI output validation failed (attempt %d):\n%s",
            attempt + 1,
            last_errors,
        )

    logger.error("AI output validation failed after %d attempts", 1 + MAX_RETRIES)
    if fallback is not None:
        return fallback
    msg = f"Validation failed after retries: {last_errors}"
    raise OutputValidationError(msg)
