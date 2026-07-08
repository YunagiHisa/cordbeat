"""Proposal storage, execution, and notification for the HEARTBEAT loop."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from cordbeat.core.gateway import GatewayServer
from cordbeat.memory.core import MemoryStore
from cordbeat.models import (
    GatewayMessage,
    HeartbeatDecision,
    MessageType,
    ProposalStatus,
    ProposalType,
    SoulCaller,
)
from cordbeat.skills.policy import (
    apply_default_skill_settings,
    delete_skill,
    delete_skill_file,
    sandbox_overrides_for_skill,
    update_skill_file,
    update_skill_settings,
)
from cordbeat.skills.registry import SkillRegistry
from cordbeat.skills.validator import SkillValidationError, validate_skill_source

from .soul import Soul

logger = logging.getLogger(__name__)

_AI_GENERATED_AUTHOR = "cordbeat-ai"
_UPDATE_SKILL_FILE_TOOL_NAME = "update_skill_file"
_DELETE_SKILL_FILE_TOOL_NAME = "delete_skill_file"
_DELETE_SKILL_TOOL_NAME = "delete_skill"
_UPDATE_SKILL_SETTINGS_TOOL_NAME = "update_skill_settings"
_PROPOSED_SKILL_PARAM_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,49}")
_PROPOSED_SKILL_PARAM_TYPES = frozenset(
    {"string", "number", "integer", "boolean"}
)


def _can_update_ai_skill(skills_dir: Path, name: str) -> bool:
    """Only AI-generated sandbox-local skills may be overwritten."""

    yaml_path = skills_dir / name / "skill.yaml"
    if not yaml_path.exists():
        return False
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        logger.warning("Could not inspect existing skill metadata: %s", name)
        return False
    if not isinstance(raw, dict):
        return False
    raw = apply_default_skill_settings(raw)
    safety = raw.get("safety") or {}
    if not isinstance(safety, dict):
        safety = {}
    return (
        raw.get("ownership") == "ai"
        and raw.get("mutable_by_ai") is True
        and raw.get("requires_approval_to_modify") is False
        and safety.get("level", "safe") == "safe"
        and safety.get("sandbox") is True
        and safety.get("network") is not True
        and safety.get("filesystem") is not True
    )


def _load_proposal_metadata(proposal: dict[str, Any]) -> dict[str, Any] | None:
    proposal_id = str(proposal.get("id", "<unknown>"))
    try:
        meta = json.loads(proposal.get("metadata") or "{}")
    except (TypeError, json.JSONDecodeError):
        logger.warning("Proposal %s has invalid metadata JSON", proposal_id)
        return None
    if not isinstance(meta, dict):
        logger.warning("Proposal %s metadata is not an object", proposal_id)
        return None
    return meta


def _format_skill_result_summary(result: Any) -> str:
    if isinstance(result, dict):
        summary_value = result.get("result", result.get("output"))
        if summary_value is not None:
            return str(summary_value)[:500]
        try:
            return json.dumps(result, ensure_ascii=False, default=str)[:500]
        except TypeError:
            return str(result)[:500]
    return str(result)[:500]


def validate_proposed_skill(
    proposed: dict[str, Any],
    skills: SkillRegistry,
) -> None:
    """Preflight an AI-generated skill before asking the user to approve it."""

    name = proposed.get("name", "")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,49}", name):
        raise ValueError(
            f"Invalid skill name: {name!r}. "
            "Must be lowercase alphanumeric with underscores."
        )

    skill_dir = skills.skills_dir / name
    if skill_dir.exists() and not _can_update_ai_skill(skills.skills_dir, name):
        raise ValueError(f"Skill '{name}' already exists.")

    code = proposed.get("code", "")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Skill code is empty.")

    description = proposed.get("description", "")
    if description is not None and (
        not isinstance(description, str) or len(description) > 500
    ):
        raise ValueError("Skill description must be a string up to 500 characters.")

    usage = proposed.get("usage", "")
    if usage is not None and (not isinstance(usage, str) or len(usage) > 2000):
        raise ValueError("Skill usage must be a string up to 2000 characters.")

    parameters = proposed.get("parameters", [])
    if parameters is None:
        parameters = []
    if not isinstance(parameters, list):
        raise ValueError("Skill parameters must be a list.")
    for index, param in enumerate(parameters, start=1):
        if not isinstance(param, dict):
            raise ValueError(f"Skill parameter #{index} must be a mapping.")
        param_name = param.get("name")
        if (
            not isinstance(param_name, str)
            or not _PROPOSED_SKILL_PARAM_NAME_RE.fullmatch(param_name)
        ):
            raise ValueError(f"Invalid skill parameter name: {param_name!r}.")
        param_type = param.get("type", "string")
        if (
            not isinstance(param_type, str)
            or param_type not in _PROPOSED_SKILL_PARAM_TYPES
        ):
            raise ValueError(f"Invalid skill parameter type: {param_type!r}.")
        required = param.get("required", True)
        if not isinstance(required, bool):
            raise ValueError(
                f"Skill parameter {param_name!r} required must be boolean."
            )
        param_description = param.get("description", "")
        if param_description is not None and (
            not isinstance(param_description, str) or len(param_description) > 500
        ):
            raise ValueError(
                f"Skill parameter {param_name!r} description must be a string "
                "up to 500 characters."
            )

    try:
        validate_skill_source(code, name)
    except SkillValidationError as exc:
        raise ValueError(str(exc)) from exc


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _metadata_matches_pending(
    metadata: dict[str, Any],
    *,
    proposal_type: str,
    skill_name: str | None = None,
    skill_params: dict[str, Any] | None = None,
    proposed_skill: dict[str, Any] | None = None,
    trait_add: list[str] | None = None,
    trait_remove: list[str] | None = None,
) -> bool:
    if metadata.get("proposal_type") != proposal_type:
        return False
    if metadata.get("status") != ProposalStatus.PENDING:
        return False
    if skill_name is not None and metadata.get("skill_name") != skill_name:
        return False
    if skill_params is not None and _stable_json(
        metadata.get("skill_params") or {}
    ) != _stable_json(skill_params):
        return False
    if proposed_skill is not None and _stable_json(
        metadata.get("proposed_skill") or {}
    ) != _stable_json(proposed_skill):
        return False
    if trait_add is not None and sorted(
        metadata.get("trait_add") or []
    ) != sorted(trait_add):
        return False
    if trait_remove is not None and sorted(
        metadata.get("trait_remove") or []
    ) != sorted(trait_remove):
        return False
    return True


async def find_duplicate_pending_proposal(
    memory: MemoryStore,
    *,
    user_id: str,
    proposal_type: str,
    skill_name: str | None = None,
    skill_params: dict[str, Any] | None = None,
    proposed_skill: dict[str, Any] | None = None,
    trait_add: list[str] | None = None,
    trait_remove: list[str] | None = None,
) -> dict[str, Any] | None:
    """Return an existing equivalent pending proposal, if one exists."""
    proposals = await memory.get_pending_proposals(
        user_id=user_id,
        status=ProposalStatus.PENDING,
    )
    for proposal in proposals:
        try:
            metadata = json.loads(proposal.get("metadata") or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if _metadata_matches_pending(
            metadata,
            proposal_type=proposal_type,
            skill_name=skill_name,
            skill_params=skill_params,
            proposed_skill=proposed_skill,
            trait_add=trait_add,
            trait_remove=trait_remove,
        ):
            return proposal
    return None


class ProposalExecutor:
    """Handles proposal storage, user notification, and execution."""

    def __init__(
        self,
        memory: MemoryStore,
        skills: SkillRegistry,
        gateway: GatewayServer,
        soul: Soul,
        adapters_options: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._memory = memory
        self._skills = skills
        self._gateway = gateway
        self._soul = soul
        self._adapters_options = adapters_options or {}

    async def _notification_metadata(
        self,
        user_id: str,
        adapter_id: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        opts = self._adapters_options.get(adapter_id, {})
        if str(opts.get("dm_policy", "reply_only")).lower() == "never":
            logger.info(
                "Proposal notification skipped (dm_policy=never) user=%s adapter=%s",
                user_id,
                adapter_id,
            )
            return None
        metadata: dict[str, Any] = dict(extra or {})
        last_seen = await self._memory.get_last_seen_channel(user_id, adapter_id)
        if last_seen is not None:
            channel_id, is_dm = last_seen
            metadata["channel_id"] = channel_id
            metadata["is_dm"] = is_dm
            metadata["allow_dm_fallback"] = False
        else:
            metadata["allow_dm_fallback"] = True
        return metadata

    async def _send_notification(
        self,
        *,
        user_id: str,
        adapter_id: str,
        platform_user_id: str,
        content: str,
        message_type: MessageType = MessageType.HEARTBEAT_MESSAGE,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        routed_metadata = await self._notification_metadata(
            user_id,
            adapter_id,
            metadata,
        )
        if routed_metadata is None:
            return False
        notification = GatewayMessage(
            type=message_type,
            adapter_id=adapter_id,
            platform_user_id=platform_user_id,
            content=content,
            metadata=routed_metadata,
        )
        await self._gateway.send_to_adapter(adapter_id, notification)
        return True

    async def store_and_notify(
        self,
        decision: HeartbeatDecision,
        proposal_type: str = ProposalType.GENERAL,
    ) -> str:
        """Store an improvement proposal and notify the target user."""
        user_id = decision.target_user_id
        adapter_id = decision.target_adapter_id
        content = decision.content

        metadata: dict[str, Any] = {
            "status": ProposalStatus.PENDING,
            "proposal_type": proposal_type,
        }
        if adapter_id:
            metadata["adapter_id"] = adapter_id

        proposal_id = await self._memory.add_certain_record(
            user_id=user_id or "__system__",
            content=content,
            record_type="proposal",
            metadata=metadata,
        )
        logger.info("Improvement proposal stored (id=%s): %s", proposal_id, content)

        if user_id and adapter_id:
            platform_user_id = await self._memory.resolve_platform_user(
                user_id, adapter_id
            )
            if platform_user_id:
                soul_snap = self._soul.get_soul_snapshot()
                sent = await self._send_notification(
                    user_id=user_id,
                    adapter_id=adapter_id,
                    platform_user_id=platform_user_id,
                    content=(
                        f"💡 {soul_snap['name']} has a suggestion:\n\n{content}"
                        f"\n\n(proposal ID: {proposal_id})"
                    ),
                )
                if sent:
                    logger.info(
                        "Proposal notification sent to %s via %s",
                        user_id,
                        adapter_id,
                    )
        return proposal_id

    async def store_skill_proposal(
        self,
        decision: HeartbeatDecision,
        skill_name: str,
    ) -> str:
        """Store a skill execution proposal for user confirmation."""
        metadata: dict[str, Any] = {
            "status": ProposalStatus.PENDING,
            "proposal_type": ProposalType.SKILL_EXECUTION,
            "skill_name": skill_name,
            "skill_params": decision.skill_params,
        }
        user_id = decision.target_user_id or "__system__"
        adapter_id = decision.target_adapter_id

        if adapter_id:
            metadata["adapter_id"] = adapter_id

        content = (
            f"Skill '{skill_name}' requires confirmation.\n"
            f"Parameters: {json.dumps(decision.skill_params)}"
        )
        duplicate = await find_duplicate_pending_proposal(
            self._memory,
            user_id=user_id,
            proposal_type=ProposalType.SKILL_EXECUTION,
            skill_name=skill_name,
            skill_params=decision.skill_params,
        )
        if duplicate is not None:
            proposal_id = str(duplicate["id"])
            logger.info(
                "Reusing pending skill proposal id=%s skill=%s",
                proposal_id,
                skill_name,
            )
            return proposal_id

        proposal_id = await self._memory.add_certain_record(
            user_id=user_id,
            content=content,
            record_type="proposal",
            metadata=metadata,
        )
        logger.info(
            "Skill proposal stored (id=%s): %s with params %s",
            proposal_id,
            skill_name,
            decision.skill_params,
        )

        if user_id != "__system__" and adapter_id:
            platform_user_id = await self._memory.resolve_platform_user(
                user_id, adapter_id
            )
            if platform_user_id:
                soul_snap = self._soul.get_soul_snapshot()
                await self._send_notification(
                    user_id=user_id,
                    adapter_id=adapter_id,
                    platform_user_id=platform_user_id,
                    message_type=MessageType.SKILL_CONFIRM,
                    content=(
                        f"🔧 {soul_snap['name']} wants to run "
                        f"skill '{skill_name}'.\n"
                        f"Parameters: {json.dumps(decision.skill_params)}\n\n"
                        f"(proposal ID: {proposal_id})"
                    ),
                    metadata={
                        "proposal_id": proposal_id,
                        "skill_name": skill_name,
                        "skill_params": decision.skill_params,
                    },
                )

        return proposal_id

    async def store_trait_proposal(
        self,
        decision: HeartbeatDecision,
    ) -> str:
        """Store a trait change proposal for user approval."""
        add = decision.trait_add
        remove = decision.trait_remove

        preview = self._soul.propose_trait_change(add=add, remove=remove)

        metadata: dict[str, Any] = {
            "status": ProposalStatus.PENDING,
            "proposal_type": ProposalType.TRAIT_CHANGE,
            "trait_add": add,
            "trait_remove": remove,
            "trait_preview": preview["preview"],
        }
        user_id = decision.target_user_id or "__system__"
        adapter_id = decision.target_adapter_id

        if adapter_id:
            metadata["adapter_id"] = adapter_id

        content = decision.content or (
            f"Trait change proposal: add {add}, remove {remove}"
        )

        duplicate = await find_duplicate_pending_proposal(
            self._memory,
            user_id=user_id,
            proposal_type=ProposalType.TRAIT_CHANGE,
            trait_add=add,
            trait_remove=remove,
        )
        if duplicate is not None:
            proposal_id = str(duplicate["id"])
            logger.info(
                "Reusing pending trait proposal id=%s add=%s remove=%s",
                proposal_id,
                add,
                remove,
            )
            return proposal_id

        proposal_id = await self._memory.add_certain_record(
            user_id=user_id,
            content=content,
            record_type="proposal",
            metadata=metadata,
        )
        logger.info(
            "Trait proposal stored (id=%s): add=%s remove=%s preview=%s",
            proposal_id,
            add,
            remove,
            preview["preview"],
        )

        if user_id != "__system__" and adapter_id:
            platform_user_id = await self._memory.resolve_platform_user(
                user_id, adapter_id
            )
            if platform_user_id:
                soul_snap = self._soul.get_soul_snapshot()
                traits_display = ", ".join(preview["preview"])
                await self._send_notification(
                    user_id=user_id,
                    adapter_id=adapter_id,
                    platform_user_id=platform_user_id,
                    content=(
                        f"🎭 {soul_snap['name']} wants to change "
                        f"personality traits.\n"
                        f"{content}\n\n"
                        f"Preview: [{traits_display}]\n\n"
                        f"(proposal ID: {proposal_id})"
                    ),
                )

        return proposal_id

    async def store_skill_creation_proposal(
        self,
        decision: HeartbeatDecision,
    ) -> str:
        """Store a skill creation proposal for user approval."""
        proposed = decision.proposed_skill
        skill_name = proposed.get("name", "unnamed_skill")
        try:
            validate_proposed_skill(proposed, self._skills)
        except ValueError as exc:
            logger.warning(
                "Rejected proposed skill %r before approval: %s",
                skill_name,
                exc,
            )
            return ""

        metadata: dict[str, Any] = {
            "status": ProposalStatus.PENDING,
            "proposal_type": ProposalType.SKILL_PROPOSAL,
            "proposed_skill": proposed,
        }
        user_id = decision.target_user_id or "__system__"
        adapter_id = decision.target_adapter_id

        if adapter_id:
            metadata["adapter_id"] = adapter_id

        content = decision.content or (
            f"New skill proposal: {skill_name}\n"
            f"Description: {proposed.get('description', '')}"
        )
        duplicate = await find_duplicate_pending_proposal(
            self._memory,
            user_id=user_id,
            proposal_type=ProposalType.SKILL_PROPOSAL,
            proposed_skill=proposed,
        )
        if duplicate is not None:
            proposal_id = str(duplicate["id"])
            logger.info(
                "Reusing pending skill creation proposal id=%s skill=%s",
                proposal_id,
                skill_name,
            )
            return proposal_id

        proposal_id = await self._memory.add_certain_record(
            user_id=user_id,
            content=content,
            record_type="proposal",
            metadata=metadata,
        )
        logger.info(
            "Skill creation proposal stored (id=%s): %s",
            proposal_id,
            skill_name,
        )

        if user_id != "__system__" and adapter_id:
            platform_user_id = await self._memory.resolve_platform_user(
                user_id, adapter_id
            )
            if platform_user_id:
                soul_snap = self._soul.get_soul_snapshot()
                params_desc = ", ".join(
                    p.get("name", "?") for p in proposed.get("parameters", [])
                )
                await self._send_notification(
                    user_id=user_id,
                    adapter_id=adapter_id,
                    platform_user_id=platform_user_id,
                    content=(
                        f"🛠️ {soul_snap['name']} wants to create "
                        f"a new skill: '{skill_name}'\n"
                        f"Description: {proposed.get('description', '')}\n"
                        f"Parameters: ({params_desc})\n\n"
                        f"(proposal ID: {proposal_id})"
                    ),
                )

        return proposal_id

    async def install_proposed_skill(
        self,
        proposed: dict[str, Any],
    ) -> None:
        """Validate and write a proposed skill to the skills directory."""
        validate_proposed_skill(proposed, self._skills)

        name = proposed.get("name", "")
        skill_dir = self._skills.skills_dir / name
        code = proposed.get("code", "")

        description = proposed.get("description") or "AI-generated skill"
        usage = proposed.get("usage") or ""
        parameters = proposed.get("parameters") or []

        yaml_data = {
            "name": name,
            "description": description,
            "version": "1.0.0",
            "author": _AI_GENERATED_AUTHOR,
            "ownership": "ai",
            "mutable_by_ai": True,
            "requires_approval_to_modify": False,
            "usage": usage,
            "parameters": parameters,
            "contexts": {"shared_voice": False},
            "safety": {
                "level": "safe",
                "sandbox": True,
                "network": False,
                "filesystem": False,
            },
        }
        yaml_content = yaml.safe_dump(
            yaml_data,
            sort_keys=False,
            allow_unicode=True,
        )

        code_header = (
            f'"""AI-generated skill: {name}."""\n\n'
            "from __future__ import annotations\n\n"
            "from typing import Any\n\n\n"
        )
        if "from __future__" not in code:
            full_code = code_header + code + "\n"
        else:
            full_code = code + "\n"

        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "skill.yaml").write_text(yaml_content, encoding="utf-8")
        (skill_dir / "main.py").write_text(full_code, encoding="utf-8")

        self._skills.load_all()
        logger.info("Installed proposed skill: %s", name)

    def _can_update_skill(self, name: str) -> bool:
        """Only AI-generated skills may be overwritten by later proposals."""
        return _can_update_ai_skill(self._skills.skills_dir, name)

    async def execute_approved(self, proposal_id: str | None = None) -> None:
        """Check for approved proposals and execute them."""
        proposals = await self._memory.get_pending_proposals(
            status=ProposalStatus.APPROVED,
        )

        for proposal in proposals:
            current_id = proposal["id"]
            if proposal_id is not None and current_id != proposal_id:
                continue
            meta = _load_proposal_metadata(proposal)
            if meta is None:
                await self._expire_proposal_safely(
                    current_id,
                    reason="invalid metadata",
                )
                continue
            proposal_type = meta.get("proposal_type", ProposalType.GENERAL)
            try:
                claimed = await self._memory.update_proposal_status(
                    current_id,
                    ProposalStatus.EXECUTING,
                )
            except ValueError:
                logger.info("Proposal %s was already claimed or completed", current_id)
                continue
            if not claimed:
                continue

            if proposal_type == ProposalType.SKILL_EXECUTION:
                await self._execute_skill_proposal(proposal, meta)
            elif proposal_type == ProposalType.TRAIT_CHANGE:
                await self._execute_trait_proposal(proposal, meta)
            elif proposal_type == ProposalType.SKILL_PROPOSAL:
                await self._execute_skill_creation(proposal, meta)
            else:
                logger.info(
                    "General proposal %s acknowledged",
                    current_id,
                )
                await self._memory.update_proposal_status(
                    current_id, ProposalStatus.EXECUTED
                )
                await self._notify_result_safely(
                    proposal,
                    "✅ Proposal acknowledged.",
                )

    async def _expire_proposal_safely(self, proposal_id: str, *, reason: str) -> bool:
        try:
            return await self._memory.update_proposal_status(
                proposal_id,
                ProposalStatus.EXPIRED,
            )
        except ValueError:
            logger.warning(
                "Could not expire proposal %s after %s",
                proposal_id,
                reason,
                exc_info=True,
            )
            return False

    async def _notify_result_safely(
        self,
        proposal: dict[str, Any],
        message: str,
    ) -> None:
        try:
            await self._notify_result(proposal, message)
        except Exception:
            logger.warning(
                "Could not send proposal result notification for %s",
                proposal.get("id"),
                exc_info=True,
            )

    async def _execute_skill_proposal(
        self,
        proposal: dict[str, Any],
        meta: dict[str, Any],
    ) -> None:
        proposal_id = proposal["id"]
        skill_name = meta.get("skill_name", "")
        skill_params = meta.get("skill_params", {})
        if skill_name == _UPDATE_SKILL_FILE_TOOL_NAME:
            await self._execute_skill_file_update(proposal, skill_params)
            return
        if skill_name == _DELETE_SKILL_FILE_TOOL_NAME:
            await self._execute_skill_file_delete(proposal, skill_params)
            return
        if skill_name == _DELETE_SKILL_TOOL_NAME:
            await self._execute_skill_delete(proposal, skill_params)
            return
        if skill_name == _UPDATE_SKILL_SETTINGS_TOOL_NAME:
            await self._execute_skill_settings_update(proposal, skill_params)
            return

        skill = self._skills.get(skill_name)
        if skill is None:
            logger.warning(
                "Approved skill '%s' not found, marking expired",
                skill_name,
            )
            await self._expire_proposal_safely(
                proposal_id,
                reason=f"missing skill {skill_name}",
            )
            await self._notify_result_safely(
                proposal,
                f"⚠️ Skill '{skill_name}' not found — proposal expired.",
            )
            return

        try:
            proposal_user_id = proposal.get("user_id")
            acting_user_id = (
                str(proposal_user_id)
                if proposal_user_id and proposal_user_id != "__system__"
                else None
            )
            result = await skill.execute(
                skill_params,
                memory=self._memory,
                acting_user_id=acting_user_id,
                sandbox_overrides=sandbox_overrides_for_skill(
                    skill_name,
                    skill_params,
                ),
            )
            logger.info(
                "Approved skill '%s' executed: %s",
                skill_name,
                result,
            )
            summary = _format_skill_result_summary(result)
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXECUTED
            )
            await self._notify_result_safely(
                proposal,
                f"✅ Skill '{skill_name}' executed successfully.\n{summary}",
            )
        except Exception:
            logger.exception("Approved skill '%s' failed", skill_name)
            await self._expire_proposal_safely(
                proposal_id,
                reason=f"skill {skill_name} failed",
            )
            await self._notify_result_safely(
                proposal,
                f"❌ Skill '{skill_name}' failed — proposal expired.",
            )

    async def _execute_skill_file_update(
        self,
        proposal: dict[str, Any],
        skill_params: dict[str, Any],
    ) -> None:
        proposal_id = proposal["id"]
        try:
            result = update_skill_file(
                self._skills.skills_dir,
                skill_name=skill_params.get("skill_name"),
                path=skill_params.get("path"),
                content=skill_params.get("content"),
                approved=True,
            )
            if result.get("error"):
                detail = str(result.get("detail") or result["error"])
                await self._expire_proposal_safely(
                    proposal_id,
                    reason="skill file update returned error",
                )
                await self._notify_result_safely(
                    proposal,
                    "❌ Skill file update failed"
                    f" ({result['error']}) — {detail[:500]}",
                )
                return
            self._skills.load_all()
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXECUTED
            )
            await self._notify_result_safely(
                proposal,
                "✅ Skill file updated successfully.\n"
                f"{result['skill_name']}/{result['path']}",
            )
        except Exception:
            logger.exception("Approved skill file update failed")
            await self._expire_proposal_safely(
                proposal_id,
                reason="skill file update failed",
            )
            await self._notify_result_safely(
                proposal,
                "❌ Skill file update failed — proposal expired.",
            )

    async def _execute_skill_settings_update(
        self,
        proposal: dict[str, Any],
        skill_params: dict[str, Any],
    ) -> None:
        proposal_id = proposal["id"]
        try:
            result = update_skill_settings(
                self._skills.skills_dir,
                skill_name=skill_params.get("skill_name"),
                ownership=skill_params.get("ownership"),
                mutable_by_ai=skill_params.get("mutable_by_ai"),
                requires_approval_to_modify=skill_params.get(
                    "requires_approval_to_modify"
                ),
            )
            self._skills.load_all()
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXECUTED
            )
            await self._notify_result_safely(
                proposal,
                "✅ Skill settings updated successfully.\n"
                f"{result['skill_name']}: ownership={result['ownership']}, "
                f"mutable_by_ai={result['mutable_by_ai']}, "
                "requires_approval_to_modify="
                f"{result['requires_approval_to_modify']}",
            )
        except Exception:
            logger.exception("Approved skill settings update failed")
            await self._expire_proposal_safely(
                proposal_id,
                reason="skill settings update failed",
            )
            await self._notify_result_safely(
                proposal,
                "❌ Skill settings update failed — proposal expired.",
            )

    async def _execute_skill_file_delete(
        self,
        proposal: dict[str, Any],
        skill_params: dict[str, Any],
    ) -> None:
        proposal_id = proposal["id"]
        try:
            result = delete_skill_file(
                self._skills.skills_dir,
                skill_name=skill_params.get("skill_name"),
                path=skill_params.get("path"),
                recursive=skill_params.get("recursive", False),
                approved=True,
            )
            self._skills.load_all()
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXECUTED
            )
            await self._notify_result_safely(
                proposal,
                "✅ Skill file deleted successfully.\n"
                f"{result['skill_name']}/{result['path']}",
            )
        except Exception:
            logger.exception("Approved skill file delete failed")
            await self._expire_proposal_safely(
                proposal_id,
                reason="skill file delete failed",
            )
            await self._notify_result_safely(
                proposal,
                "❌ Skill file delete failed — proposal expired.",
            )

    async def _execute_skill_delete(
        self,
        proposal: dict[str, Any],
        skill_params: dict[str, Any],
    ) -> None:
        proposal_id = proposal["id"]
        try:
            result = delete_skill(
                self._skills.skills_dir,
                skill_name=skill_params.get("skill_name"),
                approved=True,
            )
            self._skills.load_all()
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXECUTED
            )
            await self._notify_result_safely(
                proposal,
                f"✅ Skill '{result['skill_name']}' deleted successfully.",
            )
        except Exception:
            logger.exception("Approved skill delete failed")
            await self._expire_proposal_safely(
                proposal_id,
                reason="skill delete failed",
            )
            await self._notify_result_safely(
                proposal,
                "❌ Skill delete failed — proposal expired.",
            )

    async def _execute_trait_proposal(
        self,
        proposal: dict[str, Any],
        meta: dict[str, Any],
    ) -> None:
        proposal_id = proposal["id"]
        trait_add = meta.get("trait_add", [])
        trait_remove = meta.get("trait_remove", [])
        try:
            self._soul.apply_trait_change(
                add=trait_add,
                remove=trait_remove,
                caller=SoulCaller.SYSTEM,
            )
            logger.info(
                "Approved trait change applied: add=%s remove=%s",
                trait_add,
                trait_remove,
            )
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXECUTED
            )
            parts: list[str] = []
            if trait_add:
                parts.append(f"added: {', '.join(trait_add)}")
            if trait_remove:
                parts.append(f"removed: {', '.join(trait_remove)}")
            await self._notify_result_safely(
                proposal,
                f"✅ Personality updated — {'; '.join(parts)}.",
            )
        except Exception:
            logger.exception("Trait change failed")
            await self._expire_proposal_safely(
                proposal_id,
                reason="trait change failed",
            )
            await self._notify_result_safely(
                proposal,
                "❌ Personality change failed — proposal expired.",
            )

    async def _execute_skill_creation(
        self,
        proposal: dict[str, Any],
        meta: dict[str, Any],
    ) -> None:
        proposal_id = proposal["id"]
        proposed = meta.get("proposed_skill", {})
        skill_name = proposed.get("name", "unknown")
        try:
            await self.install_proposed_skill(proposed)
            logger.info(
                "Proposed skill '%s' installed",
                skill_name,
            )
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXECUTED
            )
            await self._notify_result_safely(
                proposal,
                f"✅ New skill '{skill_name}' installed successfully.",
            )
        except Exception as exc:
            logger.exception(
                "Proposed skill '%s' installation failed",
                skill_name,
            )
            await self._expire_proposal_safely(
                proposal_id,
                reason=f"skill {skill_name} installation failed",
            )
            detail = str(exc).strip()
            if len(detail) > 500:
                detail = detail[:497] + "..."
            detail_text = f" — {detail}" if detail else ""
            await self._notify_result_safely(
                proposal,
                f"❌ Skill '{skill_name}' installation failed{detail_text}.",
            )

    async def _notify_result(
        self,
        proposal: dict[str, Any],
        message: str,
    ) -> None:
        """Send execution result notification to the proposal owner."""
        meta = _load_proposal_metadata(proposal)
        if meta is None:
            return
        adapter_id = meta.get("adapter_id")
        user_id = proposal.get("user_id")

        if not adapter_id or not user_id:
            return

        platform_user_id = await self._memory.resolve_platform_user(user_id, adapter_id)
        if not platform_user_id:
            return

        resume = meta.get("resume_context")
        if isinstance(resume, dict) and resume.get("interrupted"):
            message = (
                f"{message}\n\n"
                "The approval-gated action from the interrupted conversation "
                "has now completed."
            )

        await self._send_notification(
            user_id=user_id,
            adapter_id=adapter_id,
            platform_user_id=platform_user_id,
            content=message,
        )
