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
    return True


async def find_duplicate_pending_proposal(
    memory: MemoryStore,
    *,
    user_id: str,
    proposal_type: str,
    skill_name: str | None = None,
    skill_params: dict[str, Any] | None = None,
    proposed_skill: dict[str, Any] | None = None,
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
    ) -> None:
        self._memory = memory
        self._skills = skills
        self._gateway = gateway
        self._soul = soul

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
                notification = GatewayMessage(
                    type=MessageType.HEARTBEAT_MESSAGE,
                    adapter_id=adapter_id,
                    platform_user_id=platform_user_id,
                    content=(
                        f"💡 {soul_snap['name']} has a suggestion:\n\n{content}"
                        f"\n\n(proposal ID: {proposal_id})"
                    ),
                )
                await self._gateway.send_to_adapter(adapter_id, notification)
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
                notification = GatewayMessage(
                    type=MessageType.SKILL_CONFIRM,
                    adapter_id=adapter_id,
                    platform_user_id=platform_user_id,
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
                await self._gateway.send_to_adapter(adapter_id, notification)

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
                notification = GatewayMessage(
                    type=MessageType.HEARTBEAT_MESSAGE,
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
                await self._gateway.send_to_adapter(adapter_id, notification)

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
                notification = GatewayMessage(
                    type=MessageType.HEARTBEAT_MESSAGE,
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
                await self._gateway.send_to_adapter(adapter_id, notification)

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

        description = proposed.get("description", "AI-generated skill")
        usage = proposed.get("usage", "")
        parameters = proposed.get("parameters", [])

        safe_desc = description.replace("\\", "\\\\").replace('"', '\\"')
        safe_usage = usage.replace("\r\n", "\n").replace("\r", "\n")
        safe_usage = safe_usage.replace("\n", "\n  ")
        yaml_lines = [
            f"name: {name}",
            f'description: "{safe_desc}"',
            'version: "1.0.0"',
            f'author: "{_AI_GENERATED_AUTHOR}"',
            "ownership: ai",
            "mutable_by_ai: true",
            "requires_approval_to_modify: false",
            "",
            f"usage: |\n  {safe_usage}",
            "",
        ]
        if parameters:
            yaml_lines.append("parameters:")
        else:
            yaml_lines.append("parameters: []")
        for param in parameters:
            yaml_lines.append(f"  - name: {param.get('name', 'arg')}")
            yaml_lines.append(f"    type: {param.get('type', 'string')}")
            yaml_lines.append(
                f"    required: {str(param.get('required', True)).lower()}"
            )
            desc = param.get("description", "")
            if desc:
                safe = desc.replace("\\", "\\\\").replace('"', '\\"')
                yaml_lines.append(f'    description: "{safe}"')
        yaml_lines.extend(
            [
                "",
                "contexts:",
                "  shared_voice: false",
                "",
                "safety:",
                "  level: safe",
                "  sandbox: true",
                "  network: false",
                "  filesystem: false",
            ]
        )
        yaml_content = "\n".join(yaml_lines) + "\n"

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
            meta = json.loads(proposal.get("metadata") or "{}")
            proposal_type = meta.get("proposal_type", ProposalType.GENERAL)
            current_id = proposal["id"]
            if proposal_id is not None and current_id != proposal_id:
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
                await self._notify_result(
                    proposal,
                    "✅ Proposal acknowledged.",
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
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXPIRED
            )
            await self._notify_result(
                proposal,
                f"⚠️ Skill '{skill_name}' not found — proposal expired.",
            )
            return

        try:
            result = await skill.execute(
                skill_params,
                memory=self._memory,
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
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXECUTED
            )
            summary_value = result.get("result", result.get("output"))
            if summary_value is None:
                summary = json.dumps(result, ensure_ascii=False)[:500]
            else:
                summary = str(summary_value)[:500]
            await self._notify_result(
                proposal,
                f"✅ Skill '{skill_name}' executed successfully.\n{summary}",
            )
        except Exception:
            logger.exception("Approved skill '%s' failed", skill_name)
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXPIRED
            )
            await self._notify_result(
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
            self._skills.load_all()
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXECUTED
            )
            await self._notify_result(
                proposal,
                "✅ Skill file updated successfully.\n"
                f"{result['skill_name']}/{result['path']}",
            )
        except Exception:
            logger.exception("Approved skill file update failed")
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXPIRED
            )
            await self._notify_result(
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
            await self._notify_result(
                proposal,
                "✅ Skill settings updated successfully.\n"
                f"{result['skill_name']}: ownership={result['ownership']}, "
                f"mutable_by_ai={result['mutable_by_ai']}, "
                "requires_approval_to_modify="
                f"{result['requires_approval_to_modify']}",
            )
        except Exception:
            logger.exception("Approved skill settings update failed")
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXPIRED
            )
            await self._notify_result(
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
            await self._notify_result(
                proposal,
                "✅ Skill file deleted successfully.\n"
                f"{result['skill_name']}/{result['path']}",
            )
        except Exception:
            logger.exception("Approved skill file delete failed")
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXPIRED
            )
            await self._notify_result(
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
            await self._notify_result(
                proposal,
                f"✅ Skill '{result['skill_name']}' deleted successfully.",
            )
        except Exception:
            logger.exception("Approved skill delete failed")
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXPIRED
            )
            await self._notify_result(
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
            await self._notify_result(
                proposal,
                f"✅ Personality updated — {'; '.join(parts)}.",
            )
        except Exception:
            logger.exception("Trait change failed")
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXPIRED
            )
            await self._notify_result(
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
            await self._notify_result(
                proposal,
                f"✅ New skill '{skill_name}' installed successfully.",
            )
        except Exception as exc:
            logger.exception(
                "Proposed skill '%s' installation failed",
                skill_name,
            )
            await self._memory.update_proposal_status(
                proposal_id, ProposalStatus.EXPIRED
            )
            detail = str(exc).strip()
            if len(detail) > 500:
                detail = detail[:497] + "..."
            detail_text = f" — {detail}" if detail else ""
            await self._notify_result(
                proposal,
                f"❌ Skill '{skill_name}' installation failed{detail_text}.",
            )

    async def _notify_result(
        self,
        proposal: dict[str, Any],
        message: str,
    ) -> None:
        """Send execution result notification to the proposal owner."""
        meta = json.loads(proposal.get("metadata") or "{}")
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

        notification = GatewayMessage(
            type=MessageType.HEARTBEAT_MESSAGE,
            adapter_id=adapter_id,
            platform_user_id=platform_user_id,
            content=message,
        )
        await self._gateway.send_to_adapter(adapter_id, notification)
