# Thinking Mode Control — Design Document

**Status**: Draft (review pending)
**Branch**: `feat/dm-channel-split-and-ai-decision-llm`
**Related code**: `voice_enable_thinking` in `src/cordbeat/ai/backend.py`,
`_generate_draw_dsl` in `src/cordbeat/core/engine.py`,
`ai_decision` config in `src/cordbeat/config.py`.

---

## Background

Thinking-capable models such as Qwen3 and DeepSeek-R1 expose a
per-request `enable_thinking: true|false` toggle. Turning it on
significantly improves accuracy on reasoning-heavy tasks at the cost of
extra tokens and latency; turning it off makes the model fast but
noticeably less accurate.

CordBeat currently exposes a single global flag
(`ai_backend.options.enable_thinking`). One narrow per-call override
already exists (`voice_enable_thinking`, used while a voice/STT context
is active) — but the rest of the engine still inherits the global flag.

Production logs from 2026-06-03 confirmed two operational regimes:

1. `enable_thinking: true` — chat replies often time out or come back
   empty because the thinking phase consumes the entire token budget
   (`content=null but reasoning_content=N chars`). Heartbeat generation
   piles up behind these stuck calls.
2. `enable_thinking: false` — chat replies recover, but **internal
   processing quality drops sharply**:
   - Memory extraction (`topic`, `keywords`, `facts`, episode
     summaries) becomes shallow and repetitive.
   - The Draw DSL generator emits geometrically valid but semantically
     meaningless output. Real example logged at 21:40:20: a request to
     "draw a dragon" produced concentric gradient circles plus tiny
     star dots — the model was unable to plan a dragon silhouette
     without a thinking phase.

Mature agent frameworks (LangGraph, AutoGen, OpenAI o1/o3, Hermes
Agent) all converge on the same pattern: **switch reasoning effort per
task category**, either via a per-call parameter
(`reasoning_effort`, `enable_thinking`) or via per-node model
selection. CordBeat should follow the same direction.

---

## Goals

- **Chat replies** stay fast (default OFF).
- **Internal processing** (memory extraction, topic detection,
  summarisation, context compression) can run with thinking ON for
  accuracy.
- **Skill helpers** (e.g. the Draw DSL generator) can declare their
  own accuracy requirements per skill.
- **Force-on skills** can keep working with full reasoning even on a
  server configured with `enable_thinking: false` globally.
- Configuration is **fully backward compatible** — existing setups
  must keep current behaviour.
- Backends that do not support a thinking toggle (Ollama, plain
  llama.cpp servers, etc.) silently no-op.

---

## Resolution Logic

```python
def effective_thinking(
    config_on: bool | None,        # ai_backend.options.enable_thinking
    skill_mode: Literal["auto", "off", "force_on"] = "auto",
) -> bool | None:
    if skill_mode == "force_on":
        return True             # overrides config OFF
    if config_on is False:
        return False            # config OFF → uniformly OFF
    if skill_mode == "off":
        return False            # config ON, but this skill opts out
    return config_on            # auto: follow config (None = let server decide)
```

| `config.enable_thinking` | `skill.thinking_mode` | Effective |
|---|---|---|
| OFF | auto | OFF |
| OFF | off | OFF |
| OFF | **force_on** | **ON** ← override |
| ON | auto | ON |
| ON | **off** | **OFF** ← skill opt-out |
| ON | force_on | ON |
| None (unset) | auto | None (server default) |
| None | force_on | ON |
| None | off | False |

---

## LLM Call Categories

Every LLM call inside CordBeat falls into one of four categories.
Each category gets its own independent control surface:

| # | Category | Example call sites | Control mechanism | Status |
|---|---|---|---|---|
| (a) | **Chat reply** | `engine._generate_response`, `_react_loop` continuation | base `enable_thinking` | existing |
| (b) | **Voice reply** | (a) when `is_voice=True` | `voice_enable_thinking` override | existing |
| (c) | **Internal processing** | `extraction.extract_facts`, `extract_diary`, `extract_topic`, context compression | **`internal_enable_thinking`** | **new** |
| (d) | **Skill helper** | `_generate_draw_dsl` (only one today) | **`thinking_mode` in `skill.yaml`** | **new** |

Implementation reuses the existing `ContextVar` pattern from
`voice_context_scope`:

```python
# ai/backend.py
_voice_context: ContextVar[bool] = ContextVar("voice_context", default=False)
_internal_context: ContextVar[bool] = ContextVar("internal_context", default=False)
_skill_thinking_context: ContextVar[Optional[Literal["auto","off","force_on"]]] = (
    ContextVar("skill_thinking_context", default=None)
)

@contextmanager
def voice_context_scope(active: bool): ...
@contextmanager
def internal_context_scope(active: bool = True): ...
@contextmanager
def skill_thinking_scope(mode: Literal["auto","off","force_on"]): ...
```

`OpenAICompatBackend._effective_enable_thinking()` becomes:

```python
def _effective_enable_thinking(self) -> bool | None:
    skill_mode = _skill_thinking_context.get()
    if skill_mode == "force_on":
        return True

    # Pick the base config: voice override > internal override > default
    if is_voice_context() and self._voice_enable_thinking is not None:
        config_on = self._voice_enable_thinking
    elif _internal_context.get() and self._internal_enable_thinking is not None:
        config_on = self._internal_enable_thinking
    else:
        config_on = self._enable_thinking

    if config_on is False:
        return False
    if skill_mode == "off":
        return False
    return config_on
```

Resolution order (top wins):
1. `skill_thinking == "force_on"` → `True`.
2. Voice override (`voice_enable_thinking`) when voice context active.
3. Internal override (`internal_enable_thinking`) when internal context
   active.
4. Base (`enable_thinking`).
5. After picking the base, `skill_thinking == "off"` collapses the
   result back to `False`.

---

## Configuration Changes

### `config.yaml` (`AIBackendConfig.options`)

```yaml
ai_backend:
  options:
    enable_thinking: false           # existing — base for chat replies
    voice_enable_thinking: false     # existing (optional) — voice override
    internal_enable_thinking: true   # NEW — override for memory extraction etc.
                                     #   unset (None) → inherit enable_thinking
```

### `skill.yaml` (`SkillManifest`)

```yaml
name: draw
safety_level: requires_confirmation
thinking_mode: force_on   # NEW — auto | off | force_on  (default: auto)
```

A skill without `thinking_mode` is treated as `auto` (identical to
current behaviour).

---

## Scope of Changes

### Files

| File | Change |
|---|---|
| `src/cordbeat/config.py` | Document `internal_enable_thinking` in `AIBackendConfig.options` (no code change — already a free-form dict). |
| `src/cordbeat/ai/backend.py` | Add `_internal_context`, `_skill_thinking_context`, `internal_context_scope()`, `skill_thinking_scope()`. Rewrite `OpenAICompatBackend._effective_enable_thinking()`. |
| `src/cordbeat/skills/loader.py` (or wherever `SkillManifest` lives) | Add `thinking_mode: Literal["auto","off","force_on"] = "auto"`. Parse from YAML. |
| `src/cordbeat/core/engine.py` | Wrap `_generate_draw_dsl` with `skill_thinking_scope(manifest.thinking_mode)`. |
| `src/cordbeat/memory/extraction.py` | Wrap `extract_facts`, `extract_diary`, `extract_topic`, `extract_keywords` with `internal_context_scope()`. |
| `src/cordbeat/memory/conversation.py` (or compression entry points) | Wrap LLM-driven compression with `internal_context_scope()`. |
| `tests/test_ai_backend.py` | Table-driven tests of the resolution logic (every row of the table above). |
| `tests/test_skills.py` | Parser tests for the new `thinking_mode` field. |
| `config.example.yaml`, `tools/config_template.yaml` | Document `internal_enable_thinking`. |
| `skills/draw/skill.yaml` | Set `thinking_mode: force_on`. |
| `CHANGELOG.md` | New entry under `[Unreleased]`. |

### Backward Compatibility

- Servers configured with only `enable_thinking` keep identical
  behaviour: `internal_enable_thinking` defaults to `None` and falls
  through to `enable_thinking`.
- Skills without `thinking_mode` default to `auto` (current
  behaviour).
- Backends that do not support thinking (Ollama and friends) read
  the ContextVars but never send `enable_thinking` to the wire — no
  visible effect.

### Breaking Changes

None.

---

## Test Strategy

### Unit tests

- Table-driven test of `_effective_enable_thinking` covering every
  row of the resolution table (~12 cases).
- Nested-scope precedence tests for `internal_context_scope` and
  `skill_thinking_scope`.
- Concurrent activation of `voice_context_scope` plus
  `skill_thinking_scope=force_on` — verify `force_on` wins.

### Integration tests

- Confirm `_generate_draw_dsl` is invoked with `skill_thinking_context`
  set to the manifest's `thinking_mode`.
- Confirm `extract_facts` runs inside `internal_context_scope`.

---

## Migration Guide

### Recommended config for existing operators

```yaml
ai_backend:
  options:
    enable_thinking: false           # fast chat
    internal_enable_thinking: true   # accurate internal processing
    voice_enable_thinking: false     # fast voice (optional)
```

### Skill author guidance

- Math, DSL synthesis, structured extraction → `thinking_mode: force_on`.
- Plain API wrappers, lookups → `auto` (default).
- Lightweight conversational skills → `off` (match chat latency).

---

## Open Questions

- **OQ1**: Final list of `extraction.*` functions that should be wrapped
  in `internal_context_scope`. Confirmed candidates: `extract_facts`,
  `extract_diary`, `extract_topic`, `extract_keywords`, context
  compression. Anything else worth including?
- **OQ2**: Will `skill_thinking_scope` be used outside
  `_generate_draw_dsl`? It is the only skill helper today. Should the
  scope generalise to all helpers, or stay draw-specific until a second
  case appears?
- **OQ3**: When a `force_on` skill runs against a backend that does not
  support thinking (Ollama, plain llama.cpp), should we silently no-op
  (current plan) or emit a `logger.info` warning ("skill requested
  thinking but backend has no toggle")?
- **OQ4**: How does this interact with the separate `ai_decision`
  backend used by `respond_mode: ai_decision_llm`? Do we want to apply
  the same context vars there, or keep `ai_decision` orthogonal?
- **OQ5**: Should `_react_loop` continuation generation (the synthesis
  call after a skill executes) be reclassified as internal (c) instead
  of chat (a)? Currently it inherits the chat flag. Out of scope for
  this iteration but worth flagging.

---

## Milestones

1. ✅ Design document review (this PR).
2. ⏭ Add `thinking_mode` to skill manifest + parser.
3. ⏭ Add new `ContextVar`s and scope helpers.
4. ⏭ Rewrite `_effective_enable_thinking`.
5. ⏭ Wrap call sites in `extraction.py` and `engine.py`.
6. ⏭ Add tests.
7. ⏭ Update `config.example.yaml`, README, CHANGELOG.
8. ⏭ Green ruff / mypy strict / pytest -q.

---

## Review Focus

- Is the per-skill + global config + force-on direction the right one,
  or are we overshooting?
- Is the four-category split ((a) chat / (b) voice / (c) internal /
  (d) skill helper) appropriate, or does it create needless surface
  area?
- Is the resolution order (force_on > voice > internal > base, then
  collapse to false on `off`) correct? Any edge cases missed?
- Is `skill_thinking_scope` worth introducing for a single call site
  (`_generate_draw_dsl`), assuming future expansion, or should we wait
  for a second case before generalising?
- Which Open Questions must be resolved before implementation begins?
