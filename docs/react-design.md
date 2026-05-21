# ReAct (Multi-Step Skill Execution) — Design Document

Status: **Draft** (v0.1, target implementation: v1.1)
Author: CordBeat core
Last updated: 2026-05-19

---

## 1. Motivation

### 1.1 Observed problem

Production logs (2026-05-19) revealed a recurring failure mode:

```text
User: "https://example.com/article  この記事を要約して"
AI  : "[SKILL: web_search | query=https://example.com/article]"
→ web_search returns search-engine hits about the URL string,
   not the article body. The AI cannot recover within the same turn.
```

The AI **wants** to fetch and summarize, but the current architecture
prevents it from:
1. Fetching the URL
2. **Observing** the fetched content
3. **Reasoning** about it
4. Producing the summary as the final reply

…all within a single user turn.

### 1.2 Architectural root cause

`engine.py::_dispatch_skill_tags` (current implementation, lines 409–466):

- Parses the **first** `[SKILL: name | k=v]` tag only
- Executes the skill
- Inlines the raw output: `[skill_name: <up to 2000 chars>]`
- Sends that text directly to the user

The AI **never sees** the skill output. Errors are silently dropped
(`exc_info=True` logged only). A second skill call in the same turn is
impossible.

### 1.3 Industry precedent

Modern function-calling agents (OpenAI tool calls, Anthropic tool_use,
Nous Hermes-Pro, Llama 3.x tool-use, Qwen2.5 function-calling) all
implement the same pattern:

```
loop:
  assistant_msg = LLM(messages)
  if assistant_msg has no tool_calls:
      break
  for call in assistant_msg.tool_calls:
      result = execute(call)
      messages.append({"role": "tool", "content": result})
  messages.append(assistant_msg)
```

CordBeat must adopt the same loop to unlock multi-step reasoning.

---

## 2. Goals & non-goals

### Goals

- **G1**: AI can chain skills (e.g., `web_search` → `fetch_url` → final
  summary) within one user turn.
- **G2**: AI observes skill outputs and adapts (including error handling).
- **G3**: Backward compatible — existing single-shot behavior reachable
  via config.
- **G4**: Works with all backends (Ollama / OpenAI / openai_compat)
  without requiring native function-calling support.
- **G5**: Bounded — predictable max latency / token cost per turn.

### Non-goals

- **NG1**: Parallel skill execution (sequential only; parallelism is
  postponed to v1.2+).
- **NG2**: Mid-loop user-confirmation interactivity (proposals still go
  through the existing queue and break the loop).
- **NG3**: Streaming partial responses (the user sees the final reply only).
- **NG4**: Native OpenAI `tools` JSON-schema dispatch (we keep the textual
  `[SKILL: ...]` tag to stay backend-agnostic).
- **NG5**: Changing the HEARTBEAT decision flow
  (`HeartbeatDecision.action=SKILL` already has its own path).

---

## 3. High-level design

### 3.1 Replace `_dispatch_skill_tags` with `_react_loop`

```python
async def _react_loop(
    self,
    initial_response: str,
    user_id: str,
    user: UserSummary,
    message: GatewayMessage,
) -> str:
    """Iteratively execute SKILL tags until the AI emits a tag-free reply."""
    response = initial_response
    trace: list[ToolCallRecord] = []

    for iteration in range(self._config.react.max_iterations):
        match = _SKILL_TAG_RE.search(response)
        if match is None:
            return response.strip()              # Final reply

        call = self._parse_skill_call(match)
        if call.skill is None or not call.skill.meta.enabled:
            response = _SKILL_TAG_RE.sub("", response, count=1)
            continue

        if call.requires_user_confirmation(self._session_allowed_skills):
            await self._request_skill_confirmation(message, call.name, call.params)
            return _SKILL_TAG_RE.sub("", response, count=1).strip()  # break loop

        result = await self._execute_call_safe(call)
        trace.append(ToolCallRecord(call=call, result=result, iteration=iteration))

        response = await self._generate_continuation(
            user_id, user, message, trace,
        )
        if response is None:
            break                                 # AI failure → fall through

    # Loop exhausted or AI gave up — strip any remaining tags
    final = _SKILL_TAG_RE.sub("", response or "", count=0).strip()
    if not final and trace:
        final = self._summarize_trace_fallback(trace)   # graceful degradation
    return final
```

### 3.1b Handling mixed text + tag and multiple tags

#### Pre-tag text (preserved & forwarded immediately)

A common pattern from Claude / ChatGPT-style agents is to acknowledge the
user **before** invoking a tool:

```text
AI iter 1: "うん、ちょっと調べてみるね [SKILL: web_search | query=X]"
```

CordBeat will:

1. **Split the response at the tag boundary.**
2. **Forward the text before the tag to the adapter immediately** (so the
   user sees a progress / acknowledgment message while the tool runs).
3. Execute the skill, feed the result back to the AI.
4. The AI produces the final integrated answer in iter 2, which is sent
   as a **second message** to the user.

The adapter therefore receives **multiple messages per user turn**. All
existing adapters already support sequential sends, so no per-adapter
change is required. This is the same UX pattern as Claude Code's
"Let me check…" → tool → "Here's what I found:" flow.

#### Post-tag text (discarded)

```text
AI iter 1: "[SKILL: fetch_url | url=X] — きっと面白いと思うよ"
```

The text **after** the tag is written **before the AI saw the tool result**
and is therefore likely to be inaccurate or to leak fabricated content.
It is discarded with a `debug` log entry.

#### Multiple tags in one response (all executed sequentially)

```text
AI iter 1: "[SKILL: web_search | query=A] [SKILL: fetch_url | url=B]"
```

**All tags are executed in order, sequentially**, matching the behavior
of Anthropic (`one or more tool_use blocks`) and OpenAI (`tool_calls: []`
array). Each tool result is collected into the trace before the next
LLM call:

```text
iter 1 assistant output: [SKILL: A] [SKILL: B]
  → execute A → result_A
  → execute B → result_B
  → feed BOTH results back to AI as <tool_response> blocks
iter 2 assistant output: final reply (or more [SKILL: ...])
```

**Execution semantics**:

- **Order**: strict document order (the order tags appear in the response)
- **Errors**: a failing tool returns `{"error": "..."}` to the AI; the
  remaining tools in the batch **still execute**. The AI sees all
  successes and failures together and decides how to recover.
- **`requires_confirmation` mid-batch**: execution **halts at that
  point**. Already-completed tool results are discarded (not committed
  to the trace) and a single proposal is queued for the user. The
  remaining tags are dropped. On user approval, a fresh user turn
  restarts the loop. (Rationale: simpler than partial-trace
  serialization across the approval boundary.)
- **Iteration count**: one iteration = one assistant turn, regardless of
  tool count. Batches don't multiply `max_iterations`.
- **Tag-after-tag text**: any prose **between** tags is discarded with a
  debug log (the AI hasn't seen any tool result yet, so inter-tag
  commentary is speculative). Tags concatenated with whitespace only is
  the canonical form.

The system prompt does **not** forbid multi-tag emission — it documents
both modes:

```text
You can call one or more tools per response by emitting one or more
[SKILL: ...] tags. Tags will be executed in order, and ALL results will
be returned to you before your next response. Prefer a single tag when
later calls depend on earlier results; emit multiple only for clearly
independent operations.
```

#### Streaming mode

A new config flag controls how the multi-message split appears:

```yaml
react:
  streaming_mode: split    # split | combined
```

- `split` (**default**, recommended): pre-tag text sent immediately as a
  separate message, final answer as a second message. Best UX —
  matches Claude / ChatGPT behavior.
- `combined`: buffer all output and send once at the end. Use for
  adapters where multiple sends are expensive (e.g., webhook-based
  platforms with strict rate limits).

### 3.2 Continuation prompt

`_generate_continuation` builds a prompt that includes the full tool
trace as conversation history. The format follows **Hermes-Pro /
Qwen2.5 function-call** conventions for maximum model compatibility:

```text
<|im_start|>system
{existing CordBeat system prompt}

You can call tools using this exact syntax inside your reply:

  [SKILL: tool_name | param1=value1 | param2=value2]

You may emit one OR multiple tags per turn — all will be executed in
order and their results returned together as <tool_response> blocks.
Prefer a single tag when later calls depend on earlier results; emit
multiple only for clearly independent operations. Based on the results,
you may either:
  - Emit further [SKILL: ...] tags to call more tools
  - Or produce the final natural-language reply to the user (no tag)
<|im_end|>

<|im_start|>user
{original user message}
<|im_end|>

<|im_start|>assistant
{first AI response, containing the original [SKILL: ...] tag}
<|im_end|>

<|im_start|>tool
<tool_response name="{skill_name}" status="{ok|error}">
{output, truncated to react.max_tool_output_chars}
</tool_response>
<|im_end|>

<|im_start|>assistant
```

Notes:
- We **do not** use OpenAI's structured `tools` parameter. Tag-based
  dispatch keeps the backend abstraction simple and works with vanilla
  `chat/completions`.
- For `openai_compat` (llama.cpp / vLLM), models that reject
  `role="tool"` can fall back to `role="user"` with a `<tool_response>`
  wrapper. The behavior is configurable.

### 3.3 Data structures

```python
@dataclass(frozen=True)
class SkillCall:
    name: str
    params: dict[str, Any]
    skill: SkillEntry | None      # Resolved registry entry (None if unknown)

@dataclass(frozen=True)
class ToolCallResult:
    status: Literal["ok", "error", "denied", "timeout"]
    output: str                   # Already truncated to max_tool_output_chars
    raw: dict[str, Any]           # Full result for logging
    duration_ms: int

@dataclass(frozen=True)
class ToolCallRecord:
    call: SkillCall
    result: ToolCallResult
    iteration: int
```

### 3.4 Config additions

`src/cordbeat/config.py`:

```python
class ReActConfig(BaseModel):
    """Multi-step skill-execution loop configuration."""

    enabled: bool = True
    """Master switch. False raises an error at startup (no legacy fallback)."""

    max_iterations: int = 3
    """Maximum LLM round-trips per user turn. Must be >= 1."""

    max_tool_output_chars: int = 4000
    """Tool output is truncated to this length before re-injection."""

    per_iteration_timeout_seconds: int = 120
    """Hard limit per skill+LLM round to bound worst-case latency."""

    tool_message_role: Literal["tool", "user"] = "tool"
    """Some backends reject role='tool'. Set to 'user' for compatibility."""

    expose_trace_to_user: bool = False
    """If True, append a debug-formatted trace block to the final reply.
    Useful during development; default False for clean UX."""

    streaming_mode: Literal["split", "combined"] = "split"
    """How to surface intermediate text. 'split' sends pre-tag text as a
    separate adapter message immediately (Claude/ChatGPT-like UX).
    'combined' buffers everything and sends once at the end."""
```

Wired into `Config.react: ReActConfig = Field(default_factory=ReActConfig)`.

---

## 4. Design decisions (decision log)

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| D1 | Native `tools` API vs textual tag? | **Textual tag** `[SKILL:...]` | Backend-agnostic; works on llama.cpp/Ollama without function-calling support. |
| D2 | Hermes/OpenAI/Anthropic format? | **Hermes-style** `<tool_response>` + ChatML roles | Best multi-backend compatibility; widely trained on by recent models. |
| D3 | Max iterations | **Default 3, configurable** | Empirically `web_search → fetch_url → reply` = 3 calls. Bounds cost. |
| D4 | Backward compatibility for legacy 1-shot behavior | **Not preserved** | ReAct becomes the only path. Removing `max_iterations=1` legacy mode keeps the code lean and avoids two parallel execution flows. |
| D5 | Confirm-required skill mid-loop | **Break loop, queue proposal** | Mid-loop async approval is a state-machine nightmare. Next user message restarts. |
| D6 | Tool error handling | **Pass `{"error": "..."}` to AI** | AI can decide retry / alternative / abandon. Errors no longer silent. |
| D7 | Where to store the trace | **Debug log only, NOT memory** | Tool noise pollutes conversation memory; the final reply is the durable artifact. |
| D8 | HEARTBEAT integration | **Out of scope** | Heartbeat has `HeartbeatDecision.action=SKILL` separate path. Adding ReAct there is v1.2+. |
| D9 | Parallel skill calls | **Sequential, but multiple per response allowed** | Causality often matters (fetch_url depends on web_search), so execution stays sequential. But the AI can emit several tags in one response and CordBeat will run them in order before re-prompting — matching Anthropic / OpenAI semantics (D15). True async-parallel is v1.2+. |
| D10 | Streaming intermediate | **No** | Adapter typing-indicator is enough; partial-stream UX is complex per platform. |
| D11 | Trace visibility to user | **Hidden by default, opt-in via config** | Clean UX; devs can flip `expose_trace_to_user=True`. |
| D12 | Fallback if loop exhausted with no reply | **Synthesize from trace** | Avoid an empty user-facing reply (e.g., "Called N tools but did not reach a conclusion; latest result: ..."). |
| D13 | Text **before** the `[SKILL:]` tag | **Forward to adapter immediately as a separate message** | Matches Claude / ChatGPT progress-message UX. Acknowledges the user during tool latency. |
| D14 | Text **after** the (last) `[SKILL:]` tag in the same response | **Discard** | Written before the AI saw any tool result; likely inaccurate or hallucinated. Same policy applies to prose **between** consecutive tags. |
| D15 | Multiple `[SKILL:]` tags in one response | **Execute ALL sequentially in document order; batch results back to AI** | Matches Anthropic (`one or more tool_use blocks`) and OpenAI (`tool_calls: []` array). Errors per-tool returned as `{"error": ...}`, batch continues. `requires_confirmation` mid-batch halts execution and queues a single proposal (remaining tags dropped, partial results discarded). |
| D16 | Streaming mode | **`split` default** (pre-tag text sent immediately as separate message); `combined` available for rate-limited adapters | Best UX matches industry leaders; combined is escape hatch. |

---

## 5. Component impact

### 5.1 `src/cordbeat/core/engine.py`

- Replace `_dispatch_skill_tags` with `_react_loop`
- Extract `_parse_skill_call(match) -> SkillCall` helper
- Extract `_execute_call_safe(call) -> ToolCallResult` (uniform error → result conversion)
- Add `_generate_continuation(user_id, user, message, trace) -> str | None`
- Add `_summarize_trace_fallback(trace) -> str`
- Update `_process_chat_message`:
  `response = await self._react_loop(response, user_id, user, message)`

### 5.2 `src/cordbeat/ai/prompt.py`

- Add `build_react_continuation_prompt(base_prompt, original_user_msg, trace) -> list[ChatMessage]`
- New constant `REACT_SYSTEM_SUFFIX` describing the tool-call protocol

### 5.3 `src/cordbeat/ai/backend.py`

- Add `generate_chat(messages: list[ChatMessage], **kwargs) -> str` if
  not already exposed (current `generate` takes a flat prompt string
  for Ollama)
- `OllamaBackend`: use `/api/chat` with `messages`
- `OpenAICompatBackend` / `OpenAIBackend`: already use `chat/completions`
- For backends that reject `role="tool"`: fall back to `role="user"`
  (configured)

### 5.4 `src/cordbeat/config.py`

- Add `ReActConfig` (§3.4)
- Add `react: ReActConfig` to `Config`

### 5.5 `src/cordbeat/models.py`

- Add `SkillCall`, `ToolCallResult`, `ToolCallRecord` dataclasses

### 5.6 Tests

- `tests/test_engine.py`:
  - `test_react_loop_single_skill` — AI emits one tag, sees the result,
    replies (legacy behavior regression with `max_iterations=3`)
  - `test_react_loop_two_skills` — `web_search` → `fetch_url` → final
    reply (both mocked)
  - `test_react_loop_error_passed_to_ai` — first skill errors; AI sees
    the error and recovers
  - `test_react_loop_max_iterations_exhausted` — AI keeps emitting tags;
    loop bound enforced
  - `test_react_loop_unknown_skill` — tag for a nonexistent skill is
    stripped; AI continues
  - `test_react_loop_confirmation_breaks_loop` — `requires_confirmation`
    skill queues a proposal and exits
  - `test_react_loop_disabled_falls_back` — `enabled=False` → legacy
    behavior
- `tests/test_config.py`: `ReActConfig` defaults and validation
- Use existing `MockAIBackend` patterns

### 5.7 Documentation

- `docs/engine.md`: Update Phase-3 description
- `docs/skills.md`: Add "How skills are invoked: the ReAct loop" section
- `docs/config-reference.md`: Document `react.*` options
- `CHANGELOG.md`: Under [Unreleased] / Added

---

## 6. Performance & cost analysis

### 6.1 Latency

Empirically (Qwen3-30B-A3B local, 2026-05-19 logs):
- Single LLM call: 30–80 s (incl. up to ~3500 reasoning tokens)
- 3-iteration ReAct: **90–240 s worst case**

Mitigation: `per_iteration_timeout_seconds=120` caps each round.

### 6.2 Token cost

Each iteration re-sends the entire conversation. With
`max_tool_output_chars=4000`:
- Iter 1: prompt ≈ 2k tokens
- Iter 2: prompt ≈ 2k + (response 1) + (tool result 4k chars ≈ 1.3k tokens) ≈ 5k
- Iter 3: ≈ 8k

Acceptable for local models with 8k–32k context. For OpenAI API users we
recommend `max_iterations=2` and `max_tool_output_chars=2000`.

### 6.3 Failure modes

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| AI never stops emitting tags | `iteration == max_iterations` | Force-strip remaining tags, synthesize trace fallback |
| Same skill called repeatedly with same params | Trace hash check (post-v1.1 enhancement) | Log warning; optionally inject "you already tried this" hint |
| Per-iteration timeout | `asyncio.wait_for` raises | Record as `status=timeout`; AI may retry or abandon |
| Backend rejects `role=tool` | Detected at first call | Auto-fallback to `role=user` (logged once) |
| Empty AI response in continuation | `response is None` | Break loop, return trace fallback |

---

## 7. Migration plan

### 7.1 Rollout

ReAct **replaces** the legacy 1-shot `_dispatch_skill_tags` path. There
is no parallel old-mode runtime to maintain. The migration is a single
PR-set (R-1 → R-4) that removes the old code in R-2.

### 7.2 Migration checklist

- ✅ Existing `[SKILL: ...]` syntax unchanged
- ✅ `requires_confirmation` proposal flow unchanged
- ✅ HEARTBEAT skill-dispatch path untouched (separate code)
- ✅ Memory storage of final reply only — no change to
  `extract_and_store_memories`
- ⚠️  Tests that asserted "single skill executed inline" must be
  rewritten — the new path produces a multi-turn trace
- ⚠️  Skill output no longer leaks raw into the user-facing reply.
  Tests/users relying on that visibility must read the trace via
  `expose_trace_to_user=True` or debug logs.

### 7.3 Telemetry

Add structured logs (use existing `logger.info` patterns):

```python
logger.info(
    "react_loop_complete user=%s iterations=%d total_ms=%d skills=%s",
    user_id, len(trace), elapsed_ms, [r.call.name for r in trace],
)
```

Future: emit OpenTelemetry spans (depends on D-2 observability work).

---

## 8. Open questions

These should be resolved before implementation begins:

- **OQ1**: Should the user's *original* message be re-included verbatim
  each iteration, or implied via the system prompt? (Recommendation:
  re-include for clarity, accept the token cost.)
- **OQ2**: Should we sanitize tool output for prompt-injection attempts
  (e.g., `</tool_response>` smuggling)? (Recommendation: yes — strip
  the closing-tag pattern from tool output before injection.)
- **OQ3**: Should we expose the trace to the **soul / emotion** subsystem?
  (Recommendation: no for v1.1. Emotion already triggers on the final
  reply.)
- **OQ4**: How does this interact with the **draw** skill (which returns
  images)? (Recommendation: image-producing skills always terminate the
  loop — AI cannot reason over PNG bytes anyway.)
- **OQ5**: Should `_react_loop` be cancellable via `/stop` mid-iteration?
  (Recommendation: yes — check `_stop_flags` before each LLM call.)
- **OQ6**: For `split` streaming mode, should we add an aggregation
  hint (e.g., a leading `…` or trailing `(続く)` marker) so the user
  knows more is coming? (Recommendation: no for v1.1, evaluate after
  user feedback.)

---

## 9. Implementation order (post-v1.0)

1. **PR R-1**: `ReActConfig` + dataclasses (`SkillCall`, `ToolCallResult`,
   `ToolCallRecord`) — additive only, no behavior change
2. **PR R-2**: Replace `_dispatch_skill_tags` with `_react_loop` —
   removes legacy 1-shot path, all tests adapted to multi-iteration
3. **PR R-3**: Continuation prompt + adapter split-streaming + tests
4. **PR R-4**: Documentation + CHANGELOG + open-question resolution
   (OQ2 prompt-injection sanitization, OQ4 image-skill terminator, OQ5
   `/stop` cancellation)

---

## 10. References

### Industry implementations surveyed

The "Multi-tool per response" column has been **verified against
primary documentation** (Anthropic, OpenAI) for the top two rows;
others are inferred from their open-source implementations and model
training cards.

| Implementation | Multi-tool per response | Pre-tool text shown | Notes |
|----------------|-------------------------|---------------------|-------|
| **Anthropic Claude API** | ✅ Verified — "one or more `tool_use` blocks" per response ([docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)) | Yes (text block streamed) | Native `tool_use` / `tool_result` content blocks |
| **OpenAI Function Calling** | ✅ Verified — `tool_calls: []` array, `finish_reason: "tool_calls"` (plural), `parallel_tool_calls=true` default ([cookbook](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models)) | Yes (`content` field alongside calls) | Native structured API |
| **Claude Code** | Yes (uses Anthropic API) | Yes — "Let me check…" pattern | In practice, 1 tool/turn is the norm except for parallel reads |
| **Cursor / Cline** | Capable but rarely emit >1 | Yes | Mostly 1 tool/turn |
| **Nous Hermes-2-Pro** | Capable (`<tool_call>` blocks) | Yes | Fine-tuned to prefer 1/turn |
| **Qwen-Agent / Qwen2.5** | Capable | Yes | Same as Hermes |
| **LangChain ReAct (classic)** | No (1 Action per step) | Thought is internal | Older paradigm |
| **OpenHands / SWE-agent** | Usually 1 | Yes | XML-block style |

**Takeaway**: Multi-tool-per-response is the industry standard (Claude,
OpenAI), used freely for independent operations. CordBeat now matches
that semantic (D15) so the model can naturally express both dependent
chains (one tag at a time) and independent batches (multiple tags
together) without architectural friction.

### Primary sources

- Nous Research, *Hermes Function Calling v1*:
  https://github.com/NousResearch/Hermes-Function-Calling
- OpenAI, *Function calling guide*:
  https://platform.openai.com/docs/guides/function-calling
- Anthropic, *Tool use*:
  https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*
  (2022)
- CordBeat current engine: `src/cordbeat/core/engine.py::_dispatch_skill_tags`
  (lines 409–466)
