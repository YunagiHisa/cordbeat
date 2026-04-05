# AI Output Validation

## Core Principle

> "Always distrust AI output."

Never save AI-generated values directly to the database.  
Always validate first. Protect existing data from anomalous values.

---

## Three Layers of Defense

### Layer 1: Minimize What AI Touches

System-managed fields (user_id, timestamps, platform info) are never AI-generated.  
The AI only generates content fields (text, scores, topics).

```
AI generates:
{
  "last_topic": "OSS project design",
  "emotional_tone": "excited",
  "attention_score": 0.8
}

System provides:
{
  "user_id": "cb_0001",
  "last_talked_at": "2026-03-31T20:00:00",
  "last_platform": "discord"
}
```

### Layer 2: Validation

Check AI output against type, range, and required field constraints.  
On failure, collect all errors and pass them to a retry prompt.

### Layer 3: Append/Merge Saves

Never overwrite — always merge.  
Even if an anomalous value slips through, existing data isn't destroyed.

---

## Retry Flow

```
AI generates (attempt 1)
  ↓
Validation
  ↓ FAIL → Collect error reasons
AI regenerates (with error feedback, max 2 retries)
  ↓ FAIL (attempt 3)
Keep previous value, log the failure
```

---

## Error Feedback Format

Multiple errors are batched into a single retry to save inference:

```
Your previous output was invalid. Please fix and regenerate.

Errors:
- attention_score is out of range (1.5 → must be 0–1)
- last_topic is too long (300 chars → max 50 chars)
```

---

## Fallback When Summary Fails

Summaries are "nice to have" — not required.  
On failure, the global HEARTBEAT falls back to DB-level minimum info (last conversation time).

| Scenario | Behavior |
|---|---|
| Validation OK | Save as-is |
| Fail (attempt 1–2) | Retry with error feedback |
| Fail (attempt 3) | Keep previous value, log failure |
| Summary unavailable | Fall back to DB minimum info |
