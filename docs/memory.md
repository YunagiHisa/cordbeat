# MEMORY — Per-User Isolated Context

## Concept

MEMORY enables CordBeat to "remember the user."  
Not mere data storage — it is **designed to mimic human memory structures**.

---

## 4-Layer Architecture

```
┌─────────────────────────────────────┐
│  Layer 1: Core Profile (SQLite)     │
│  No forgetting, always loaded       │
│  Name, language, basic traits       │
├─────────────────────────────────────┤
│  Layer 2: Semantic Memory (ChromaDB)│
│  Forgetting + reinforcement         │
│  ├ Preferences, goals, hobbies      │
│  ├ Procedural memory                │
│  │  "Prefers short replies"         │
│  └ Context-dependent memory         │
│     "Different vibe late at night"  │
├─────────────────────────────────────┤
│  Layer 3: Episodic Memory (ChromaDB)│
│  Forgetting + emotional weighting   │
│  ├ Normal episodes (with forgetting)│
│  └ Flashbulb memories               │
│     "The day we first talked"       │
│     emotional_weight=1.0, no decay  │
├─────────────────────────────────────┤
│  Layer 4: Reliable Records          │
│  (SQLite + text)                    │
│  No forgetting, searchable          │
│  ├ Diary (auto-generated in sleep)  │
│  └ Important logs (explicit)        │
└─────────────────────────────────────┘
```

---

## Memory Confidence Levels

| Level | Description |
|---|---|
| 3 | Certain (recorded in diary/logs) |
| 2 | Normal (vector memory, high strength) |
| 1 | Vague (low strength) |
| 0 | Nearly forgotten (archived) |

---

## Forgetting Model

Based on Ebbinghaus forgetting curve:

```
strength = base_strength × (1 / (1 + decay_rate × elapsed_days))

# Reinforced when referenced
strength += reinforcement_bonus

# Emotional memories fade slower
decay_rate *= (1 - emotional_weight)
```

### Strength-Based State Transitions

| Strength | State | Behavior |
|---|---|---|
| 1.0 – 0.3 | Vivid | Loaded into context |
| 0.3 – 0.1 | Vague | Loaded only on search |
| 0.1 – 0.01 | Nearly forgotten | Not loaded, kept in DB |
| < 0.01 | Forgotten | Archived (never fully deleted) |

---

## Sleep Phase Memory Consolidation

During quiet hours, the AI automatically:
1. Reviews the day's conversations
2. Generates a diary entry
3. Consolidates and adjusts memory strengths
4. Archives decayed memories
