# SKILL — Pluggable Actions

## Concept

A SKILL is a unit of action CordBeat can perform.  
One skill = one directory. Drop a directory in, and it's auto-detected.

---

## Directory Structure

```
skills/
  my_custom_skill/
    skill.yaml
    main.py
  another_skill/
    skill.yaml
    main.py
```

Skills are user-created. Drop a directory containing `skill.yaml` and
`main.py` into the skills directory, and CordBeat auto-detects it.

---

## skill.yaml Format

```yaml
name: web_search
description: "Performs web searches and returns results"
version: "1.0.0"
author: "cordbeat"

usage: |
  Use when you need to search for information on the web.
  Effective for latest news, weather, etc.

parameters:
  - name: query
    type: string
    required: true
    description: "Search query"
  - name: max_results
    type: integer
    required: false
    default: 5

safety:
  level: safe
  sandbox: false
  network: true
  filesystem: false
```

---

## Safety Levels

| Level | Description | Examples |
|---|---|---|
| `safe` | AI can execute autonomously | Web search, weather, timer |
| `requires_confirmation` | Requires user approval | File writes, API calls |
| `dangerous` | Blocked from HEARTBEAT execution entirely | Shell commands, system ops |

### `requires_confirmation` Flow

When a HEARTBEAT decision attempts to run a `requires_confirmation` skill,
the system creates a **proposal** instead of executing immediately:

1. A `skill_execution` proposal is stored with `status=pending`
2. The user is notified with the skill name, parameters, and proposal ID
3. The user can approve or reject the proposal
4. On the next heartbeat tick, approved proposals are executed
5. Executed proposals are marked `executed`; failures are marked `expired`

See [Heartbeat — Proposal Approval System](heartbeat.md#proposal-approval-system)
for the full lifecycle.

---

## Integrity Verification

Skills can declare a SHA-256 hash in `skill.yaml` for integrity checks.
When present, CordBeat verifies `main.py` against the hash before loading:

```yaml
integrity:
  sha256: "a1b2c3d4..."
```

If the hash does not match, the skill is rejected with an error. Skills
without an `integrity` field are loaded normally (backwards compatible).
