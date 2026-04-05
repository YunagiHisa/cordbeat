# SKILL — Pluggable Actions

## Concept

A SKILL is a unit of action CordBeat can perform.  
One skill = one directory. Drop a directory in, and it's auto-detected.

---

## Directory Structure

```
skills/
  web_search/        ← Built-in
    skill.yaml
    main.py
  file_ops/          ← Built-in
    skill.yaml
    main.py
  weather/           ← Built-in
    skill.yaml
    main.py
  shell_exec/        ← Built-in (dangerous, disabled by default)
    skill.yaml
    main.py
  my_custom_skill/   ← User-added
    skill.yaml
    main.py
```

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
| `dangerous` | Disabled by default, must be explicitly enabled | Shell commands, system ops |

---

## Built-in Skills

| Skill | Level | Description |
|---|---|---|
| web_search | safe | Web search and information retrieval |
| weather | safe | Weather and news |
| timer | safe | Timers and reminders |
| file_read | safe | File reading |
| file_write | requires_confirmation | File writing |
| api_call | requires_confirmation | External API calls |
| shell_exec | dangerous | Command execution (off by default) |
| read_diary | safe | Diary/log reference (MEMORY integration) |
