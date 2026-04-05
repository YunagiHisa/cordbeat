# HEARTBEAT — The Autonomous Loop

## Concept

The HEARTBEAT is CordBeat's pulse.  
Not a loop for executing tasks — it's **proof that this AI is alive and thinking**.

---

## Two-Layer Structure

### Global HEARTBEAT (lightweight, periodic)
Reviews a summary of all users and decides: "Should I take action toward someone?"

```
All user summaries (lightweight):
┌──────────────────────────────────┐
│ Alex:  6 hours no response, dev  │
│ User A: 1 day no response, normal│
│ User B: Chatted 30 mins ago     │
└──────────────────────────────────┘
        ↓ AI decides
"Alex is the one I'm most concerned about"
        ↓
per-user HEARTBEAT triggered
```

### Per-User HEARTBEAT (detailed, targeted)
Loads only the context for one target user to decide on action.  
User count doesn't cause context explosion.

---

## One Cycle

```
1. Gather Context
   - Time since last conversation
   - Current time and day
   - User's recent memory/state
   - Character's emotional state
   - Any pending events

2. AI Evaluates & Decides (one inference)
   "Should I do something now?"
   "If so, what?"
   "When should the next HEARTBEAT be?"

3. Take Action
   A. Send a message
   B. Execute a skill
   C. Propose a self-improvement
   D. Do nothing (stay quiet)

4. Update State
   - Update emotional state
   - Log the action
   - Set next HEARTBEAT time

5. Sleep → Wake at scheduled time
```

---

## AI Response Format

```json
{
  "action": "message",
  "content": "Hey, how's the project coming along? I've been curious.",
  "next_heartbeat_minutes": 120
}
```

### Action Types

| Action | Description |
|---|---|
| `message` | Send a message to the user |
| `skill` | Execute a skill |
| `propose_change` | Propose a self-improvement |
| `skip` | Do nothing (stay quiet) |
