# Voice input and output

CordBeat keeps speech recognition and speech synthesis independent:

```text
Discord/Telegram audio -> faster-whisper STT -> Core/LLM
Core reply -> speech direction -> voice-design TTS -> Discord/Telegram audio
```

## Prompt-capable TTS

`tts.backend: voice_design` connects to an OpenAI-compatible
`POST /v1/audio/speech` endpoint and expects PCM WAV output. The model remains
loaded in the external API server; CordBeat is only an asynchronous client.

The public configuration is provider-neutral. The provider-specific request
extension is built inside the backend and is not exposed as a configuration
section.

```yaml
ai_backend:
  options:
    voice_enable_thinking: false

tts:
  enabled: true
  backend: "voice_design"
  api_url: "http://127.0.0.1:8088"
  model: "irodori-tts"
  timeout: 120.0
  response_format: "wav"

  voice_profile:
    description: >
      A calm, approachable adult voice with a warm and conversational personality.
    language_source: "soul"
    language: ""

  speech_direction:
    enabled: true
    history_turns: 3
    max_tokens: 128
    temperature: 0.2
    timeout: 3.0
    max_direction_chars: 180
    fallback: "Speak in a natural, conversational manner."

  streaming:
    enabled: true
    chunk_min_chars: 15
    chunk_max_chars: 45
    max_queue_size: 20

  backend_options:
    num_steps: 16
    retries: 1
    voice: "none"
```

`voice_profile.description` is free-form. It may describe gender, age, pitch,
energy, distance, and personality, or omit gender entirely. The profile stays
stable across a response so content-aware delivery does not replace the
speaker's identity.

When `language_source` is `soul`, the spoken language comes from
`data/soul/soul.yaml`:

```yaml
identity:
  language: ja
```

## Content-aware delivery

For voice replies only, Core sends a second, independent request to the loaded
main LLM after the reply is complete. The request has thinking disabled in the
voice context, a small output budget, and a strict timeout. Its system and user
prompts are English and its JSON output never enters the visible reply.

The speech director considers recent conversation, the latest user message,
the completed assistant response, Soul state, and the configured voice profile.
It returns an intent, tone, pace, energy, and a short English delivery direction.
On timeout or invalid JSON, CordBeat derives a safe direction from Soul state.

The final voice prompt is:

```text
stable voice profile + selected language + response delivery direction
```

## Chunked Discord playback

CordBeat normalizes chat markup, replaces URLs with `link`, and splits long
responses at sentence and punctuation boundaries. The TTS API receives one
chunk at a time. Discord starts the first WAV immediately while the next WAV is
being generated.

When a new reply arrives, the current chunk is allowed to finish. Generation
and queued playback for the older reply are cancelled, then playback switches
to the newer reply. Disconnecting or stopping the adapter cancels all remaining
work. Synthesis is serialized in CordBeat, and the external server should also
enforce a concurrency limit of one.

Telegram continues its existing behavior: it sends an audio reply only when
the user's latest input was a voice message. CordBeat merges compatible WAV
chunks by PCM frames before uploading one WAV file to Telegram.

## Verification

Check the external API:

```bash
curl -i http://127.0.0.1:8088/health
```

Start Core and the adapters in separate shells:

```bash
uv run cordbeat
uv run cordbeat-discord config.yaml
uv run cordbeat-telegram config.yaml
```

Run the voice tests:

```bash
uv run pytest tests/test_speech.py tests/test_voice.py \
  tests/test_discord_vc.py tests/test_adapters.py
```
