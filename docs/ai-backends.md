# AI Backends

CordBeat uses an abstracted AI backend that supports multiple providers.
The backend is configured in `config.yaml` under the `ai_backend` section.

---

## Supported Providers

### Ollama (default)

Local inference via the [Ollama](https://ollama.ai) HTTP API.

```yaml
ai_backend:
  provider: ollama
  base_url: "http://localhost:11434"
  model: "qwen3.5:9b"
  options:
    num_predict: 512
    temperature: 0.8
```

Uses the `/api/generate` endpoint with non-streaming mode.

### OpenAI / Strict OpenAI-compatible APIs

Works with hosted APIs that implement the standard OpenAI Chat Completions API,
including OpenAI, Google Gemini/Gemma OpenAI-compatible API, OpenRouter, Groq,
Together, and DeepInfra. Use `compatibility_mode: strict_openai` for these
providers so CordBeat does not send llama.cpp/vLLM-only fields.

```yaml
ai_backend:
  provider: openai_compat
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  model: "gemma-4-31b-it"
  timeout: 300.0
  max_tokens: 2048
  options:
    api_key: "${GEMINI_API_KEY}"
    compatibility_mode: "strict_openai"
    reasoning_mode: "none"
```

In `strict_openai` mode, CordBeat does not include `enable_thinking` or
`chat_template_kwargs` in the request payload, does not run no-think retries,
and does not drop a successful response just because its content looks like
reasoning or model-control text.

### llama.cpp

The [llama.cpp](https://github.com/ggerganov/llama.cpp) built-in server
exposes an OpenAI-compatible API. CordBeat auto-detects it at
`localhost:8080` during setup.

```yaml
ai_backend:
  provider: openai_compat
  base_url: "http://localhost:8080/v1"
  model: "default"
  max_tokens: 4096
  options:
    compatibility_mode: "llama_cpp"
    enable_thinking: false
    reasoning_content_keys: ["reasoning_content", "reasoning"]
    reasoning_strip_tags: ["think", "thinking", "analysis", "reasoning"]
```

Start the llama.cpp server:

```bash
llama-server -m your-model.gguf --port 8080
```

`llama_cpp` mode permits both `chat_template_kwargs.enable_thinking` and the
legacy top-level `enable_thinking` field. CordBeat may also retry once with
thinking disabled and drop reasoning-like output when the retry still fails.

### vLLM

vLLM also exposes an OpenAI-compatible API, but it is safer to avoid the
legacy top-level `enable_thinking` field.

```yaml
ai_backend:
  provider: openai_compat
  base_url: "http://localhost:8000/v1"
  model: "Qwen/Qwen3-..."
  max_tokens: 4096
  options:
    compatibility_mode: "vllm"
    enable_thinking: false
    reasoning_content_keys: ["reasoning", "reasoning_content"]
    reasoning_strip_tags: ["think", "thinking", "analysis", "reasoning"]
```

`vllm` mode permits `chat_template_kwargs.enable_thinking`, but not top-level
`enable_thinking`. CordBeat may run no-think retries for reasoning-only or
reasoning-like responses.

When `compatibility_mode` is omitted, CordBeat defaults to `strict_openai` for
remote OpenAI-compatible URLs. For existing local setups, `localhost`,
`127.0.0.1`, `0.0.0.0`, and `::1` are inferred as `llama_cpp` unless you set a
mode explicitly.

---

## Thinking Models

Some models (e.g., `qwen3.5:9b`) use internal "thinking" tokens before
producing visible output. For these models, set `num_predict` to 512 or
higher — lower values may result in the model exhausting its token budget
on thinking tokens with no visible response.

```yaml
ai_backend:
  options:
    num_predict: 512
```

---

## JSON Generation

`AIBackend.generate_json()` generates structured output by:

1. Calling `generate()` with a lower temperature (0.3 default)
2. Stripping markdown code fences if present
3. Parsing the result as JSON

This is used internally by the HEARTBEAT evaluation loop.

---

## Adding a New Provider

1. Create a class that inherits from `AIBackend`
2. Implement the `generate()` method
3. Register it in the `create_backend()` factory function

```python
class MyBackend(AIBackend):
    def __init__(self, config: AIBackendConfig) -> None:
        ...

    async def generate(self, prompt, system="", temperature=0.7, max_tokens=1024):
        ...
```

Then add a case to `create_backend()`:

```python
case "my_provider":
    return MyBackend(config)
```
