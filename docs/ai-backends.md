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

### OpenAI-compatible

Works with any server that implements the OpenAI Chat Completions API:
vLLM, LM Studio, text-generation-webui, LocalAI, etc.

```yaml
ai_backend:
  provider: openai
  base_url: "http://localhost:8000"
  model: "my-model"
  options:
    api_key: "sk-..."  # optional, depends on server
```

Uses the `/v1/chat/completions` endpoint.

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
