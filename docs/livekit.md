# Using Fast-MoE as the LLM in a LiveKit voice agent

Fast-MoE needs no extra code for this. While a model runs, llama-server serves an
OpenAI-compatible API at `http://<host>:8080/v1` (streaming and tool calls included), and
LiveKit Agents' OpenAI plugin can talk to any such endpoint.

Tested 2026-10-08 with `livekit-agents` / `livekit-plugins-openai` 1.8.5 and Qwen3.6-35B-A3B
Q4_K_M on an RTX 3060 Laptop (6 GB): streamed replies, a tool call
(`get_weather({"city": "Paris"})`), and a 401 for a wrong API key.

## 1. Start the model server

Use the headless launcher; arguments after `--` go straight to llama-server:

```bash
python -m engine.serve --ctx 16384 -- \
  --alias qwen3.6 \
  --api-key "$FAST_MOE_API_KEY" \
  --chat-template-kwargs '{"enable_thinking":false}'
```

| Flag | Why |
|---|---|
| `--alias qwen3.6` | the model name clients send; any name works |
| `--api-key ...` | rejects requests without `Authorization: Bearer <key>`; set it whenever the port is reachable from other machines |
| `--chat-template-kwargs '{"enable_thinking":false}'` | Qwen3.6 thinks before answering by default (about 1,200 tokens, a minute here, for one tool call); a voice turn can't wait that long |
| `--ctx 16384` | a voice conversation rarely needs more; smaller context leaves more VRAM for expert weights |

Check it: `curl http://127.0.0.1:8080/health` returns `{"status":"ok"}`.

**Agent on another machine?** By default the server listens on `127.0.0.1` only. Add
`--host 0.0.0.0` (or set `FAST_MOE_API_HOST=0.0.0.0`) and always set `--api-key`. With
Docker, `docker-compose.yml` publishes `127.0.0.1:8080:8080`; change it to `8080:8080` to
reach it from the network. The dashboard (`python -m ui.app`) starts the same server, but
without these extra flags.

## 2. Point the LiveKit agent at it

```bash
pip install "livekit-agents[openai]"
```

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai

llm = openai.LLM(
    model="qwen3.6",                      # the --alias above
    base_url="http://127.0.0.1:8080/v1",  # or http://<fast-moe-host>:8080/v1
    api_key="<same key as --api-key>",     # any string if no key is set
)

session = AgentSession(
    llm=llm,
    # stt=..., tts=..., vad=..., turn_detection=...  (your usual providers)
)

agent = Agent(instructions=(
    "You are a voice assistant. Answer in one short sentence. "
    "You have no live data: whenever a tool can answer, call the tool instead of guessing."
))
```

Everything else (rooms, STT, TTS, `@function_tool`s) works as in any LiveKit agent; only the
LLM changes.

## Tool calls need a direct instruction

With thinking off, Qwen3.6 sometimes answers from memory instead of calling a tool. Asked
"What's the weather in Paris right now?" with a `get_weather` tool, it replied "The weather in
Paris is currently clear." Adding *"You have no live data: whenever a tool can answer, call
the tool instead of guessing"* to the instructions made it call the tool in every run (3 of 3
raw, plus through the LiveKit plugin), about 1.3 s per call. Keep a line like that in your
agent's instructions.

## Measured latency (RTX 3060 Laptop 6 GB, Qwen3.6-35B-A3B, thinking off)

| Request | First token | Total |
|---|---|---|
| "Hi, who are you?" | 1.0–2.0 s | 1.5–2.3 s |
| Tool call | 1.3 s warm, up to 4.9 s for a new prompt | same (the call is the whole reply) |

The first request after start and any new system prompt pay for prompt processing; later
turns reuse the cached prefix.

## Troubleshooting

- **401 Invalid API Key**: `api_key` in the agent doesn't match `--api-key`.
- **Connection refused from another machine**: the server is on `127.0.0.1`; see step 1.
- **Long silence before each reply**: thinking is on; add the `--chat-template-kwargs` flag.
- **Model invents answers instead of calling tools**: add the instruction above.
