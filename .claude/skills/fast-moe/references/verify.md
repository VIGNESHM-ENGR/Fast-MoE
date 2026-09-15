# Verify

Run the checks that match what changed. Report the numbers you observed, not that it "should work".

## Every change

```bash
.venv/bin/ruff check engine ui tests
.venv/bin/python -m pytest -q
```

Both must pass. Tests need no GPU, llama.cpp build or model.

## Engine change: real server

```bash
python -m engine.serve --model models/<repo-name>/<file>.gguf --port 8082
curl -s http://127.0.0.1:8082/health
curl -s http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":40}'
```

Check the printed plan (layers on GPU / split / RAM, context) and the response `timings`
(`predicted_per_second`). Stop with Ctrl+C and confirm `pgrep -x llama-server` is empty.

## UI change: browser walkthrough

Use Playwright in a scratch venv, outside the repo:

```bash
python3 -m venv /tmp/pw && /tmp/pw/bin/pip install playwright && /tmp/pw/bin/python -m playwright install chromium
```

Walk the golden path against a running dashboard, collect `pageerror` events, and screenshot
desktop (1440 wide) and mobile (390 wide):

```python
page.goto("http://127.0.0.1:7860/")
page.wait_for_selector(".fm-visualizer-wrapper")
page.get_by_role("button", name="Preview Placement").click()
page.wait_for_selector(".fm-layer-item", timeout=300_000)
page.get_by_role("button", name="Apply & Start").click()
page.wait_for_selector(".fm-status-pill.state-live", timeout=600_000)
page.get_by_role("tab", name="💬 Chat Studio").click()
page.get_by_placeholder("Type a message to the running model...").fill("Hello")
page.get_by_role("button", name="Send", exact=True).click()
page.wait_for_selector(".fm-hud-state-pill.state-completed", timeout=600_000)
```

Open every screenshot and look at it. Confirm: no page errors, numbers match the real plan,
empty states say what to do next, nothing overflows at 390 px.

## Docker change

```bash
docker compose build && docker compose up -d
curl -sf http://127.0.0.1:7860/ && docker compose ps
```

Run the walkthrough against port 7860, and check the host reaches the API at
`http://127.0.0.1:8080/v1/models` while a model runs. Repeat with
`docker compose -f docker-compose.cpu.yml` for CPU changes. Finish with `docker compose down`.

## start.sh change

Run `shellcheck start.sh scripts/*.sh` (or `docker run --rm -v "$PWD:/mnt" koalaman/shellcheck:stable
/mnt/start.sh`). Then drive it like a user from Python in its own process group: wait for
"Press Ctrl+C", check `http://127.0.0.1:7860/` returns 200, send SIGINT to the group, and
confirm "Stopped.", exit code 130 and no `fast-moe` containers left. Cover `auto`, `gpu`,
`cpu`, a busy port 7860, and a hidden GPU (a fake failing `nvidia-smi` first on `PATH`):
`gpu` must refuse and `auto` must fall back to CPU.

## CI

After pushing: `gh run watch <run-id> --exit-status`, then
`gh run view <run-id> --json jobs --jq '.jobs[] | "\(.name): \(.conclusion)"'`. Jobs: `lint`,
`test (3.10)`, `test (3.12)`, `docker` (CPU image build plus a smoke test run from `/`).

## Reference measurements

RTX 3060 Laptop GPU (6 GB), i5-11400H, 38 GB RAM, NVMe. A result far from these on the same
machine is a regression to explain.

| Model (Q4_K_M) | Placement at 4K context | Live after Apply | Decode |
|---|---|---|---|
| Qwen3.6-35B-A3B | layers 0-3 GPU, 4 split, 5-39 RAM | 19 s | 18.3 tok/s native, 18.1 Docker |
| Qwen3.6-35B-A3B, CPU-only Docker | all 40 layers RAM, 262,144 ctx | 54 s | 3.9 tok/s |
| Qwen3-30B-A3B | layers 0-7 GPU, 8 split, 9-47 RAM | 7 s (`engine.serve`) | 16-18 tok/s |
| Gemma 4 26B-A4B (UD-Q4_K_M) | layers 0-1 GPU, 2 split, 3-29 RAM | 15 s (preview 22 s) | 10.2 tok/s, TTFT 3.6 s |
