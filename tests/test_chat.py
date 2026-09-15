"""Chat Studio request flow against a fake OpenAI stream (no server)."""

from types import SimpleNamespace

import openai
import pytest

from ui import app


def chunk(content=None, reasoning=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None, model_extra={})


class FakeStream:
    def __init__(self, chunks, on_chunk=None):
        self.chunks, self.on_chunk, self.closed = chunks, on_chunk, False

    def __iter__(self):
        for i, c in enumerate(self.chunks):
            if self.closed:
                raise RuntimeError("stream closed")
            if self.on_chunk:
                self.on_chunk(i)
            yield c

    def close(self):
        self.closed = True


@pytest.fixture
def server(monkeypatch):
    sent = {}
    monkeypatch.setattr(app.runner, "server", SimpleNamespace(url="http://x", proc=SimpleNamespace(poll=lambda: None)))
    monkeypatch.setattr(app.runner, "plan", None)
    fake = SimpleNamespace(stream=FakeStream([chunk(reasoning="hmm"), chunk(content="Red.")]))

    class Client:
        def __init__(self, **_):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            sent.update(kwargs)
            return fake.stream

    monkeypatch.setattr(openai, "OpenAI", Client)
    fake.sent = sent
    return fake


def run_chat(message="Colours?", history=(), reasoning=False, temperature=0.3):
    return list(app.chat(message, list(history), "", temperature, 0.9, 20, 0.0, 0.0, 1.0, 512, reasoning))


def test_reasoning_off_is_sent_as_off(server):
    # Regression: the checkbox value was overwritten by the thinking message, so reasoning was always on.
    run_chat(reasoning=False)
    assert server.sent["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert server.sent["temperature"] == 0.3 and server.sent["extra_body"]["top_k"] == 20


def test_thinking_collapses_when_the_answer_starts_and_settings_are_shown(server):
    _, history, hud = run_chat(reasoning=True)[-1]
    thought = next(m for m in history if (m.get("metadata") or {}).get("title"))
    assert thought["metadata"]["status"] == "done"
    assert history[-1]["content"] == "Red."
    assert "reasoning on" in hud and "temp 0.3" in hud


def test_stop_closes_the_stream_and_keeps_the_partial_answer(server):
    server.stream = FakeStream([chunk(content="Red"), chunk(content=", blue"), chunk(content=", green")],
                               on_chunk=lambda i: app.stop_chat() if i == 1 else None)
    _, history, hud = run_chat()[-1]
    assert server.stream.closed
    assert history[-1]["content"].startswith("Red") and "Stopped" in history[-1]["content"]
    assert "green" not in history[-1]["content"]
    assert "state-stopped" in hud and not app.active_chat["busy"]


def test_retry_regenerates_from_the_chosen_user_message(server):
    history = [{"role": "user", "content": "Colours?"}, {"role": "assistant", "content": "Old."}]
    message, kept = app.prepare_retry(history, SimpleNamespace(index=0))
    final = run_chat(message, kept)[-1][1]
    assert final[0]["content"] == "Colours?" and final[-1]["content"] == "Red." and "Old." not in str(final)


def test_send_while_streaming_changes_nothing(server):
    app.active_chat["busy"] = True
    try:
        out = run_chat()
    finally:
        app.active_chat["busy"] = False
    assert len(out) == 1 and all(isinstance(x, type(app.gr.skip())) for x in out[0])
    assert not server.sent
