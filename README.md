# agentreplay

**Time-travel debugger for AI agents.** Record a run, click any step, edit its output (or input), and replay the agent from that point. Local-first, zero dependencies, no account, no SaaS.

```
step 0  plan        {"city": "Paris, Texas"}   <- the bug. Edit it to "Paris, France" and fork.
step 1  get_weather cached/live ...
step 2  summarize   ...
```

Forking reuses every step before the edit (no repeated LLM calls or cost), applies your change, and runs the rest live. The UI shows which steps changed against the parent run.

## Try it in 30 seconds (no API key)

```bash
pip install -e .
agentreplay demo
```

This records a demo agent that picks the wrong "Paris" and opens the UI. Select step **0 · plan**, change `"Paris, Texas"` to `"Paris, France"`, and click **Fork & replay**. You'll see the new answer, with the downstream steps re-run and the changed steps highlighted.

## Use it with your own agent

```python
from agentreplay import agent, step

@step(kind="llm")                      # record model calls
def plan(question): ...                # your OpenAI / Anthropic / local call

@step                                  # record tool calls
def search(query): ...

@agent                                 # one recorded run per call
def my_agent(question):
    decision = plan(question)
    return search(decision["query"])

my_agent("hello")                      # saved to ./.agentreplay/runs/
```

```bash
agentreplay serve --module my_module --open
```

`--module` imports your code so agents are registered; replay needs it to re-run your agent.
Sync and `async` functions both work.

## How replay works

For a fork at step `N`:

| Steps | Behavior |
|---|---|
| `0 .. N-1` | **cached**: recorded output returned, function not executed |
| `N` | **edited**: your output is used (skip), or your input is used (re-execute) |
| `N+1 ..` | **live**: executed for real |

Forked runs are saved as new runs linked to their parent, so you can branch repeatedly.

## Limits (v0.1)

- Agents must call steps in a deterministic order. If a replay diverges from the recorded order, caching stops and steps run live.
- Step outputs must be JSON-serializable (dataclasses and pydantic models are converted; other objects fall back to `repr`, which makes them unfaithful on replay).
- Only top-level steps are recorded. Calls nested inside another step are part of the outer step.
- Cached steps skip their side effects. Steps that write to external systems will not re-run when cached.
- The UI server binds to `127.0.0.1` and replay executes your agent code. Don't expose it to untrusted networks.

## Roadmap ideas

- OpenTelemetry / OpenInference trace import (LangChain, LlamaIndex, OpenAI Agents SDK)
- Auto-instrumentation for the `openai` and `anthropic` clients
- Step-level diff view, token/cost per step
- Mocking tools by recorded output in tests (`pytest` plugin)

## Development

```bash
pip install -e ".[dev]"
pytest
```

MIT licensed. Contributions welcome.
