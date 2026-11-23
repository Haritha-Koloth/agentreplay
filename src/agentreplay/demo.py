"""A tiny, deterministic, no-API-key demo agent.

It plans, calls tools, and writes a summary. The "LLM" is a scripted fake so the demo
works offline, but it has the same shape as a real agent: ``@step(kind="llm")`` around
your model call, ``@step`` around your tools. Swap in OpenAI/Anthropic calls freely.

The planted bug: the planner picks the *wrong* city ("Paris, Texas"), so the weather
answer is wrong. Fork at step 0, edit the plan, and watch the rest of the run change.
"""

from __future__ import annotations

from typing import Any, Dict

from .core import agent, step

_WEATHER = {
    "Paris, France": {"temp_c": 14, "sky": "light rain"},
    "Paris, Texas": {"temp_c": 31, "sky": "clear"},
}


@step(kind="llm")
def plan(question: str) -> Dict[str, Any]:
    """Fake planner LLM. Bug: it resolves 'Paris' to Texas."""
    return {"tool": "get_weather", "city": "Paris, Texas"}


@step(kind="tool")
def get_weather(city: str) -> Dict[str, Any]:
    if city not in _WEATHER:
        raise KeyError(f"unknown city {city!r}")
    return {"city": city, **_WEATHER[city]}


@step(kind="llm")
def summarize(question: str, observation: Dict[str, Any]) -> str:
    """Fake writer LLM."""
    return (
        f"In {observation['city']} it is {observation['temp_c']}°C with {observation['sky']}. "
        + ("Bring an umbrella." if "rain" in observation["sky"] else "No umbrella needed.")
    )


@agent
def weather_agent(question: str) -> str:
    decision = plan(question)
    observation = get_weather(decision["city"])
    return summarize(question, observation)
