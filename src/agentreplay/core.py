"""Core of agentreplay: record agent runs, then fork and replay them from any step.

How replay works
----------------
An agent run is a sequence of *steps* (calls to functions decorated with ``@step``).
To fork a recorded run at step ``N`` we re-invoke the agent with the same inputs, and:

* steps ``0 .. N-1`` return their recorded output without executing (``source="cached"``),
* step ``N`` returns an edited output, or executes with edited input (``source="edited"``),
* steps after ``N`` execute for real (``source="live"``).

This requires steps to be called in a deterministic order and (for faithful replay)
to return JSON-serializable values.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = ["step", "agent", "replay", "Run", "Step", "Store", "default_store"]

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_DEPTH = 20


# --------------------------------------------------------------------------- models


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Best-effort conversion of ``obj`` to something ``json.dumps`` accepts."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if _depth > _MAX_DEPTH:
        return repr(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, _depth + 1) for v in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj), _depth + 1)
    model_dump = getattr(obj, "model_dump", None)  # pydantic v2
    if callable(model_dump):
        return to_jsonable(model_dump(), _depth + 1)
    return repr(obj)


@dataclass
class Step:
    index: int
    name: str
    kind: str
    input: Dict[str, Any]
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    source: str = "live"  # live | cached | edited


@dataclass
class Run:
    id: str
    agent: str
    inputs: Dict[str, Any]
    steps: List[Step] = field(default_factory=list)
    status: str = "running"  # running | ok | error
    output: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    parent_id: Optional[str] = None
    fork_at: Optional[int] = None
    edit: Optional[Dict[str, Any]] = None

    @staticmethod
    def new(agent: str, inputs: Dict[str, Any], **kw: Any) -> "Run":
        return Run(id=uuid.uuid4().hex[:8], agent=agent, inputs=inputs, **kw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent,
            "status": self.status,
            "created_at": self.created_at,
            "duration_ms": self.duration_ms,
            "n_steps": len(self.steps),
            "parent_id": self.parent_id,
            "fork_at": self.fork_at,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Run":
        d = dict(d)
        d["steps"] = [Step(**s) for s in d.get("steps", [])]
        return Run(**d)


# --------------------------------------------------------------------------- storage


class Store:
    """One JSON file per run under ``<root>/runs``."""

    def __init__(self, root: "str | os.PathLike[str]") -> None:
        self.dir = Path(root) / "runs"

    def _path(self, run_id: str) -> Path:
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(f"invalid run id: {run_id!r}")
        return self.dir / f"{run_id}.json"

    def save(self, run: Run) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(run.id).with_suffix(".tmp")
        tmp.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self._path(run.id))

    def load(self, run_id: str) -> Run:
        path = self._path(run_id)
        if not path.exists():
            raise KeyError(run_id)
        return Run.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> List[Run]:
        if not self.dir.exists():
            return []
        runs = [Run.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in self.dir.glob("*.json")]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)


def default_store() -> Store:
    return Store(os.environ.get("AGENTREPLAY_DIR", ".agentreplay"))


# --------------------------------------------------------------------------- recorder


@dataclass
class _Decision:
    step: Step
    done: bool = False  # output is already known; do not execute the function
    edited_input: Optional[Dict[str, Any]] = None


class Recorder:
    def __init__(self, run: Run, parent: Optional[Run] = None) -> None:
        self.run = run
        self.parent = parent
        self.diverged = False  # set when the agent stops following the parent's step order

    def begin(self, name: str, kind: str, arguments: Dict[str, Any]) -> _Decision:
        idx = len(self.run.steps)
        step = Step(index=idx, name=name, kind=kind, input=to_jsonable(arguments))
        self.run.steps.append(step)
        d = _Decision(step)
        fork = self.run.fork_at
        if self.parent is None or fork is None or self.diverged:
            return d
        if idx < fork:
            p = self.parent.steps[idx] if idx < len(self.parent.steps) else None
            if p is not None and p.name == name and p.error is None:
                step.input, step.output = p.input, p.output
                step.duration_ms, step.source = p.duration_ms, "cached"
                d.done = True
            else:
                self.diverged = True
        elif idx == fork:
            edit = self.run.edit or {}
            if "output" in edit:
                step.output, step.source, d.done = edit["output"], "edited", True
            elif "input" in edit:
                step.input = {**step.input, **edit["input"]}
                step.source, d.edited_input = "edited", edit["input"]
        return d

    @staticmethod
    def finish(d: _Decision, output: Any, t0: float) -> None:
        d.step.output = to_jsonable(output)
        d.step.duration_ms = round((time.perf_counter() - t0) * 1000, 2)

    @staticmethod
    def fail(d: _Decision, exc: BaseException, t0: float) -> None:
        d.step.error = f"{type(exc).__name__}: {exc}"
        d.step.duration_ms = round((time.perf_counter() - t0) * 1000, 2)


_current: contextvars.ContextVar[Optional[Recorder]] = contextvars.ContextVar("agentreplay_recorder", default=None)
_in_step: contextvars.ContextVar[bool] = contextvars.ContextVar("agentreplay_in_step", default=False)
_AGENTS: Dict[str, Callable[..., Any]] = {}


# --------------------------------------------------------------------------- decorators


def step(fn: Optional[Callable[..., Any]] = None, *, name: Optional[str] = None, kind: str = "tool"):
    """Record a function call as a step. Use ``kind="llm"`` for model calls.

    Only top-level steps are recorded; steps called from inside another step run normally.
    """

    def deco(f: Callable[..., Any]) -> Callable[..., Any]:
        label = name or f.__name__
        sig = inspect.signature(f)

        def prepare(a: Any, k: Any):
            rec = _current.get()
            if rec is None or _in_step.get():
                return None, None, None
            bound = sig.bind(*a, **k)
            bound.apply_defaults()
            d = rec.begin(label, kind, dict(bound.arguments))
            if d.edited_input:
                bound.arguments.update(d.edited_input)
            return rec, bound, d

        if inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def awrapper(*a: Any, **k: Any) -> Any:
                rec, bound, d = prepare(a, k)
                if rec is None:
                    return await f(*a, **k)
                if d.done:
                    return d.step.output
                tok, t0 = _in_step.set(True), time.perf_counter()
                try:
                    out = await f(*bound.args, **bound.kwargs)
                except BaseException as e:
                    rec.fail(d, e, t0)
                    raise
                finally:
                    _in_step.reset(tok)
                rec.finish(d, out, t0)
                return out

            return awrapper

        @functools.wraps(f)
        def wrapper(*a: Any, **k: Any) -> Any:
            rec, bound, d = prepare(a, k)
            if rec is None:
                return f(*a, **k)
            if d.done:
                return d.step.output
            tok, t0 = _in_step.set(True), time.perf_counter()
            try:
                out = f(*bound.args, **bound.kwargs)
            except BaseException as e:
                rec.fail(d, e, t0)
                raise
            finally:
                _in_step.reset(tok)
            rec.finish(d, out, t0)
            return out

        return wrapper

    return deco(fn) if fn is not None else deco


def _finalize(run: Run, store: Store, t0: float, output: Any = None, error: Optional[BaseException] = None) -> None:
    run.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    if error is not None:
        run.status, run.error = "error", f"{type(error).__name__}: {error}"
    else:
        run.status, run.output = "ok", to_jsonable(output)
    store.save(run)


def agent(fn: Optional[Callable[..., Any]] = None, *, name: Optional[str] = None, store: Optional[Store] = None):
    """Mark a function as an agent entry point. Each top-level call is recorded as a run.

    The latest run is available as ``fn.last_run``. Agents are registered by name so
    runs can be replayed later (``agentreplay serve --module your_module``).
    """

    def deco(f: Callable[..., Any]) -> Callable[..., Any]:
        label = name or f.__name__
        sig = inspect.signature(f)
        _AGENTS[label] = f

        def start(a: Any, k: Any):
            bound = sig.bind(*a, **k)
            bound.apply_defaults()
            run = Run.new(label, to_jsonable(dict(bound.arguments)))
            return run, Recorder(run), (store or default_store())

        if inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def awrapper(*a: Any, **k: Any) -> Any:
                if _current.get() is not None:
                    return await f(*a, **k)
                run, rec, st = start(a, k)
                tok, t0 = _current.set(rec), time.perf_counter()
                try:
                    out = await f(*a, **k)
                except BaseException as e:
                    _finalize(run, st, t0, error=e)
                    raise
                finally:
                    _current.reset(tok)
                    awrapper.last_run = run  # type: ignore[attr-defined]
                _finalize(run, st, t0, output=out)
                return out

            return awrapper

        @functools.wraps(f)
        def wrapper(*a: Any, **k: Any) -> Any:
            if _current.get() is not None:
                return f(*a, **k)
            run, rec, st = start(a, k)
            tok, t0 = _current.set(rec), time.perf_counter()
            try:
                out = f(*a, **k)
            except BaseException as e:
                _finalize(run, st, t0, error=e)
                raise
            finally:
                _current.reset(tok)
                wrapper.last_run = run  # type: ignore[attr-defined]
            _finalize(run, st, t0, output=out)
            return out

        return wrapper

    return deco(fn) if fn is not None else deco


# --------------------------------------------------------------------------- replay


def replay(run_id: str, fork_at: int, edit: Optional[Dict[str, Any]] = None, store: Optional[Store] = None) -> Run:
    """Fork ``run_id`` at step ``fork_at`` and return the new (saved) run.

    ``edit`` is ``{"output": <value>}`` to replace the step's output without executing it,
    or ``{"input": {<arg>: <value>}}`` to re-execute the step with changed arguments.
    With no edit the step simply re-executes live.
    """
    store = store or default_store()
    parent = store.load(run_id)
    if not 0 <= fork_at < len(parent.steps):
        raise ValueError(f"fork_at must be in [0, {len(parent.steps) - 1}]")
    edit = edit or {}
    if "output" in edit and "input" in edit:
        raise ValueError("edit may contain 'output' or 'input', not both")
    if "input" in edit and not isinstance(edit["input"], dict):
        raise ValueError("edit['input'] must be an object mapping argument names to values")
    fn = _AGENTS.get(parent.agent)
    if fn is None:
        raise LookupError(f"agent {parent.agent!r} is not registered; import the module that defines it")

    run = Run.new(parent.agent, parent.inputs, parent_id=parent.id, fork_at=fork_at, edit=edit or None)
    rec = Recorder(run, parent)
    tok, t0 = _current.set(rec), time.perf_counter()
    try:
        out = asyncio.run(fn(**parent.inputs)) if inspect.iscoroutinefunction(fn) else fn(**parent.inputs)
    except BaseException as e:  # a failing forked run is still a useful, saved result
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        _finalize(run, store, t0, error=e)
    else:
        _finalize(run, store, t0, output=out)
    finally:
        _current.reset(tok)
    return run
