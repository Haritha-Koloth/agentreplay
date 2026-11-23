import json
import threading
import urllib.error
import urllib.request

import pytest

import agentreplay.core as core
from agentreplay import Store, agent, replay, step
from agentreplay.server import serve


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


def make_agent(store, calls):
    @step(kind="llm")
    def plan(q):
        calls.append("plan")
        return {"city": "Paris, Texas"}

    @step
    def lookup(city):
        calls.append("lookup")
        return {"city": city, "temp": 31 if "Texas" in city else 14}

    @step(kind="llm")
    def write(obs):
        calls.append("write")
        return f"{obs['city']}: {obs['temp']}C"

    @agent(name="t_agent", store=store)
    def run_agent(q):
        d = plan(q)
        o = lookup(d["city"])
        return write(o)

    return run_agent


def test_records_steps(store):
    calls = []
    a = make_agent(store, calls)
    assert a("hi") == "Paris, Texas: 31C"
    run = store.load(a.last_run.id)
    assert [s.name for s in run.steps] == ["plan", "lookup", "write"]
    assert [s.kind for s in run.steps] == ["llm", "tool", "llm"]
    assert run.status == "ok" and run.output == "Paris, Texas: 31C"
    assert run.inputs == {"q": "hi"}


def test_fork_edit_output_reuses_prefix_and_reruns_suffix(store):
    calls = []
    a = make_agent(store, calls)
    a("hi")
    calls.clear()
    forked = replay(a.last_run.id, 0, {"output": {"city": "Paris, France"}}, store)
    assert calls == ["lookup", "write"]  # plan skipped (edited), downstream executed live
    assert forked.output == "Paris, France: 14C"
    assert [s.source for s in forked.steps] == ["edited", "live", "live"]
    assert forked.parent_id == a.last_run.id and forked.fork_at == 0


def test_fork_midway_caches_earlier_steps(store):
    calls = []
    a = make_agent(store, calls)
    a("hi")
    calls.clear()
    forked = replay(a.last_run.id, 2, {"output": "custom"}, store)
    assert calls == []  # nothing executed at all
    assert [s.source for s in forked.steps] == ["cached", "cached", "edited"]
    assert forked.output == "custom"


def test_fork_edit_input_reexecutes_step(store):
    calls = []
    a = make_agent(store, calls)
    a("hi")
    calls.clear()
    forked = replay(a.last_run.id, 1, {"input": {"city": "Paris, France"}}, store)
    assert calls == ["lookup", "write"]
    assert forked.steps[1].input == {"city": "Paris, France"}
    assert forked.output == "Paris, France: 14C"


def test_fork_without_edit_reruns_live(store):
    calls = []
    a = make_agent(store, calls)
    a("hi")
    calls.clear()
    forked = replay(a.last_run.id, 1, None, store)
    assert calls == ["lookup", "write"]
    assert [s.source for s in forked.steps] == ["cached", "live", "live"]


def test_errors_are_recorded_and_agent_still_raises(store):
    @step
    def boom():
        raise RuntimeError("nope")

    @agent(name="boom_agent", store=store)
    def bad():
        return boom()

    with pytest.raises(RuntimeError):
        bad()
    run = store.load(bad.last_run.id)
    assert run.status == "error" and "nope" in run.error
    assert run.steps[0].error == "RuntimeError: nope"


def test_forked_run_that_fails_is_saved(store):
    calls = []
    a = make_agent(store, calls)
    a("hi")

    @step
    def lookup(city):  # noqa: F841 - ensure unrelated names don't matter
        return 1

    forked = replay(a.last_run.id, 1, {"input": {"city": None}}, store)
    assert forked.status == "error"
    assert store.load(forked.id).status == "error"


def test_nested_steps_are_not_double_recorded(store):
    @step
    def inner(x):
        return x + 1

    @step
    def outer(x):
        return inner(x) * 2

    @agent(name="nested", store=store)
    def n(x):
        return outer(x)

    assert n(1) == 4
    assert [s.name for s in store.load(n.last_run.id).steps] == ["outer"]


def test_steps_outside_agent_just_run():
    @step
    def f(x):
        return x * 2

    assert f(3) == 6


def test_async_agent_and_replay(store):
    calls = []

    @step
    async def fetch(x):
        calls.append("fetch")
        return x * 2

    @step
    async def fmt(v):
        calls.append("fmt")
        return f"v={v}"

    @agent(name="async_agent", store=store)
    async def a(x):
        return await fmt(await fetch(x))

    import asyncio

    assert asyncio.run(a(2)) == "v=4"
    calls.clear()
    forked = replay(a.last_run.id, 0, {"output": 10}, store)
    assert forked.output == "v=10" and calls == ["fmt"]


def test_diverged_agent_stops_using_cache(store):
    @step
    def a_step(x):
        return x

    @step
    def b_step(x):
        return x

    flip = {"v": False}

    @agent(name="diverge", store=store)
    def d(x):
        if flip["v"]:
            a_step(x)  # different order than the recorded run
            b_step(x)
        else:
            b_step(x)
            a_step(x)
        return x

    d(1)
    flip["v"] = True
    forked = replay(d.last_run.id, 1, None, store)
    assert [s.source for s in forked.steps] == ["live", "live"]


def test_replay_validation(store):
    calls = []
    a = make_agent(store, calls)
    a("hi")
    rid = a.last_run.id
    with pytest.raises(ValueError):
        replay(rid, 9, None, store)
    with pytest.raises(ValueError):
        replay(rid, 0, {"output": 1, "input": {}}, store)
    with pytest.raises(ValueError):
        replay(rid, 0, {"input": [1]}, store)
    with pytest.raises(ValueError):
        store.load("../etc/passwd")
    core._AGENTS.pop("t_agent")
    with pytest.raises(LookupError):
        replay(rid, 0, None, store)


def test_to_jsonable_handles_odd_types():
    from dataclasses import dataclass

    @dataclass
    class P:
        a: int

    out = core.to_jsonable({"p": P(1), "s": {1}, "o": object(), 1: (1, 2)})
    json.dumps(out)
    assert out["p"] == {"a": 1}


def test_http_api(store):
    calls = []
    a = make_agent(store, calls)
    a("hi")
    rid = a.last_run.id
    httpd = serve(store, "127.0.0.1", 0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def call(path, data=None):
        req = urllib.request.Request(
            base + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    try:
        code, body = call("/")
        assert code == 200 and b"agentreplay" in body
        code, body = call("/api/runs")
        assert code == 200 and json.loads(body)[0]["id"] == rid
        code, body = call(f"/api/runs/{rid}")
        assert code == 200 and len(json.loads(body)["steps"]) == 3
        assert call("/api/runs/nope")[0] == 404
        assert call("/api/runs/..%2Fx")[0] == 404
        code, body = call(f"/api/runs/{rid}/fork", {"fork_at": 0, "edit": {"output": {"city": "Paris, France"}}})
        assert code == 200 and json.loads(body)["output"] == "Paris, France: 14C"
        assert call(f"/api/runs/{rid}/fork", {"fork_at": 99})[0] == 400
        assert call(f"/api/runs/{rid}/fork", {})[0] == 400
        assert call("/api/runs/nope/fork", {"fork_at": 0})[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
