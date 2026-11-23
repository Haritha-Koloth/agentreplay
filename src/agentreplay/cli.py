"""Command line: ``agentreplay demo | serve | runs | show``."""

from __future__ import annotations

import argparse
import importlib
import sys
import webbrowser
from typing import List, Optional

from .core import default_store
from .server import serve


def _cmd_serve(args: argparse.Namespace) -> int:
    for mod in args.module or []:
        importlib.import_module(mod)
    store = default_store()
    httpd = serve(store, args.host, args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"agentreplay UI: {url}   (runs in {store.dir})   Ctrl+C to stop")
    if args.open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import weather_agent

    answer = weather_agent("What's the weather in Paris?")
    print(f"Agent answered: {answer}\n(That is the wrong Paris. Fork step 0 in the UI to fix it.)\n")
    args.module, args.open = None, True
    return _cmd_serve(args)


def _cmd_runs(_: argparse.Namespace) -> int:
    for r in default_store().list():
        fork = f" <- {r.parent_id}@{r.fork_at}" if r.parent_id else ""
        print(f"{r.id}  {r.status:<7} {r.agent:<20} {len(r.steps)} steps{fork}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    run = default_store().load(args.run_id)
    print(f"{run.id} {run.agent} [{run.status}] {run.duration_ms}ms")
    for s in run.steps:
        print(f"  {s.index}. {s.name} ({s.kind}, {s.source}) -> {s.error or s.output}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="agentreplay", description="Time-travel debugger for AI agents.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def net(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--host", default="127.0.0.1")
        sp.add_argument("--port", type=int, default=8765)

    sp = sub.add_parser("demo", help="record the built-in demo agent and open the UI")
    net(sp)
    sp.set_defaults(fn=_cmd_demo)
    sp = sub.add_parser("serve", help="serve the UI for recorded runs")
    net(sp)
    sp.add_argument("--module", "-m", action="append", help="module defining your @agent (repeatable); needed to replay")
    sp.add_argument("--open", action="store_true", help="open a browser tab")
    sp.set_defaults(fn=_cmd_serve)
    sub.add_parser("runs", help="list recorded runs").set_defaults(fn=_cmd_runs)
    sp = sub.add_parser("show", help="print a run")
    sp.add_argument("run_id")
    sp.set_defaults(fn=_cmd_show)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except KeyError as e:
        print(f"run not found: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
