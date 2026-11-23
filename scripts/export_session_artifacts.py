#!/usr/bin/env python3
"""Export everything a running marimo session needs to rebuild its static HTML.

marimo's on-disk session JSON (__marimo__/session/<nb>.json) is incomplete:

  * it drops cell outputs that reference in-memory virtual files
    (./@file/<len>-<name>, e.g. mo.image or large charts),
  * it holds none of the virtual files' bytes,
  * it holds no widget (anywidget / ipywidgets) model state,
  * it holds none of the widgets' ESM code, which are virtual files too.

This script talks to a running marimo server over HTTP, runs a snippet in the
kernel scratchpad, and writes a bundle to --out-dir:

  session.json   the session JSON with the dropped outputs restored
                 (links left as ./@file/<len>-<name>)
  models.json    one `model-lifecycle` open notification per live widget, in
                 the format static HTML export embeds (window.__MARIMO_STATIC__)
  files/         every referenced virtual file, named <len>-<name>

Standard library only. Requires the notebook to be open (an active session).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

BEGIN, END = "<<<ARTIFACTS-JSON>>>", "<<<END-ARTIFACTS-JSON>>>"

# Executed inside the marimo kernel scratchpad.
KERNEL_CODE = r'''
import base64 as _b64, json as _json, re as _re
import marimo._code_mode as _cm
from marimo._runtime.virtual_file import read_virtual_file as _read

_REF = _re.compile(r"\./@file/(\d+)-([A-Za-z0-9_.\-]+)")
with open(__SESSION_PATH__) as _f:
    _session = _json.load(_f)

_cells, _files, _missing = {}, {}, []

def _grab(_len, _name):
    _key = f"{_len}-{_name}"
    if _key in _files:
        return
    try:
        _files[_key] = _b64.b64encode(_read(_name, int(_len))).decode()
    except Exception as _e:
        _missing.append([_key, repr(_e)])

async with _cm.get_context() as _ctx:
    for _c in _ctx.cells:
        _o = _c.output
        if _o is None or not isinstance(_o.data, str) or "./@file/" not in _o.data:
            continue
        _cells[_c.id] = {"name": _c.name, "mimetype": _o.mimetype, "data": _o.data}
        for _len, _name in _REF.findall(_o.data):
            _grab(_len, _name)

# Widget models: the server replays these to the browser from its session view,
# which the kernel can't reach. Rebuild one `open` per live comm from the
# widget's current state instead.
_models = []
try:
    import ipywidgets as _ipw
    from ipywidgets.widgets.widget import _remove_buffers
    from marimo._plugins.ui._impl.anywidget.init import WIDGET_COMM_MANAGER as _M
    from marimo._messaging.notification import ModelLifecycleNotification as _N, ModelOpen as _Open
    _instances = _ipw.widgets.widget._instances
    for _cid, _comm in _M.comms.items():
        _w = _instances.get(_cid)
        if _comm._closed or _w is None:
            continue
        _state, _paths, _bufs = _remove_buffers(_w.get_state())
        _state.pop("_esm", None)  # code travels only through esm_spec
        # Widget state may hold numpy scalars/arrays, which msgspec can't encode.
        _state = _json.loads(_json.dumps(_state, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        _spec = _comm.esm_spec
        if _spec is not None and (_m := _REF.fullmatch(_spec.url)):
            _grab(*_m.groups())
        _models.append(_N(model_id=_cid, message=_Open(
            state=_state, buffer_paths=_paths, buffers=[bytes(_b) for _b in _bufs], esm_spec=_spec,
        )).to_json_serializable())
except ImportError:
    pass

import os as _os, tempfile as _tf
with _tf.NamedTemporaryFile("w", suffix=".json", delete=False) as _out:
    _json.dump({"session": _session, "cells": _cells, "files": _files,
                "models": _models, "missing": _missing}, _out)
print(__BEGIN__, _out.name, _os.path.getsize(_out.name), __END__)
'''

# marimo caps a cell's stdout (runtime.std_stream_max_bytes, 1 MB by default),
# so the payload is written to a temp file and read back in chunks.
CHUNK = 256 * 1024  # base64 of this stays well under the cap
CHUNK_CODE = r'''
with open(__PATH__, "rb") as _f:
    _f.seek(__START__)
    print(__BEGIN__ + __import__("base64").b64encode(_f.read(__SIZE__)).decode() + __END__)
if __LAST__:
    import os as _os; _os.remove(__PATH__)
'''


def _between(stdout: str) -> str:
    if BEGIN not in stdout or END not in stdout:
        sys.exit(f"unexpected kernel output:\n{stdout[-2000:]}")
    return stdout.split(BEGIN, 1)[1].split(END, 1)[0]


def _request(url: str, token: str | None, data: bytes | None = None, headers: dict | None = None):
    # Some proxies reject urllib's default User-Agent.
    h = {"Content-Type": "application/json", "User-Agent": "marimo-session-artifacts", **(headers or {})}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h))


def pick_session(base: str, token: str | None, wanted: str | None) -> str:
    if wanted:
        return wanted
    with _request(f"{base}/api/sessions", token) as r:
        sessions = json.load(r)
    if len(sessions) != 1:
        listing = "\n".join(f"  {k}  {v.get('filename', '')}" for k, v in sessions.items())
        sys.exit(f"expected exactly one active session, found {len(sessions)}; pass --session\n{listing}")
    return next(iter(sessions))


def execute(base: str, token: str | None, session_id: str, code: str) -> str:
    """Run code in the kernel scratchpad; return its stdout (SSE stream)."""
    body = json.dumps({"code": code}).encode()
    out, event = [], None
    with _request(f"{base}/api/kernel/execute", token, body, {"Marimo-Session-Id": session_id}) as r:
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                if event == "stdout":
                    out.append(payload.get("data", ""))
                elif event == "stderr":
                    sys.stderr.write(payload.get("data", ""))
                elif event == "done":
                    if payload.get("success") is False:
                        sys.exit(f"kernel error: {payload.get('error', {}).get('msg')}")
                    break
    return "".join(out)


def _fill(template: str, **values: object) -> str:
    for key, value in values.items():
        template = template.replace(f"__{key}__", value if isinstance(value, str) else repr(value))
    return template


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="marimo server URL, e.g. http://localhost:2718")
    p.add_argument("--token", default=os.environ.get("MARIMO_TOKEN"), help="auth token (or $MARIMO_TOKEN)")
    p.add_argument("--session", help="session id (default: the only active session)")
    p.add_argument("--session-path", default="__marimo__/session/notebook.py.json",
                   help="session JSON path on the server, relative to the kernel cwd")
    p.add_argument("--out-dir", default="__marimo__/export")
    args = p.parse_args()

    base = args.url.rstrip("/")
    session_id = pick_session(base, args.token, args.session)
    code = _fill(KERNEL_CODE, SESSION_PATH=repr(args.session_path), BEGIN=repr(BEGIN), END=repr(END))
    path, size = _between(execute(base, args.token, session_id, code)).split()
    size, raw = int(size), bytearray()
    for start in range(0, size, CHUNK):
        chunk_code = _fill(CHUNK_CODE, PATH=repr(path), START=str(start), SIZE=str(CHUNK),
                           LAST=str(start + CHUNK >= size), BEGIN=repr(BEGIN), END=repr(END))
        raw += base64.b64decode(_between(execute(base, args.token, session_id, chunk_code)))
    payload = json.loads(raw)

    out = Path(args.out_dir)
    files = out / "files"
    files.mkdir(parents=True, exist_ok=True)
    for stale in files.iterdir():
        stale.unlink()
    for key, b64 in payload["files"].items():
        (files / key).write_bytes(base64.b64decode(b64))

    session = payload["session"]
    by_id = {c["id"]: c for c in session["cells"]}
    restored = []
    for cid, info in payload["cells"].items():
        if cid in by_id:
            by_id[cid]["outputs"] = [{"type": "data", "data": {info["mimetype"]: info["data"]}}]
            restored.append(info["name"])
    (out / "session.json").write_text(json.dumps(session, indent=2))
    (out / "models.json").write_text(json.dumps(payload["models"]))

    print(f"{out}/: {len(payload['files'])} file(s), {len(payload['models'])} widget model(s), "
          f"{len(restored)} restored output(s): {', '.join(restored) or '-'}")
    for key, err in payload["missing"]:
        print(f"could not read {key}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
