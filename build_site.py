# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo==0.25.2"]
# ///
"""Build the static page from notebook.py + an export bundle, without running it.

The bundle comes from marimo-session-artifacts' export_session_artifacts.py:
session.json (outputs), models.json (widget state) and files/ (virtual files,
including widget ESM). The page is marimo's own static frontend, mounted the
way `marimo export html` mounts it, with two differences:

  * virtual files are written next to the page under @file/ instead of being
    inlined as data URIs, so the HTML stays small and the browser fetches
    images and widget code lazily;
  * custom.css (the pirate theme) is appended to the head.

Usage: uv run build_site.py [--bundle __marimo__/export] [--out site]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from marimo._config.config import DEFAULT_CONFIG
from marimo._config.utils import deep_copy
from marimo._export.exporter import get_html_contents
from marimo._server.tokens import SkewProtectionToken
from marimo._session.notebook.loader import load_notebook
from marimo._session.state.serialize import serialize_notebook
from marimo._session.state.session_view import SessionView
from marimo._templates import static_notebook_template
from marimo._utils.code import hash_code

ROOT = Path(__file__).parent


class _Notification:
    """models.json already holds the serialized form the template embeds."""

    def __init__(self, data: dict) -> None:
        self.data = data

    def to_json_serializable(self) -> dict:
        return self.data


def notebook_snapshot(path: Path):
    # Same trick as marimo-website's read_notebook_session: a SessionView that
    # only knows each cell's code yields the notebook structure, no outputs.
    manager = load_notebook(path)
    view = SessionView()
    for cell_id in manager.app.cell_manager.cell_ids():
        view.last_executed_code[cell_id] = manager.app.cell_manager.get_cell_data(cell_id).code
    return manager, serialize_notebook(view, manager.app.cell_manager)


def build(notebook: Path, bundle: Path, out: Path) -> Path:
    manager, notebook_snap = notebook_snapshot(notebook)
    session = json.loads((bundle / "session.json").read_text())
    models = [_Notification(m) for m in json.loads((bundle / "models.json").read_text())]

    # A .py file doesn't store cell ids: loading it assigns them by position,
    # while the live kernel keeps the random ids of cells created during the
    # session. Match cells by position, check their code, and adopt the file's ids.
    cells = notebook_snap["cells"]
    assert len(session["cells"]) == len(cells), "session.json and notebook.py differ in cell count"
    for s_cell, n_cell in zip(session["cells"], cells):
        if s_cell.get("code_hash") not in (None, hash_code(n_cell["code"])):
            raise SystemExit(f"cell {n_cell.get('name') or n_cell['id']}: code differs from session.json; re-export")
        s_cell["id"] = n_cell["id"]

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(bundle / "files", out / "@file")
    (out / ".nojekyll").touch()  # GitHub Pages: serve as is

    # Identity map: lookups that need an entry find one, and the browser
    # fetches the real file from ./@file/ next to the page.
    files = {f"./@file/{p.name}": f"./@file/{p.name}" for p in (out / "@file").iterdir()}

    config = deep_copy(DEFAULT_CONFIG)
    source = notebook.read_text(encoding="utf-8")
    html = static_notebook_template(
        html=get_html_contents(),
        user_config=config,
        config_overrides={},
        server_token=SkewProtectionToken("static"),
        app_config=manager.app.config,
        filepath=str(notebook),
        code=source,
        code_hash=hash_code(source),
        session_snapshot=session,
        notebook_snapshot=notebook_snap,
        files=files,
        model_notifications=models,
    )
    css = (ROOT / "custom.css").read_text()
    html = html.replace("</head>", f"<style>{css}</style></head>", 1)

    page = out / "index.html"
    page.write_text(html, encoding="utf-8")
    return page


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--notebook", type=Path, default=ROOT / "notebook.py")
    p.add_argument("--bundle", type=Path, default=ROOT / "__marimo__" / "export")
    p.add_argument("--out", type=Path, default=ROOT / "site")
    args = p.parse_args()
    page = build(args.notebook, args.bundle, args.out)
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"wrote {page} ({page.stat().st_size / 1e6:.1f} MB; site {size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
