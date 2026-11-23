# jack-sparse-row

A marimo notebook on how language models represent numbers, distributions and
places, told through a pirate story. Published at
<https://rambip.github.io/jack-sparse-row/>.

## How the page is built

The notebook needs a GPU, so it is never run in CI. Instead, a session that ran
on a GPU (molab) is exported once, committed, and turned into a static page by
GitHub Actions.

```
running notebook ──export_session_artifacts.py──▶ __marimo__/export/ ──build_site.py──▶ site/
   (molab, GPU)                                     (committed)          (CI, no GPU)
```

**1. Run the notebook to completion** on a GPU machine. Heavy steps are wrapped
in `mo.persistent_cache` and load the model through `get_model()` only on a
cache miss, so a rerun with a warm cache is cheap.

**2. Export the session** with
[`scripts/export_session_artifacts.py`](scripts/export_session_artifacts.py)
(standard library only) while the notebook is open:

```sh
python scripts/export_session_artifacts.py --url <server url> --token <token> --out-dir __marimo__/export
```

This writes what marimo's own session file leaves out:

| File | Contents |
|---|---|
| `session.json` | cell outputs, including the ones that point at in-memory files (`./@file/…`) |
| `models.json` | the state of every widget (anywidget / ipywidgets) |
| `files/` | the in-memory files themselves: images, and the widgets' JavaScript |

Save the notebook's `notebook.py` from the same session next to it: the export
and the notebook must match cell for cell.

**3. Build the page**, which CI does on every push to `main`:

```sh
uv run build_site.py          # writes site/
python -m http.server -d site # preview
```

`build_site.py` renders marimo's own static frontend, the one
`marimo export html` uses, from `notebook.py` (the cells) and the export (the
outputs and widgets). It differs from `marimo export html` in two ways: the
in-memory files are written to `site/@file/` rather than inlined into the HTML,
and `custom.css` (the pirate theme) is added to the page.

Only the export is committed; `site/` is built output and is ignored.
