# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.2.2",
#     "anywidget==0.11.0",
#     "folium==0.20.0",
#     "ipyleaflet==0.20.0",
#     "leafmap==0.63.0",
#     "manim==0.20.1",
#     "manim-widget==0.2.0",
#     "mapwidget==0.2.1",
#     "marimo>=0.23.13",
#     "matplotlib==3.11.0",
#     "numpy==2.5.1",
#     "pillow==12.3.0",
#     "polars==1.42.1",
#     "python-dotenv==1.2.2",
#     "requests==2.34.2",
#     "scikit-learn==1.9.0",
#     "torch==2.12.1",
#     "traitlets==5.15.1",
#     "transformers==5.13.0",
#     "wigglystuff==0.5.11",
# ]
#
# [tool.uv.sources]
# wigglystuff = { git = "https://github.com/koaning/wigglystuff/" }
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import marimo as mo
    import json
    import anywidget
    import traitlets
    from wigglystuff import Excalidraw, CurveEditor
    import altair as alt
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import polars as pl
    from dataclasses import dataclass
    from collections.abc import Sequence
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.model_selection import train_test_split
    from dotenv import load_dotenv
    import leafmap.foliumap as leafmap
    import leafmap.maplibregl as leafmap3d
    from manim_widget import ManimWidget, patch_tex
    from pathlib import Path
    from urllib.request import urlretrieve

    patch_tex()
    from manim import (
        ThreeDAxes,
        VMobject,
        PMobject,
        Axes,
        VGroup,
        Dot,
        Dot3D,
        Line,
        DashedLine,
        Arrow,
        Create,
        Write,
        FadeIn,
        FadeOut,
        Transform,
        AnimationGroup,
        Text,
        WHITE,
        ORANGE,
        GRAY,
        DOWN,
        LEFT,
        RIGHT,
        UP,
        OUT,
        IN,
        Arrow3D,
        DEGREES,
        linear,
    )

    load_dotenv(".env")
    from importlib.metadata import version
    if "rc" in version("manim-widget"):
        raise ValueError("Version bug in molab: Please upgrade manim widget to\n manim-widget==0.2.0 ")

    _REPO_RAW_BASE = "https://raw.githubusercontent.com/rambip/jack-sparse-row/main"
    _REQUIRED_REPO_FILES = (
        "assets/jack.png",
        "assets/jar.png",
        "assets/parrot.png",
        "assets/experimental-setup-simple.excalidraw",
        "assets/experimental-setup.excalidraw",
        "assets/llm-intro.excalidraw",
        "data/geo_cities.json",
        "data/geo_monument_examples.json",
        "data/geo_monuments.json",
    )

    def ensure_repo_assets_and_data(root="."):
        root = Path(root)
        downloaded = []
        for rel in _REQUIRED_REPO_FILES:
            path = root / rel
            if path.exists() and path.stat().st_size > 0:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            urlretrieve(f"{_REPO_RAW_BASE}/{rel}", tmp)
            tmp.replace(path)
            downloaded.append(rel)
        return downloaded

    _downloaded_repo_files = ensure_repo_assets_and_data()
    if _downloaded_repo_files:
        mo.status.toast(f"Downloaded {len(_downloaded_repo_files)} missing asset/data files")


@app.cell(hide_code=True)
def widgets():
    class CSSInjector(anywidget.AnyWidget):
        _esm = """
        function injectCSS(vars, styles, root) {
            const varText = Object.entries(vars).map(([k,v]) => `${k}: ${v} !important`).join('; ');
            const styleText = Object.entries(styles).map(([k,v]) => `${k}:${v}`).join(';');
            const id = 'pirate-css-override';

            if (root instanceof ShadowRoot) {
                let tag = root.querySelector('#' + id);
                if (!tag) { tag = document.createElement('style'); tag.id = id; root.appendChild(tag); }
                tag.textContent = `:host { ${varText} } * { ${varText} } .markdown { ${styleText} }`;
            }

            for (const el of (root.querySelectorAll ? root.querySelectorAll('*') : [])) {
                if (el.shadowRoot) injectCSS(vars, styles, el.shadowRoot);
            }
        }

        function render({ model, el }) {
            const vars = model.get('vars');
            const styles = model.get('styles');

            const globalId = 'pirate-global-styles';
            let globalStyle = document.getElementById(globalId);
            if (!globalStyle) {
                globalStyle = document.createElement('style');
                globalStyle.id = globalId;
                document.head.appendChild(globalStyle);
            }
            const varText = Object.entries(vars).map(([k,v]) => `${k}: ${v} !important`).join('; ');
            const styleText = Object.entries(styles).map(([k,v]) => `${k}:${v}`).join(';');
            const fontImport = styles['font-family']
                ? `@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital@0;1&display=swap');`
                : '';
            globalStyle.textContent = fontImport + ` :root { ${varText} } .markdown { ${styleText} }`;

            injectCSS(vars, styles, document);
            const observer = new MutationObserver(() => injectCSS(vars, styles, document));
            observer.observe(document.body, { childList: true, subtree: true });

            el.textContent = 'CSS injected';
            return () => observer.disconnect();
        }
        export default { render };
        """
        vars = traitlets.Dict({}).tag(sync=True)
        styles = traitlets.Dict({}).tag(sync=True)


    class Ticker(anywidget.AnyWidget):
        """A custom widget with play/stop/restart buttons that animates through steps."""

        current_step = traitlets.Int(0).tag(sync=True)
        ticks = traitlets.List([]).tag(sync=True)
        label_start = traitlets.Unicode("Play").tag(sync=True)
        label_stop = traitlets.Unicode("Stop").tag(sync=True)
        label_restart = traitlets.Unicode("Restart").tag(sync=True)

        def __init__(self, ticks, label_start="Play", label_stop="Stop", label_restart="Restart", **kwargs):
            super().__init__(**kwargs)
            self.ticks = ticks
            self.label_start = label_start
            self.label_stop = label_stop
            self.label_restart = label_restart

        _esm = """
        function render({ model, el }) {
            el.innerHTML = `
            <div class="animator-box">
                <div class="button-row">
                    <button class="animator-btn main-btn"></button>
                    <button class="animator-btn restart-btn" style="display: none;">↻ Restart</button>
                </div>
                <div class="progress-row">
                    <div class="animator-track">
                        <div class="animator-fill"></div>
                    </div>
                    <div class="animator-text">0/0</div>
                </div>
            </div>
            `;

            const mainBtn = el.querySelector('.main-btn');
            const restartBtn = el.querySelector('.restart-btn');
            const text = el.querySelector('.animator-text');
            const fill = el.querySelector('.animator-fill');

            let playing = false;
            let timer = null;

            function getLabel(labelType) {
                const label = model.get(labelType);
                if (labelType === 'label_start') return label || 'Play';
                if (labelType === 'label_stop') return label || 'Stop';
                if (labelType === 'label_restart') return label || 'Restart';
                return label || '';
            }

            function updateButtons() {
                const step = model.get('current_step');
                const ticks = model.get('ticks') || [];
                const total = ticks.length;
                const isComplete = step >= total && total > 0;

                if (isComplete && !playing) {
                    // Animation complete - show restart button
                    mainBtn.style.display = 'none';
                    restartBtn.style.display = 'block';
                } else {
                    // Animation in progress or not started
                    mainBtn.style.display = 'block';
                    restartBtn.style.display = 'none';
                    mainBtn.textContent = playing ? ('⏹ ' + getLabel('label_stop')) : ('▶ ' + getLabel('label_start'));
                    mainBtn.classList.toggle('stop', playing);
                }
            }

            function update() {
                const step = model.get('current_step');
                const ticks = model.get('ticks') || [];
                const total = ticks.length;
                const pct = total > 0 ? (step / total) * 100 : 0;

                fill.style.width = pct + '%';
                text.textContent = `${step}/${total}`;
                updateButtons();
            }

            function tick() {
                const ticks = model.get('ticks') || [];
                const total = ticks.length;
                let step = model.get('current_step');

                if (!playing || step >= total) {
                    playing = false;
                    updateButtons();
                    return;
                }

                // next step
                step = step + 1;
                model.set('current_step', step);
                model.save_changes();
                update();

                // schedule next
                if (step < total) {
                    timer = setTimeout(tick, ticks[step - 1] || 500);
                } else {
                    playing = false;
                    updateButtons();
                }
            }

            function startAnimation() {
                const total = (model.get('ticks') || []).length;
                if (model.get('current_step') >= total) {
                    model.set('current_step', 0);
                    model.save_changes();
                    update();
                }
                playing = true;
                updateButtons();
                tick();
            }

            function stopAnimation() {
                playing = false;
                clearTimeout(timer);
                updateButtons();
            }

            function restartAnimation() {
                model.set('current_step', 0);
                model.save_changes();
                update();
                startAnimation();
            }

            // Main button click handler (play/stop)
            mainBtn.onclick = () => {
                if (playing) {
                    stopAnimation();
                } else {
                    startAnimation();
                }
            };

            // Restart button click handler
            restartBtn.onclick = () => {
                restartAnimation();
            };

            model.on('change:current_step', update);
            model.on('change:ticks', update);
            model.on('change:label_start', update);
            model.on('change:label_stop', update);
            model.on('change:label_restart', update);
            update();

            return () => {
                playing = false;
                clearTimeout(timer);
            };
        }

        export default { render };
        """

        _css = """
        .animator-box {
            font-family: system-ui, sans-serif;
            padding: 16px;
            background: #f5f5f5;
            border-radius: 8px;
            width: 400px;
        }

        .button-row {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-bottom: 16px;
        }

        .animator-btn {
            padding: 6px 12px;
            font-size: 13px;
            font-weight: 600;
            color: white;
            background: #5C8020;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .animator-btn:hover {
            transform: translateY(-1px);
        }

        .animator-btn.stop {
            background: #E87F24;
        }

        .restart-btn {
            background: #5C8020;
            min-width: 80px;
        }

        .restart-btn:hover {
            background: var(--grass-9);
        }

        .progress-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .animator-text {
            font-size: 13px;
            color: #555;
            white-space: nowrap;
            min-width: 50px;
            text-align: right;
        }

        .animator-track {
            flex: 1;
            height: 6px;
            background: #ddd;
            border-radius: 3px;
            overflow: hidden;
        }

        .animator-fill {
            height: 100%;
            width: 0%;
            background: #5C8020;
            border-radius: 3px;
            transition: width 0.2s ease;
        }
        """


    return CSSInjector, Ticker


@app.cell(hide_code=True)
def images_and_theme(CSSInjector):
    def jack(size=150, float=True):
        style = "float:left; margin-right:15px; margin-bottom:10px;" if float else ""
        return mo.Html(f'<div style="{style}">{mo.image("assets/jack.png", width=size).text}</div>')

    jar_image = mo.Html(
        f'<div style="float:left; margin-right:15px; margin-bottom:10px;">{mo.image("assets/jar.png", width=150).text}</div>'
    )

    parrot_image = mo.Html(
        f'<div style="float:left; margin-right:15px; margin-bottom:10px;">{mo.image("assets/parrot.png", width=200).text}</div>'
    )

    md_text_start = mo.Html('<span style="display:block; height:3em;"></span>')

    md_character_break = mo.Html('<br style="clear:both">')


    PIRATE_BLUE = "#73A5CA"
    PIRATE_SKY = "#9FEAE5"
    PIRATE_ORANGE = "#E87F24"
    PIRATE_YELLOW = "#FFC81E"
    PIRATE_BEIGE = "#FEFDDF"
    PIRATE_GREY = "#F9EFDB"
    PIRATE_PURPLE = "#840D4D"

    PIRATE_GREEN = "#5C8020"
    PIRATE_RED = "#a8071f"


    _css = CSSInjector(
        vars={
            "--red-2": "#FFD5C0",
            "--red-8": PIRATE_ORANGE,
            "--red-9": PIRATE_ORANGE,
            "--grass-2": "#EFF6D0",
            "--grass-8": PIRATE_GREEN,
            "--grass-9": PIRATE_GREEN,
            "--blue-2": "#C8DFF0",
            "--blue-8": PIRATE_BLUE,
            "--blue-9": PIRATE_BLUE,
            "--blue-10": "#5A8FB5",
        },
        styles={
            "font-family": "'Crimson Text', serif",
            "font-size": "20px",
        },
    )
    _css
    return (
        PIRATE_BLUE,
        PIRATE_GREEN,
        PIRATE_ORANGE,
        PIRATE_PURPLE,
        PIRATE_RED,
        jack,
        jar_image,
        md_character_break,
        md_text_start,
        parrot_image,
    )


@app.cell(hide_code=True)
def header(jack):
    mo.hstack([
        jack(200),
        mo.md("# Jack Sparse'Row and the shape of Beliefs").center()
    ])
    return


@app.cell(hide_code=True)
def tldr():
    mo.md(r"""
    > **TL;DR**:
    > In this notebook, we replicate and extend the work of [Shape of Belief (Sarfati et Al, 2026)](https://arxiv.org/abs/2602.02315).

    It is well established that numbers in activation space are represented on a non-linear manifold, which can be detected using linear and categorical probes. This work shows that higher-level statistical concepts like the mean and variance of a time series have a similar representation. Additionally, it shows that existing steering techniques cannot change the belief of a model without changing it's uncertainty, and compare different steering methods that respect the manifold curvature. We extend this work by applying it to geographical data.
    """)
    return


@app.cell(hide_code=True)
def introduction_title():
    mo.md(r"""
    # Introduction
    """)
    return


@app.cell(hide_code=True)
def intro():
    mo.md("""
    > Feel free to jump directly to **Experimental setup**

    You are currently on Tensor Bay, a small but very popular island near the coasts of Silicium Valley. Among the farmers, the fishermen and the sea entrepreneurs, you try to find your way. Suddenly, you see before you a particularly dense crowd of people. They are all gathered around a stall and shouts ring out.
    """)
    return


@app.cell(hide_code=True)
def jar_scene(jar_image, md_character_break, md_text_start):
    mo.md(f"""
    {jar_image}
    {md_text_start}
    « 50 coins for the one among you that finds the exact number of beads in this jar ! »

    — *1438!*

    — *800!*

    — *1540!*

    — *910!*

    ...

    {md_character_break}
    """)
    return


@app.cell(hide_code=True)
def parrot_stranger():
    mo.md("""
    Near the crowd, you immediately notice someone. He is tall and has a mechanical parrot on his shoulder. He does not even look at the jar. He smiles at you. A few moments later, after everyone made the predictions and the calm started to come back, he comes to the jar, takes the parrot off his shoulder and says, *"I think Gemma wants to try."*
    """)
    return


@app.cell(hide_code=True)
def silence():
    mo.md("""
    Silence. Everyone is looking at the mechanical parrot. I noticed something and I think I am not the only one. The parrot seems to have no eyes at all.
    """)
    return


@app.cell(hide_code=True)
def parrot_answer(md_character_break, md_text_start, parrot_image):
    mo.md(f"""
    {parrot_image}
    {md_text_start}
    # — "863 ± 13.5"

    {md_character_break}
    """)
    return


@app.cell(hide_code=True)
def merchant_coins():
    mo.md("""
    After an even longer silence, I see, dumbfounded, the merchant hand the 50 gold coins to the stranger and his parrot. Intrigued, I walked straight to him and asked him who he is.
    """)
    return


@app.cell(hide_code=True)
def jack_intro(jack, md_character_break, md_text_start):
    mo.md(f"""
    {jack(200)}
    {md_text_start}
    — Hi! I'm Captain Jack Sparse'Row, pirate of the activation space, explorer of the manifolds. To be honest, I'm quite a legend here, and not without reason. I can steer any enemy with one-neuron attacks. I reverse engineer vessels from their wrecks. I survived a gradient overflow by hiding in the KV cache, losing one eye for it.

    {md_character_break}

    You seem to be quite interested in my stochastic parrot, Gemma. It is a LLM, very good at speech, but still a bit dumb for vision.
    As you noticed, Gemma has no eyes, but very good hears. And it is very good at modeling uncertainty, even though it was not designed for it. I just discovered yesterday that it had this skill ! 

    It's very easy to reproduce: take any LLM, pick any number $x$, generate a sequence of numbers with $x$ as the mean (typically using a gaussian distribution with some standard deviation $\sigma$) and convert the sequence to text.


    > `530,396,575,594,305,370,513,468,498,415,588,578,507,613,547,414,537,404,588,495`

    <details>
        <summary>
        Can you tell which x was used in this example?
        </summary>
        {mo.md("It was generated with $x=500$ and $\sigma=100$").text}
    </details>

    Then, feed the first $K$ numbers in the sequence to your LLM, and look at which number it predicts next.

    ///note
    For stability, we don't just look at the number the LLM predicts with the highest probability, but we consider the M top predicted numbers and weight them using the model's probability.
    ///
    """)
    return


@app.cell(hide_code=True)
def model_selector():
    model_selector = mo.ui.dropdown(
        [
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3.1-8B",
        ],
        value="meta-llama/Llama-3.2-1B",
        label="Model",
    )
    model_selector
    return (model_selector,)


@app.cell(hide_code=True)
def jack_demo_bundle(bundle):
    jack_bundle = bundle
    return (jack_bundle,)


@app.cell(hide_code=True)
def jack_demo_helpers(
    SUPPORT,
    ints_to_prompt,
    jack_bundle,
    sample_gaussian_ints,
    scan_batch,
    tokenize,
):
    @mo.persistent_cache
    def convergence_soft(mu: float, sigma: float, n: int, seed: int) -> np.ndarray:
        """Real softmax over the 1000 integer tokens at every comma position of one
        Gaussian sequence, from the fixed demo model (jack_bundle)."""
        rng = np.random.default_rng(seed)
        xs = sample_gaussian_ints(mu, sigma, n, rng)
        ids = tokenize(jack_bundle, [ints_to_prompt(xs)])
        _, soft = scan_batch(jack_bundle, ids)
        return soft[0]


    @mo.persistent_cache
    def switch_soft(mu_before: float, mu_after: float, sigma: float, n_each: int, seed: int) -> np.ndarray:
        """Same as convergence_soft, but the generating mean switches from mu_before
        to mu_after halfway through the sequence."""
        rng = np.random.default_rng(seed)
        xs = np.concatenate([
            sample_gaussian_ints(mu_before, sigma, n_each, rng),
            sample_gaussian_ints(mu_after, sigma, n_each, rng),
        ])
        ids = tokenize(jack_bundle, [ints_to_prompt(xs)])
        _, soft = scan_batch(jack_bundle, ids)
        return soft[0]


    def topk_predictions(soft: np.ndarray) -> pl.DataFrame:
        """Every predicted integer and its probability at every position, ranked by
        probability (rank 1 = most likely) — the full distribution, not truncated."""
        order = np.argsort(-soft, axis=1)
        probs = np.take_along_axis(soft, order, axis=1)
        n_positions, n_values = soft.shape
        return pl.DataFrame({
            "position": np.repeat(np.arange(n_positions), n_values),
            "rank": np.tile(np.arange(1, n_values + 1), n_positions),
            "value": order.flatten(),
            "probability": probs.flatten(),
        })


    TOPK_CHOICES = ["1", "5", "20", "50", "all"]


    def resolve_topk(choice: str) -> int:
        return SUPPORT if choice == "all" else int(choice)


    def weighted_topk_mean_curve(df: pl.DataFrame, choice: str) -> np.ndarray:
        """Probability-weighted mean of the top-k predicted values at each position
        (renormalized within the top-k subset); 'all' reduces to the full posterior
        expectation, matching the steering section's estimator exactly."""
        k = resolve_topk(choice)
        return (
            df.filter(pl.col("rank") <= k)
            .group_by("position", maintain_order=True)
            .agg((pl.col("value") * pl.col("probability")).sum() / pl.col("probability").sum())
            .sort("position")["value"]
            .to_numpy()
        )

    return (
        TOPK_CHOICES,
        convergence_soft,
        switch_soft,
        topk_predictions,
        weighted_topk_mean_curve,
    )


@app.cell(hide_code=True)
def topk_dropdown_cell(TOPK_CHOICES):
    topk_dropdown = mo.ui.dropdown(
        TOPK_CHOICES, value="1",
        label="Prediction = weighted mean of the most likely numbers",
    )
    topk_dropdown
    return (topk_dropdown,)


@app.cell(hide_code=True)
def jack_convergence_data(convergence_soft, topk_predictions):
    mo.stop(
        not torch.cuda.is_available(),
        mo.callout(mo.md("No GPU available — demo skipped."), kind="warn"),
    )

    convergence_topk_df = topk_predictions(convergence_soft(500.0, 100.0, 200, 42))
    return (convergence_topk_df,)


@app.cell(hide_code=True)
def jack_convergence_demo(
    PIRATE_BLUE,
    convergence_topk_df,
    topk_dropdown,
    weighted_topk_mean_curve,
):
    mo.stop(not torch.cuda.is_available())

    _curve = weighted_topk_mean_curve(convergence_topk_df, topk_dropdown.value)

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(np.arange(1, len(_curve) + 1), _curve, color=PIRATE_BLUE, linewidth=1.5)
    _ax.axhline(500, color="black", linestyle=":", linewidth=1.3)
    _ax.set_xlabel("numbers shown so far")
    _ax.set_ylabel("predicted number")
    _ax.spines[["top", "right"]].set_visible(False)
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def gaussian_switch_note():
    mo.md(r"""
    You can even make the task harder and change the mean and $\sigma$ value mid-way:
    """)
    return


@app.cell(hide_code=True)
def jack_switch_data(switch_soft, topk_predictions):
    mo.stop(
        not torch.cuda.is_available(),
        mo.callout(mo.md("No GPU available — demo skipped."), kind="warn"),
    )

    switch_topk_df = topk_predictions(switch_soft(300.0, 700.0, 100.0, 100, 42))
    return (switch_topk_df,)


@app.cell(hide_code=True)
def jack_switch_demo(
    PIRATE_ORANGE,
    switch_topk_df,
    topk_dropdown,
    weighted_topk_mean_curve,
):
    mo.stop(not torch.cuda.is_available())

    _curve = weighted_topk_mean_curve(switch_topk_df, topk_dropdown.value)

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(np.arange(1, len(_curve) + 1), _curve, color=PIRATE_ORANGE, linewidth=1.5)
    _ax.axhline(300, color="black", linestyle=":", linewidth=1.3)
    _ax.axhline(700, color="black", linestyle=":", linewidth=1.3)
    _ax.set_xlabel("numbers shown so far")
    _ax.set_ylabel("predicted number")
    _ax.spines[["top", "right"]].set_visible(False)
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def gemma_curiosity_note(jack):
    mo.md(rf"""
    {jack(100)}
    I'm still investigating how Gemma is able to do that, and it's actually super interesting. You seem to be the kind of person that is curious and like adventure. Ready for the ride ?
    """)
    return


@app.cell(hide_code=True)
def probe_title():
    mo.md(r"""
    # How to probe LLMs
    """)
    return


@app.cell(hide_code=True)
def llm_basics():
    mo.md(r"""
    I assume you already met that kind of beast before, right ? Gemma is a tiny one compared to the monumental god-like species that you see in the ports, like CrabGPT or Clawde, but they work in the same way inside.
    Let's start with the basics to make sure we're on the same boat.

    A LLM (Large Language model) is trained to predict tokens (pieces of text, often words) with a very generic architecture.

    Picture it like this: the words are floating at the top, they are embedded in the network (i.e, they become vectors), and one layer after the other, the vectors for each of the words are updated to get richer meaning. Let me draw you a diagram on the sand:
    """)
    return


@app.cell
def llm_diagram():
    diagram_e1 = mo.ui.anywidget(
        Excalidraw.from_file("assets/llm-intro.excalidraw", height=600)
    )
    diagram_e1
    return


@app.cell(hide_code=True)
def save_diagram():
    #diagram_e1.save()
    return


@app.cell(hide_code=True)
def probes_intro():
    mo.md(r"""
    As you can imagine, we're mainly interested in the vectors associated with the last token, because that is what will be used to predict the next word.

    The interesting part starts here: how do you go from a word to a representation so complex that you can write the next word of a book, find the right answer to a biology problem, or count beeds ? How do concepts form inside the LLM ?

    Don't expect me to answer this question. Thousands of pirates are working on the problem right now, chasing a treasure without knowing how far away it lies. But we've made some progress, thanks to a very simple trick: probes.
    """)
    return


@app.function(hide_code=True)
def generate_labeled_points(
    n_points: int = 100,
    centroid_1: tuple[float, float] = (2.0, 2.0),
    centroid_2: tuple[float, float] = (8.0, 8.0),
    std: float = 1.5,
    random_state: int = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate random 2D points with boolean labels and associated text.

    Args:
        n_points: Number of points to generate
        centroid_1: Center for label=False (negative) points
        centroid_2: Center for label=True (positive) points
        std: Standard deviation for Gaussian clusters
        random_state: Random seed for reproducibility

    Returns:
        x: X coordinates (n_points,)
        y: Y coordinates (n_points,)
        labels: Boolean labels (n_points,)
        texts: Associated text for each point (n_points,)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Predefined positive and negative sentences
    positive_sentences = [
        "Magnificent plunder today.",
        "Splendid winds indeed.",
        "Excellent rum aboard.",
        "Glorious treasure found.",
        "Outstanding crew today.",
        "Fine weather ahead.",
        "Brilliant sword work.",
        "Perfect map reading.",
    ]

    negative_sentences = [
        "Dreadful weather today.",
        "Terrible compass reading.",
        "Rotten provisions aboard.",
        "Pathetic plunder today.",
        "Awful navigation today.",
        "Dismal treasure haul.",
        "Horrible storm tonight.",
        "Wretched map indeed.",
    ]

    # Generate random boolean labels
    labels = np.random.choice([True, False], size=n_points)

    # Generate points from Gaussians around centroids based on labels
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    texts = []

    for i, label in enumerate(labels):
        if label:  # Positive class - centroid_2
            x[i] = np.random.normal(centroid_2[0], std)
            y[i] = np.random.normal(centroid_2[1], std)
            texts.append(np.random.choice(positive_sentences))
        else:  # Negative class - centroid_1
            x[i] = np.random.normal(centroid_1[0], std)
            y[i] = np.random.normal(centroid_1[1], std)
            texts.append(np.random.choice(negative_sentences))

    return x, y, labels, texts


@app.function(hide_code=True)
def plot_points_with_centroid_line(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    centroid_pos: tuple[float, float],
    centroid_neg: tuple[float, float],
    n: int = None,
    show_centroids: bool = True,
    show_perpendiculars: bool = False,
    figsize: tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot points with centroid line and optional perpendicular segments.

    Args:
        x: X coordinates
        y: Y coordinates
        labels: Boolean labels for each point
        centroid_pos: (x, y) tuple for positive centroid
        centroid_neg: (x, y) tuple for negative centroid
        n: Number of points to display (if None, shows all)
        show_centroids: Whether to show centroids and the line connecting them
        show_perpendiculars: Whether to draw perpendicular segments from points to line
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    # Use first n points if specified
    if n is not None:
        x = x[:n]
        y = y[:n]
        labels = labels[:n]

    # Separate points by label
    mask_pos = labels == True
    mask_neg = labels == False

    x_pos = x[mask_pos]
    y_pos = y[mask_pos]
    x_neg = x[mask_neg]
    y_neg = y[mask_neg]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot points
    ax.scatter(x_pos, y_pos, c='#73A5CA', alpha=0.6, s=80, label='Positive', edgecolors='white', linewidth=0.5)
    ax.scatter(x_neg, y_neg, c='#E87F24', alpha=0.6, s=80, label='Negative', edgecolors='white', linewidth=0.5)

    # Plot centroids and line only if requested
    if show_centroids:
        # Plot centroids
        ax.scatter(*centroid_pos, c='#4A7FA0', s=200, marker='*', edgecolors='white', linewidth=2, zorder=5, label='Positive Centroid')
        ax.scatter(*centroid_neg, c='#E87F24', s=200, marker='*', edgecolors='white', linewidth=2, zorder=5, label='Negative Centroid')

        # Compute extended line through centroids
        dx = centroid_pos[0] - centroid_neg[0]
        dy = centroid_pos[1] - centroid_neg[1]

        # Extend line beyond centroids
        extend_factor = 0.5
        line_start = (centroid_neg[0] - dx * extend_factor, centroid_neg[1] - dy * extend_factor)
        line_end = (centroid_pos[0] + dx * extend_factor, centroid_pos[1] + dy * extend_factor)

        # Draw the line
        ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 
                'k--', linewidth=2, alpha=0.7, label='Centroid Line')

        # Draw perpendicular segments if requested
        if show_perpendiculars:
            # Line direction vector (unit)
            line_len = np.sqrt(dx**2 + dy**2)
            if line_len > 0:
                udx, udy = dx / line_len, dy / line_len

                # Perpendicular direction
                perp_x, perp_y = -udy, udx

                for xi, yi, label in zip(x, y, labels):
                    # Project point onto line
                    # Vector from line start to point
                    vx, vy = xi - line_start[0], yi - line_start[1]

                    # Projection onto line direction
                    proj = vx * udx + vy * udy

                    # Closest point on line
                    closest_x = line_start[0] + proj * udx
                    closest_y = line_start[1] + proj * udy

                    # Draw perpendicular segment
                    color = '#2563eb' if label else '#dc2626'
                    ax.plot([xi, closest_x], [yi, closest_y], 
                            color=color, alpha=0.3, linewidth=0.8, linestyle='-')

    # Styling
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    title = f'Point Cloud (n={len(x)})'
    if show_centroids:
        title = f'Point Cloud with Centroid Line (n={len(x)})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


@app.function(hide_code=True)
def add_annotations(ax, centroid_pos, centroid_neg):
    """Add annotation arrows pointing to centroids.

    Args:
        ax: Matplotlib axes object
        centroid_pos: (x, y) tuple for positive centroid
        centroid_neg: (x, y) tuple for negative centroid
    """
    # Compute average centroid
    centroid_avg = ((centroid_pos[0] + centroid_neg[0]) / 2,
                   (centroid_pos[1] + centroid_neg[1]) / 2)

    # All arrows gray
    arrow_style = dict(arrowstyle='->', color='#666', lw=2)

    # 'yes' and 'no' from top left
    ax.annotate('yes', xy=centroid_pos, 
                xytext=(centroid_pos[0] - 2.0, centroid_pos[1] + 2.0),
                arrowprops=arrow_style,
                fontsize=12, color='#444', fontweight='bold')

    ax.annotate('no', xy=centroid_neg, 
                xytext=(centroid_neg[0] - 2.0, centroid_neg[1] + 2.0),
                arrowprops=arrow_style,
                fontsize=12, color='#444', fontweight='bold')

    # 'uncertain' from bottom right
    ax.annotate('uncertain', xy=centroid_avg, 
                xytext=(centroid_avg[0] + 2.0, centroid_avg[1] - 2.0),
                arrowprops=arrow_style,
                fontsize=12, color='#444', fontweight='bold')


@app.cell(hide_code=True)
def probe_ingredients():
    mo.md(r"""
    To create a probe, you need a few ingredients:
    - a concept you want to track. Let's say, *the sentiment of a sentence*
    - positive and negative examples of this concept
    - access to the **activations of the model** (the vectors of each token in between layers)

    With these, you can create small detectors to know if the model is thinking about a specific concept. We feed Gemma short pirate sentences, positive or negative, look at the last token's activation vector, and learn the pattern.

    (note: the animation below requires a good internet connection, or to run the notebook locally. sorry)
    """)
    return


@app.cell(hide_code=True)
def animation_config():
    N_POINTS = 50  # Total number of points
    SLOW_STEPS = 10  # First K points go slower
    SLOW_TICK = 2000  # ms for slow ticks
    FAST_TICK = 300  # ms for fast ticks
    return FAST_TICK, N_POINTS, SLOW_STEPS, SLOW_TICK


@app.cell(hide_code=True)
def ticker(
    FAST_TICK,
    N_POINTS,
    SLOW_STEPS,
    SLOW_TICK,
    Ticker,
    probe_animation_precomputed,
):
    ticks = [SLOW_TICK] * SLOW_STEPS + [FAST_TICK] * (N_POINTS - SLOW_STEPS)

    # Artificial dependency: keep the run control downstream of precomputation.
    _probe_animation_ready = probe_animation_precomputed

    # Instantiate the widget with the generated ticks
    ticker = mo.ui.anywidget(
        Ticker(
            ticks=ticks,
            label_start="Start animation",
            label_stop="Halt",
            label_restart="Restart"
        )
    )
    return (ticker,)


@app.cell(hide_code=True)
def ticker_display(ticker):
    ticker
    return


@app.cell
def _(ticker):
    ticker.step = 10
    return


@app.cell(hide_code=True)
def probe_animation(N_POINTS, probe_animation_precomputed, ticker):
    # Use the precomputed animation data; per tick only indexes the current frame.
    step = ticker.current_step
    _probe_frame = probe_animation_precomputed

    x = _probe_frame["x"]
    y = _probe_frame["y"]
    labels = _probe_frame["labels"]
    texts = _probe_frame["texts"]
    n_show = int(_probe_frame["n_by_step"][step])
    centroid_pos = _probe_frame["centroid_pos_by_step"][step]
    centroid_neg = _probe_frame["centroid_neg_by_step"][step]

    fig = plot_points_with_centroid_line(
        x,
        y,
        labels,
        centroid_pos=centroid_pos,
        centroid_neg=centroid_neg,
        n=n_show,
        show_centroids=(step > 0),
        show_perpendiculars=(step == N_POINTS),
        figsize=(6, 6),
    )

    ax = fig.axes[0]
    ax.set_xlim(*_probe_frame["xlim"])
    ax.set_ylim(*_probe_frame["ylim"])

    if step == N_POINTS:
        add_annotations(ax, centroid_pos, centroid_neg)

    mo.hstack(
        [
            mo.callout(
                texts[min(step, N_POINTS - 1)],
                kind="info" if labels[min(step, N_POINTS - 1)] else "danger",
            ).center(),
            ax,
        ]
    )
    return


@app.cell(hide_code=True)
def probe_math():
    mo.md(r"""
    As you can see, making a probe is very easy mathematically: just a regression. You can even get a good approximation by calculating the average of the vectors in the 'positive' and 'negative' groups, and it usually works fine.

    You have no idea how many times these small measuring tools saved my life. With probes, you can detect the [direction of refusal](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) inside a model, its [emotional behaviour](https://transformer-circuits.pub/2026/emotions/index.html), [wether it is lying](https://www.antischeming.ai/), etc. And importantly for us, we can even detect ... numbers !

    ///note
    Using a regression to train a probe has a major flaw: **overfitting**.
    Indeed, the activation vectors of a LLM typically have $2048$ coefficients or more. Because of the [Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality), we can perfectly fit a probe on up to arbitrary 2049 points, with arbitrary labels !

    To fight overfitting, there are 2 main solutions:
    1. Compute the direction using the difference of the mean point of the 2 groups (it is a better estimator because it reduces noise, see [The Geometry of Truth](https://arxiv.org/abs/2310.06824))
    2. Use L2 regularization during the regression.

    It is not a detail: without these techniques, you can get a probe that is pretty much unusable. We will use solution 2., like the paper does.
    ///
    """)
    return


@app.cell(hide_code=True)
def experimental_setup_title():
    mo.md(r"""
    # Experimental setup
    """)
    return


@app.cell(hide_code=True)
def experiment_description():
    mo.md(r"""
    Here is the experiment: we will chose beforehand a few values $\mu_1$,  $\mu_2$ ... and sample values around them.
    Then, we will convert them to text and fed it to the parrot. Since we want to know how Gemmma represent uncertainty, we will also vary the variability of the sequences we generate. Finally, we will extract the activation at a specific layer.

    ///note
    **Which token**:
    In order to get the vector, we have to chose the token we're looking at. We could chose the last of the sequence, doing the average of all tokens, or even more complex strategies. For better stability, the paper uses the last K tokens of the seqence, and average them.
    ///
    """)
    return


@app.cell(hide_code=True)
def diagram_experiment():
    diagram_e2 = mo.ui.anywidget(Excalidraw.from_file("assets/experimental-setup.excalidraw"))
    diagram_e2
    return


@app.cell(hide_code=True)
def save_diagram_experiment():
    #diagram_e2.save()
    return


@app.cell(hide_code=True)
def simple_setup_intro():
    mo.md(r"""
    In order to understand the basics, we will a simpler setup first: take any 3-digit number, embed it in a sentence, and look at the corresponding representation.

    Note that we chose **random templates** to add variability, and see the representation of the number in various situations.
    """)
    return


@app.cell
def diagram_simple():
    diagram_e3 = mo.ui.anywidget(Excalidraw.from_file("assets/experimental-setup-simple.excalidraw"))
    diagram_e3
    return


@app.cell
def save_diagram_simple():
    #diagram_e3.save()
    return


@app.cell(hide_code=True)
def collection_intro():
    mo.md(r"""
    We'll first generate these vectors and store them (at each layer). Click to start !
    """)
    return


@app.cell(hide_code=True)
def experiment_form():
    _fields = mo.md("""
    **Mu values**
    {mu_values}

    **Sigma values**
    {sigma_values}

    **Sequences per (mu, sigma)**
    {n_sequences}

    **Averaging window**
    {convergence_window}

    **Random seed**
    {seed}
    """).batch(
        mu_values=mo.ui.text(value="350,450,550,650", label=""),
        sigma_values=mo.ui.text(value="20,50,100,200", label=""),
        n_sequences=mo.ui.slider(start=8, stop=32, step=8, value=16, label=""),
        convergence_window=mo.ui.range_slider(
            start=0, stop=400, step=10, value=[100, 200], label=""
        ),
        seed=mo.ui.number(start=0, stop=9999, step=1, value=33, label=""),
    )
    experiment_form = _fields.form(submit_button_label="Compute")
    experiment_form
    return (experiment_form,)


@app.cell(hide_code=True)
def experiment_config(experiment_form):
    @dataclass
    class ExperimentConfig:
        mu_values: tuple[float, ...]
        sigma_values: tuple[float, ...]
        n_sequences: int
        n_numbers: int
        convergence_start: int
        seed: int

    mo.stop(
        experiment_form.value is None,
        mo.callout(
            mo.md("Submit the experiment form above to continue."), kind="warn"
        ),
    )

    _v = experiment_form.value
    experiment_config = ExperimentConfig(
        mu_values=tuple(float(x) for x in _v["mu_values"].split(",")),
        sigma_values=tuple(float(x) for x in _v["sigma_values"].split(",")),
        n_sequences=int(_v["n_sequences"]),
        convergence_start=int(_v["convergence_window"][0]),
        n_numbers=int(_v["convergence_window"][1]),
        seed=int(_v["seed"]),
    )
    return ExperimentConfig, experiment_config


@app.cell(hide_code=True)
def collect_imports():
    SUPPORT = 1000
    DEVICE = torch.device("cuda")

    def gpu_batch_size(default: int) -> int:
        """Shrink to a conservative batch size on small GPUs to avoid OOM; keep
        `default` on GPUs with enough headroom for the model + full-vocab logits."""
        if not torch.cuda.is_available():
            return default
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return 4 if total_gb < 8 else default

    @dataclass(frozen=True, kw_only=True, slots=True)
    class ModelBundle:
        model: object
        tokenizer: object
        int_token_ids: torch.Tensor
        comma_token_id: int
        n_layers: int
        d_model: int

    def sample_gaussian_ints(
        mu: float, sigma: float, n: int, rng
    ) -> np.ndarray:
        x = rng.normal(loc=mu, scale=sigma, size=n)
        return np.clip(np.round(x), 0, SUPPORT - 1).astype(np.int64)

    def ints_to_prompt(xs: np.ndarray) -> str:
        return ",".join(str(int(x)) for x in xs)

    def group_prompts(
        mu: float, sigma: float, n_numbers: int, n_sequences: int, seed: int
    ) -> list[str]:
        rng = np.random.default_rng(seed)
        return [
            ints_to_prompt(sample_gaussian_ints(mu, sigma, n_numbers, rng))
            for _ in range(n_sequences)
        ]

    def load_model(model_name: str) -> ModelBundle:
        import logging
        import warnings
        import transformers
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        logging.getLogger("httpx").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(DEVICE)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        int_ids = [
            tokenizer.encode(str(n), add_special_tokens=False)
            for n in range(SUPPORT)
        ]
        comma = tokenizer.encode(",", add_special_tokens=False)
        return ModelBundle(
            model=model,
            tokenizer=tokenizer,
            int_token_ids=torch.tensor([e[0] for e in int_ids], device=DEVICE),
            comma_token_id=comma[0],
            n_layers=model.config.num_hidden_layers,
            d_model=model.config.hidden_size,
        )

    def tokenize(bundle: ModelBundle, prompts: Sequence[str]) -> torch.Tensor:
        return bundle.tokenizer(
            list(prompts), return_tensors="pt", add_special_tokens=True
        ).input_ids

    def _comma_positions(
        bundle: ModelBundle, ids: torch.Tensor
    ) -> torch.Tensor:
        return (ids == bundle.comma_token_id).nonzero(as_tuple=True)[0]

    def _int_softmax(
        bundle: ModelBundle, logits: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        int_probs = probs.index_select(-1, bundle.int_token_ids)
        return int_probs / int_probs.sum(dim=-1, keepdim=True)

    @torch.inference_mode()
    def scan_batch(bundle: ModelBundle, ids: torch.Tensor):
        ids = ids.to(DEVICE)
        pos = _comma_positions(bundle, ids[0])
        out = bundle.model(input_ids=ids, output_hidden_states=True)
        # shrink each layer to just the comma positions BEFORE stacking, instead of
        # stacking all layers at full sequence length first: cuts peak GPU memory
        # roughly seq_len / len(pos) with an identical result.
        resid = torch.stack(
            [h.index_select(1, pos) for h in out.hidden_states], dim=1
        )
        soft = _int_softmax(bundle, out.logits.index_select(1, pos))
        return resid.float().cpu().numpy(), soft.cpu().numpy()

    def _steering_module(bundle: ModelBundle, layer: int):
        """Module whose output is hidden_states[layer] (0 = embeddings, i = block i-1)."""
        if layer == 0:
            return bundle.model.model.embed_tokens
        return bundle.model.model.layers[layer - 1]

    @torch.inference_mode()
    def steered_scan_batch(
        bundle: ModelBundle, ids: torch.Tensor, layer: int, vector: np.ndarray
    ):
        """Like scan_batch, but adds `vector` to the residual stream ONLY at the last
        comma position of hidden_states[layer] (not broadcast to every position --
        steering every position measurably inflates sigma, see EXPERIMENT.md D3/H2),
        then reads off the real softmax over int tokens at that same position."""
        ids = ids.to(DEVICE)
        pos = _comma_positions(bundle, ids[0])
        last_pos = pos[-1]
        vec = torch.as_tensor(vector, dtype=torch.float16, device=DEVICE)

        def _hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs.clone()
            hs[:, last_pos, :] = hs[:, last_pos, :] + vec
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs

        handle = _steering_module(bundle, layer).register_forward_hook(_hook)
        try:
            out = bundle.model(input_ids=ids)
        finally:
            handle.remove()
        return (
            _int_softmax(bundle, out.logits[:, last_pos : last_pos + 1])
            .cpu()
            .numpy()
        )

    return (
        DEVICE,
        SUPPORT,
        gpu_batch_size,
        group_prompts,
        ints_to_prompt,
        load_model,
        sample_gaussian_ints,
        scan_batch,
        steered_scan_batch,
        tokenize,
    )


@app.cell(hide_code=True)
def model_bundle(load_model, model_selector):
    mo.stop(
        not torch.cuda.is_available(),
        mo.callout(
            mo.md("No GPU available — model loading skipped."), kind="warn"
        ),
    )
    with mo.status.progress_bar(total=1, title="Loading model") as _bar:
        bundle = load_model(model_selector.value)
        _bar.update()
    return (bundle,)


@app.cell(hide_code=True)
def simple_embeddings(DEVICE, bundle, tokenize):
    mo.stop(
        not torch.cuda.is_available(),
        mo.callout(mo.md("No GPU available — embedding skipped."), kind="warn"),
    )

    TEMPLATES = [
        "I think {}",
        "The number {}",
        "There are {}",
        "I counted {}",
        "Approximately {}",
        "I see {}",
        "Count: {}",
        "Value {}",
    ]

    @torch.inference_mode()
    def _embed_number_templates(n: int) -> np.ndarray:
        """Per-template activations, not averaged. Returns [templates, layers, d]."""
        acts = []
        for template in TEMPLATES:
            prompt = template.format(n)
            ids = tokenize(bundle, [prompt]).to(DEVICE)
            n_digit_toks = len(bundle.tokenizer.encode(str(n), add_special_tokens=False))
            out = bundle.model(input_ids=ids, output_hidden_states=True)
            hidden = torch.stack(out.hidden_states, dim=1)[0]
            acts.append(hidden[:, -n_digit_toks:, :].mean(dim=1).float().cpu().numpy())
        return np.stack(acts)  # [templates, layers, d]

    with mo.persistent_cache("simple_embeddings_templates", save_path="./checkpoints/cache"):
        simple_embeddings_templates = {
            n: _embed_number_templates(n)
            for n in mo.status.progress_bar(range(100, 1000), title="Embedding numbers (per template)")
        }

    simple_embeddings = {n: v.mean(axis=0) for n, v in simple_embeddings_templates.items()}
    return TEMPLATES, simple_embeddings_templates


@app.cell(hide_code=True)
def collect_activations(
    bundle,
    experiment_config,
    gpu_batch_size,
    ints_to_prompt,
    sample_gaussian_ints,
    scan_batch,
    tokenize,
):
    mo.stop(
        not torch.cuda.is_available(),
        mo.callout(
            mo.md("No GPU available — collection skipped."), kind="warn"
        ),
    )

    _collection_batch_size = gpu_batch_size(8)

    def _collect_group(
        mu: float, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect converged-mean residuals and softmax for one (mu, sigma) pair."""
        rng = np.random.default_rng(experiment_config.seed)
        prompts = [
            ints_to_prompt(
                sample_gaussian_ints(
                    mu, sigma, experiment_config.n_numbers, rng
                )
            )
            for _ in range(experiment_config.n_sequences)
        ]
        resid_chunks, soft_chunks = [], []
        for i in range(0, len(prompts), _collection_batch_size):
            ids = tokenize(bundle, prompts[i : i + _collection_batch_size])
            resid, soft = scan_batch(bundle, ids)
            conv = slice(experiment_config.convergence_start, None)
            resid_chunks.append(resid[:, :, conv, :].mean(axis=2))
            soft_chunks.append(soft[:, conv, :].mean(axis=1))
        return np.concatenate(resid_chunks), np.concatenate(soft_chunks)

    _settings = [
        (mu, sigma)
        for mu in experiment_config.mu_values
        for sigma in experiment_config.sigma_values
    ]

    with mo.persistent_cache("activations", save_path="./checkpoints/cache"):
        activation_groups = {
            (mu, sigma): _collect_group(mu, sigma)
            for mu, sigma in mo.status.progress_bar(
                _settings, title="Collecting activations"
            )
        }
    return (activation_groups,)


@app.cell(hide_code=True)
def numbers_and_curves_title():
    mo.md(r"""
    # Numbers and curves
    """)
    return


@app.cell(hide_code=True)
def simple_probe_intro():
    mo.md(r"""
    Now, we have everything to start studying our LLM. Let's focus on the simpler setup: 3 digits numbers.

    Let's train a single probe to predict the magnitude of the number.

    ///note
    Usually, probes detect binary labels (presence or absence of a concept) but we can of course predict values directly ! But as we will see, that's not the best way to do it.
    ///
    """)
    return


@app.cell(hide_code=True)
def simple_layer_slider(bundle):
    simple_layer_slider = mo.ui.slider(0, bundle.n_layers, value=12, label="Layer", debounce=True, show_value=True)
    simple_layer_slider
    return (simple_layer_slider,)


@app.function(hide_code=True)
def build_flat_split(embeddings_templates, layer, test_size=0.2, seed=42, n_templates_train=3):
    """Flatten per-template activations at a layer.

    Train keeps a random sample of `n_templates_train` templates per number
    (multiple rows/number); test keeps exactly one random template per number.
    """
    numbers = np.array(sorted(embeddings_templates.keys()))
    train_numbers, test_numbers = train_test_split(numbers, test_size=test_size, random_state=seed)
    rng = np.random.default_rng(seed)
    n_t = next(iter(embeddings_templates.values())).shape[0]

    train_rows = [
        (n, t, embeddings_templates[n][t, layer])
        for n in train_numbers
        for t in rng.choice(n_t, size=min(n_templates_train, n_t), replace=False)
    ]
    numbers_train = np.array([r[0] for r in train_rows])
    template_idx_train = np.array([r[1] for r in train_rows])
    X_train = np.stack([r[2] for r in train_rows])

    test_template_idx = {n: int(rng.integers(n_t)) for n in test_numbers}
    numbers_test = test_numbers
    template_idx_test = np.array([test_template_idx[n] for n in test_numbers])
    X_test = np.stack([embeddings_templates[n][test_template_idx[n], layer] for n in test_numbers])

    return numbers_train, template_idx_train, X_train, numbers_test, template_idx_test, X_test


@app.cell(hide_code=True)
def simple_probe_scatter(
    PIRATE_BLUE,
    PIRATE_ORANGE,
    TEMPLATES,
    simple_embeddings_templates,
    simple_layer_slider,
):
    mo.stop(
        not simple_embeddings_templates,
        mo.callout(mo.md("No embeddings — run simple embedding first."), kind="warn"),
    )

    _layer = simple_layer_slider.value
    _n_tr, _t_tr, _X_tr, _n_te, _t_te, _X_te = build_flat_split(simple_embeddings_templates, _layer)

    _y_tr = _n_tr.astype(np.float64)
    _y_te = _n_te.astype(np.float64)

    _clf = LinearRegression().fit(_X_tr, _y_tr)
    _r2_train = _clf.score(_X_tr, _y_tr)
    _r2_test = _clf.score(_X_te, _y_te)

    _color_scale = alt.Scale(domain=["train", "test"], range=[PIRATE_BLUE, PIRATE_ORANGE])

    _df_tr = pl.DataFrame({
        "true_value": _y_tr,
        "predicted_value": _clf.predict(_X_tr),
        "split": "train",
    })
    _df_te = pl.DataFrame({
        "true_value": _y_te,
        "predicted_value": _clf.predict(_X_te),
        "split": "test",
        "prompt": [TEMPLATES[t].format(n) for n, t in zip(_n_te, _t_te)],
    })

    _train_layer = alt.Chart(_df_tr).mark_circle(size=40, opacity=0.6).encode(
        alt.X("true_value:Q", title="True value"),
        alt.Y("predicted_value:Q", title="Predicted value"),
        alt.Color("split:N", scale=_color_scale, legend=alt.Legend(title=None)),
    )
    _test_layer = alt.Chart(_df_te).mark_circle(size=40, opacity=0.6).encode(
        alt.X("true_value:Q"),
        alt.Y("predicted_value:Q"),
        alt.Color("split:N", scale=_color_scale, legend=alt.Legend(title=None)),
        [alt.Tooltip("prompt:N", title="prompt")],
    )

    (_train_layer + _test_layer).properties(
        width=600, height=350,
        title=f"Simple linear probe — layer {_layer} | train R²={_r2_train:.2f}  test R²={_r2_test:.2f}",
    ).configure_view(strokeWidth=0).configure_axis(grid=False)
    return


@app.cell(hide_code=True)
def simple_probe_scatter_note():
    mo.md(r"""
    Well, it seems to work extremely well ! Note that we are working in dimension $2048$, it's pretty easy to fit pretty much anything in this space ... That's why we added some validation points, and they seem to hold up.

    So we can detect how big a number is. But that's not the only thing we could want to detect ! What about this: "does the number end with a $7$" ? Let's see.
    """)
    return


@app.cell(hide_code=True)
def ends_with_7_probe(
    PIRATE_BLUE,
    PIRATE_ORANGE,
    simple_embeddings_templates,
    simple_layer_slider,
):
    mo.stop(
        not simple_embeddings_templates,
        mo.callout(mo.md("No embeddings — run simple embedding first."), kind="warn"),
    )

    _layer = simple_layer_slider.value
    _n_tr, _t_tr, _X_tr, _n_te, _t_te, _X_te = build_flat_split(simple_embeddings_templates, _layer)

    _y_tr = (_n_tr % 10 == 7).astype(int)
    _y_te = (_n_te % 10 == 7).astype(int)

    _clf = LogisticRegression(max_iter=1000).fit(_X_tr, _y_tr)
    _acc_train = _clf.score(_X_tr, _y_tr)
    _acc_test = _clf.score(_X_te, _y_te)

    _proba_tr = _clf.predict_proba(_X_tr)[:, 1]
    _proba_te = _clf.predict_proba(_X_te)[:, 1]

    _df = pl.concat([
        pl.DataFrame({"last_digit": _n_tr % 10, "p_ends_7": _proba_tr, "split": "train"}),
        pl.DataFrame({"last_digit": _n_te % 10, "p_ends_7": _proba_te, "split": "test"}),
    ])
    _agg = _df.group_by(["last_digit", "split"]).agg(pl.mean("p_ends_7")).sort(["last_digit", "split"])

    alt.Chart(_agg).mark_bar(opacity=0.85).encode(
        alt.X("last_digit:O", title="Last digit", axis=alt.Axis(labelAngle=0)),
        alt.Y("p_ends_7:Q", title="Mean P(ends with 7)", stack="zero"),
        alt.Color("split:N", scale=alt.Scale(
            domain=["train", "test"],
            range=[PIRATE_BLUE, PIRATE_ORANGE],
        ), legend=alt.Legend(title=None)),
        [alt.Tooltip("last_digit:O", title="last digit"),
         alt.Tooltip("split:N", title="split"),
         alt.Tooltip("p_ends_7:Q", title="mean P(ends 7)", format=".2f")],
    ).properties(
        width=400, height=300,
        title=f"Probe: does number end with 7? — layer {_layer} | train acc={_acc_train:.2f}  test acc={_acc_test:.2f}",
    ).configure_view(strokeWidth=0).configure_axis(grid=False)
    return


@app.cell(hide_code=True)
def digit_probes_puzzle():
    mo.md(r"""
    It works perfectly too !

    Let's call the probes we build $P_{size}$ and $P_7$

    I have a puzzle for you: **«can you organize the numbers in space, such that you can detect $P_{size}$ in one direction, $P_7$ in another ?»** What shape do you get ? Think about it.
    """)
    return


@app.cell(hide_code=True)
def helix_probe_display():
    HelixProbe(is_3d=True, autoplay=False, js="remote")
    return


@app.cell(hide_code=True)
def helix_basis_note():
    mo.md(r"""
    And with a bit more of probes magic, we can find a basis in which the numbes have exactly this shape:
    """)
    return


@app.cell(hide_code=True)
def digit_angle_probes(
    TEMPLATES,
    simple_embeddings_templates,
    simple_layer_slider,
):
    mo.stop(
        not simple_embeddings_templates,
        mo.callout(mo.md("No embeddings — run simple embedding first."), kind="warn"),
    )

    _layer = simple_layer_slider.value
    _n_tr, _t_tr, _X_tr, _n_te, _t_te, _X_te = build_flat_split(simple_embeddings_templates, _layer)

    _angles_tr = 2 * np.pi * (_n_tr % 10) / 10
    _cos_tr = np.cos(_angles_tr)
    _sin_tr = np.sin(_angles_tr)

    probe_cos_digit = LinearRegression().fit(_X_tr, _cos_tr)
    probe_sin_digit = LinearRegression().fit(_X_tr, _sin_tr)

    _cos_pred = probe_cos_digit.predict(_X_te)
    _sin_pred = probe_sin_digit.predict(_X_te)

    _pred_digit = np.round(np.arctan2(_sin_pred, _cos_pred) * 10 / (2 * np.pi)).astype(int) % 10
    _acc = np.mean(_pred_digit == (_n_te % 10))

    _example_texts = [TEMPLATES[t].format(n) for n, t in zip(_n_te, _t_te)]

    _df_scatter = pl.DataFrame({
        "cos_pred": _cos_pred,
        "sin_pred": _sin_pred,
        "last_digit": (_n_te % 10).astype(str),
        "number": _n_te,
        "prompt": _example_texts,
    })

    alt.Chart(_df_scatter).mark_circle(size=50, opacity=0.5).encode(
        alt.X("cos_pred:Q", title="cos probe", scale=alt.Scale(domain=[-1.5, 1.5])),
        alt.Y("sin_pred:Q", title="sin probe", scale=alt.Scale(domain=[-1.5, 1.5])),
        alt.Color("last_digit:N",
            scale=alt.Scale(scheme="tableau10"),
            sort=[str(d) for d in range(10)],
            legend=alt.Legend(title="last digit"),
        ),
        [alt.Tooltip("number:Q", title="number"),
         alt.Tooltip("last_digit:N", title="last digit"),
         alt.Tooltip("prompt:N", title="prompt")],
    ).properties(
        width=350, height=350,
        title=f"Digit angle probes — layer {_layer} (digit acc={_acc:.2f})",
    ).configure_view(strokeWidth=0).configure_axis(grid=False).properties(height=500, width=500)
    return probe_cos_digit, probe_sin_digit


@app.class_definition(hide_code=True)
class HelixProbe(ManimWidget):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            z_range=[-3.5, 3.5, 1],
            x_length=4, y_length=4, z_length=6,
            axis_config={"color": WHITE, "stroke_width": 1.5},
        )
        x_label = axes.get_x_axis_label(Text("detects a 7", font_size=20, color=WHITE))
        z_label = axes.get_z_axis_label(Text("magnitude", font_size=20, color=WHITE))
        self.add(axes, x_label, z_label)

        _n_turns = 3
        _n_pts = 600
        _ts = [i / _n_pts for i in range(_n_pts + 1)]
        _coords = [
            (
                np.cos(2 * np.pi * _n_turns * t),
                np.sin(2 * np.pi * _n_turns * t),
                6 * t - 3,
            )
            for t in _ts
        ]
        _coords_manim = [axes.c2p(x, y, z) for x, y, z in _coords]

        helix = VMobject()
        helix.set_points_smoothly(_coords_manim)
        helix.set_color("#E87F24")
        helix.set_stroke(width=4)
        self.play(Create(helix), run_time=5)

        self.camera.background_color = "#1a1a2e"
        self.set_camera_orientation(phi=60 * DEGREES, theta=-75 * DEGREES, zoom=1)


@app.cell(hide_code=True)
def probe_helpers():
    _EPS = 1e-12

    def train_probe(x, y, seed=33, test_size=0.2):
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=test_size, random_state=seed, stratify=y
        )
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(x_tr, y_tr)
        weights = clf.coef_.astype(np.float32)
        if weights.shape[0] == 1:
            weights = np.concatenate([-weights, weights], axis=0)
        return weights, float(clf.score(x_te, y_te))

    def transfer_accuracy(x_train, y_train, x_test, y_test, seed=33):
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(x_train, y_train)
        return float(clf.score(x_test, y_test))

    def cosine_gram(w):
        unit = w / (np.linalg.norm(w, axis=-1, keepdims=True) + _EPS)
        return unit @ unit.T

    def kernel_pca_embedding(gram, n_components=3):
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1][:n_components]
        lam = np.clip(eigvals[order], 0.0, None)
        return (eigvecs[:, order] * np.sqrt(lam)).astype(np.float32), eigvals[order]

    def linear_interp(w_left, w_right, frac):
        return (1.0 - frac) * w_left + frac * w_right

    def slerp(w_left, w_right, frac):
        n_l, n_r = np.linalg.norm(w_left), np.linalg.norm(w_right)
        u_l = w_left / (n_l + _EPS)
        u_r = w_right / (n_r + _EPS)
        omega = np.arccos(np.clip(u_l @ u_r, -1.0, 1.0))
        if omega < _EPS:
            direction = u_l
        else:
            s = np.sin(omega)
            direction = (np.sin((1 - frac) * omega) * u_l + np.sin(frac * omega) * u_r) / s
        return ((1.0 - frac) * n_l + frac * n_r) * direction

    def kernel_interp(params, w, query, bandwidth):
        weights = np.exp(-0.5 * ((params - query) / bandwidth) ** 2)
        weights /= weights.sum() + _EPS
        return (weights[:, None] * w).sum(axis=0)

    def cosine(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + _EPS))

    def pooled_mean(groups: dict, mu: float, layer: int) -> np.ndarray:
        """Mean residual-stream activation across all sigma groups for one mu, at one layer."""
        chunks = [r[:, layer, :] for (m, _s), (r, _) in groups.items() if m == mu]
        return np.concatenate(chunks).astype(np.float64).mean(axis=0)

    def pooled_mean_by_sigma(groups: dict, sigma: float, layer: int) -> np.ndarray:
        """Mean residual-stream activation across all mu groups for one sigma, at one layer
        (mirrors pooled_mean, pooling the other way)."""
        chunks = [r[:, layer, :] for (_m, s), (r, _) in groups.items() if s == sigma]
        return np.concatenate(chunks).astype(np.float64).mean(axis=0)

    return (
        kernel_interp,
        linear_interp,
        pooled_mean,
        pooled_mean_by_sigma,
        train_probe,
    )


@app.cell
def simple_layer_slider_display(simple_layer_slider):
    simple_layer_slider
    return


@app.cell(hide_code=True)
def helix_template_selector():
    helix_template_dropdown = mo.ui.dropdown(
        options=[
            "I think {}",
            "The number {}",
            "I counted {}",
            "Approximately {}",
            "I see {}",
            "Value {}",
        ],
        value="I counted {}",
        label="Chose the template used for embedding:",
    )
    helix_template_dropdown
    return (helix_template_dropdown,)


@app.cell(hide_code=True)
def digit_angle_helix(
    TEMPLATES,
    helix_template_dropdown,
    probe_cos_digit,
    probe_sin_digit,
    simple_embeddings_templates,
    simple_layer_slider,
):
    mo.stop(
        not simple_embeddings_templates,
        mo.callout(mo.md("No embeddings — run simple embedding first."), kind="warn"),
    )

    _layer = simple_layer_slider.value
    _TEMPLATE = helix_template_dropdown.value
    _numbers = np.array(range(100, 150))

    _template_idx = TEMPLATES.index(_TEMPLATE)
    _X_all = np.stack([
        simple_embeddings_templates[n][_template_idx, _layer] for n in _numbers
    ]).astype(np.float64)
    _cos_all = probe_cos_digit.predict(_X_all)
    _sin_all = probe_sin_digit.predict(_X_all)
    _z_all = (_numbers - _numbers.min()) / (_numbers.max() - _numbers.min()) * 6 - 3

    _DIGIT_COLORS = [
        "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
        "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac",
    ]

    class _DigitHelix(ManimWidget):
        def __init__(self, numbers, cos_preds, sin_preds, z_vals, **kwargs):
            self._numbers = numbers
            self._cos = cos_preds
            self._sin = sin_preds
            self._z = z_vals
            super().__init__(**kwargs)

        def construct(self):
            _axes = ThreeDAxes(
                x_range=[-2, 2, 1], y_range=[-2, 2, 1], z_range=[-3.5, 3.5, 1],
                x_length=4, y_length=4, z_length=6,
                axis_config={"color": WHITE, "stroke_width": 1.5},
            )
            self.add(_axes)
            self.add(_axes.get_x_axis_label(Text("cos probe", font_size=18, color=WHITE)))
            self.add(_axes.get_y_axis_label(Text("sin probe", font_size=18, color=WHITE)))
            self.add(_axes.get_z_axis_label(Text("magnitude", font_size=18, color=WHITE)))

            _groups = [[] for _ in range(10)]
            for i, n in enumerate(self._numbers):
                _groups[int(n % 10)].append(_axes.c2p(self._cos[i], self._sin[i], self._z[i]))

            for digit, pts in enumerate(_groups):
                if not pts:
                    continue
                pm = PMobject()
                pm.add_points(pts, color=_DIGIT_COLORS[digit])
                pm.stroke_width = 12
                pm.set_opacity(0.35)
                self.add(pm)

            _all_pts = [_axes.c2p(self._cos[i], self._sin[i], self._z[i])
                        for i in range(len(self._numbers))]
            spline = VMobject()
            spline.set_points_smoothly(_all_pts)
            spline.set_color_by_gradient(*_DIGIT_COLORS)
            spline.set_stroke(width=4)
            self.add(spline)

            self.camera.background_color = "#1a1a2e"
            self.set_camera_orientation(phi=80 * DEGREES, theta=-75 * DEGREES, zoom=1.8)

    _DigitHelix(_numbers, _cos_all, _sin_all, _z_all, is_3d=True, js="remote")
    return


@app.cell(hide_code=True)
def curvy_numbers_note():
    mo.md(r"""
    If you can remember only one thing, let it be that **inside LLMs, numbers are curvy**.

    You may wonder:

    > «Chosing to detect magnitude and last digit was pretty arbitrary, does that mean we can chose the probes to represent any shape we want ? Or is there a **fundamental shape underneath** ?»

    You are right to be careful: probes are really useful to understand one **particular facet** of the model. You chose one direction to look at among the ten of hundreds of possible directions.

    But probes can't completely make up reality : There are some concepts that probes just can't detect (see example below).
    """)
    return


@app.cell(hide_code=True)
def parity_probe(
    PIRATE_GREEN,
    PIRATE_RED,
    simple_embeddings_templates,
    simple_layer_slider,
):
    mo.stop(
        not simple_embeddings_templates,
        mo.callout(mo.md("No embeddings — run simple embedding first."), kind="warn"),
    )

    def _digit_sum_parity(n):
        _d = (n // 100) + (n // 10) % 10 + n % 10
        return _d % 2

    _layer = simple_layer_slider.value
    _n_tr, _t_tr, _X_tr, _n_te, _t_te, _X_te = build_flat_split(simple_embeddings_templates, _layer)

    _y_tr = np.array([_digit_sum_parity(n) for n in _n_tr])
    _y_te = np.array([_digit_sum_parity(n) for n in _n_te])

    _clf = LogisticRegression(max_iter=1000).fit(_X_tr, _y_tr)
    _acc_train = _clf.score(_X_tr, _y_tr)
    _acc_test = _clf.score(_X_te, _y_te)

    _df = pl.DataFrame({
        "number": _n_te,
        "true_parity": np.where(_y_te == 1, "odd", "even"),
        "predicted_prob": _clf.predict_proba(_X_te)[:, 1],
    })

    _chart = alt.Chart(_df).mark_point(size=40, filled=True).encode(
        alt.X("number:N", title="Number", axis=alt.Axis(labels=False, ticks=False)),
        alt.Y("predicted_prob:Q", title="P(odd)"),
        alt.Color("true_parity:N", scale=alt.Scale(
            domain=["even", "odd"], range=[PIRATE_GREEN, PIRATE_RED],
        ), legend=alt.Legend(title="true parity")),
        [alt.Tooltip("number:N", title="number"),
         alt.Tooltip("true_parity:N", title="true parity"),
         alt.Tooltip("predicted_prob:Q", title="P(odd)", format=".2f")],
    ).properties(
        width=900, height=200,
        title=f"Parity probe — train acc={_acc_train:.3f}  test acc={_acc_test:.3f}",
    ).configure_view(strokeWidth=0).configure_axis(grid=False)

    mo.vstack([_chart, mo.md("> *A probe can't detect wether the sum of the digits is even or odd*")])
    return


@app.cell(hide_code=True)
def manifolds_title():
    mo.md(r"""
    # The beautiful world of manifolds
    """)
    return


@app.cell(hide_code=True)
def manifolds_intro():
    mo.md(r"""
    So far, each number we considered was associated with a signle point in the latent space (the space of activations). That was fine for the simple version of the experiment, but not for the version with mean **and** variability.

    Let's say we train a probe $P_{big}$: «*Is this number big, according to a human ?*». Of course, the answer will depend on context:
    - "204 km" will fire it
    - "204 cm" will not

    This shows that we can't think about $204$ as a single point. Instead, it's a **set**, or a **region**.

    It's the same thing for our $(\mu, \sigma)$ experiment: there is not such a thing as the "$\mu=204$" point in our latent space. Instead, we can think of it as the set of all points with "$\mu=204$", with different values of $\sigma$.

    It's worth trying to visualize what we mean by that.
    """)
    return


@app.class_definition(hide_code=True)
class ManifoldWorld(ManimWidget):
    def setup_scene(self):
        self.camera.background_color = "#1a1a2e"
        self.set_camera_orientation(
            phi=65 * DEGREES, theta=-60 * DEGREES, zoom=1.5
        )

        self._rng = np.random.default_rng(7)

        self.n_values = [10, 60, 220, 800]
        self.n_regions = len(self.n_values)

        _lo, _hi = np.sqrt(min(self.n_values)), np.sqrt(max(self.n_values))

        def _x_pos(n):
            return -7.0 + 14.0 * (np.sqrt(n) - _lo) / (_hi - _lo)

        self.xs = [_x_pos(n) for n in self.n_values]

    def construct(self):
        self.setup_scene()
        self.section1_points()
        self.section2_spine()
        self.section3_regions()
        self.section4_splines()
        self.section5_morph()

    def ensure_axes(self, animate=True):
        if getattr(self, "axes", None) is None:
            self.axes = ThreeDAxes(
                x_range=[-8, 8, 2],
                y_range=[-3, 3, 1],
                z_range=[-3, 3, 1],
                x_length=9,
                y_length=6,
                z_length=6,
                axis_config={"color": GRAY, "stroke_width": 0.8},
            )
            if animate:
                self.play(Create(self.axes), run_time=1.8)
            else:
                self.add(self.axes)
            return True
        return False

    def section1_points(self):
        self.next_section("Simple case: points")
        title1 = Text(
            "Simple case:\neach number is a point",
            font_size=64,
            color=WHITE,
        )
        self.add_fixed_in_frame_mobjects(title1)
        title1.set_opacity(0)
        self.play(title1.animate.set_opacity(1), run_time=1.5)
        self.wait(0.8)
        self.play(FadeOut(title1), run_time=1.2)

        self.ensure_axes()

        self._y_offsets = self._rng.normal(0, 1.0, self.n_regions)
        self._z_offsets = self._rng.normal(0, 1.0, self.n_regions)

        self.dots = []
        self.labels = []
        for n, x, y, z in zip(
            self.n_values, self.xs, self._y_offsets, self._z_offsets
        ):
            pos = self.axes.c2p(x, y, z)
            dot = Dot3D(pos, radius=0.18, color=ORANGE)
            lbl = Text(f"n={n}", font_size=20, color=WHITE).next_to(
                dot, UP + OUT, buff=0.15
            )
            self.add_fixed_orientation_mobjects(lbl)
            self.play(FadeIn(dot), Write(lbl), run_time=1.0)
            self.wait(0.6)
            self.dots.append(dot)
            self.labels.append(lbl)

        self.wait(0.5)
        self.play(*[FadeOut(l) for l in self.labels], run_time=0.9)

    def section2_spine(self):
        self.next_section("Spine through the points")
        spine_pts = [
            self.axes.c2p(x, y, z)
            for x, y, z in zip(self.xs, self._y_offsets, self._z_offsets)
        ]
        self.spine = VMobject(color=ORANGE, stroke_width=5)
        self.spine.set_points_smoothly(spine_pts)
        self.play(Create(self.spine), run_time=2.2)
        self.wait(0.8)

        self.play(
            *[FadeOut(d) for d in self.dots],
            FadeOut(self.spine),
            run_time=1.2,
        )

    def _compute_regions(self):
        """Build patatoids/points for every region. Pure computation: creates the
        Mobjects but does not add or animate them into the scene."""
        _base_r = 1.1
        _base_noise = 0.24  # one-time perturbation that turns the circle into a "potato"
        _region_noise = (
            0.10  # small extra x,y jitter re-drawn for every region
        )
        _n_blob_pts = 10
        _angles = np.linspace(0, 2 * np.pi, _n_blob_pts, endpoint=False)

        _inner_angles = np.array([90, 210, 330]) * DEGREES
        _inner_r = 0.44
        self.inner_colors = [ORANGE, "#5C8020", "#a8071f"]

        _arc_amp = 1.5  # first-order movement: blob centers trace cos(x) as an arc in Y
        _arc_freq = 0.35

        # --- generate the base shape ONCE, as local (x, y) coordinates in the disc plane ---
        _base_radii = _base_r + self._rng.normal(
            0, _base_noise, _n_blob_pts
        )
        _base_perimeter_xy = np.stack(
            [_base_radii * np.cos(_angles), _base_radii * np.sin(_angles)],
            axis=1,
        )
        _base_inner_xy = np.stack(
            [
                _inner_r * np.cos(_inner_angles),
                _inner_r * np.sin(_inner_angles),
            ],
            axis=1,
        )

        self.patatoids = []
        self.region_dot_groups = []
        self.inner_pts_by_region = []
        self.region_raw_perimeter = []
        self.region_raw_inner = []
        for i, x in enumerate(self.xs):
            # slight random tilt of the patatoid's plane so it's not perfectly perpendicular to X
            _tilt_y, _tilt_z = self._rng.normal(0, 0.55, 2)
            _normal = np.array([1.0, _tilt_y, _tilt_z])
            _normal /= np.linalg.norm(_normal)
            _ref = (
                np.array([0.0, 1.0, 0.0])
                if abs(_normal[1]) < 0.9
                else np.array([0.0, 0.0, 1.0])
            )
            _u = np.cross(_normal, _ref)
            _u /= np.linalg.norm(_u)
            _v = np.cross(_normal, _u)
            _center = np.array([x, _arc_amp * np.cos(_arc_freq * x), 0.0])
            if i == 2:
                # third blob shifted up
                _center = _center + np.array([0.0, 0.8, 0.0])

            # per-region variation: only add noise to the local (x, y) coordinates of the base shape
            _perimeter_xy = _base_perimeter_xy + self._rng.normal(
                0, _region_noise, _base_perimeter_xy.shape
            )
            _inner_xy = _base_inner_xy + self._rng.normal(
                0, _region_noise, _base_inner_xy.shape
            )

            _raw_perimeter = np.array(
                [_center + px * _u + py * _v for px, py in _perimeter_xy]
            )
            _raw_inner = np.array(
                [_center + px * _u + py * _v for px, py in _inner_xy]
            )
            self.region_raw_perimeter.append(_raw_perimeter)
            self.region_raw_inner.append(_raw_inner)

            _pts3d = [self.axes.c2p(*p) for p in _raw_perimeter]
            blob = VMobject()
            blob.set_points_smoothly(_pts3d + [_pts3d[0]])
            blob.set_fill("#73A5CA", opacity=0.45)
            blob.set_stroke("#73A5CA", width=2)
            self.patatoids.append(blob)

            _inner_3d = [self.axes.c2p(*p) for p in _raw_inner]
            self.inner_pts_by_region.append(_inner_3d)

            _dots3 = [
                Dot3D(p, radius=0.12, color=c)
                for p, c in zip(_inner_3d, self.inner_colors)
            ]
            self.region_dot_groups.append(_dots3)

    def section3_regions(self):
        self.next_section("General case: regions")
        title2 = Text(
            "General case:\neach number is a region",
            font_size=64,
            color=WHITE,
        )
        self.add_fixed_in_frame_mobjects(title2)
        title2.set_opacity(0)

        _axes_pre_existing = getattr(self, "axes", None) is not None
        if _axes_pre_existing:
            self.play(
                title2.animate.set_opacity(1),
                FadeOut(self.axes),
                run_time=1.5,
            )
            self.wait(0.8)
            self.play(FadeOut(title2), FadeIn(self.axes), run_time=1.2)
        else:
            self.play(title2.animate.set_opacity(1), run_time=1.5)
            self.wait(0.8)
            self.play(FadeOut(title2), run_time=0.6)
            self.ensure_axes()

        self._compute_regions()
        self.region_labels = []
        for blob, dots3, n in zip(
            self.patatoids, self.region_dot_groups, self.n_values
        ):
            lbl = Text(f"n={n}", font_size=20, color=WHITE).move_to(
                blob.get_top() + UP * 0.3
            )
            self.add_fixed_orientation_mobjects(lbl)
            self.play(
                FadeIn(blob),
                *[FadeIn(d) for d in dots3],
                Write(lbl),
                run_time=1.3,
            )
            self.region_labels.append(lbl)

        self.wait(0.5)

    def section3_regions_instant(self):
        """Same end-state as section3_regions, but with everything already placed
        instead of animated in: no title, no axes-create animation, no per-region
        FadeIn."""
        self.ensure_axes(animate=False)
        self._compute_regions()
        for blob, dots3 in zip(self.patatoids, self.region_dot_groups):
            self.add(blob, *dots3)

    def section4_splines(self):
        self.next_section("Splines through the regions")
        self.splines = []
        self.sigma_labels = []
        for k in range(3):
            _pts_k = [
                self.inner_pts_by_region[r][k]
                for r in range(self.n_regions)
            ]
            _spl = VMobject(color=self.inner_colors[k], stroke_width=5)
            _spl.set_points_smoothly(_pts_k)
            self.splines.append(_spl)

            _lbl = Text(
                f"sigma {k + 1}", font_size=26, color=self.inner_colors[k]
            )
            _lbl.to_corner(UP + LEFT, buff=0.5).shift(DOWN * 0.55 * k)
            self.add_fixed_in_frame_mobjects(_lbl)
            self.sigma_labels.append(_lbl)

            self.play(Create(_spl), Write(_lbl), run_time=1.8)
            self.wait(0.5)

        self.wait(0.6)

    def section5_morph(self):
        self.next_section("Morph patatoids along the splines")
        morph_blob = self.patatoids[0]
        self.play(
            *[FadeOut(b) for b in self.patatoids[1:]],
            *[FadeOut(d) for grp in self.region_dot_groups for d in grp],
            *[FadeOut(l) for l in self.region_labels[1:]],
            run_time=1.3,
        )

        for _target in self.patatoids[1:]:
            self.play(
                Transform(morph_blob, _target.copy()),
                run_time=4.0,
                rate_func=linear,
            )


@app.cell
def manifold_world_display():
    ManifoldWorld(is_3d=True, autoplay=False, canvas_width=900. js="remote")
    return


@app.class_definition(hide_code=True)
class ManifoldWorldProjection(ManifoldWorld):
    def construct(self):
        self.setup_scene()
        self.section3_regions_instant()
        self.section_rotate_and_draw_probe_line()
        self.section5_project_to_line()

    def section_rotate_and_draw_probe_line(self):
        self.next_section("Rotate towards the probe line")

        # only change theta: tilt partway towards "Y as depth", keep phi as-is
        self.move_camera(theta=-80 * DEGREES, run_time=2.0, zoom=1)
        self.wait(0.3)

        self._line_z = -3.0
        self.probe_line = Line(
            self.axes.c2p(self.xs[0] - 1, 0, self._line_z),
            self.axes.c2p(self.xs[-1] + 1, 0, self._line_z),
            color=GRAY, stroke_width=3,
        )
        self.probe_label = Text("probe", font_size=26, color=GRAY).next_to(
            self.axes.c2p(self.xs[0] - 1, 0, self._line_z), LEFT, buff=0.3
        )
        self.add_fixed_orientation_mobjects(self.probe_label)
        self.play(Create(self.probe_line), Write(self.probe_label), run_time=1.5)
        self.wait(0.3)

    def section5_project_to_line(self):
        self.next_section("Project blobs onto a line")

        for i in range(self.n_regions):
            blob = self.patatoids[i]
            dots = self.region_dot_groups[i]

            blob_clone = blob.copy()
            dot_clones = [d.copy() for d in dots]
            self.add(blob_clone, *dot_clones)

            # orthogonal projection onto the line: keep x, snap z to the line, y to 0
            _raw_perimeter = self.region_raw_perimeter[i].copy()
            _raw_inner = self.region_raw_inner[i].copy()
            _raw_perimeter[:, 1] = 0.0
            _raw_perimeter[:, 2] = self._line_z
            _raw_inner[:, 1] = 0.0
            _raw_inner[:, 2] = self._line_z

            _target_pts = [self.axes.c2p(*p) for p in _raw_perimeter]
            target_blob = VMobject()
            target_blob.set_points_smoothly(_target_pts + [_target_pts[0]])
            target_blob.set_stroke(blob.get_stroke_color(), width=10)
            target_blob.set_fill(opacity=0)

            _target_dot_positions = [self.axes.c2p(*p) for p in _raw_inner]

            self.play(
                Transform(blob_clone, target_blob),
                *[dc.animate.move_to(pos) for dc, pos in zip(dot_clones, _target_dot_positions)],
                run_time=1.6,
            )
            self.wait(0.2)

        self.wait(0.6)


@app.cell
def _():
    ManifoldWorldProjection(is_3d=True, autoplay=False, canvas_width=900, js="remote")
    return


@app.cell(hide_code=True)
def submanifold_probe_note():
    mo.md(r"""
    Each $\mu=\mu_0$ is called a *submanifold*: a smooth region of the latent space.

    There is one last thing to understand here: where are our probes in this ?

    Easy: it's any direction that can **linearly separate** the submanifolds.
    """)
    return


@app.cell(hide_code=True)
def probe_information_loss_note():
    mo.md(r"""
    Notice how much information the probe loses: we can't know if the blobs are aranged in a straight line, or if they form a very complex pattern.

    We could improve this by computing *local probes*, but more on that later.


    And that's what the next experiment will show: **stering !**
    """)
    return


@app.cell(hide_code=True)
def steering_title():
    mo.md(r"""
    # Steering
    """)
    return


@app.cell(hide_code=True)
def steering_intro():
    mo.md(r"""
    The next experiment is the most interesting one I think.

    We will use the probe to directly **modify the LLM internal representations**. In the previous figure, it corresponds to translating a point in the direction given by the probe.
    This concept is called *steering*, and it's used to brain-wash LLMs in very different ways 🏴‍☠️

    Let's see that in action: take $(\sigma, \mu)$, look at the model prediction, push the activation vector, and see how that changes the model behaviour. Let's start very naive and use a linear probe.

    ///note
    Since a LLM compute the next token from every previous token, we could either steer the activations for the *last token*, or for *multiple tokens*. We also have to chose the layer at which we steer.

    Unfortunately, the paper stays unclear about what it used and why. Looking at the original code, it used layer 15 with steering on multiple tokens. We tried different settings and we found that:
    1. Steering works broadly better when using *the last* token only. For this reason, that's what we do in this notebook.
    2. Layer 15 is the second to last layer, which is not always where the richest internal representation are. We tried with layers `12` to `15`, and *for layer 12, the conclusions of the paper do not hold !*. We still use layer 15 in this notebook, but it's still an open problem to know why steering behaves differently in earlier layers.
    """)
    return


@app.cell
def steer_controls():
    LAYER = 15
    steer_mu = 350.0  # most illustrative starting point: farthest from the probe direction's midpoint
    steer_max_alpha = 1000
    return LAYER, steer_max_alpha, steer_mu


@app.cell(hide_code=True)
def steering_helpers(
    ExperimentConfig,
    PIRATE_BLUE,
    PIRATE_GREEN,
    PIRATE_ORANGE,
    PIRATE_PURPLE,
    SUPPORT,
    bundle,
    gpu_batch_size,
    group_prompts,
    pooled_mean,
    pooled_mean_by_sigma,
    steered_scan_batch,
    tokenize,
):
    def steering_direction(
        mu_coef: np.ndarray,
        mu_lo: float,
        mu_hi: float,
        groups: dict,
        layer: int,
    ) -> np.ndarray:
        """Linear mu-probe coefficient direction, scaled to the real observed centroid
        displacement between the two extreme mu groups — the most naive steering
        direction to try first."""
        unit = mu_coef / np.linalg.norm(mu_coef)
        scale = np.linalg.norm(
            pooled_mean(groups, mu_hi, layer)
            - pooled_mean(groups, mu_lo, layer)
        )
        return unit * (scale / (mu_hi - mu_lo))

    def sigma_steering_direction(
        sigma_coef: np.ndarray,
        sigma_lo: float,
        sigma_hi: float,
        groups: dict,
        layer: int,
    ) -> np.ndarray:
        """Linear sigma-probe coefficient direction, scaled to the real observed centroid
        displacement between the two extreme sigma groups — mirrors steering_direction,
        pooling over mu instead of sigma."""
        unit = sigma_coef / np.linalg.norm(sigma_coef)
        scale = np.linalg.norm(
            pooled_mean_by_sigma(groups, sigma_hi, layer)
            - pooled_mean_by_sigma(groups, sigma_lo, layer)
        )
        return unit * (scale / (sigma_hi - sigma_lo))

    @mo.persistent_cache
    def steered_moments(
        layer: int,
        direction: tuple[float, ...],
        alpha: float,
        mu: float,
        sigma: float,
        n_numbers: int,
        n_sequences: int,
        convergence_start: int,
        seed: int,
    ) -> tuple[float, float]:
        """Real softmax mean/std for (mu, sigma) prompts after adding alpha*direction to the
        residual stream at `layer` via a forward hook — the model is actually re-run end to end,
        no synthetic probe readout involved."""
        prompts = group_prompts(mu, sigma, n_numbers, n_sequences, seed)
        vector = alpha * np.array(direction)
        values = np.arange(SUPPORT)
        _batch = gpu_batch_size(16)
        chunks = []
        for i in range(0, len(prompts), _batch):
            ids = tokenize(bundle, prompts[i : i + _batch])
            soft = steered_scan_batch(bundle, ids, layer, vector)
            chunks.append(
                soft[:, 0, :]
            )  # single last-position readout per sequence
        probs = np.concatenate(chunks)
        pred_mu = probs @ values
        pred_var = (probs * (values[None, :] - pred_mu[:, None]) ** 2).sum(
            axis=1
        )
        return float(pred_mu.mean()), float(np.sqrt(pred_var).mean())

    def collect_steering_curve(
        layer: int,
        direction: np.ndarray,
        alphas: Sequence[float],
        groups: Sequence[tuple[float, float]],
        config: ExperimentConfig,
        mu_max: float | None = None,
    ) -> pl.DataFrame:
        """Sweep alpha for each (mu, sigma) group into one tidy dataframe; each point is an
        independently cached call to steered_moments, so re-sweeping a wider alpha range only
        computes the new points. If mu_max is set, stops sweeping a group as soon as its
        measured pred_mu reaches mu_max, instead of pre-guessing a safe alpha range per scheme."""
        direction_t = tuple(direction.tolist())
        rows = []
        for mu, sigma in groups:
            for alpha in mo.status.progress_bar(
                alphas, title=f"Steering @ layer {layer}"
            ):
                pred_mu, pred_sigma = steered_moments(
                    layer,
                    direction_t,
                    alpha,
                    mu,
                    sigma,
                    config.n_numbers,
                    config.n_sequences,
                    config.convergence_start,
                    config.seed,
                )
                rows.append(
                    {
                        "alpha": alpha,
                        "true_mu": mu,
                        "true_sigma": sigma,
                        "pred_mu": pred_mu,
                        "pred_sigma": pred_sigma,
                    }
                )
                if mu_max is not None and pred_mu >= mu_max:
                    break
        return pl.DataFrame(rows)

    def plot_steering_curve(
        df: pl.DataFrame,
        start_mu: float,
        layer: int,
        accuracy: float,
        title: str | None = None,
    ):
        """(pred_mu, pred_sigma) trajectory per sigma group; the alpha=0 point (the true,
        unsteered baseline) is marked with a black-outlined dot and an annotation."""
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        palette = [PIRATE_BLUE, PIRATE_ORANGE, PIRATE_GREEN, PIRATE_PURPLE]
        for sigma_val, color in zip(
            sorted(df["true_sigma"].unique().to_list()), palette
        ):
            sub = df.filter(pl.col("true_sigma") == sigma_val).sort("alpha")
            ax.plot(
                sub["pred_mu"],
                sub["pred_sigma"],
                "-o",
                color=color,
                markersize=4,
                linewidth=1.5,
                label=f"σ={sigma_val:g}",
            )
            start = sub.filter(pl.col("alpha") == 0.0)
            ax.plot(
                start["pred_mu"],
                start["pred_sigma"],
                "o",
                color=color,
                markersize=11,
                markeredgecolor="black",
                markeredgewidth=1.5,
                zorder=5,
            )

        start_row = (
            df.filter(pl.col("alpha") == 0.0)
            .sort("true_sigma")
            .row(0, named=True)
        )
        ax.annotate(
            f"start (α=0, μ={start_mu:g})",
            xy=(start_row["pred_mu"], start_row["pred_sigma"]),
            xytext=(15, -15),
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"),
        )
        ax.set_xlabel("Model's predicted μ after steering")
        ax.set_ylabel("Model's predicted σ after steering")
        ax.legend(title="Original σ", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        if title:
            ax.set_title(title)
        plt.close(fig)
        return fig

    def plot_alpha_mu_curve(
        df: pl.DataFrame, start_mu: float, layer: int, accuracy: float
    ):
        """pred_mu trajectory vs alpha per sigma group — isolates how much the steered
        mean moves with steering strength, independent of the (separately shown) sigma
        distortion."""
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        palette = [PIRATE_BLUE, PIRATE_ORANGE, PIRATE_GREEN, PIRATE_PURPLE]
        for sigma_val, color in zip(
            sorted(df["true_sigma"].unique().to_list()), palette
        ):
            sub = df.filter(pl.col("true_sigma") == sigma_val).sort("alpha")
            ax.plot(
                sub["alpha"],
                sub["pred_mu"],
                "-o",
                color=color,
                markersize=4,
                linewidth=1.5,
                label=f"σ={sigma_val:g}",
            )
        ax.axhline(start_mu, color="black", linestyle=":", linewidth=1.3)
        ax.set_xlabel("alpha (steering strength)")
        ax.set_ylabel("Model's predicted μ")
        ax.legend(title="Original σ", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        plt.close(fig)
        return fig

    return (
        collect_steering_curve,
        plot_alpha_mu_curve,
        plot_steering_curve,
        steered_moments,
        steering_direction,
    )


@app.cell(hide_code=True)
def steering_data(
    LAYER,
    activation_groups,
    collect_steering_curve,
    experiment_config,
    mu_linear_probe_weights,
    steer_max_alpha,
    steer_mu,
    steering_direction,
):
    mo.stop(
        not activation_groups,
        mo.callout(
            mo.md("No activations — run collection first."), kind="warn"
        ),
    )

    _layer = LAYER
    _mu_classes = sorted(experiment_config.mu_values)
    _mu_lo, _mu_hi = _mu_classes[0], _mu_classes[-1]
    _direction = steering_direction(
        mu_linear_probe_weights[_layer],
        _mu_lo,
        _mu_hi,
        activation_groups,
        _layer,
    )
    _alphas = np.linspace(0.0, steer_max_alpha, 21)
    _groups = [g for g in activation_groups if g[0] == steer_mu]

    steering_layer = _layer
    steering_df = collect_steering_curve(
        _layer, _direction, _alphas, _groups, experiment_config, mu_max=700.0
    )
    return steering_df, steering_layer


@app.cell(hide_code=True)
def probe_weights(activation_groups, experiment_config, train_probe):
    mo.stop(
        not activation_groups,
        mo.callout(mo.md("No activations — run collection first."), kind="warn"),
    )

    _resid_all = np.concatenate([r for r, _ in activation_groups.values()])
    _mu_all = np.concatenate([
        np.full(r.shape[0], mu)
        for (mu, _sigma), (r, _) in activation_groups.items()
    ])
    _n_layers = next(iter(activation_groups.values()))[0].shape[1]

    @mo.persistent_cache
    def _train_probe_layer(layer: int) -> tuple[torch.Tensor, float]:
        w, acc = train_probe(_resid_all[:, layer, :].astype(np.float64), _mu_all,
                             seed=experiment_config.seed)
        return torch.from_numpy(w), acc

    _trained = {
        layer: _train_probe_layer(layer)
        for layer in mo.status.progress_bar(range(_n_layers), title="Training probes")
    }
    probe_accuracy = {layer: acc for layer, (_w, acc) in _trained.items()}
    return (probe_accuracy,)


@app.cell(hide_code=True)
def sigma_probe_weights(activation_groups, experiment_config):
    mo.stop(
        not activation_groups,
        mo.callout(mo.md("No activations — run collection first."), kind="warn"),
    )

    _resid_all = np.concatenate([r for r, _ in activation_groups.values()])
    _sigma_all = np.concatenate([
        np.full(r.shape[0], sigma)
        for (_mu, sigma), (r, _) in activation_groups.items()
    ])
    _n_layers = next(iter(activation_groups.values()))[0].shape[1]

    @mo.persistent_cache
    def _train_sigma_probe_layer(layer: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        x = _resid_all[:, layer, :].astype(np.float64)
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, _sigma_all, test_size=0.2, random_state=experiment_config.seed, stratify=_sigma_all
        )
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(x_tr, y_tr)
        w = clf.coef_.astype(np.float32)
        b = clf.intercept_.astype(np.float32)
        if w.shape[0] == 1:
            w = np.concatenate([-w, w], axis=0)
            b = np.concatenate([-b, b], axis=0)
        return torch.from_numpy(w), torch.from_numpy(b), float(clf.score(x_te, y_te))

    _sigma_trained = {
        layer: _train_sigma_probe_layer(layer)
        for layer in mo.status.progress_bar(range(_n_layers), title="Training sigma probes")
    }
    sigma_probe_weights = {layer: w for layer, (w, _b, _acc) in _sigma_trained.items()}
    sigma_probe_intercepts = {layer: b for layer, (_w, b, _acc) in _sigma_trained.items()}
    sigma_probe_accuracy = {layer: acc for layer, (_w, _b, acc) in _sigma_trained.items()}
    return sigma_probe_intercepts, sigma_probe_weights


@app.cell
def steering_alpha_mu_plot(
    plot_alpha_mu_curve,
    probe_accuracy,
    steer_mu,
    steering_df,
    steering_layer,
):
    plot_alpha_mu_curve(steering_df, steer_mu, steering_layer, probe_accuracy[steering_layer])
    return


@app.cell(hide_code=True)
def steering_result_note():
    mo.md(r"""
    It works ! When we steer stronger (we translate the latent vector further), the mean outputed by the model increases. We're able to change its thoughts. Interestingly, the effect of the steering depends on $\sigma$, but we're still able to change $\mu$ as much as we want.

    Something we can't see from this experiment is *how uncertain the model is*. When we use $\sigma=200$, are the predictions of the model more spread-out than when $\sigma=20$ ?

    We can easily check, and it is indeed the case. The model learns to predict $\mu$, but also the correct $\sigma$. But now, look at what happens when we steer:
    """)
    return


@app.cell
def steering_sigma_plot(
    plot_steering_curve,
    probe_accuracy,
    steer_mu,
    steering_df,
    steering_layer,
):
    plot_steering_curve(steering_df, steer_mu, steering_layer, probe_accuracy[steering_layer])
    return


@app.cell(hide_code=True)
def steering_sigma_disruption_note():
    mo.md(r"""
    We wanted to brain-wash the model so that it thinks the numbers have mean $\mu_0$, but at the same time we completely disrupted $\sigma$. Now the model thinks the mean is higher than it really is, but also more spread-out.

    Remember the 2 truths about Manifold exploration:
    1. Paths are curvy
    2. Probes show you a tiny part of reality

    Basically, we tried to travel in a straight line, in a wavy ocean. It did not work.

    Below, I represented the 3d PCA of all the points. Remember that like probes, PCA shows only a tiny part of reality !
    """)
    return


@app.cell(hide_code=True)
def mu_linear_probe_weights(activation_groups, experiment_config):
    mo.stop(
        not activation_groups,
        mo.callout(mo.md("No activations — run collection first."), kind="warn"),
    )

    _resid_all = np.concatenate([r for r, _ in activation_groups.values()])
    _mu_all_lin = np.concatenate([
        np.full(r.shape[0], mu)
        for (mu, _sigma), (r, _) in activation_groups.items()
    ])
    _n_layers = next(iter(activation_groups.values()))[0].shape[1]

    @mo.persistent_cache
    def _train_mu_linear_probe_layer(layer: int) -> tuple[np.ndarray, float, float]:
        x = _resid_all[:, layer, :].astype(np.float64)
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, _mu_all_lin, test_size=0.2, random_state=experiment_config.seed
        )
        reg = LinearRegression().fit(x_tr, y_tr)
        return reg.coef_.astype(np.float64), float(reg.intercept_), float(reg.score(x_te, y_te))

    _mu_linear_trained = {
        layer: _train_mu_linear_probe_layer(layer)
        for layer in mo.status.progress_bar(range(_n_layers), title="Training linear mu probes")
    }
    mu_linear_probe_weights = {layer: w for layer, (w, _b, _r2) in _mu_linear_trained.items()}
    mu_linear_probe_intercepts = {layer: b for layer, (_w, b, _r2) in _mu_linear_trained.items()}
    mu_linear_probe_accuracy = {layer: r2 for layer, (_w, _b, r2) in _mu_linear_trained.items()}
    return (mu_linear_probe_weights,)


@app.cell(hide_code=True)
def belief_geometry_helpers():
    def ellipse_points(mean: np.ndarray, cov: np.ndarray, n_std: float = 1.5, resolution: int = 100):
        """2D confidence-ellipse boundary coordinates at n_std standard deviations."""
        eigvals, eigvecs = np.linalg.eigh(cov)
        radii = n_std * np.sqrt(np.clip(eigvals, 0.0, None))
        t = np.linspace(0, 2 * np.pi, resolution)
        circle = np.stack([np.cos(t), np.sin(t)], axis=-1)
        pts = (circle * radii) @ eigvecs.T + mean
        return pts[:, 0], pts[:, 1]


    def belief_geometry_points(
        groups: dict, layer: int, mu_coef: np.ndarray, mu_intercept: float,
        sigma_coef: np.ndarray, sigma_intercept: float,
    ) -> dict[tuple[float, float], np.ndarray]:
        """Project every sequence's activation at `layer` onto 2 axes: the mu-probe's
        linear-regression readout, and the sigma-probe's linear-regression readout."""
        points = {}
        for key, (resid, _softmax) in groups.items():
            acts = resid[:, layer, :].astype(np.float64)
            x = acts @ mu_coef + mu_intercept
            y = acts @ sigma_coef + sigma_intercept
            points[key] = np.stack([x, y], axis=1)
        return points


    def belief_geometry_arrows(
        groups: dict, layer: int, mu_coef: np.ndarray, mu_intercept: float,
        sigma_coef: np.ndarray, sigma_intercept: float, direction: np.ndarray, alpha: float,
    ) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
        """Shift each group's real pooled activation by a single small alpha*direction
        directly in latent space (no model re-run: the steering hook just adds `direction`
        uniformly), then read both the unshifted and shifted vector off through the same
        linear mu/sigma probes. Both readouts are now exactly affine in alpha by
        construction, since they're plain linear projections of the shifted activation."""
        arrows = {}
        for key, (resid, _softmax) in groups.items():
            base = resid[:, layer, :].astype(np.float64).mean(axis=0)
            shifted = base + alpha * direction

            def _readout(v):
                x = v @ mu_coef + mu_intercept
                y = v @ sigma_coef + sigma_intercept
                return np.array([x, y])

            arrows[key] = (_readout(base), _readout(shifted))
        return arrows


    def belief_geometry_sigma_pca_axes(
        groups: dict, layer: int, mu_coef: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Average the (mu, sigma) group centroids by sigma (pooling over mu, to denoise),
        project the mu-probe direction out of those sigma centroids, then take the top-2
        principal components of what's left: an unsupervised 2D basis for how sigma is
        actually represented, instead of a supervised sigma-probe + orthogonalized residual."""
        sigma_values = sorted({sigma for _mu, sigma in groups})
        sigma_centroids = np.stack([
            np.mean([
                r[:, layer, :].astype(np.float64).mean(axis=0)
                for (_mu, s), (r, _softmax) in groups.items() if s == sigma
            ], axis=0)
            for sigma in sigma_values
        ])
        center = sigma_centroids.mean(axis=0)

        unit_mu = mu_coef / np.linalg.norm(mu_coef)
        centered = sigma_centroids - center
        residual = centered - np.outer(centered @ unit_mu, unit_mu)

        _, _, vt = np.linalg.svd(residual, full_matrices=False)
        return center, vt[0], vt[1]


    def belief_geometry_centroids_3d(
        groups: dict, layer: int, mu_coef: np.ndarray, mu_intercept: float,
        center: np.ndarray, pc1: np.ndarray, pc2: np.ndarray,
    ) -> dict[tuple[float, float], np.ndarray]:
        """Each group's mean activation projected onto (mu-probe readout, sigma-PCA-1,
        sigma-PCA-2) — the 3D coordinate used for the constant-sigma splines."""
        points = {}
        for key, (resid, _softmax) in groups.items():
            mean_act = resid[:, layer, :].astype(np.float64).mean(axis=0)
            x = mean_act @ mu_coef + mu_intercept
            centered = mean_act - center
            y = centered @ pc1
            z = centered @ pc2
            points[key] = np.array([x, y, z])
        return points

    return


@app.cell(hide_code=True)
def belief_geometry_splines_3d(
    LAYER,
    PIRATE_BLUE,
    PIRATE_GREEN,
    PIRATE_ORANGE,
    PIRATE_PURPLE,
    activation_groups,
    sigma_probe_intercepts,
    sigma_probe_weights,
):
    mo.stop(
        not activation_groups,
        mo.callout(
            mo.md("No activations — run collection first."), kind="warn"
        ),
    )

    _layer = LAYER
    _sigma_values = sorted({sigma for _mu, sigma in activation_groups})
    _mu_values = sorted({mu for mu, _sigma in activation_groups})

    # plain, unsupervised PCA of the 16 (mu, sigma) centroids -- no probe direction,
    # no gradient: just the raw activation geometry.
    _raw_centroids = {
        key: resid[:, _layer, :].astype(np.float64).mean(axis=0)
        for key, (resid, _soft) in activation_groups.items()
    }
    _matrix = np.stack(list(_raw_centroids.values()))
    _pca_mean = _matrix.mean(axis=0)
    _, _, _vt = np.linalg.svd(_matrix - _pca_mean, full_matrices=False)
    _pc1, _pc2, _pc3 = _vt[0], _vt[1], _vt[2]

    def _project(v):
        c = v - _pca_mean
        return np.array([c @ _pc1, c @ _pc2, c @ _pc3])

    _centroids = {key: _project(v) for key, v in _raw_centroids.items()}

    # decision boundary between EVERY adjacent pair of sigma classes, read directly
    # off the trained softmax sigma-probe hyperplane at this layer (sklearn's
    # LogisticRegression.classes_ is always sorted ascending, matching _sigma_values).
    _boundary_curves = []
    for _i in range(len(_sigma_values) - 1):
        _s_from, _s_to = _sigma_values[_i], _sigma_values[_i + 1]
        _direction = (
            (
                sigma_probe_weights[_layer][_i + 1]
                - sigma_probe_weights[_layer][_i]
            )
            .double()
            .numpy()
        )
        _bias = float(
            sigma_probe_intercepts[_layer][_i + 1]
            - sigma_probe_intercepts[_layer][_i]
        )
        _curve = []
        for _mu in _mu_values:
            _c_from = _raw_centroids[(_mu, _s_from)]
            _c_to = _raw_centroids[(_mu, _s_to)]
            _delta = _c_to - _c_from
            _denom = _direction @ _delta
            # point along the centroid-to-centroid segment where the probe's logit
            # difference crosses zero — the actual softmax decision boundary, not a
            # heuristic lerp fraction
            _t = -(_direction @ _c_from + _bias) / _denom if _denom else 0.5
            _t = float(np.clip(_t, 0.0, 1.0))
            _curve.append(tuple(_project(_c_from + _t * _delta)))
        _boundary_curves.append(_curve)

    _xs = [p[0] for p in _centroids.values()] + [
        pt[0] for curve in _boundary_curves for pt in curve
    ]
    _ys = [p[1] for p in _centroids.values()] + [
        pt[1] for curve in _boundary_curves for pt in curve
    ]
    _zs = [p[2] for p in _centroids.values()] + [
        pt[2] for curve in _boundary_curves for pt in curve
    ]

    def _pad(lo, hi, frac=0.2):
        span = (hi - lo) or 1.0
        return lo - frac * span, hi + frac * span

    _x_lo, _x_hi = _pad(min(_xs), max(_xs))
    _y_lo, _y_hi = _pad(min(_ys), max(_ys))
    _z_lo, _z_hi = _pad(min(_zs), max(_zs))

    _SIGMA_COLORS = [PIRATE_BLUE, PIRATE_ORANGE, PIRATE_GREEN, PIRATE_PURPLE]
    _sigma_color = {s: c for s, c in zip(_sigma_values, _SIGMA_COLORS)}

    class _BeliefSplines3D(ManimWidget):
        def __init__(
            self,
            centroids,
            mu_values,
            sigma_values,
            sigma_color,
            boundary_curves,
            x_range,
            y_range,
            z_range,
            **kwargs,
        ):
            self._centroids = centroids
            self._mu_values = mu_values
            self._sigma_values = sigma_values
            self._sigma_color = sigma_color
            self._boundary_curves = boundary_curves
            self._x_range = x_range
            self._y_range = y_range
            self._z_range = z_range
            super().__init__(**kwargs)

        def construct(self):
            _axes = ThreeDAxes(
                x_range=[
                    self._x_range[0],
                    self._x_range[1],
                    (self._x_range[1] - self._x_range[0]) / 4,
                ],
                y_range=[
                    self._y_range[0],
                    self._y_range[1],
                    (self._y_range[1] - self._y_range[0]) / 4,
                ],
                z_range=[
                    self._z_range[0],
                    self._z_range[1],
                    (self._z_range[1] - self._z_range[0]) / 4,
                ],
                x_length=6,
                y_length=6,
                z_length=4,
                axis_config={"color": WHITE, "stroke_width": 1.5},
            )
            self.add(_axes)
            self.add(
                _axes.get_x_axis_label(Text("PC1", font_size=18, color=WHITE))
            )
            self.add(
                _axes.get_y_axis_label(Text("PC2", font_size=18, color=WHITE))
            )
            self.add(
                _axes.get_z_axis_label(Text("PC3", font_size=18, color=WHITE))
            )

            self.camera.background_color = "#1a1a2e"
            self.set_camera_orientation(
                phi=70 * DEGREES, theta=-60 * DEGREES, zoom=1.3
            )
            self.next_section("Sigma splines & boundaries")

            # sigma color legend, fixed on screen regardless of camera orientation
            _legend = VGroup()
            for sigma in self._sigma_values:
                color = self._sigma_color[sigma]
                _swatch = Dot(radius=0.08, color=color)
                _label = Text(f"σ={sigma:g}", font_size=20, color=WHITE)
                _label.next_to(_swatch, RIGHT, buff=0.15)
                _legend.add(VGroup(_swatch, _label))
            _legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            _legend.to_corner(UP + LEFT, buff=0.3)
            self.add_fixed_in_frame_mobjects(_legend)

            # draw the 4 sigma curves one at a time; the first one reveals each mu
            # point with a label, slowly -- the rest are unlabeled and quicker
            for i, sigma in enumerate(self._sigma_values):
                color = self._sigma_color[sigma]
                is_first = i == 0
                pts = []
                for mu in self._mu_values:
                    x, y, z = self._centroids[(mu, sigma)]
                    p = _axes.c2p(x, y, z)
                    pts.append(p)
                    dot = Dot3D(point=p, radius=0.06, color=color)
                    if is_first:
                        label = Text(f"μ={mu:g}", font_size=16, color=WHITE)
                        label.next_to(dot, UP + OUT, buff=0.15)
                        self.add_fixed_orientation_mobjects(label)
                        self.play(FadeIn(dot), Write(label), run_time=0.9)
                    else:
                        self.play(FadeIn(dot), run_time=0.25)
                spline = VMobject()
                spline.set_points_smoothly(pts)
                spline.set_color(color)
                spline.set_stroke(width=4)
                self.play(Create(spline), run_time=2.2 if is_first else 0.8)

            # once all 4 curves are in place, draw every sigma-adjacent decision
            # boundary in gray
            for curve in self._boundary_curves:
                _pts = [_axes.c2p(*pt) for pt in curve]
                _boundary = VMobject()
                _boundary.set_points_smoothly(_pts)
                _boundary.set_color(GRAY)
                _boundary.set_stroke(width=3, opacity=0.9)
                self.add(_boundary)
                self.play(Create(_boundary), run_time=1.8)

            # straight (chord) direction, for sigma=20: mu=350 -> mu=650
            _straight_start = _axes.c2p(*self._centroids[(350.0, 20.0)])
            _straight_end = _axes.c2p(*self._centroids[(650.0, 20.0)])
            _straight_line = DashedLine(
                _straight_start, _straight_end, color=WHITE, stroke_width=3
            )
            self.play(Create(_straight_line), run_time=1.5)

            _straight_mid = (_straight_start + _straight_end) / 2
            _moving_dot = Dot3D(
                point=_straight_start, radius=0.08, color=WHITE
            )
            self.add(_moving_dot)
            self.play(
                _moving_dot.animate.move_to(_straight_mid),
                run_time=2.5,
                rate_func=linear,
            )
            self.play(
                _moving_dot.animate.set_color(PIRATE_ORANGE), run_time=0.3
            )
            for _ in range(5):
                self.play(_moving_dot.animate.set_opacity(0.15), run_time=0.25)
                self.play(_moving_dot.animate.set_opacity(1.0), run_time=0.25)

    belief_splines_widget = _BeliefSplines3D(
        _centroids,
        _mu_values,
        _sigma_values,
        _sigma_color,
        _boundary_curves,
        (_x_lo, _x_hi),
        (_y_lo, _y_hi),
        (_z_lo, _z_hi),
        is_3d=True,
        canvas_width=900,
        autoplay=False,
        js="remote"
    )
    belief_splines_widget
    return


@app.cell(hide_code=True)
def _(jack):
    mo.md(rf"""
    {jack()}
    ///note
    We're taking a path that is *uncharted*. The probes have no idea what is there, because they were never trained with points from this region. The manifold is so curvy that there may be all other sigmas on that path.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Zoo of steering techniques
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    As we saw previously, steering in a straight line does not work well. Can we do better ?

    Ideally, we would like to steer while *following the curvature of the manifold*.

    There are 2 main ways to do it:
    - chosing the direction that goes from one centroid to the next (piecewise-linear probes)
    - fitting a spline through the centroids to have a more precise direction at each point. This is what is used in the paper
    """)
    return


@app.class_definition(hide_code=True)
class ManifoldWorldPiecewiseLinear(ManifoldWorld):
    def construct(self):
        self.setup_scene()
        self.section_piecewise_arrows()

    def section_piecewise_arrows(self):
        # 1. build the scene, then orient the camera
        self.section3_regions_instant()
        self.move_camera(theta=-80 * DEGREES, run_time=2.0, zoom=1)
        self.wait(0.3)

        # 2. hide -> title -> show back -> draw the piecewise arrows
        self.next_section(
            "Piecewise-linear probes: one direction per region"
        )
        title1 = Text("Piecewise linear probes", font_size=56, color=WHITE)
        self.add_fixed_in_frame_mobjects(title1)
        title1.set_opacity(0)
        self.play(
            title1.animate.set_opacity(1),
            FadeOut(self.axes),
            *[FadeOut(b) for b in self.patatoids],
            *[FadeOut(d) for grp in self.region_dot_groups for d in grp],
            run_time=1.5,
        )
        self.wait(0.8)
        self.play(
            FadeOut(title1),
            FadeIn(self.axes),
            *[FadeIn(b) for b in self.patatoids],
            *[FadeIn(d) for grp in self.region_dot_groups for d in grp],
            run_time=1.2,
        )

        # local direction for each region: centroid of its blob
        self._region_centers = [
            self.axes.c2p(*perimeter.mean(axis=0))
            for perimeter in self.region_raw_perimeter
        ]

        self.local_arrows = []
        for i in range(self.n_regions - 1):
            start, end = (
                self._region_centers[i],
                self._region_centers[i + 1],
            )
            _direction = end - start
            _axis = _direction / np.linalg.norm(_direction)
            _stub_end = start + 0.05 * _direction

            stub = Arrow(start=start, end=_stub_end, color=WHITE, buff=0)
            full = Arrow(start=start, end=end, color=WHITE, buff=0)
            stub.rotate(90 * DEGREES, axis=_axis)
            full.rotate(90 * DEGREES, axis=_axis)
            self.add(stub)
            self.play(Transform(stub, full), run_time=1.3)
            self.wait(0.25)
            self.local_arrows.append(stub)

        self.wait(0.6)

        # 3. hide (destroying the arrows) -> title -> show back -> draw the spline
        self.next_section("A smooth spline probe: one tangent direction")
        title2 = Text("A smooth spline probe", font_size=56, color=WHITE)
        self.add_fixed_in_frame_mobjects(title2)
        title2.set_opacity(0)
        self.play(
            title2.animate.set_opacity(1),
            FadeOut(self.axes),
            *[FadeOut(b) for b in self.patatoids],
            *[FadeOut(d) for grp in self.region_dot_groups for d in grp],
            *[FadeOut(a) for a in self.local_arrows],
            run_time=1.5,
        )
        self.wait(0.8)
        self.play(
            FadeOut(title2),
            FadeIn(self.axes),
            *[FadeIn(b) for b in self.patatoids],
            *[FadeIn(d) for grp in self.region_dot_groups for d in grp],
            run_time=1.2,
        )

        # smooth spline through the centroid of the 3 colored points in each region
        _spline_pts = [
            np.mean(self.inner_pts_by_region[r], axis=0)
            for r in range(self.n_regions)
        ]
        self.probe_spline = VMobject(color=GRAY, stroke_width=5)
        self.probe_spline.set_points_smoothly(_spline_pts)
        self.play(Create(self.probe_spline), run_time=2.0)
        self.wait(0.5)

        # point between blob 2 and blob 3 (region index 1 and 2), tangent to the spline there
        _alpha = 0.5
        _d = 1e-3
        _p0 = self.probe_spline.point_from_proportion(_alpha - _d)
        _p1 = self.probe_spline.point_from_proportion(_alpha + _d)
        _point = self.probe_spline.point_from_proportion(_alpha)
        _direction = _p1 - _p0
        _axis = _direction / np.linalg.norm(_direction)

        _length = 1.2
        _end = _point + _length * _axis
        _stub_end = _point + 0.05 * (_end - _point)

        self.tangent_dot = Dot3D(_point, radius=0.1, color=WHITE)
        stub = Arrow(start=_point, end=_stub_end, color=WHITE, buff=0)
        full = Arrow(start=_point, end=_end, color=WHITE, buff=0)
        stub.rotate(90 * DEGREES, axis=_axis)
        full.rotate(90 * DEGREES, axis=_axis)

        self.play(FadeIn(self.tangent_dot), run_time=0.4)
        self.add(stub)
        self.play(Transform(stub, full), run_time=1.3)
        self.tangent_arrow = stub
        self.wait(0.6)


@app.cell(hide_code=True)
def manifold_world_piecewise_display():
    ManifoldWorldPiecewiseLinear(is_3d=True, autoplay=False, canvas_width=900, js="remote")
    return


@app.class_definition(hide_code=True)
class ManifoldWorldMultiCategory(ManifoldWorld):
    def construct(self):
        self.setup_scene()
        self.section_multi_category_probes()

    def section_multi_category_probes(self):
        self.next_section(
            "Multi-category probes: one direction per blob from the origin"
        )

        # build the scene, then move the camera straight to the final viewing spot
        self.section3_regions_instant()
        self.move_camera(theta=-80 * DEGREES, zoom=0.85, run_time=2.0)
        self.wait(0.3)

        # the origin point, with a fixed-orientation label
        self.origin_point = self.axes.c2p(0, 0, -4)
        self.origin_dot = Dot3D(
            self.origin_point, radius=0.14, color=WHITE
        )
        self.origin_label = Text(
            "origin", font_size=24, color=WHITE
        ).next_to(self.origin_point, DOWN, buff=0.2)
        self.add_fixed_orientation_mobjects(self.origin_label)
        self.play(
            FadeIn(self.origin_dot), Write(self.origin_label), run_time=1.0
        )
        self.wait(0.4)

        # local direction for each region: centroid of its blob
        self._region_centers = [
            self.axes.c2p(*perimeter.mean(axis=0))
            for perimeter in self.region_raw_perimeter
        ]

        # one arrow from the origin to each blob's center, with its n= label
        self.origin_arrows = []
        self.region_labels = []
        for n, blob, center in zip(
            self.n_values, self.patatoids, self._region_centers
        ):
            _direction = center - self.origin_point
            _axis = _direction / np.linalg.norm(_direction)
            _stub_end = self.origin_point + 0.05 * _direction

            stub = Arrow(
                start=self.origin_point, end=_stub_end, color=WHITE, buff=0
            )
            full = Arrow(
                start=self.origin_point, end=center, color=WHITE, buff=0
            )
            stub.rotate(90 * DEGREES, axis=_axis)
            full.rotate(90 * DEGREES, axis=_axis)

            lbl = Text(f"n={n}", font_size=20, color=WHITE).move_to(
                blob.get_top() + UP * 0.3
            )
            self.add_fixed_orientation_mobjects(lbl)

            self.add(stub)
            self.play(Transform(stub, full), Write(lbl), run_time=1.3)
            self.wait(0.25)
            self.origin_arrows.append(stub)
            self.region_labels.append(lbl)

        self.wait(0.6)


@app.cell(hide_code=True)
def multi_category_aside():
    mo.md(rf"""
    ///warning
    Quick aside: I lied a little bit about the way we construct probes.

    It was not strictly necessary in order to understand representation and steering, but it's an important detail if you want to dig deeper later.

    We construct a set of probes, one for each $\mu$. The role of each probe is to detect it's assigned mu, and to react to nothing else.

    {mo.as_html(ManifoldWorldMultiCategory(is_3d=True, autoplay=False, canvas_width=900, js="remote"))}

    It's essentially a training trick. You train the probes $W_{{\mu}}$  such that $Softmax(W_{{\mu}} x_0)$ gives you almost one for the $\mu$ associated with $x_0$ and 0 for the other.

    It is not very different from computing the centroids of each blob and working on them for steering. One reason why multi-category probes seeems to work better is that in the activation space, *the norm of the vector is often useless*. We only care about which *direction* the blobs are in.

    The paper is not 100% clear on this subject. Quoting: 

    > In practice, we find one-vs-rest probes [the straight direction] to be less efficient to separate a specific class from all others. Hence we rely on the multiclass probe in what follows.

    and 

    > The steering directions are calculated from differences of point-probes [i.e, straight direction]. 

    That means multi-class probes were used for preliminary analysis, supposedly because it worked better to separate classes, but then don't use them for steering. It can be true, but it would be worth explaining the difference between the 2 geometrically.
    ///
    """)
    return


@app.cell(hide_code=True)
def steering_variant_helpers(
    ExperimentConfig,
    PIRATE_BLUE,
    kernel_interp,
    linear_interp,
    pooled_mean,
    steered_moments,
):
    def shade_color(color_hex: str, factor: float) -> str:
        """Blend a hex color toward white; factor=1.0 keeps it as-is, smaller
        factors give progressively lighter shades."""
        r, g, b = (
            int(color_hex[1:3], 16),
            int(color_hex[3:5], 16),
            int(color_hex[5:7], 16),
        )
        r = int(r + (255 - r) * (1 - factor))
        g = int(g + (255 - g) * (1 - factor))
        b = int(b + (255 - b) * (1 - factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    def mean_sigma_distance(df: pl.DataFrame) -> float:
        """Mean |pred_sigma - true_sigma| across every (true_sigma, alpha) point
        actually swept for this method — a single-number summary of how much a
        steering scheme disrupts sigma, averaged over all sigma groups."""
        return float((df["pred_sigma"] - df["true_sigma"]).abs().mean())

    def plot_sigma_distance_bar(
        dfs: dict[str, pl.DataFrame],
        title: str = "Sigma disruption by steering scheme",
    ):
        labels = list(dfs.keys())
        distances = [mean_sigma_distance(df) for df in dfs.values()]
        shades = [1.0, 0.8, 0.55, 0.3]
        colors = [shade_color(PIRATE_BLUE, s) for s in shades[: len(labels)]]

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(
            labels, distances, color=colors, edgecolor="black", linewidth=1.2
        )
        for i, d in enumerate(distances):
            ax.text(i, d, f"{d:.1f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Mean |predicted σ − true σ|")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        plt.close(fig)
        return fig

    def truncate_at_target_mu(
        df: pl.DataFrame, target_mu: float
    ) -> pl.DataFrame:
        """Keep only each true_sigma trajectory's initial portion, up to and including
        the first alpha at which pred_mu reaches target_mu (curves that never reach it
        keep everything) — an effect-matched comparison (same achieved mu) instead of
        an alpha-matched one, since alpha-matching lets an under-steering method look
        artificially clean simply by not moving mu much."""
        keep = []
        for sigma_val in df["true_sigma"].unique().to_list():
            sub = df.filter(pl.col("true_sigma") == sigma_val).sort("alpha")
            reached = sub.filter(pl.col("pred_mu") >= target_mu)
            if reached.height > 0:
                cutoff_alpha = reached["alpha"].min()
                keep.append(sub.filter(pl.col("alpha") <= cutoff_alpha))
            else:
                keep.append(sub)
        return pl.concat(keep)

    def pooled_centroids(
        mu_classes: Sequence[float], groups: dict, layer: int
    ) -> np.ndarray:
        """Stacked pooled-mean centroid per mu class, at one layer — the raw material
        for both the piecewise-linear and spline steering directions below."""
        return np.stack([pooled_mean(groups, mu, layer) for mu in mu_classes])

    def _piecewise_path_position(
        mu_classes: Sequence[float], centroids: np.ndarray, query_mu: float
    ) -> np.ndarray:
        """Position on the piecewise-linear centroid path at query_mu (extrapolates
        past the ends along the nearest segment's slope, like a chord continuing straight)."""
        classes = np.asarray(mu_classes)
        i = int(
            np.clip(
                np.searchsorted(classes, query_mu) - 1, 0, len(classes) - 2
            )
        )
        frac = (query_mu - classes[i]) / (classes[i + 1] - classes[i])
        return linear_interp(centroids[i], centroids[i + 1], frac)

    def piecewise_direction_at(
        mu_classes: Sequence[float],
        centroids: np.ndarray,
        base_mu: float,
        query_mu: float,
    ) -> np.ndarray:
        """Displacement from the piecewise-linear path's position at base_mu to its
        position at query_mu — EXPERIMENT.md's `curve` scheme (walk the manifold, not
        a straight chord), but along straight segments between known mu classes
        instead of a smooth fit. This is an absolute displacement to inject, not a
        per-unit rate: steered_moments multiplies it by alpha=1.0 unchanged."""
        return _piecewise_path_position(
            mu_classes, centroids, query_mu
        ) - _piecewise_path_position(mu_classes, centroids, base_mu)

    def spline_direction_at(
        mu_classes: Sequence[float],
        centroids: np.ndarray,
        base_mu: float,
        query_mu: float,
        bandwidth: float,
    ) -> np.ndarray:
        """Displacement from the kernel-smoothed centroid curve's position at base_mu
        to its position at query_mu — EXPERIMENT.md's `curve` scheme with a smooth fit
        through every mu class instead of straight segments. Also an absolute
        displacement, matching piecewise_direction_at's contract."""
        classes = np.asarray(mu_classes)
        p_base = kernel_interp(classes, centroids, base_mu, bandwidth)
        p_query = kernel_interp(classes, centroids, query_mu, bandwidth)
        return p_query - p_base

    def chord_direction_at(
        mu_classes: Sequence[float],
        centroids: np.ndarray,
        base_mu: float,
        query_mu: float,
    ) -> np.ndarray:
        """Displacement along the raw straight-line centroid chord (c_lo -> c_hi
        slope applied uniformly for the whole range, ignoring every intermediate
        centroid) — EXPERIMENT.md's `chord` scheme: off-manifold, no probe involved."""
        c_lo, c_hi = centroids[0], centroids[-1]
        mu_lo, mu_hi = mu_classes[0], mu_classes[-1]
        rate = (c_hi - c_lo) / (mu_hi - mu_lo)
        return rate * (query_mu - base_mu)

    def chord_direction_at(
        mu_classes: Sequence[float],
        centroids: np.ndarray,
        base_mu: float,
        query_mu: float,
    ) -> np.ndarray:
        """Displacement along the raw straight-line centroid chord (c_lo -> c_hi
        slope applied uniformly for the whole range, ignoring every intermediate
        centroid) — EXPERIMENT.md's `chord` scheme: off-manifold, no probe involved."""
        c_lo, c_hi = centroids[0], centroids[-1]
        mu_lo, mu_hi = mu_classes[0], mu_classes[-1]
        rate = (c_hi - c_lo) / (mu_hi - mu_lo)
        return rate * (query_mu - base_mu)

    def collect_variable_direction_steering_curve(
        layer: int,
        direction_fn,
        alphas: Sequence[float],
        groups: Sequence[tuple[float, float]],
        config: ExperimentConfig,
        mu_max: float | None = None,
    ) -> pl.DataFrame:
        """Like collect_steering_curve, but the injected vector is recomputed at every
        alpha via direction_fn(mu + alpha) — an absolute displacement from the baseline
        mu to the target mu along the piecewise/spline centroid path — instead of
        scaling one fixed global direction by alpha. If mu_max is set, stops sweeping a
        group as soon as its measured pred_mu reaches mu_max."""
        rows = []
        for mu, sigma in groups:
            for alpha in mo.status.progress_bar(
                alphas, title=f"Steering @ layer {layer}"
            ):
                pred_mu, pred_sigma = steered_moments(
                    layer,
                    tuple(direction_fn(mu + alpha).tolist()),
                    1.0,
                    mu,
                    sigma,
                    config.n_numbers,
                    config.n_sequences,
                    config.convergence_start,
                    config.seed,
                )
                rows.append(
                    {
                        "alpha": alpha,
                        "true_mu": mu,
                        "true_sigma": sigma,
                        "pred_mu": pred_mu,
                        "pred_sigma": pred_sigma,
                    }
                )
                if mu_max is not None and pred_mu >= mu_max:
                    break
        return pl.DataFrame(rows)

    return (
        chord_direction_at,
        collect_variable_direction_steering_curve,
        piecewise_direction_at,
        pooled_centroids,
        shade_color,
        spline_direction_at,
        truncate_at_target_mu,
    )


@app.cell(hide_code=True)
def steering_variant_data(
    activation_groups,
    chord_direction_at,
    collect_variable_direction_steering_curve,
    experiment_config,
    piecewise_direction_at,
    pooled_centroids,
    spline_direction_at,
    steer_max_alpha,
    steer_mu,
    steering_layer,
):
    _mu_classes = sorted(experiment_config.mu_values)
    _centroids = pooled_centroids(
        _mu_classes, activation_groups, steering_layer
    )
    _bandwidth = float(np.mean(np.diff(_mu_classes)))
    _alphas = np.linspace(0.0, steer_max_alpha, 21)
    _groups = [g for g in activation_groups if g[0] == steer_mu]

    chord_steering_df = collect_variable_direction_steering_curve(
        steering_layer,
        lambda m: chord_direction_at(_mu_classes, _centroids, steer_mu, m),
        _alphas,
        _groups,
        experiment_config,
        mu_max=700.0,
    )
    piecewise_steering_df = collect_variable_direction_steering_curve(
        steering_layer,
        lambda m: piecewise_direction_at(_mu_classes, _centroids, steer_mu, m),
        _alphas,
        _groups,
        experiment_config,
        mu_max=700.0,
    )
    spline_steering_df = collect_variable_direction_steering_curve(
        steering_layer,
        lambda m: spline_direction_at(
            _mu_classes, _centroids, steer_mu, m, _bandwidth
        ),
        _alphas,
        _groups,
        experiment_config,
        mu_max=700.0,
    )
    return chord_steering_df, piecewise_steering_df, spline_steering_df


@app.cell
def steering_variant_sigma_selector(experiment_config):
    steering_variant_sigma = mo.ui.dropdown(
        [str(s) for s in sorted(experiment_config.sigma_values)],
        value="50.0",
        label="σ",
    )
    steering_variant_sigma
    return (steering_variant_sigma,)


@app.cell(hide_code=True)
def steering_variant_plot(
    PIRATE_BLUE,
    PIRATE_GREEN,
    PIRATE_ORANGE,
    PIRATE_PURPLE,
    chord_steering_df,
    piecewise_steering_df,
    shade_color,
    spline_steering_df,
    steer_mu,
    steering_df,
    steering_variant_sigma,
    truncate_at_target_mu,
):
    def _plot_steering_variants(
        dfs: dict[str, pl.DataFrame], sigma: float, start_mu: float
    ):
        """(pred_mu, pred_sigma) trajectory for one sigma group, one curve per steering
        method — same base color as that sigma group gets in plot_steering_curve,
        distinguished here by shade + line dash pattern + marker shape instead of by color."""
        palette = [PIRATE_BLUE, PIRATE_ORANGE, PIRATE_GREEN, PIRATE_PURPLE]
        any_df = next(iter(dfs.values()))
        sigma_order = sorted(any_df["true_sigma"].unique().to_list())
        base_color = palette[sigma_order.index(sigma) % len(palette)]

        shades = [1.0, 0.8, 0.55, 0.3]
        linestyles = ["-", "--", ":", "-."]
        markers = ["o", "s", "^", "X"]

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for (label, df), shade, linestyle, marker in zip(
            dfs.items(), shades, linestyles, markers
        ):
            color = shade_color(base_color, shade)
            sub = df.filter(pl.col("true_sigma") == sigma).sort("alpha")
            ax.plot(
                sub["pred_mu"],
                sub["pred_sigma"],
                linestyle=linestyle,
                marker=marker,
                color=color,
                markersize=6,
                linewidth=1.8,
                label=label,
            )
            start = sub.filter(pl.col("alpha") == 0.0)
            ax.plot(
                start["pred_mu"],
                start["pred_sigma"],
                marker=marker,
                color=color,
                markersize=13,
                markeredgecolor="black",
                markeredgewidth=1.5,
                zorder=5,
            )
        ax.set_xlabel("Model's predicted μ after steering")
        ax.set_ylabel("Model's predicted σ after steering")
        ax.legend(title=f"Steering method (σ={sigma:g})", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        plt.close(fig)
        return fig

    steering_variant_target_mu = 450.0

    _plot_steering_variants(
        {
            "Global": truncate_at_target_mu(
                steering_df, steering_variant_target_mu
            ),
            "Straight": truncate_at_target_mu(
                chord_steering_df, steering_variant_target_mu
            ),
            "Piecewise-linear": truncate_at_target_mu(
                piecewise_steering_df, steering_variant_target_mu
            ),
            "Splines": truncate_at_target_mu(
                spline_steering_df, steering_variant_target_mu
            ),
        },
        sigma=float(steering_variant_sigma.value),
        start_mu=steer_mu,
    )
    return


@app.cell(hide_code=True)
def steering_variant_sigma_distance_bounded(
    PIRATE_BLUE,
    chord_steering_df,
    piecewise_steering_df,
    shade_color,
    spline_steering_df,
    steering_df,
):
    def _max_reach_and_error(df: pl.DataFrame):
        """For each true_sigma group: the maximum pred_mu actually reached anywhere
        in the sweep (robust to non-monotonic curves that saturate or collapse),
        paired with the WORST sigma error |pred_sigma - true_sigma| seen anywhere
        in the sweep -- not necessarily at the same alpha as the max-mu point."""
        rows = []
        for sigma_val in sorted(df["true_sigma"].unique().to_list()):
            sub = df.filter(pl.col("true_sigma") == sigma_val)
            max_mu = sub["pred_mu"].max()
            max_error = (sub["pred_sigma"] - sigma_val).abs().max()
            rows.append(
                {
                    "true_sigma": sigma_val,
                    "max_mu": max_mu,
                    "error": max_error,
                }
            )
        return rows

    def _plot_reach_error_scatter(dfs: dict[str, pl.DataFrame]):
        shades = [1.0, 0.75, 0.5, 0.3]
        markers = ["o", "s", "^", "X"]

        fig, ax = plt.subplots(figsize=(6.5, 5))
        for (label, df), shade, marker in zip(dfs.items(), shades, markers):
            points = _max_reach_and_error(df)
            mean_mu = sum(p["max_mu"] for p in points) / len(points)
            mean_error = sum(p["error"] for p in points) / len(points)
            color = shade_color(PIRATE_BLUE, shade)
            ax.scatter(
                mean_mu,
                mean_error,
                marker=marker,
                color=color,
                s=140,
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )
            ax.annotate(
                label,
                (mean_mu, mean_error),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
            )

        ax.set_xlabel("max μ reached")
        ax.set_ylabel("max σ error")
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        plt.close(fig)
        return fig

    _plot_reach_error_scatter(
        {
            "Global": steering_df,
            "Straight": chord_steering_df,
            "Piecewise-linear": piecewise_steering_df,
            "Splines": spline_steering_df,
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Extensions

    Now you know the state of the art pirate knowledge about steering in uncertainty.

    But we could go deeper, a lot deeper.

    ## Steering using the gradient

    First, the best steering method we found was still disrupting $\sigma$ a lot. Since we have a white-box access to the model, we can do far better: we can compute the gradient w.r.t the sigma value the model will predict, and  **steer orthogonaly from the gradient**.

    So we take the previous 'straight' steering direction, but at each steering step, we project it so that it's orthogonal to the gradient at that point.
    """)
    return


@app.cell(hide_code=True)
def gradient_illustration(PIRATE_BLUE, PIRATE_ORANGE):
    def _gradient_illustration():
        """Toy illustration (fake data, not real model output): the sigma=20 curve is
        a level curve of sigma, so by multivariable calculus grad(sigma) is ALWAYS
        orthogonal to it -- that's the 'toward sigma' arrow. The second arrow is
        tangent to the curve: the direction that (to first order) does NOT change
        sigma, i.e. the clean direction we want to steer along."""
        x = np.linspace(-6, 6, 200)
        y20 = 0.05 * x**2 + 3.0 + 0.15 * x
        y50 = 0.05 * x**2

        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot(x, y20, color=PIRATE_BLUE, linewidth=2.5, label="σ=20")
        ax.plot(x, y50, color=PIRATE_ORANGE, linewidth=2.5, label="σ=50")

        x0 = 3.0
        y0 = 0.05 * x0**2 + 3.0 + 0.15 * x0
        ax.scatter([x0], [y0], color="black", s=90, zorder=5)

        _mu_x = -3.0
        _mu_y = 0.05 * _mu_x**2 + 3.0 + 0.15 * _mu_x
        ax.scatter([_mu_x], [_mu_y], color="black", marker="x", s=90, zorder=5)
        ax.text(
            _mu_x - 0.3,
            _mu_y + 0.6,
            "μ=350",
            color="black",
            fontsize=10,
            ha="right",
        )

        _slope = 0.1 * x0 + 0.15  # d/dx of 0.05*x^2 + 3 + 0.15*x
        _tangent = np.array([1.0, _slope])
        _tangent /= np.linalg.norm(_tangent)
        _normal = np.array([-_tangent[1], _tangent[0]])
        if _normal[1] < 0:
            _normal = -_normal

        # grad(sigma): orthogonal to the level curve, by multivariable calculus
        ax.annotate(
            "",
            xy=(x0 + _normal[0] * 2.2, y0 + _normal[1] * 2.2),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="red", linewidth=2.5),
        )
        ax.text(
            x0 - 0.3,
            y0 + 1.4,
            "∇σ",
            color="red",
            fontsize=9,
        )

        # tangent to the curve: preserves sigma to first order
        ax.annotate(
            "",
            xy=(x0 + _tangent[0] * 2.2, y0 + _tangent[1] * 2.2),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="black", linewidth=2.5),
        )
        ax.text(
            x0 - 1.8,
            y0 - 1.3,
            "tangent to the curve\n(preserves σ)",
            color="black",
            fontsize=9,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, loc="upper left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
        plt.close(fig)
        return fig

    _gradient_illustration()
    return


@app.cell(hide_code=True)
def gradient_steering_helpers(
    DEVICE,
    SUPPORT,
    bundle,
    group_prompts,
    tokenize,
):
    def exact_grad_at_last_pos(bundle, layer, ids, last_pos, base_vector=None):
        """Exact gradient of sigma_hat at the last comma position, with base_vector
        (if any) injected ONLY at that position before computing the gradient --
        real backprop through the remaining decoder blocks + final norm + unembedding.
        Also returns the real (pred_mu, pred_sigma) read off at this same point, since
        they fall out of the same forward pass for free."""
        module = (
            bundle.model.model.embed_tokens
            if layer == 0
            else bundle.model.model.layers[layer - 1]
        )
        vec = (
            None
            if base_vector is None
            else torch.as_tensor(
                base_vector, dtype=torch.float16, device=DEVICE
            )
        )
        captured = {}

        def _hook(_module, _inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs.clone()
            if vec is not None:
                hs[:, last_pos, :] = hs[:, last_pos, :] + vec
            hs.requires_grad_(True)
            captured["resid"] = hs
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs

        handle = module.register_forward_hook(_hook)
        try:
            out = bundle.model(input_ids=ids)
        finally:
            handle.remove()

        logits = out.logits[0, last_pos].float()
        probs = torch.softmax(logits, dim=-1)
        int_probs = probs.index_select(-1, bundle.int_token_ids)
        int_probs = int_probs / int_probs.sum()
        values = torch.arange(SUPPORT, device=DEVICE, dtype=torch.float32)
        pred_mu = (values * int_probs).sum()
        pred_var = (int_probs * (values - pred_mu) ** 2).sum()
        sigma_hat = torch.sqrt(pred_var)

        resid = captured["resid"]
        (grad,) = torch.autograd.grad(sigma_hat, resid)
        return (
            grad[0, last_pos].detach().cpu().float(),
            float(pred_mu.item()),
            float(sigma_hat.item()),
        )

    def gradient_discovery_trajectory(
        layer,
        mu,
        sigma,
        rate,
        n_numbers,
        seed,
        seg_alpha=12.5,
        target_dmu=50.0,
        max_segments=80,
    ):
        """Segmented exact-gradient-orthogonalized discovery walk, last-position-only:
        re-anchors to the real model (a fresh forward+backward with the accumulated
        vector already injected) at every step. Unlike a fixed total_alpha, the walk
        keeps extending (fixed-size segments) until pred_mu has actually moved by
        target_dmu -- some (mu, sigma) starting points are much harder to steer than
        others (a real model property, not a construction artifact: e.g. tight, low-
        sigma distributions resist steering far more than diffuse, high-sigma ones),
        so a shared total_alpha silently under-steers the resistant groups. Returns
        the walk's own measured (alpha, pred_mu, pred_sigma) at each step -- a single
        real sweep, not a separate re-verification. max_segments is a safety cap."""
        prompts = group_prompts(mu, sigma, n_numbers, 1, seed)
        ids = tokenize(bundle, prompts).to(DEVICE)
        last_pos = (ids[0] == bundle.comma_token_id).nonzero(as_tuple=True)[0][
            -1
        ]

        rate_t = torch.tensor(rate, dtype=torch.float32)
        accumulated = np.zeros_like(rate)
        rows = []
        positions = []
        base_mu = None
        i = 0
        for i in range(max_segments):
            grad, pred_mu, pred_sigma = exact_grad_at_last_pos(
                bundle, layer, ids, last_pos, accumulated if i > 0 else None
            )
            if base_mu is None:
                base_mu = pred_mu
            rows.append(
                {
                    "alpha": i * seg_alpha,
                    "pred_mu": pred_mu,
                    "pred_sigma": pred_sigma,
                }
            )
            positions.append(accumulated.copy())
            if pred_mu - base_mu >= target_dmu:
                break
            denom = float(grad @ grad) + 1e-8
            d_orth = rate_t - (float(rate_t @ grad) / denom) * grad
            accumulated = accumulated + (d_orth * seg_alpha).numpy()
        else:
            i += 1  # ran out of segments without reaching target_dmu

        # one more real measurement at the final accumulated point
        _, final_mu, final_sigma = exact_grad_at_last_pos(
            bundle, layer, ids, last_pos, accumulated
        )
        rows.append(
            {
                "alpha": (i + 1) * seg_alpha,
                "pred_mu": final_mu,
                "pred_sigma": final_sigma,
            }
        )
        positions.append(accumulated.copy())
        return rows, positions

    return (gradient_discovery_trajectory,)


@app.cell(hide_code=True)
def gradient_steering_data(
    activation_groups,
    experiment_config,
    gradient_discovery_trajectory,
    pooled_centroids,
    steer_mu,
    steering_layer,
):
    _mu_classes = sorted(experiment_config.mu_values)
    _centroids = pooled_centroids(
        _mu_classes, activation_groups, steering_layer
    )
    _rate = (_centroids[-1] - _centroids[0]) / (
        _mu_classes[-1] - _mu_classes[0]
    )
    _sigma_values = sorted(experiment_config.sigma_values)

    _rows = []
    gradient_trajectory_positions = {}
    for _sigma in mo.status.progress_bar(
        _sigma_values, title="Gradient discovery sweeps"
    ):
        _sigma_rows, _sigma_positions = gradient_discovery_trajectory(
            steering_layer,
            steer_mu,
            _sigma,
            _rate,
            experiment_config.n_numbers,
            experiment_config.seed,
        )
        for _row in _sigma_rows:
            _rows.append({"true_mu": steer_mu, "true_sigma": _sigma, **_row})
        gradient_trajectory_positions[_sigma] = _sigma_positions

    gradient_steering_df = pl.DataFrame(_rows)
    return gradient_steering_df, gradient_trajectory_positions


@app.cell(hide_code=True)
def gradient_steering_plot(
    gradient_steering_df,
    plot_steering_curve,
    probe_accuracy,
    steer_mu,
    steering_layer,
):
    plot_steering_curve(
        gradient_steering_df,
        steer_mu,
        steering_layer,
        probe_accuracy[steering_layer],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's see how curvy the trajectory was (respecting curvature was the entire point after all)
    """)
    return


@app.cell(hide_code=True)
def gradient_trajectory_pca_plot(PIRATE_GREEN, gradient_trajectory_positions):
    def _plot_trajectory_pca(positions):
        P = np.stack(positions)
        mean = P.mean(axis=0, keepdims=True)
        centered = P - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T

        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            "-",
            color=PIRATE_GREEN,
            linewidth=1.2,
            alpha=0.6,
        )
        ax.scatter(
            coords[:, 0], coords[:, 1], color=PIRATE_GREEN, s=25, zorder=3
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.spines[["top", "right"]].set_visible(False)
        plt.close(fig)
        return fig

    _plot_trajectory_pca(gradient_trajectory_positions[20.0])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **It's almost Flat !**

    ///note
    It's a quite surprising result. After all this adventure, the steering technique that works best give an almost flat trajectory.

    Here is the interpretation: the manifold space is huge. There are hundreds of different directions you can use to change $\mu$, not just one path. Among all these paths, there is one that is straight AND does not change $\sigma$. But the hard part is finding it.

    Also, note that the path depends on your starting point. there is no universal way to move on the manifold to preserve sigma.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Geographical belief
    """)
    return


@app.cell(hide_code=True)
def _(jack):
    mo.md(rf"""
    {jack()}

    Hopefuly, bold explorer, now you understand how LLMs represent uncertainty.

    But that's not the end of the journey, I have something left to show you ...

    The $(\mu, \sigma)$ experiment was what I would call a *toy model*: the smallest experiment that shows the phenomenon. But uncertainty and belief shows up everywhere. I will not stop at toy models: That's not the kind of thing that **Jack Sparse'Row** does !

    In the next experiment, I took 90 cities across the earth, with their position on the globe. I also collected some monuments for each of these cities.

    Then, I pick a city (let's say *London*). I construct one of 2 different prompts: 

    **Coordinate-based prompt**

    I take a point near London (randomly in a radius of 500km), translate it to Latitude/Longitude, and create the prompt:

    > "Lat 50.6508, Long -0.7508 is near the city"

    **Monument-based prompt**

    I pick one random monument in the city and craft: 

    > "Eltham Palace is a famous landmark near the city"
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Then, I forward the prompt to the LLM, and I extract the activation vector from the last layer.

    I train a **multi-category probe** (90 categories) to predict which city I created the prompt from.

    Once the probe is trained, I take a new monument the probe has never seen, and look at the probability distribution over all the cities. From this probability distribution, I can compute:
    - the mean position on the sphere
    - the uncertainty about this position (spread)

    And I can even do this at different scales by varying the temperature of the softmax.

    The intuition is that if the model and the probe are well calibrated, and if you ask where the [Monaco](https://en.wikipedia.org/wiki/Monaco) is, you will get around 0.5 for "Paris" and 0.5 for "Rome", which is enough to get an approximate position.
    """)
    return


@app.cell(hide_code=True)
def geo_cities():
    with open("data/geo_cities.json") as _f:
        GEO_CITIES = tuple(
            (row["name"], row["lat"], row["lon"]) for row in json.load(_f)
        )

    EARTH_RADIUS_KM = 6371.0

    def geo_names():
        return tuple(name for name, _, _ in GEO_CITIES)

    def geo_latlon():
        return np.array(
            [[lat, lon] for _, lat, lon in GEO_CITIES], dtype=np.float64
        )

    def geo_latlon_to_xyz(coords):
        lat = np.deg2rad(coords[..., 0])
        lon = np.deg2rad(coords[..., 1])
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        return np.stack([x, y, z], axis=-1)

    def geo_xyz_to_latlon(xyz):
        unit = xyz / (np.linalg.norm(xyz, axis=-1, keepdims=True) + 1e-12)
        lat = np.rad2deg(np.arcsin(np.clip(unit[..., 2], -1.0, 1.0)))
        lon = np.rad2deg(np.arctan2(unit[..., 1], unit[..., 0]))
        return np.stack([lat, lon], axis=-1)

    def geo_haversine_km(a, b):
        lat1, lon1 = np.deg2rad(a[..., 0]), np.deg2rad(a[..., 1])
        lat2, lon2 = np.deg2rad(b[..., 0]), np.deg2rad(b[..., 1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
        return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))

    def geo_destination_point(lat, lon, distance_km, bearing_deg):
        """Great-circle destination point, correct near poles and across the dateline."""
        lat1 = np.radians(lat)
        lon1 = np.radians(lon)
        theta = np.radians(bearing_deg)
        d_r = distance_km / EARTH_RADIUS_KM
        lat2 = np.arcsin(
            np.sin(lat1) * np.cos(d_r)
            + np.cos(lat1) * np.sin(d_r) * np.cos(theta)
        )
        lon2 = lon1 + np.arctan2(
            np.sin(theta) * np.sin(d_r) * np.cos(lat1),
            np.cos(d_r) - np.sin(lat1) * np.sin(lat2),
        )
        lat2_deg = np.degrees(lat2)
        lon2_deg = (np.degrees(lon2) + 540) % 360 - 180
        return lat2_deg, lon2_deg

    GEO_QUERY_TEMPLATES = (
        "{name} is a famous place near the city",
        "{name} is located in the city",
        "{name} is near the city",
        "\"{name}\" can be found in the city",
    )

    with open("data/geo_monuments.json") as _f:
        GEO_MONUMENTS_BY_CITY = json.load(_f)
    return (
        EARTH_RADIUS_KM,
        GEO_CITIES,
        GEO_MONUMENTS_BY_CITY,
        GEO_QUERY_TEMPLATES,
        geo_destination_point,
        geo_haversine_km,
        geo_latlon,
        geo_latlon_to_xyz,
        geo_names,
        geo_xyz_to_latlon,
    )


@app.cell(hide_code=True)
def geo_jitter_training_data(
    GEO_CITIES,
    GEO_MONUMENTS_BY_CITY,
    GEO_QUERY_TEMPLATES,
    geo_destination_point,
):
    GEO_SEED = 33
    GEO_JITTER_PER_CITY = 8
    GEO_JITTER_MIN_KM = 10.0
    GEO_JITTER_MAX_KM = 2000.0

    _rng = np.random.default_rng(GEO_SEED)
    _n_coord = GEO_JITTER_PER_CITY // 2
    _n_monument = GEO_JITTER_PER_CITY - _n_coord

    geo_training_prompts = []
    geo_training_labels = []
    for _city_idx, (_name, _lat, _lon) in enumerate(GEO_CITIES):
        for _ in range(_n_coord):
            _dist = np.exp(
                _rng.uniform(
                    np.log(GEO_JITTER_MIN_KM), np.log(GEO_JITTER_MAX_KM)
                )
            )
            _bearing = _rng.uniform(0.0, 360.0)
            _jlat, _jlon = geo_destination_point(_lat, _lon, _dist, _bearing)
            geo_training_prompts.append(
                f"Lat {_jlat:.4f}, Long {_jlon:.4f} is near the city"
            )
            geo_training_labels.append(_city_idx)

        _monuments = _rng.choice(
            GEO_MONUMENTS_BY_CITY[_name], size=_n_monument, replace=False
        )
        for _i, _monument in enumerate(_monuments):
            _template = GEO_QUERY_TEMPLATES[_i % len(GEO_QUERY_TEMPLATES)]
            geo_training_prompts.append(_template.format(name=_monument))
            geo_training_labels.append(_city_idx)

    geo_training_labels = np.array(geo_training_labels)

    geo_train_button = mo.ui.run_button(
        label=f"Collect activations & train probe ({len(geo_training_prompts)} prompts)", kind="warn"
    )
    geo_train_button
    return (
        GEO_SEED,
        geo_train_button,
        geo_training_labels,
        geo_training_prompts,
    )


@app.cell(hide_code=True)
def geo_collect_activations(
    DEVICE,
    bundle,
    geo_train_button,
    geo_training_prompts,
):
    mo.stop(
        not geo_train_button.value,
        mo.callout(
            mo.md(
                "Click **Collect activations & train probe** above to (re)build the geo classifier."
            ),
            kind="info",
        ),
    )

    @mo.persistent_cache
    def geo_collect_geo_activations(prompts):
        @torch.inference_mode()
        def _last_token_resid(prompt):
            ids = bundle.tokenizer(prompt, return_tensors="pt").input_ids.to(
                DEVICE
            )
            out = bundle.model(input_ids=ids, output_hidden_states=True)
            hidden = torch.stack(
                out.hidden_states, dim=1
            )  # [1, layers+1, seq, d]
            return hidden[0, :, -1, :].float().cpu().numpy()

        rows = [
            _last_token_resid(_p)
            for _p in mo.status.progress_bar(
                prompts, title="Collecting geo activations"
            )
        ]
        return np.stack(rows)  # [n_prompts, layers+1, d]

    geo_training_resid = geo_collect_geo_activations(
        tuple(geo_training_prompts)
    )
    return geo_collect_geo_activations, geo_training_resid


@app.cell(hide_code=True)
def geo_probe_layer_sweep(geo_training_resid):
    geo_best_layer = (
        geo_training_resid.shape[1] - 1
    )
    return (geo_best_layer,)


@app.cell(hide_code=True)
def geo_final_probe(
    GEO_SEED,
    geo_best_layer,
    geo_collect_geo_activations,
    geo_training_labels,
    geo_training_prompts,
):
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    def _fit_sgd_with_progress(
        x, y, classes, title, n_epochs=60, batch_size=64, seed=GEO_SEED
    ):
        clf = SGDClassifier(loss="log_loss", alpha=1e-3, random_state=seed)
        rng = np.random.default_rng(seed)
        n = len(y)
        batches_per_epoch = max(1, int(np.ceil(n / batch_size)))
        steps = list(range(n_epochs * batches_per_epoch))
        for step in mo.status.progress_bar(steps, title=title):
            epoch, batch_i = divmod(step, batches_per_epoch)
            if batch_i == 0:
                perm = rng.permutation(n)
            lo, hi = batch_i * batch_size, min((batch_i + 1) * batch_size, n)
            idx = perm[lo:hi]
            clf.partial_fit(x[idx], y[idx], classes=classes)
        return clf

    @mo.persistent_cache
    def geo_train_city_classifier(prompts, labels, layer, seed):
        resid = geo_collect_geo_activations(prompts)[:, layer, :]
        labels = np.array(labels)
        classes = np.unique(labels)

        scaler = StandardScaler().fit(resid)
        resid_scaled = scaler.transform(resid)

        x_tr, x_te, y_tr, y_te = train_test_split(
            resid_scaled,
            labels,
            test_size=0.25,
            random_state=seed,
            stratify=labels,
        )

        clf = _fit_sgd_with_progress(
            x_tr, y_tr, classes, "Training geo probe (held-out split)"
        )
        test_acc = float(clf.score(x_te, y_te))

        clf_full = _fit_sgd_with_progress(
            resid_scaled, labels, classes, "Training geo probe (full data)"
        )
        return clf_full, test_acc, scaler

    geo_city_classifier_full, geo_city_classifier_test_acc, geo_city_scaler = (
        geo_train_city_classifier(
            tuple(geo_training_prompts),
            tuple(int(v) for v in geo_training_labels),
            geo_best_layer,
            GEO_SEED,
        )
    )
    return geo_city_classifier_full, geo_city_scaler


@app.cell(hide_code=True)
def geo_belief_helpers(
    DEVICE,
    EARTH_RADIUS_KM,
    GEO_QUERY_TEMPLATES,
    PIRATE_ORANGE,
    bundle,
    geo_best_layer,
    geo_city_classifier_full,
    geo_city_scaler,
    geo_destination_point,
    geo_haversine_km,
    geo_latlon,
    geo_latlon_to_xyz,
    geo_names,
    geo_xyz_to_latlon,
):
    @torch.inference_mode()
    def geo_activation_at_layer(prompt, layer):
        ids = bundle.tokenizer(prompt, return_tensors="pt").input_ids.to(
            DEVICE
        )
        out = bundle.model(input_ids=ids, output_hidden_states=True)
        return out.hidden_states[layer][0, -1, :].float().cpu().numpy()

    def geo_belief_from_prompt(
        prompt, layer=None, classifier=None, scaler=None
    ):
        layer = geo_best_layer if layer is None else layer
        classifier = (
            geo_city_classifier_full if classifier is None else classifier
        )
        scaler = geo_city_scaler if scaler is None else scaler
        resid = geo_activation_at_layer(prompt, layer)
        resid_scaled = scaler.transform(resid[None, :])
        logits = classifier.decision_function(resid_scaled)[0]
        probs = classifier.predict_proba(resid_scaled)[0]

        xyz = geo_latlon_to_xyz(geo_latlon())
        mean_xyz = (probs[:, None] * xyz).sum(axis=0)
        mean_xyz = mean_xyz / (np.linalg.norm(mean_xyz) + 1e-12)
        mean_latlon = geo_xyz_to_latlon(mean_xyz[None, :])[0]

        dists = geo_haversine_km(
            np.tile(mean_latlon, (len(probs), 1)), geo_latlon()
        )
        sigma_km = float((probs * dists).sum())

        return {
            "probs": probs,
            "logits": logits,
            "mean_latlon": mean_latlon,
            "sigma_km": sigma_km,
            "argmax_city": geo_names()[int(np.argmax(probs))],
            "argmax_prob": float(probs.max()),
        }

    def geo_softmax_temperature(logits, temperature):
        scaled = logits / temperature
        scaled = scaled - scaled.max()
        p = np.exp(scaled)
        return p / p.sum()

    def geo_belief_at_temperature(logits, temperature):
        probs = geo_softmax_temperature(logits, temperature)
        xyz = geo_latlon_to_xyz(geo_latlon())
        mean_xyz = (probs[:, None] * xyz).sum(axis=0)
        mean_xyz = mean_xyz / (np.linalg.norm(mean_xyz) + 1e-12)
        mean_latlon = geo_xyz_to_latlon(mean_xyz[None, :])[0]
        dists = geo_haversine_km(
            np.tile(mean_latlon, (len(probs), 1)), geo_latlon()
        )
        sigma_km = float((probs * dists).sum())
        return mean_latlon, sigma_km

    def geo_circle_polygon(center_lat, center_lon, radius_km, n_points=64):
        if radius_km < 1.0:
            radius_km = 1.0
        bearings = np.linspace(0.0, 360.0, n_points)
        lats, lons = geo_destination_point(
            center_lat, center_lon, radius_km, bearings
        )
        coords = [[float(lo), float(la)] for la, lo in zip(lats, lons)]
        coords.append(coords[0])
        return coords

    GEO_TEMPERATURES = (100, 50, 25, 10.0, 5.)

    def geo_render_belief_globe(logits, true_latlon=None, true_label=None):
        _pred_latlon, _ = geo_belief_at_temperature(logits, 1.0)
        if true_latlon is None:
            _center_latlon = _pred_latlon
        else:
            _mid_xyz = geo_latlon_to_xyz(true_latlon) + geo_latlon_to_xyz(
                _pred_latlon
            )
            _mid_xyz = _mid_xyz / (np.linalg.norm(_mid_xyz) + 1e-12)
            _center_latlon = geo_xyz_to_latlon(_mid_xyz[None, :])[0]

        belief_map = leafmap3d.Map(
            center=(_center_latlon[1], _center_latlon[0]),
            zoom=2.5,
            projection="globe",
            style="dark-matter",
            height="600px",
        )
        belief_map.add_globe_control()

        for _temp in GEO_TEMPERATURES:
            _mean_ll, _sigma_km = geo_belief_at_temperature(logits, _temp)
            _sigma_km = min(max(_sigma_km, 150.0), EARTH_RADIUS_KM * 0.95)
            _ring = geo_circle_polygon(_mean_ll[0], _mean_ll[1], _sigma_km)
            _polygon_geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [_ring],
                        },
                        "properties": {
                            "temperature": _temp,
                            "sigma_km": _sigma_km,
                        },
                    }
                ],
            }
            _middle_temp = sorted(GEO_TEMPERATURES)[len(GEO_TEMPERATURES) // 2]
            _color = PIRATE_ORANGE if _temp == _middle_temp else "gray"
            belief_map.add_geojson(
                _polygon_geojson,
                layer_type="fill",
                name=f"belief T={_temp}",
                fit_bounds=False,
                paint={"fill-color": _color, "fill-opacity": 0.35},
            )
            belief_map.add_geojson(
                _polygon_geojson,
                layer_type="line",
                name=f"belief outline T={_temp}",
                fit_bounds=False,
                paint={
                    "line-color": _color,
                    "line-width": 1.5,
                    "line-opacity": 0.8,
                },
            )

        if true_latlon is not None:
            _true_geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [true_latlon[1], true_latlon[0]],
                        },
                        "properties": {"name": true_label},
                    }
                ],
            }
            belief_map.add_geojson(
                _true_geojson,
                layer_type="circle",
                name="true location",
                fit_bounds=False,
                paint={
                    "circle-radius": 7,
                    "circle-color": "red",
                    "circle-opacity": 1.0,
                },
            )
            belief_map.add_geojson(
                _true_geojson,
                layer_type="symbol",
                name="true location label",
                fit_bounds=False,
                layout={
                    "text-field": ["get", "name"],
                    "text-offset": [0, 1.4],
                    "text-size": 13,
                    "text-anchor": "top",
                },
                paint={
                    "text-color": "red",
                    "text-halo-color": "black",
                    "text-halo-width": 1.5,
                },
            )
        return belief_map

    def geo_belief_ensemble(
        name, templates=GEO_QUERY_TEMPLATES, layer=None, classifier=None
    ):
        """Average logits across several prompt phrasings for the same query, since a
        single formulation's zero-shot transfer can be surprisingly phrasing-sensitive
        (verified: identical monument, wildly different answers depending on template).
        Also reports formulation_spread_km: how far each individual template's own
        belief-mean lands from the ensemble mean -- a second, complementary notion of
        uncertainty (disagreement across phrasings, not just within one distribution).
        """
        per_template = []
        logits_list = []
        for _tpl in templates:
            _prompt = _tpl.format(name=name)
            _belief = geo_belief_from_prompt(
                _prompt, layer=layer, classifier=classifier
            )
            per_template.append({"prompt": _prompt, **_belief})
            logits_list.append(_belief["logits"])

        avg_logits = np.mean(np.stack(logits_list), axis=0)
        avg_probs = geo_softmax_temperature(avg_logits, 1.0)

        xyz = geo_latlon_to_xyz(geo_latlon())
        mean_xyz = (avg_probs[:, None] * xyz).sum(axis=0)
        mean_xyz = mean_xyz / (np.linalg.norm(mean_xyz) + 1e-12)
        mean_latlon = geo_xyz_to_latlon(mean_xyz[None, :])[0]
        dists = geo_haversine_km(
            np.tile(mean_latlon, (len(avg_probs), 1)), geo_latlon()
        )
        sigma_km = float((avg_probs * dists).sum())

        # exclude near-uniform (uninformative) templates: a flat distribution's
        # spherical mean is essentially arbitrary noise, not a coherent alternative
        # opinion, and including it corrupts the spread with meaningless outliers
        _confident = [
            t
            for t in per_template
            if t["argmax_prob"] > 5.0 / len(geo_names())
        ]
        if len(_confident) == 0:
            # none cleared the confidence bar -- fall back to all templates rather
            # than reporting nan; if they're all uninformative they likely agree
            # with each other too (verified: identical uniform outputs -> spread 0)
            _confident = per_template
        per_template_means = np.stack([t["mean_latlon"] for t in _confident])
        formulation_dists = geo_haversine_km(
            np.tile(mean_latlon, (len(per_template_means), 1)),
            per_template_means,
        )
        formulation_spread_km = float(formulation_dists.mean())
        formulation_spread_std_km = (
            float(formulation_dists.std(ddof=1))
            if len(formulation_dists) > 1
            else 0.0
        )

        return {
            "probs": avg_probs,
            "logits": avg_logits,
            "mean_latlon": mean_latlon,
            "sigma_km": sigma_km,
            "argmax_city": geo_names()[int(np.argmax(avg_probs))],
            "argmax_prob": float(avg_probs.max()),
            "formulation_spread_km": formulation_spread_km,
            "formulation_spread_std_km": formulation_spread_std_km,
            "per_template": per_template,
        }

    return geo_belief_ensemble, geo_render_belief_globe


@app.cell(hide_code=True)
def geo_monument_dropdown():
    with open("data/geo_monument_examples.json") as _f:
        _geo_monument_examples_raw = json.load(_f)

    GEO_MONUMENT_EXAMPLES = tuple(
        (
            row["name"],
            row["city"],
            row["lat"],
            row["lon"],
            row["in_training_cities"],
        )
        for row in _geo_monument_examples_raw
    )

    geo_monument_dropdown = mo.ui.dropdown(
        options=[m[0] for m in GEO_MONUMENT_EXAMPLES],
        value="Taj Mahal",
        label="Query location (monument)",
    )
    geo_monument_text_input = mo.ui.text(
        label="Enter any place or monument name",
        placeholder="e.g. Golden Gate Bridge",
    )
    geo_query_mode_tabs = mo.ui.tabs(
        {
            "Preselected monument": geo_monument_dropdown,
            "Your own (no ground truth)": geo_monument_text_input,
        }
    )
    geo_query_mode_tabs
    return (
        GEO_MONUMENT_EXAMPLES,
        geo_monument_dropdown,
        geo_monument_text_input,
        geo_query_mode_tabs,
    )


@app.cell(hide_code=True)
def geo_monument_belief_compute(
    GEO_MONUMENT_EXAMPLES,
    geo_belief_ensemble,
    geo_monument_dropdown,
    geo_monument_text_input,
    geo_query_mode_tabs,
):
    if geo_query_mode_tabs.value == "Preselected monument":
        _m = next(
            m
            for m in GEO_MONUMENT_EXAMPLES
            if m[0] == geo_monument_dropdown.value
        )
        _mon_name, _mon_city, _mon_lat, _mon_lon, _mon_in_training = _m
        geo_monument_true_latlon = np.array([_mon_lat, _mon_lon])
        _true_line = f"True location: {_mon_name}, {_mon_city} ({_mon_lat:.2f}, {_mon_lon:.2f})"
    else:
        _mon_name = geo_monument_text_input.value
        geo_monument_true_latlon = None
        _true_line = "No ground truth (custom query)."

    mo.stop(
        not _mon_name,
        mo.callout(mo.md("Enter a place name."), kind="info"),
    )

    geo_monument_belief = geo_belief_ensemble(_mon_name)
    geo_monument_query_name = _mon_name

    mo.md(
        f"""
    Model's top guess: **{geo_monument_belief["argmax_city"]}**
    """
    )
    return (
        geo_monument_belief,
        geo_monument_query_name,
        geo_monument_true_latlon,
    )


@app.cell(hide_code=True)
def geo_monument_globe_plot(
    geo_monument_belief,
    geo_monument_query_name,
    geo_monument_true_latlon,
    geo_render_belief_globe,
):
    geo_monument_map = geo_render_belief_globe(
        geo_monument_belief["logits"],
        geo_monument_true_latlon,
        geo_monument_query_name,
    )
    geo_monument_map
    return


@app.cell(hide_code=True)
def additional_material_title():
    mo.md(r"""
    # Additional material
    """)
    return


@app.cell(hide_code=True)
def additional_material_intro():
    mo.md(r"""
    Writing this notebook was a lot of experimentation, both with the experiences of the paper itself, extensions we wanted to add, and ways to explain the concept that did not end up in the final version. This is bonus content, enjoy 😉
    """)
    return


@app.cell(hide_code=True)
def intrinsic_dimension_title():
    mo.md(r"""
    ## Intrinsic dimension and curviness
    """)
    return


@app.cell(hide_code=True)
def intrinsic_dimension_pca_stats(activation_groups):
    def _joint_belief_centroids_for_pca(groups, layer: int):
        _pairs = sorted(groups.keys())
        _centroids = np.stack([
            groups[pair][0][:, layer, :].mean(axis=0)
            for pair in _pairs
        ])
        return _pairs, _centroids


    def _joint_pca_intrinsic_rows(pairs, centroids, layer: int):
        _centered = centroids - centroids.mean(axis=0, keepdims=True)
        _rank_cap = max(0, min(len(pairs) - 1, centroids.shape[1]))
        if _rank_cap == 0:
            return []

        _singular_values = np.linalg.svd(
            _centered, full_matrices=False, compute_uv=False
        )
        _variance = _singular_values[:_rank_cap] ** 2
        _total_variance = float(_variance.sum())
        _explained = (
            np.zeros_like(_variance)
            if _total_variance <= 0
            else _variance / _total_variance
        )
        _cumulative = np.cumsum(_explained)

        return [
            {
                "layer": layer,
                "k": k,
                "n_centroids": len(pairs),
                "n_mu": len({mu for mu, _sigma in pairs}),
                "n_sigma": len({sigma for _mu, sigma in pairs}),
                "variance_axis_k": float(_explained[k - 1]),
                "variance_first_k": float(_cumulative[k - 1]),
                "variance_after_k": float(1.0 - _cumulative[k - 1]),
            }
            for k in range(1, _rank_cap + 1)
        ]


    _intrinsic_rows = []
    _n_layers = next(iter(activation_groups.values()))[0].shape[1]
    for _layer in range(_n_layers):
        _pairs, _centroids = _joint_belief_centroids_for_pca(
            activation_groups, _layer
        )
        _intrinsic_rows.extend(
            _joint_pca_intrinsic_rows(_pairs, _centroids, _layer)
        )

    intrinsic_dimension_pca_df = pl.DataFrame(_intrinsic_rows)
    return (intrinsic_dimension_pca_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    On thing the paper did and not us was measuring the "curviness" of the $(\sigma, \mu)$ space. One cool think they observed was that the curviness (measured by the number of PCA dimensions needed to reach 95% explained variance) increased with the layers, and then decreased near the end. We did not reproduce this result, because we did not use enough $\mu$ values.
    """)
    return


@app.cell(hide_code=True)
def intrinsic_dimension_pca_chart(intrinsic_dimension_pca_df):
    _nearest_layer = alt.selection_point(
        nearest=True,
        on="pointerover",
        fields=["layer"],
        empty=False,
    )

    _base = alt.Chart(intrinsic_dimension_pca_df).encode(
        x=alt.X(
            "k:O",
            title="PCA rank k",
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y(
            "variance_first_k:Q",
            title="Cumulative explained variance",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"),
        ),
        color=alt.Color(
            "layer:Q",
            title="Layer",
            scale=alt.Scale(scheme="turbo"),
        ),
        detail="layer:N",
        tooltip=[
            alt.Tooltip("layer:Q", title="layer", format=".0f"),
            alt.Tooltip("k:Q", title="rank", format=".0f"),
            alt.Tooltip("variance_axis_k:Q", title="axis variance", format=".1%"),
            alt.Tooltip("variance_first_k:Q", title="cumulative", format=".1%"),
            alt.Tooltip("variance_after_k:Q", title="remaining", format=".1%"),
            alt.Tooltip("n_centroids:Q", title="centroids", format=".0f"),
        ],
    )

    _lines = _base.mark_line(point=True, strokeWidth=2.25).encode(
        opacity=alt.condition(_nearest_layer, alt.value(0.98), alt.value(0.16)),
    ).add_params(_nearest_layer)

    _hover_targets = _base.mark_point(size=160, opacity=0).add_params(_nearest_layer)

    (_lines + _hover_targets).properties(
        width=940,
        height=380,
        title="Joint (mu, sigma) centroid PCA intrinsic dimension by layer",
    )
    return


@app.cell(hide_code=True)
def eye_probe_analogy_note():
    mo.md(r"""
    ## Probes in your eyeballs

    At one point, I thought that the best way to explain probes was to go with an analogy with the human eye: you have 3 detectors, and you want to reconstruct a single wavelength from them.
    """)
    return


@app.cell(hide_code=True)
def cone_probe_scene():
    class _ConeProbe(ManimWidget):
        def construct(self):
            self.camera.background_color = "#1a1a2e"

            def _gov(lam, lam_max):
                x = lam_max / lam
                a = 0.8795 + 0.0459 * np.exp(-(lam_max - 300)**2 / 11940)
                val = np.exp(69.7*(a - x)) + np.exp(28.0*(0.922 - x)) + np.exp(-14.9*(1.104 - x)) + 0.674
                return float(np.clip(1.0 / val, 0, None))

            _lams = np.linspace(380, 720, 500)
            def _norm(lam_max):
                pk = max(_gov(l, lam_max) for l in _lams)
                return lambda l: _gov(l, lam_max) / pk

            s_sens = _norm(420)
            m_sens = _norm(530)
            l_sens = _norm(560)

            axes = Axes(
                x_range=[380, 720, 50], y_range=[0, 1.1, 0.5],
                x_length=8, y_length=4,
                axis_config={"color": WHITE, "stroke_width": 1.5},
                x_axis_config={"numbers_to_include": [400, 450, 500, 550, 600, 650, 700]},
            ).shift(LEFT * 0.8 + DOWN * 0.5)
            x_label = Text("λ (nm)", font_size=22, color=WHITE).next_to(axes.x_axis, DOWN)

            self.play(Create(axes), Write(x_label), run_time=1.5)

            s_curve = axes.plot(s_sens, x_range=[380, 720], color="#6699dd")
            m_curve = axes.plot(m_sens, x_range=[380, 720], color="#44bb66")
            l_curve = axes.plot(l_sens, x_range=[380, 720], color="#dd4444")

            s_lbl = Text("S", font_size=22, color="#6699dd").move_to(axes.c2p(420, 1.08))
            m_lbl = Text("M", font_size=22, color="#44bb66").move_to(axes.c2p(530, 1.08))
            l_lbl = Text("L", font_size=22, color="#dd4444").move_to(axes.c2p(560, 1.08))

            self.play(Create(s_curve), Write(s_lbl), run_time=2.5)
            self.play(Create(m_curve), Write(m_lbl), run_time=2.5)
            self.play(Create(l_curve), Write(l_lbl), run_time=2.5)

            self.next_section("Wavelength probe")

            bar_lam = 580
            s_val = s_sens(bar_lam)
            m_val = m_sens(bar_lam)
            l_val = l_sens(bar_lam)

            bar = Line(
                axes.c2p(bar_lam, 0), axes.c2p(bar_lam, 1.05),
                color=ORANGE, stroke_width=5,
            )
            self.play(Create(bar), run_time=1.2)

            s_dot = Dot(axes.c2p(bar_lam, s_val), color="#6699dd", radius=0.13)
            m_dot = Dot(axes.c2p(bar_lam, m_val), color="#44bb66", radius=0.13)
            l_dot = Dot(axes.c2p(bar_lam, l_val), color="#dd4444", radius=0.13)
            self.play(FadeIn(s_dot), FadeIn(m_dot), FadeIn(l_dot), run_time=1.2)
            self.wait(0.6)

            right_x = axes.get_right()[0] + 1.6
            s_target = [right_x, axes.c2p(bar_lam, s_val)[1], 0]
            m_target = [right_x, axes.c2p(bar_lam, m_val)[1], 0]
            l_target = [right_x, axes.c2p(bar_lam, l_val)[1], 0]

            self.play(
                s_dot.animate.move_to(s_target),
                m_dot.animate.move_to(m_target),
                l_dot.animate.move_to(l_target),
                run_time=2.0,
            )
            self.play(
                s_curve.animate.set_opacity(0.15),
                m_curve.animate.set_opacity(0.15),
                l_curve.animate.set_opacity(0.15),
                FadeOut(bar),
                run_time=1.0,
            )

            s_txt = Text(f"{s_val:.2f}", font_size=20, color="#6699dd").next_to(s_dot, RIGHT, buff=0.2)
            m_txt = Text(f"{m_val:.2f}", font_size=20, color="#44bb66").next_to(m_dot, RIGHT, buff=0.2)
            l_txt = Text(f"{l_val:.2f}", font_size=20, color="#dd4444").next_to(l_dot, RIGHT, buff=0.2)
            qmark = Text("?", font_size=72, color=ORANGE).move_to(axes.c2p(bar_lam, 0.55))

            self.play(
                FadeIn(s_txt), FadeIn(m_txt), FadeIn(l_txt),
                Write(qmark),
                run_time=1.5,
            )
            self.wait(0.5)

    _ConeProbe(autoplay=False, canvas_width=700)
    return


@app.cell(hide_code=True)
def channel_merge_note():
    mo.md(r"""
    I wanted to use them to explain how the probes for M (green) and L(red) are not very different, and that you could merge them into a single detector without losing too much information. It's related to how you can do a PCA on multi-linear field probes to get the most important variations.
    """)
    return


@app.cell(hide_code=True)
def channel_sensitivity_helpers():
    import requests, io
    from PIL import Image

    def make_channel_grey_fig(source):
        if str(source).startswith("http://") or str(source).startswith("https://"):
            _resp = requests.get(source, timeout=15, headers={"User-Agent": "morphos-notebook/1.0 (educational)"})
            _img = Image.open(io.BytesIO(_resp.content)).convert("RGB")
        else:
            _img = Image.open(source).convert("RGB")
        _img.thumbnail((400, 400))
        _arr = np.asarray(_img).astype(np.float64) / 255.0
        _R, _G, _B = _arr[..., 0], _arr[..., 1], _arr[..., 2]

        _fig, _axes = plt.subplots(2, 3, figsize=(11, 7))
        _axes[0, 0].imshow(_B, cmap="gray"); _axes[0, 0].set_title("S")
        _axes[0, 1].imshow(_G, cmap="gray"); _axes[0, 1].set_title("M")
        _axes[0, 2].imshow(_R, cmap="gray"); _axes[0, 2].set_title("L")

        _axes[1, 0].imshow(_B, cmap="gray"); _axes[1, 0].set_title("S")
        _axes[1, 1].imshow((_R + _G) / 2, cmap="gray"); _axes[1, 1].set_title("M+L")
        _axes[1, 2].imshow(_R - _G, cmap="gray", vmin=-1, vmax=1); _axes[1, 2].set_title("M-L (diff)")

        for _row in _axes:
            for _ax in _row:
                _ax.axis("off")
        _fig.tight_layout()
        return _fig

    return (make_channel_grey_fig,)


@app.cell
def channel_sensitivity_display(make_channel_grey_fig):
    make_channel_grey_fig("./assets/parrot.png")
    return


@app.cell(hide_code=True)
def linear_probe_limits_title():
    mo.md(r"""
    ## Limitations of linear probes

    There are really good reasons why we don't use linear probes everywhere, and instead use more complicated stuff.

    Indeed, linear probes can find directions that are not optimal in the sense of "Maximum variance direction", ie the first axis of a PCA.

    Here is the example Claude came up with (I'm pretty sure it's a well-known fact in the litterature, but I did not check)
    """)
    return


@app.cell(hide_code=True)
def sigma_probe_curvature_plot(
    PIRATE_BLUE,
    PIRATE_GREEN,
    PIRATE_ORANGE,
    PIRATE_PURPLE,
):

    _sigmas = np.array([20.0, 50.0, 100.0, 200.0])
    _t = np.log10(_sigmas)
    _t = _t - _t.mean()

    _x1 = _t
    _x2 = _t**2
    _X = np.stack([_x1, _x2], axis=1)

    _Xd = np.hstack([_X, np.ones((4, 1))])
    _beta, *_ = np.linalg.lstsq(_Xd, _sigmas, rcond=None)
    _coef = _beta[:2]
    _unit_probe = _coef / np.linalg.norm(_coef)
    _pred = _Xd @ _beta
    _r2 = 1 - ((_sigmas - _pred) ** 2).sum() / ((_sigmas - _sigmas.mean()) ** 2).sum()

    _orth = np.array([-_unit_probe[1], _unit_probe[0]])

    _probe_readout = _X @ _unit_probe
    _residual_readout = _X @ _orth
    _corr = np.corrcoef(_probe_readout, _residual_readout)[0, 1]

    # Calculate principal components for maximum variance line
    _data = np.stack([_probe_readout, _residual_readout], axis=1)
    _cov = np.cov(_data.T)
    _eigvals, _eigvecs = np.linalg.eigh(_cov)
    _max_var_dir = _eigvecs[:, -1]  # Eigenvector corresponding to largest eigenvalue

    _fig, _axes2 = plt.subplots(1, 2, figsize=(11, 4.5))

    _ax = _axes2[0]
    _curve_t = np.linspace(_t.min() - 0.15, _t.max() + 0.15, 200)
    _ax.plot(_curve_t, _curve_t**2, color="gray", linewidth=1.5, alpha=0.6, label="true manifold (x1, x1²)")
    _palette = [PIRATE_BLUE, PIRATE_ORANGE, PIRATE_GREEN, PIRATE_PURPLE]
    for _xi1, _xi2, _s, _c in zip(_x1, _x2, _sigmas, _palette):
        _ax.plot(_xi1, _xi2, "o", color=_c, markersize=10, markeredgecolor="black", zorder=5, label=f"σ={_s:g}")
    _scale = 0.55
    _center_pt = np.array([_x1.mean(), _x2.mean()])
    _ax.annotate("", xy=tuple(_center_pt + _scale * _unit_probe), xytext=tuple(_center_pt - _scale * _unit_probe),
                 arrowprops=dict(arrowstyle="-|>", color="black", linewidth=2))
    _ax.text(*(_center_pt + _scale * 1.15 * _unit_probe), "probe dir", fontsize=9, ha="center")
    _ax.annotate("", xy=tuple(_center_pt + _scale * _orth), xytext=tuple(_center_pt - _scale * _orth),
                 arrowprops=dict(arrowstyle="-|>", color="crimson", linewidth=2))
    _ax.text(*(_center_pt + _scale * 1.25 * _orth), "orthogonal dir", fontsize=9, ha="center", color="crimson")
    _ax.set_xlabel("x1 = log10(σ), centered")
    _ax.set_ylabel("x2 = x1²")
    _ax.set_title(f"linear probe on curved manifold  (R²={_r2:.3f})")
    _ax.legend(fontsize=8, frameon=False, loc="upper center")
    _ax.spines[["top", "right"]].set_visible(False)

    _ax2 = _axes2[1]
    for _p, _r, _s, _c in zip(_probe_readout, _residual_readout, _sigmas, _palette):
        _ax2.plot(_p, _r, "o", color=_c, markersize=12, markeredgecolor="black", zorder=5)
        _ax2.annotate(f"σ={_s:g}", (_p, _r), textcoords="offset points", xytext=(6, 6), fontsize=8)

    # Add dotted line for axis of maximum variance
    _probe_range = np.array([_probe_readout.min(), _probe_readout.max()])
    _residual_range = np.array([_residual_readout.min(), _residual_readout.max()])
    _data_center = _data.mean(axis=0)
    _max_var_slope = _max_var_dir[1] / _max_var_dir[0]
    _probe_extent = np.linspace(_probe_range[0] - 0.1, _probe_range[1] + 0.1, 100)
    _residual_max_var = _data_center[1] + _max_var_slope * (_probe_extent - _data_center[0])
    _ax2.plot(_probe_extent, _residual_max_var, ':', color='gray', linewidth=1.5, alpha=0.8, label="max variance axis")

    _ax2.set_xlabel("probe readout  (X · probe_dir)")
    _ax2.set_ylabel("residual readout  (X · orthogonal_dir)")
    _ax2.set_title(f"orthogonal-as-vectors, correlated-as-readouts (r={_corr:.2f})")
    _ax2.spines[["top", "right"]].set_visible(False)
    _ax2.axhline(0, color="gray", linewidth=0.5)
    _ax2.axvline(0, color="gray", linewidth=0.5)
    _ax2.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def linear_probe_limits_note():
    mo.md(r"""
    Here, you see that the network learns a direction that align the points well with their associated values, not the direction in which the points are most spread-out.

    It also illustrate why $L_2$ normalization works here: if you force the weights to be smaller, the network will converge to the dotted line.
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def faq_title():
    mo.md(r"""
    # FAQ
    """)
    return


@app.function(hide_code=True)
def faq_detail(title: str, element: mo.Html) -> mo.Html:
    """Wrap a FAQ entry in a <details>/<summary>. The <h2> is forced inline
    so it doesn't clash with the <summary>::before disclosure marker."""
    return mo.md(
        f'<details><summary><h2 style="display: inline; margin: 0">{title}</h2></summary>{element.text}</details>'
    )


@app.cell(hide_code=True)
def faq_3d_graphics():
    faq_detail("How did you make the gorgeous 3D graphics and animations?", mo.md(r"""
    Most of the 2d plots were done with matplotlib. A good color scheme and a few layout tweaks really go a long way.

    The 3d scenes are all done using [Manim](https://github.com/ManimCommunity/manim).
    However, you may have noticed that they are not pre-recorded, and even that some of them are interactive.

    That's the project I've been working on for months: [manim-widget](https://github.com/rambip/manim-widget).
    It's now in alpha, I hope you will love it.
    """))
    return


@app.cell(hide_code=True)
def faq_ai_usage():
    faq_detail("How did you use AI?", mo.md(r"""

    We really care a lot about responsible AI use in this team. One of us work as a AI safety researcher, the other in explainable AI and advocacy (via [pauseIA](https://pauseai.info/))

    But of course, we used AI extensively (with marimo-pair) for this project. We followed the following principle:
    - promote Human culture and fullfilment: no AI generated art, all images are hand-drawn
    - 0% AI writing. AI are so bad at writting and narrative work, when they don't clearly gaslight you
    - avoid code SLOP: review the core functions, be clear about architecture
    """))
    return


@app.cell(hide_code=True)
def faq_marimo_suggestions():
    faq_detail("Do you have suggestions to improve marimo?", mo.md(r"""

    Yes, a lot !

    ### Navigation

    Currently you can't use `#sectionName` to jump to a header with marimo. that would be really cool if you could do that, to point to a specific section of the notebook with a url.

    ### Style

    The default style of marimo is good, not great. For this notebook we increased font size, font family and a few other tweaks. It usually works well with custom CSS sheets, but unfortunately it does not work on molab. And sometimes it makes more sense to have the style of an element be defined in your notebook directly.

    We hacked together a "style-injection widget", but it would be nice to have that kind of thing built-in to marimo.


    ### Elements

    In this notebook there are a few custom widget that could live inside [wigglystuff](https://github.com/koaning/wigglystuff) or even marimo core elements.
    - the `Ticker` widget that reruns descendents a given number of times
    - a `spoiler` element that hides your widget before the user clicks 'reveal'


    ### Core and animations

    Better support for streaming animations ? I don't know what that could look like but being able to construct matplotlib animations easily would be great.
    """))
    return


@app.cell(hide_code=True)
def _():
    faq_detail("Who are you ?", mo.md("""
    We're 2 students, both aspiring researchers. Find us here:
    - https://sckathach.github.io
    - https://rambip.github.io

    If by any chance you want to propose us an internship or a thesis, reach out at antonin.peronnet@telecom-paris.fr
    """))
    return


@app.cell(hide_code=True)
def _(N_POINTS):
    # Precompute all animation data that does not depend on the live ticker step.
    _x, _y, _labels, _texts = generate_labeled_points(
        n_points=N_POINTS,
        centroid_1=(2.0, 2.0),
        centroid_2=(8.0, 8.0),
        std=1.5,
        random_state=42,
    )

    _x_min, _x_max = _x.min() - 1, _x.max() + 1
    _y_min, _y_max = _y.min() - 1, _y.max() + 1

    _n_by_step = np.minimum(np.arange(N_POINTS + 1) + 1, len(_x))
    _centroid_pos_by_step = []
    _centroid_neg_by_step = []

    for _n_show in _n_by_step:
        _x_shown = _x[:_n_show]
        _y_shown = _y[:_n_show]
        _labels_shown = _labels[:_n_show]

        _mask_pos = _labels_shown == True
        _mask_neg = _labels_shown == False

        _x_pos = _x_shown[_mask_pos]
        _y_pos = _y_shown[_mask_pos]
        _x_neg = _x_shown[_mask_neg]
        _y_neg = _y_shown[_mask_neg]

        _centroid_pos_by_step.append((
            np.mean(_x_pos) if len(_x_pos) > 0 else 8.0,
            np.mean(_y_pos) if len(_y_pos) > 0 else 8.0,
        ))
        _centroid_neg_by_step.append((
            np.mean(_x_neg) if len(_x_neg) > 0 else 2.0,
            np.mean(_y_neg) if len(_y_neg) > 0 else 2.0,
        ))

    probe_animation_precomputed = {
        "x": _x,
        "y": _y,
        "labels": _labels,
        "texts": _texts,
        "xlim": (_x_min, _x_max),
        "ylim": (_y_min, _y_max),
        "n_by_step": _n_by_step,
        "centroid_pos_by_step": tuple(_centroid_pos_by_step),
        "centroid_neg_by_step": tuple(_centroid_neg_by_step),
    }
    return (probe_animation_precomputed,)


if __name__ == "__main__":
    app.run()
