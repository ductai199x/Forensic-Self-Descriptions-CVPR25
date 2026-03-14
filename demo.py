"""Gradio demo for FSD: Detecting AI-Generated Images via Forensic Self-Descriptions.

Usage:
    uv run demo.py
    uv run demo.py --share
    uv run demo.py --device cpu
"""

import argparse

import gradio as gr
from PIL import Image
# Register HEIF/HEIC support before any image loading
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

from fsd import FSDDetector, DetectionResult, AttributionResult

# ---------------------------------------------------------------------------
# Palette  (dark theme, user-provided from coolors.co)
# ---------------------------------------------------------------------------
JET_BLACK = "#2d3142"       # dark bg
BEIGE = "#e9edde"           # primary text on dark
BANANA_CREAM = "#e7e247"    # warning / uncertain accent
GLAUCOUS = "#5c80bc"        # links, buttons, secondary accent
PEARL_AQUA = "#69d1c5"      # positive / real accent

CARD_BG = "#363b50"         # slightly lighter than jet for cards
MUTED = "#9a9eb0"           # subdued text


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

def _verdict(z: float, threshold: float):
    if z >= -1.0:
        return ("Real", "Forensic signature is consistent with real photographs.", "verdict-real")
    if z >= threshold:
        return ("Likely Real", "Leans toward real, but not a definitive match.", "verdict-likely-real")
    if z >= threshold - 1.0:
        return ("Likely AI", "Shows signs of AI generation in its forensic signature.", "verdict-likely-ai")
    return ("AI-Generated", "Forensic signature strongly deviates from real photographs.", "verdict-ai")


def _prob_fake(z: float, threshold: float = -2.0, k: float = 2.0) -> float:
    """Sigmoid centered at the decision threshold so z=threshold -> exactly 50%.

    k=2.0 gives: z=0 -> 2%, z=-1 -> 12%, z=-2 -> 50%, z=-3 -> 88%, z=-4 -> 98%.
    """
    import math
    return 1.0 / (1.0 + math.exp(-k * (threshold - z)))


def _build_attribution_html(attr_result: AttributionResult) -> str:
    """Build horizontal bar chart HTML for source attribution scores."""
    # Sort sources by confidence (softmax of log-likelihoods)
    import torch
    names = list(attr_result.scores.keys())
    ll_tensor = torch.tensor([attr_result.scores[n] for n in names], dtype=torch.float64)
    probs = torch.softmax(ll_tensor, dim=0).tolist()
    ranked = sorted(zip(names, probs), key=lambda x: x[1], reverse=True)

    bars_html = ""
    for name, prob in ranked:
        pct = prob * 100
        is_best = (name == attr_result.source)
        bar_cls = "attr-bar-best" if is_best else ""
        label_cls = "attr-label-best" if is_best else ""
        bars_html += f"""
        <div class="attr-row">
          <span class="attr-name {label_cls}">{name}</span>
          <div class="attr-track">
            <div class="attr-fill {bar_cls}" style="width:{pct:.1f}%"></div>
          </div>
          <span class="attr-pct {label_cls}">{pct:.1f}%</span>
        </div>"""

    return f"""
    <div class="attribution-section">
      <div class="attr-header">Source Attribution</div>
      <div class="attr-predicted">
        Predicted source: <strong>{attr_result.source}</strong>
        ({attr_result.confidence:.1%} confidence)
      </div>
      <div class="attr-chart">{bars_html}
      </div>
    </div>"""


def build_result_html(result, attr_result=None) -> str:
    z = result.z_score
    threshold = result.threshold
    label, desc, css_cls = _verdict(z, threshold)
    p_fake = _prob_fake(z, threshold)
    pct = max(0, min(100, (z + 5.0) / 6.0 * 100))

    attribution_html = ""
    if attr_result is not None:
        attribution_html = _build_attribution_html(attr_result)

    return f"""
    <div class="result-card {css_cls}">
      <span class="verdict-label">{label}</span>
      <p class="verdict-desc">{desc}</p>

      <div class="gauge">
        <div class="gauge-track">
          <div class="gauge-marker" style="left:{pct:.1f}%"></div>
        </div>
        <div class="gauge-labels">
          <span>AI-Generated</span>
          <span>Real</span>
        </div>
      </div>

      <div class="stats-row">
        <div class="stat">
          <div class="stat-value">{p_fake:.0%}</div>
          <div class="stat-label">Probability AI-Generated</div>
        </div>
        <div class="stat">
          <div class="stat-value">{z:.3f}</div>
          <div class="stat-label">Z-Score</div>
        </div>
        <div class="stat">
          <div class="stat-value">{result.raw_score:.2f}</div>
          <div class="stat-label">Raw Score</div>
        </div>
      </div>
      {attribution_html}
    </div>
    """


PLACEHOLDER_HTML = """
<div class="result-card placeholder">
  <p>Upload an image to check if it is AI-generated and identify its source.</p>
</div>
"""


# ---------------------------------------------------------------------------
# CSS — palette accents on top of Soft theme (which handles dark mode)
# ---------------------------------------------------------------------------

CSS = f"""
/* light mode: tint page background so white cards have contrast */
:root:not(.dark) {{
  --body-background-fill: #e4e6df !important;
  --background-fill-primary: #e4e6df !important;
  --background-fill-secondary: #ffffff !important;
  --block-background-fill: #ffffff !important;
  --panel-background-fill: #ffffff !important;
}}

/* layout */
.gradio-container {{ max-width: 1000px !important; margin: auto; }}
footer {{ display: none !important; }}


/* header */
.app-header {{ text-align:center; padding:28px 0 16px; }}
.app-header h1 {{ font-size:28px; font-weight:800; margin:0; color: var(--body-text-color); }}
.app-header p  {{ margin:8px 0 0; font-size:15px; color: var(--body-text-color); opacity:0.7; }}
.app-header a  {{ color:{GLAUCOUS}; text-decoration:underline; text-underline-offset:3px; }}

/* result card */
.result-card {{
  border-radius: 14px;
  padding: 28px;
  background: var(--background-fill-secondary);
  border: 1px solid var(--border-color-accent);
  box-shadow: 0 2px 8px rgba(0,0,0,.06);
  min-height: 240px;
  display: flex; flex-direction: column; justify-content: center;
}}
.result-card.placeholder {{
  text-align: center; min-height: 320px; align-items: center;
  border: 2px dashed var(--border-color-accent);
  background: transparent;
  box-shadow: none;
}}
.result-card.placeholder p {{
  margin: 0; font-size: 16px; color: var(--body-text-color); opacity:0.6;
}}

/* verdict accent stripe — matches gauge gradient */
.verdict-real        {{ border-left: 5px solid #22c55e; }}
.verdict-likely-real {{ border-left: 5px solid #84cc16; }}
.verdict-likely-ai   {{ border-left: 5px solid #f97316; }}
.verdict-ai          {{ border-left: 5px solid #ef4444; }}

.verdict-real .verdict-label        {{ color: #22c55e; }}
.verdict-likely-real .verdict-label {{ color: #84cc16; }}
.verdict-likely-ai .verdict-label   {{ color: #f97316; }}
.verdict-ai .verdict-label          {{ color: #ef4444; }}

/* verdict text */
.verdict-label {{ font-size:26px; font-weight:800; line-height:1; }}
.verdict-desc  {{ font-size:15px; margin:8px 0 22px; color: var(--body-text-color); opacity:0.8; }}

/* gauge */
.gauge {{ margin-bottom:24px; }}
.gauge-track {{
  height:8px; border-radius:4px; position:relative;
  background: linear-gradient(to right, #ef4444, #f97316, #eab308, #84cc16, #22c55e);
}}
.gauge-marker {{
  position:absolute; top:-6px;
  width:6px; height:20px; border-radius:3px;
  background: var(--body-text-color);
  transform:translateX(-50%);
  box-shadow: 0 1px 4px rgba(0,0,0,.4);
}}
.gauge-labels {{
  display:flex; justify-content:space-between;
  font-size:12px; margin-top:5px;
  color: var(--body-text-color-subdued); text-transform:uppercase; letter-spacing:.04em;
}}

/* stats */
.stats-row {{ display:flex; gap:10px; flex-wrap:wrap; }}
.stat {{
  flex:1; min-width:80px; text-align:center;
  padding:12px 8px; border-radius:10px;
  background: var(--background-fill-primary);
  border: 1px solid var(--border-color-accent);
  box-shadow: 0 1px 3px rgba(0,0,0,.04);
}}
.stat-value {{
  font-size: clamp(14px, 3.5vw, 20px);
  font-weight:700; font-variant-numeric:tabular-nums;
  color: var(--body-text-color);
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}}
.stat-label {{
  font-size:12px; text-transform:uppercase; letter-spacing:.05em;
  color: var(--body-text-color-subdued); margin-top:2px;
}}

/* info box */
.info-box {{
  font-size:14px; line-height:1.7;
  padding:18px 22px; border-radius:10px; margin-top:12px;
  background: var(--background-fill-secondary);
  color: var(--body-text-color);
  border: 1px solid var(--border-color-accent);
  box-shadow: 0 2px 8px rgba(0,0,0,.06);
  opacity: 0.85;
}}
.info-box b {{ opacity:1; }}
.info-box .steps {{
  margin:8px 0 16px; padding-left:20px; line-height:1.8;
}}
.info-box .steps li {{ margin-bottom:4px; }}
.info-box .interpret-table {{
  width:100%; border-collapse:collapse; margin:8px 0 12px;
}}
.info-box .interpret-table td {{
  padding:8px 10px; border-bottom:1px solid var(--border-color-accent); vertical-align:top;
  font-size:14px;
}}
.info-box .interpret-table tr:last-child td {{ border-bottom:none; }}
.info-box .interpret-table td:first-child {{ white-space:nowrap; width:170px; }}
.info-box .note {{
  margin:10px 0 0; font-size:13px; font-style:italic; opacity:0.7;
}}

/* attribution */
.attribution-section {{
  margin-top: 20px;
  padding-top: 18px;
  border-top: 1px solid var(--border-color-accent);
}}
.attr-header {{
  font-size: 16px; font-weight: 700;
  color: var(--body-text-color);
  margin-bottom: 6px;
}}
.attr-predicted {{
  font-size: 14px;
  color: var(--body-text-color); opacity: 0.85;
  margin-bottom: 14px;
}}
.attr-chart {{ display: flex; flex-direction: column; gap: 6px; }}
.attr-row {{
  display: flex; align-items: center; gap: 8px;
}}
.attr-name {{
  width: 140px; min-width: 140px;
  font-size: 12px; text-align: right;
  color: var(--body-text-color-subdued);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.attr-track {{
  flex: 1; height: 14px; border-radius: 7px;
  background: var(--background-fill-primary);
  border: 1px solid var(--border-color-accent);
  overflow: hidden;
}}
.attr-fill {{
  height: 100%; border-radius: 7px;
  background: {GLAUCOUS}; opacity: 0.5;
  transition: width 0.4s ease;
}}
.attr-fill.attr-bar-best {{
  background: #ef4444; opacity: 0.9;
}}
.attr-pct {{
  width: 48px; min-width: 48px;
  font-size: 12px; font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--body-text-color-subdued);
}}
.attr-label-best {{
  color: var(--body-text-color) !important;
  font-weight: 700 !important;
}}

/* button */
.analyze-btn {{
  background: {GLAUCOUS} !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
}}
.analyze-btn:hover {{ background: #4a6da6 !important; }}

/* image input elevation */
#img-input {{
  box-shadow: 0 8px 32px rgba(92,128,188,.35) !important;
  border: 1px solid var(--border-color-accent) !important;
}}
:root:not(.dark) #img-input {{
  box-shadow: 0 8px 32px rgba(0,0,0,.25) !important;
}}

"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_demo(device: str = "cpu") -> gr.Blocks:
    print(f"Loading FSD detector on device={device} ...")
    try:
        detector = FSDDetector.load(device=device, attribution=True)
        has_attribution = True
        print("Detector ready (with attribution).")
    except Exception:
        detector = FSDDetector.load(device=device)
        has_attribution = False
        print("Detector ready (detection only, attribution weights not found).")

    def analyze(image):
        if image is None:
            return PLACEHOLDER_HTML
        try:
            pil_img = Image.open(image)
        except Exception as exc:
            return f'<div class="result-card placeholder"><p>Could not open image: {exc}</p></div>'
        result = detector.score(pil_img)
        attr_result = None
        if has_attribution and result.is_fake:
            attr_result = detector.attribute(pil_img)
        return build_result_html(result, attr_result)

    with gr.Blocks(title="FSD - AI Image Detector") as demo:

        gr.HTML("""
        <div class="app-header">
          <h1>Forensic Self-Descriptions</h1>
          <p>
            Zero-shot AI-generated image detection &amp; source attribution &mdash;
            trained only on real photos, generalizes to any generator &mdash;
            <a href="https://arxiv.org/abs/2503.21003" target="_blank">CVPR 2025</a>
          </p>
        </div>
        """)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="filepath",
                    sources=["upload", "clipboard"],
                    label="Input Image",
                    height=360,
                    elem_id="img-input",
                    format="png",
                )
                analyze_btn = gr.Button("Analyze", variant="primary", elem_classes=["analyze-btn"])

            with gr.Column(scale=1):
                result_output = gr.HTML(value=PLACEHOLDER_HTML)

        gr.HTML("""
        <div class="info-box">
          <b>How it works</b>
          <ol class="steps">
            <li><b>Forensic Residual Extraction</b> &mdash; Learned prediction-error filters
                capture subtle pixel-level traces that differ between real cameras and AI generators.</li>
            <li><b>Self-Description Computation</b> &mdash; Multi-scale patch analysis produces
                a compact 960-dimensional forensic fingerprint of the image.</li>
            <li><b>Statistical Scoring</b> &mdash; A Gaussian Mixture Model, trained exclusively
                on real photographs, measures how well the fingerprint matches natural image statistics.</li>
            <li><b>Z-Score &amp; Decision</b> &mdash; The score is normalized into a z-score
                (standard deviations from the real-image mean). More negative = less like a real photo.</li>
            <li><b>Source Attribution</b> &mdash; If an image is flagged as AI-generated,
                per-source statistical models identify which generator most likely produced it.</li>
          </ol>

          <b>Interpreting the results</b>
          <table class="interpret-table">
            <tr><td><b>Z-score above &minus;1</b></td>
                <td>Forensic signature matches real photographs &mdash; very likely real.</td></tr>
            <tr><td><b>Z-score &minus;1 to &minus;2</b></td>
                <td>Still within the real range &mdash; likely a genuine photograph.</td></tr>
            <tr><td><b>Z-score &minus;2 to &minus;3</b></td>
                <td>Crosses the detection threshold &mdash; likely AI-generated.</td></tr>
            <tr><td><b>Z-score below &minus;3</b></td>
                <td>Far beyond the threshold &mdash; very likely AI-generated.</td></tr>
          </table>

          <p class="note">
            This detector is trained only on real photographs and has never seen AI-generated images.
            It generalizes to new generators zero-shot, but accuracy may vary with heavy JPEG
            compression, screenshots, or other post-processing.
          </p>
        </div>
        """)

        image_input.change(fn=analyze, inputs=[image_input], outputs=[result_output])
        analyze_btn.click(fn=analyze, inputs=[image_input], outputs=[result_output])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSD Gradio Demo")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = create_demo(device=args.device)
    theme = gr.themes.Soft(
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    )
    # Force dark mode unless user explicitly overrides with ?__theme=light
    js_dark = """() => {
        if (!window.location.search.includes('__theme=light')) {
            document.querySelector('body').classList.toggle('dark', true);
        }
    }"""
    demo.launch(share=args.share, server_port=args.port, show_error=True, theme=theme, css=CSS, js=js_dark)
