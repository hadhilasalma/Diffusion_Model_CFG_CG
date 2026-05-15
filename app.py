"""
Streamlit demo — MNIST Diffusion Model (CFG vs Classifier Guidance)
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models.unet import SimpleUNet
from models.classifier import SimpleClassifier
from diffusion.scheduler_ddpm import DDPMScheduler

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Diffusion Model Demo", page_icon="✦", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHOD_CFG   = "Classifier-Free Guidance"
METHOD_CG    = "Classifier Guidance"
METHOD_UNCOND = "Unconditional"

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f5f0ff;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

/* ── Header ── */
.hero {
    background: linear-gradient(135deg, #e8d5ff 0%, #ffd6e8 50%, #d6f0ff 100%);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "✦";
    position: absolute;
    right: 2.5rem; top: 1.2rem;
    font-size: 4rem;
    opacity: 0.12;
    color: #7c3aed;
}
.hero-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: #2d1b69;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 0.82rem;
    color: #7c6fa0;
    margin: 0;
    letter-spacing: 0.3px;
}
.device-pill {
    display: inline-block;
    background: rgba(124,58,237,0.12);
    color: #7c3aed;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-top: 0.7rem;
    font-family: 'DM Mono', monospace;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0e8ff 0%, #fce8f3 100%);
    border-right: 1px solid #e8d5ff;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }
.sidebar-heading {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #9b72cf;
    margin-bottom: 1rem;
}

/* ── Generate button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #a78bfa, #f472b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1.2rem !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 15px rgba(167,139,250,0.35) !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(167,139,250,0.45) !important;
}

/* ── Cards ── */
.card {
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 16px rgba(124,58,237,0.07);
    border: 1px solid #f0e8ff;
}
.card-label {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.card-body {
    font-size: 0.83rem;
    color: #374151;
    line-height: 1.7;
}

/* ── Formula block ── */
.formula {
    background: #fef9ee;
    border: 1px solid #fde68a;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.55rem 1rem;
    margin: 0.6rem 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.79rem;
    color: #92400e;
    letter-spacing: 0.3px;
}

/* ── Stat row ── */
.stat-row {
    display: flex;
    gap: 0.7rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.stat-pill {
    background: white;
    border: 1px solid #e8d5ff;
    border-radius: 10px;
    padding: 0.45rem 0.9rem;
    box-shadow: 0 1px 6px rgba(124,58,237,0.06);
}
.stat-k { font-size: 0.58rem; color: #9b72cf; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; }
.stat-v { font-size: 0.95rem; font-weight: 700; color: #2d1b69; margin-top: 1px; }

/* ── Method overview cards ── */
.method-card {
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
    border: 1.5px solid transparent;
}
.method-cfg  { background: #f3eeff; border-color: #c4b5fd; }
.method-cg   { background: #ecfdf5; border-color: #6ee7b7; }
.method-uncond { background: #fff7ed; border-color: #fcd34d; }
.method-tag {
    display: inline-block;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.tag-cfg    { background: #7c3aed; color: white; }
.tag-cg     { background: #059669; color: white; }
.tag-uncond { background: #d97706; color: white; }
.method-name { font-size: 0.92rem; font-weight: 700; color: #1f2937; margin-bottom: 0.3rem; }
.method-desc { font-size: 0.78rem; color: #6b7280; line-height: 1.6; }

/* ── Placeholder ── */
.placeholder {
    background: linear-gradient(135deg, #fdf4ff, #f0fdf4, #fefce8);
    border: 2px dashed #d8b4fe;
    border-radius: 18px;
    padding: 3.5rem 2rem;
    text-align: center;
}
.placeholder-icon { font-size: 3.5rem; margin-bottom: 0.8rem; }
.placeholder-text { color: #9b72cf; font-size: 0.85rem; font-weight: 500; }

/* ── Step badge ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px; height: 22px;
    border-radius: 50%;
    background: linear-gradient(135deg, #a78bfa, #f472b6);
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    margin-right: 0.5rem;
}

/* ── code inline ── */
code {
    background: #f0e8ff;
    color: #7c3aed;
    border-radius: 4px;
    padding: 1px 5px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8em;
}

/* ── progress bar colour ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #a78bfa, #f472b6);
}
</style>
""", unsafe_allow_html=True)


# ── Model loaders ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CFG model…")
def load_cfg_model():
    m = SimpleUNet(in_channels=1, out_channels=1, channels=64, num_classes=10)
    c = torch.load("checkpoints/model_final.pth", map_location=DEVICE)
    m.load_state_dict(c["model_state_dict"] if isinstance(c, dict) else c)
    return m.to(DEVICE).eval()

@st.cache_resource(show_spinner="Loading CG model…")
def load_cg_model():
    m = SimpleUNet(in_channels=1, out_channels=1, channels=64, num_classes=10)
    c = torch.load("checkpoints_cg/model_final.pth", map_location=DEVICE)
    m.load_state_dict(c["model_state_dict"] if isinstance(c, dict) else c)
    return m.to(DEVICE).eval()

@st.cache_resource(show_spinner="Loading classifier…")
def load_classifier():
    clf = SimpleClassifier(num_classes=10)
    clf.load_state_dict(torch.load("checkpoints/classifier_final.pth", map_location=DEVICE))
    return clf.to(DEVICE).eval()

@st.cache_resource
def load_scheduler():
    return DDPMScheduler(num_timesteps=1000).to(DEVICE)


# ── Sampling ───────────────────────────────────────────────────────────────────
def sample_cfg(model, scheduler, num_samples, class_labels, guidance_scale, num_steps, pb):
    x = torch.randn(num_samples, 1, 28, 28, device=DEVICE)
    null = torch.full((num_samples,), 10, dtype=torch.long, device=DEVICE)
    steps = list(range(num_steps - 1, -1, -1))
    with torch.no_grad():
        for i, t in enumerate(steps):
            tb  = torch.full((num_samples,), t, dtype=torch.long, device=DEVICE)
            e_u = model(x, tb, null)
            if class_labels is not None and guidance_scale != 1.0:
                e_c = model(x, tb, class_labels)
                e   = e_u + guidance_scale * (e_c - e_u)
            else:
                e = e_u
            a, ab = scheduler.alphas[t], scheduler.alphas_cumprod[t]
            x = (x - (1 - a) / torch.sqrt(1 - ab) * e) / torch.sqrt(a)
            if t > 0:
                x = x + torch.sqrt(scheduler.posterior_variance[t]) * torch.randn_like(x)
            if i % max(1, len(steps) // 40) == 0 or i == len(steps) - 1:
                pb.progress((i + 1) / len(steps), text=f"Denoising… step {i+1}/{len(steps)}")
    return x


def sample_cg(model, classifier, scheduler, num_samples, class_labels, guidance_scale, num_steps, pb):
    x = torch.randn(num_samples, 1, 28, 28, device=DEVICE)
    null = torch.full((num_samples,), 10, dtype=torch.long, device=DEVICE)
    steps = list(range(num_steps - 1, -1, -1))
    for i, t in enumerate(steps):
        tb = torch.full((num_samples,), t, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            e = model(x, tb, null)
        xg = x.clone().requires_grad_(True)
        F.log_softmax(classifier(xg, tb), dim=1)[range(num_samples), class_labels].sum().backward()
        grad = torch.clamp(xg.grad, -10.0, 10.0)
        with torch.no_grad():
            a, ab = scheduler.alphas[t], scheduler.alphas_cumprod[t]
            e = e - guidance_scale * torch.sqrt(1 - ab) * grad
            x = (x - (1 - a) / torch.sqrt(1 - ab) * e) / torch.sqrt(a)
            if t > 0:
                x = x + torch.sqrt(scheduler.posterior_variance[t]) * torch.randn_like(x)
        if i % max(1, len(steps) // 40) == 0 or i == len(steps) - 1:
            pb.progress((i + 1) / len(steps), text=f"Denoising… step {i+1}/{len(steps)}")
    return x


# ── Image grid ─────────────────────────────────────────────────────────────────
def render_grid(samples, method):
    n    = samples.shape[0]
    cols = 3 if n == 9 else min(n, 4)
    rows = (n + cols - 1) // cols

    bg_colors = {
        METHOD_CFG:    "#faf5ff",
        METHOD_CG:     "#f0fdf4",
        METHOD_UNCOND: "#fffbeb",
    }
    bg = bg_colors.get(method, "#fafafa")

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.7))
    fig.patch.set_facecolor(bg)
    axes = np.array(axes).reshape(rows, cols)

    for r in range(rows):
        for c in range(cols):
            ax  = axes[r][c]
            idx = r * cols + c
            if idx < n:
                img = ((samples[idx].cpu().squeeze().numpy() + 1) / 2).clip(0, 1)
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_facecolor(bg)
            ax.axis("off")

    plt.tight_layout(pad=0.4)
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-heading">✦ Settings</p>', unsafe_allow_html=True)

    method = st.selectbox(
        "Generation method",
        [METHOD_CFG, METHOD_CG, METHOD_UNCOND],
    )

    digit = st.selectbox(
        "Target digit",
        list(range(10)),
        index=3,
        format_func=lambda d: f"Digit  {d}",
        disabled=(method == METHOD_UNCOND),
    )

    if method == METHOD_CFG:
        scale = st.slider("Guidance scale  s", 1.0, 15.0, 7.5, 0.5)
    elif method == METHOD_CG:
        scale = st.slider("Guidance scale  s", 0.1, 3.0, 0.5, 0.1)
    else:
        scale = 1.0
        st.caption("No guidance scale for unconditional mode.")

    n_samples = st.select_slider("Samples to generate", [4, 9, 16], value=9)

    n_steps = st.radio(
        "Denoising steps",
        [200, 1000],
        format_func=lambda x: f"{x} steps  {'⚡ fast' if x == 200 else '✨ full quality'}",
        index=0,
    )

    st.markdown("<br/>", unsafe_allow_html=True)
    generate = st.button("✦  Generate Images", use_container_width=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:white;border-radius:10px;padding:0.7rem 1rem;border:1px solid #e8d5ff;">'
        f'<p style="margin:0;font-size:0.65rem;color:#9b72cf;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">Device</p>'
        f'<p style="margin:0;font-size:0.85rem;font-weight:700;color:#2d1b69;font-family:\'DM Mono\',monospace">{str(DEVICE).upper()}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">MNIST Diffusion Model Demo</p>
  <p class="hero-sub">
    Denoising Diffusion Probabilistic Models &nbsp;·&nbsp;
    Classifier-Free Guidance &nbsp;·&nbsp; Classifier Guidance &nbsp;·&nbsp;
    Ho et al. 2020 &nbsp;·&nbsp; Dhariwal &amp; Nichol 2021
  </p>
  <span class="device-pill">DDPM · T = 1000 · MNIST 28×28</span>
</div>
""", unsafe_allow_html=True)


# ── Main layout ────────────────────────────────────────────────────────────────
col_out, col_notes = st.columns([3, 2], gap="large")


# ══════════════════ NOTES COLUMN ══════════════════════════════════════════════
with col_notes:

    # ── Method card ──
    if method == METHOD_CFG:
        st.markdown("""
        <div class="card" style="border-left: 4px solid #a78bfa;">
          <div class="card-label" style="color:#7c3aed;">✦ Classifier-Free Guidance (CFG)</div>
          <div class="card-body">
            A <strong>single U-Net</strong> is trained with <strong>10% label dropout</strong>
            — randomly replacing class labels with a null token during training. This teaches
            the model to denoise both <em>conditionally</em> and <em>unconditionally</em>
            from one set of weights.<br/><br/>
            At inference, the model runs <strong>twice per step</strong> and the predictions are blended:
          </div>
          <div class="formula">ε̂ = ε_uncond + s · (ε_cond − ε_uncond)</div>
          <div class="card-body">
            where <code>s</code> is the guidance scale. Larger <code>s</code> pushes
            samples toward the target class but reduces diversity.
          </div>
        </div>
        """, unsafe_allow_html=True)

    elif method == METHOD_CG:
        st.markdown("""
        <div class="card" style="border-left: 4px solid #6ee7b7;">
          <div class="card-label" style="color:#059669;">✦ Classifier Guidance (CG)</div>
          <div class="card-body">
            Uses <strong>two separate models</strong>: an unconditional diffusion U-Net
            and a <strong>noise-aware classifier</strong> trained to identify digits
            even in heavily noisy images.<br/><br/>
            At each denoising step, the classifier gradient steers the trajectory
            toward the target class:
          </div>
          <div class="formula">ε̂ = ε_θ(x_t) − √(1−ᾱ_t) · s · ∇log p(y | x_t)</div>
          <div class="card-body">
            The gradient <code>∇log p(y|x_t)</code> literally points toward
            pixels that increase the target class probability.
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card" style="border-left: 4px solid #fcd34d;">
          <div class="card-label" style="color:#d97706;">✦ Unconditional Generation</div>
          <div class="card-body">
            The CFG model is run with the <strong>null token only</strong> — no class
            label is provided. The model generates random MNIST-style digits from
            pure Gaussian noise without any class conditioning.
          </div>
          <div class="formula">x_{t-1} = (x_t − β_t/√(1−ᾱ_t) · ε_θ) / √α_t + σ_t · z</div>
          <div class="card-body">
            Pure reverse diffusion — each step removes a small amount of noise
            guided only by what the model learned about MNIST digit structure.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Guidance scale interpretation ──
    if method == METHOD_CFG:
        if scale >= 12:
            interp_color, interp_icon, interp_text = "#fce7f3", "⚠️", "Very high — samples will be extremely class-faithful but near-identical. Low diversity, possible artifacts."
        elif scale >= 7:
            interp_color, interp_icon, interp_text = "#f3e8ff", "✓", "Optimal range — strong class fidelity with reasonable diversity. Recommended for MNIST."
        elif scale >= 3:
            interp_color, interp_icon, interp_text = "#ecfdf5", "~", "Moderate — balanced between class accuracy and variety."
        else:
            interp_color, interp_icon, interp_text = "#fff7ed", "↓", "Low — weakly conditioned. Samples may not resemble the target digit."

        st.markdown(
            f'<div class="card" style="background:{interp_color};border-left:4px solid #a78bfa;">'
            f'<div class="card-label" style="color:#7c3aed;">Scale s = {scale}  →  Interpretation</div>'
            f'<div class="card-body">{interp_icon} {interp_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    elif method == METHOD_CG:
        if scale >= 2:
            interp_color, interp_icon, interp_text = "#fce7f3", "⚠️", "Very strong steering — possible over-saturation or checkerboard artifacts."
        elif scale >= 1:
            interp_color, interp_icon, interp_text = "#ecfdf5", "✓", "Strong — high class accuracy, slightly less naturalistic samples."
        elif scale >= 0.3:
            interp_color, interp_icon, interp_text = "#f0fdf4", "~", "Gentle — natural-looking samples with moderate class accuracy. Recommended."
        else:
            interp_color, interp_icon, interp_text = "#fff7ed", "↓", "Minimal guidance — output close to unconditional generation."

        st.markdown(
            f'<div class="card" style="background:{interp_color};border-left:4px solid #6ee7b7;">'
            f'<div class="card-label" style="color:#059669;">Scale s = {scale}  →  Interpretation</div>'
            f'<div class="card-body">{interp_icon} {interp_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Architecture note ──
    st.markdown("""
    <div class="card" style="border-left: 4px solid #f9a8d4;">
      <div class="card-label" style="color:#db2777;">✦ Model Architecture</div>
      <div class="card-body">
        <strong>U-Net</strong> with encoder–decoder skip connections.
        Each block: <code>ResBlock</code> (GroupNorm + SiLU + AdaGN time conditioning)
        + <code>Self-Attention</code> at the bottleneck.<br/><br/>
        Time step <code>t</code> and class label are injected via sinusoidal
        positional encoding → projected into scale &amp; shift for AdaGN.<br/><br/>
        <strong>~3.3M parameters</strong> &nbsp;·&nbsp; Null token = label 10 &nbsp;·&nbsp; Image range [−1, 1]
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Training note ──
    st.markdown("""
    <div class="card" style="border-left: 4px solid #93c5fd;">
      <div class="card-label" style="color:#2563eb;">✦ Training Details</div>
      <div class="card-body">
        Dataset: <strong>MNIST</strong> (60k images, 28×28, 10 classes)<br/>
        Epochs: 20 &nbsp;·&nbsp; Batch: 64 &nbsp;·&nbsp; Adam lr = 1e-4<br/>
        Cosine LR decay &nbsp;·&nbsp; Mixed precision (AMP)<br/>
        CFG dropout: 10% label → null token during training<br/>
        T = 1000 timesteps &nbsp;·&nbsp; Linear β schedule (0.0001 → 0.02)
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════ OUTPUT COLUMN ═════════════════════════════════════════════
with col_out:

    if generate:
        scheduler = load_scheduler()

        method_colors = {
            METHOD_CFG:    ("#7c3aed", "#f3eeff"),
            METHOD_CG:     ("#059669", "#ecfdf5"),
            METHOD_UNCOND: ("#d97706", "#fffbeb"),
        }
        accent, bg = method_colors[method]

        # stat row
        label_str = f"Digit  {digit}" if method != METHOD_UNCOND else "—"
        scale_str = str(scale) if method != METHOD_UNCOND else "—"
        st.markdown(
            f'<div class="stat-row">'
            f'<div class="stat-pill"><div class="stat-k">Method</div><div class="stat-v" style="color:{accent}">{method.split()[0]}</div></div>'
            f'<div class="stat-pill"><div class="stat-k">Target</div><div class="stat-v">{label_str}</div></div>'
            f'<div class="stat-pill"><div class="stat-k">Scale</div><div class="stat-v">{scale_str}</div></div>'
            f'<div class="stat-pill"><div class="stat-k">Steps</div><div class="stat-v">{n_steps}</div></div>'
            f'<div class="stat-pill"><div class="stat-k">Samples</div><div class="stat-v">{n_samples}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        pb = st.progress(0, text="Initialising…")

        if method == METHOD_UNCOND:
            model   = load_cfg_model()
            samples = sample_cfg(model, scheduler, n_samples,
                                 None, 1.0, n_steps, pb)
        elif method == METHOD_CFG:
            model  = load_cfg_model()
            labels = torch.full((n_samples,), digit, dtype=torch.long, device=DEVICE)
            samples = sample_cfg(model, scheduler, n_samples,
                                 labels, scale, n_steps, pb)
        else:
            model      = load_cg_model()
            classifier = load_classifier()
            labels     = torch.full((n_samples,), digit, dtype=torch.long, device=DEVICE)
            samples    = sample_cg(model, classifier, scheduler, n_samples,
                                   labels, scale, n_steps, pb)
        pb.empty()

        fig = render_grid(samples, method)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        caption = (
            f"Generated {n_samples} samples"
            + (f" of digit **{digit}**" if method != METHOD_UNCOND else "")
            + f" · {method} · s = {scale_str} · {n_steps} denoising steps"
        )
        st.caption(caption)

    else:
        # welcome state
        st.markdown("""
        <div class="placeholder">
          <div class="placeholder-icon">✦</div>
          <p class="placeholder-text">Your generated images will appear here</p>
          <p style="color:#c4b5fd;font-size:0.75rem;margin-top:0.3rem;">
            Configure the settings in the sidebar and click Generate
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # method overview cards
        st.markdown('<p style="font-size:0.7rem;font-weight:700;color:#9b72cf;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:0.8rem;">Available Methods</p>', unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card method-cfg">
          <span class="method-tag tag-cfg">CFG</span>
          <div class="method-name">Classifier-Free Guidance</div>
          <div class="method-desc">Single model · 10% label dropout · Two forward passes at inference · Best quality</div>
        </div>
        <div class="method-card method-cg">
          <span class="method-tag tag-cg">CG</span>
          <div class="method-name">Classifier Guidance</div>
          <div class="method-desc">Two models · Gradient-based steering · Explicit guidance mechanism · Dhariwal & Nichol 2021</div>
        </div>
        <div class="method-card method-uncond">
          <span class="method-tag tag-uncond">UNCOND</span>
          <div class="method-name">Unconditional</div>
          <div class="method-desc">No class conditioning · Pure DDPM reverse process · Random digit generation</div>
        </div>
        """, unsafe_allow_html=True)
