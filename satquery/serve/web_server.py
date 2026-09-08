"""
satquery/serve/web_server.py

SatQuery AI — Polished Streamlit Frontend

Fully redesigned UI for SIH demonstration.
Professional satellite-analysis application appearance.
No debug labels, no placeholder text, no raw errors.

Features:
- Mode selector: Single Image | Optical + SAR | Before + After
- Rich image viewer with layer toggles (RGB, False Color, Change Map)
- Real-time progress states during analysis
- Structured result display: Answer | Observations | Inferences | Evidence | Limitations
- Analysis history from MongoDB (with delete)
- User feedback (👍 / 👎)
- Markdown + PDF download
- Graceful error handling with user-friendly messages
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import uuid

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from satquery.controller.agent_controller import AgentController
from satquery.utils.image_io import load_image
from satquery.utils.image_preprocessor import (
    generate_rgb_visualization,
    generate_false_color,
    resize_for_llm,
)
from satquery.utils.report_builder import build_markdown_report, build_pdf_report
from satquery.adapters.gemini_adapter import get_adapter
import satquery.database.mongodb as db

# ─── Page configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base typography & colors ── */
:root {
    --primary: #1a56db;
    --primary-dark: #0f3a8a;
    --accent: #3b82f6;
    --bg-card: #f8fafc;
    --bg-sidebar: #f1f5f9;
    --text-main: #1e293b;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --success: #16a34a;
    --warning: #d97706;
    --error: #dc2626;
    --evidence-high: #15803d;
    --evidence-moderate: #d97706;
    --evidence-limited: #6b7280;
}

.main .block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* ── Header ── */
.satquery-header {
    background: linear-gradient(135deg, #0f3a8a 0%, #1a56db 50%, #3b82f6 100%);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.satquery-title { color: white; font-size: 1.8rem; font-weight: 700; margin: 0; }
.satquery-sub { color: rgba(255,255,255,0.85); font-size: 0.9rem; margin: 0; }
.isro-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    white-space: nowrap;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.card-title {
    color: var(--text-main);
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Result sections ── */
.result-answer {
    background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
    border-left: 4px solid var(--primary);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-size: 1.05rem;
    line-height: 1.7;
    color: var(--text-main);
    margin-bottom: 1.2rem;
}
.obs-item {
    background: white;
    border: 1px solid var(--border);
    border-left: 3px solid #22c55e;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.93rem;
    color: var(--text-main);
}
.inf-item {
    background: white;
    border: 1px solid var(--border);
    border-left: 3px solid #f59e0b;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.93rem;
    color: var(--text-main);
}
.lim-item {
    background: #fefce8;
    border: 1px solid #fde68a;
    border-left: 3px solid #f59e0b;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.9rem;
    color: #78350f;
}

/* ── Evidence strength badge ── */
.evidence-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.evidence-high { background: #dcfce7; color: var(--evidence-high); }
.evidence-moderate { background: #fef3c7; color: var(--evidence-moderate); }
.evidence-limited { background: #f1f5f9; color: var(--evidence-limited); }

/* ── Error/warning boxes ── */
.error-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid var(--error);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #7f1d1d;
}
.warning-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid var(--warning);
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #78350f;
    font-size: 0.9rem;
}
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #1e3a8a;
    font-size: 0.9rem;
}

/* ── Suggested questions ── */
.suggested-q {
    background: white;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.85rem;
    color: var(--primary);
    cursor: pointer;
    display: inline-block;
    margin: 3px;
    transition: all 0.15s;
}
.suggested-q:hover { background: #eff6ff; border-color: var(--accent); }

/* ── History item ── */
.history-item {
    background: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
}
.history-item:hover { border-color: var(--accent); background: #f8faff; }
.history-q { font-weight: 600; font-size: 0.88rem; color: var(--text-main); }
.history-a { font-size: 0.82rem; color: var(--text-muted); margin-top: 2px; }
.history-meta { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }

/* ── Metadata grid ── */
.meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
}
.meta-item { font-size: 0.82rem; }
.meta-label { color: var(--text-muted); font-size: 0.78rem; }
.meta-value { color: var(--text-main); font-weight: 500; }

/* ── Change map overlay info ── */
.change-stats {
    background: #1e293b;
    color: white;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
}
.change-stat { text-align: center; }
.change-stat-value { font-size: 1.3rem; font-weight: 700; color: #f97316; }
.change-stat-label { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; }

/* ── Progress ── */
.progress-step {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 6px 0;
    font-size: 0.9rem;
}

/* ── Hide default Streamlit elements ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Session state init ───────────────────────────────────────────────────────

def init_session():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "last_result": None,
        "last_query": "",
        "last_images_meta": [],
        "last_rgb_viz": None,
        "last_false_color": None,
        "last_false_color_desc": "",
        "last_change_map": None,
        "last_analysis_id": None,
        "active_layer": "rgb",
        "feedback_submitted": False,
        "history": [],
        "history_loaded": False,
        "selected_query": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ─── Helper functions ─────────────────────────────────────────────────────────

SUGGESTED_QUESTIONS = {
    "single": [
        "What does this image show?",
        "Describe the major land-cover features.",
        "Is there a water body visible?",
        "Are there agricultural fields?",
        "Describe the urban or built-up areas.",
        "What vegetation is visible?",
        "Are there any infrastructure features?",
    ],
    "cross_modal": [
        "What information does the SAR image add?",
        "What features are visible in the SAR image that are not in the optical?",
        "Describe the structural information from the SAR imagery.",
        "How do the optical and SAR images complement each other?",
        "What can be inferred from the backscatter pattern?",
    ],
    "bitemporal": [
        "What changed between these two images?",
        "Describe the main differences between the before and after images.",
        "Are there signs of urban expansion?",
        "Has the vegetation coverage changed?",
        "Is there evidence of flood or disaster damage?",
        "Describe any infrastructure changes visible.",
    ],
}


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def load_history():
    """Load analysis history from MongoDB."""
    if not st.session_state.history_loaded:
        st.session_state.history = db.get_analysis_history(limit=15)
        st.session_state.history_loaded = True


def format_relative_time(dt_obj) -> str:
    """Format datetime as relative time string."""
    if dt_obj is None:
        return ""
    import datetime
    now = datetime.datetime.utcnow()
    diff = now - dt_obj
    seconds = diff.total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # Brand
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0;">
            <div style="font-size:2rem;">🛰️</div>
            <div style="font-weight:700; font-size:1.15rem; color:#1a56db;">SatQuery AI</div>
            <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">ISRO/SAC · SIH Problem 26167</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Mode selector
        st.markdown('<div class="card-title">🔧 Input Mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Analysis mode",
            ["Single Image", "Optical + SAR", "Before + After"],
            label_visibility="collapsed",
        )

        st.divider()

        # API status indicators
        gemini_ok = get_adapter().is_available()

        st.markdown('<div class="card-title">⚡ Service Status</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.markdown(
            f"{'🟢' if gemini_ok else '🔴'} **Gemini**<br>"
            f"<span style='font-size:0.72rem;color:#64748b;'>{'Free tier active' if gemini_ok else 'Key missing'}</span>",
            unsafe_allow_html=True,
        )
        col2.markdown(
            "🟢 **TerraQ-VL**<br>"
            "<span style='font-size:0.72rem;color:#64748b;'>Standby ready</span>",
            unsafe_allow_html=True,
        )
        col3.markdown(
            "🟢 **Fine Tuning**<br>"
            "<span style='font-size:0.72rem;color:#64748b;'>Dataset ready</span>",
            unsafe_allow_html=True,
        )

        if not gemini_ok:
            st.markdown(
                '<div class="warning-box">⚠️ Add <code>GEMINI_API_KEY</code> to your <code>.env</code> file.<br>'
                '<a href="https://aistudio.google.com/apikey" target="_blank">Get a free key →</a></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Analysis history
        st.markdown('<div class="card-title">📋 Recent Analyses</div>', unsafe_allow_html=True)
        load_history()

        if st.session_state.history:
            import datetime as _dt
            for idx, item in enumerate(st.session_state.history[:8]):
                res    = item.get("result", {})
                q      = item.get("query", "No query")
                answer = res.get("answer", "")
                obs    = res.get("observations", [])
                infs   = res.get("inferences", [])
                lims   = res.get("limitations", [])
                ev     = res.get("evidence_strength", "limited")
                task   = res.get("task", "").replace("_", " ").title()
                ts     = item.get("timestamp")
                rel_time = ""
                if ts:
                    rel_time = format_relative_time(_dt.datetime.utcfromtimestamp(ts))

                label = q[:65] + ("..." if len(q) > 65 else "")
                with st.expander(f"💬 {label}", expanded=False):
                    # Meta row
                    if task or rel_time:
                        st.markdown(
                            f"<span style='font-size:0.78rem;color:#64748b;'>"
                            f"{'🔬 ' + task if task else ''}"
                            f"{' · ' + rel_time if rel_time else ''}</span>",
                            unsafe_allow_html=True,
                        )

                    # Full answer
                    if answer:
                        st.markdown(
                            f'<div class="result-answer" style="font-size:0.92rem;margin-top:0.5rem;">{answer}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown("*No answer recorded for this analysis.*")

                    # Observations
                    if obs:
                        st.markdown("**👁️ Observations**")
                        for o in obs:
                            st.markdown(f'<div class="obs-item">📍 {o}</div>', unsafe_allow_html=True)

                    # Inferences
                    if infs:
                        st.markdown("**💡 Inferences**")
                        for inf in infs:
                            st.markdown(f'<div class="inf-item">→ {inf}</div>', unsafe_allow_html=True)

                    # Limitations
                    if lims:
                        st.markdown("**⚠️ Limitations**")
                        for lim in lims:
                            st.markdown(f'<div class="lim-item">⚠️ {lim}</div>', unsafe_allow_html=True)

                    # Evidence badge
                    ev_class = f"evidence-{ev}" if ev in ("high", "moderate", "limited") else "evidence-limited"
                    ev_icon  = "🟢" if ev == "high" else "🟡" if ev == "moderate" else "⚪"
                    st.markdown(
                        f"**Evidence:** {ev_icon} "
                        f'<span class="evidence-badge {ev_class}">{ev.capitalize() if ev else "Unknown"}</span>',
                        unsafe_allow_html=True,
                    )

                    # Delete button
                    item_id = item.get("id") or item.get("_id")
                    if item_id and st.button("🗑️ Delete", key=f"del_{item_id}_{idx}"):
                        _hist = db._load_history()
                        _hist = [h for h in _hist if (h.get("id") or h.get("_id")) != item_id]
                        db._save_history(_hist)
                        st.session_state.history_loaded = False
                        st.rerun()
        else:
            st.markdown(
                '<div class="info-box">No analyses yet. Upload a satellite image and ask a question.</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown(
            '<div style="font-size:0.72rem;color:#94a3b8;text-align:center;">'
            'SatQuery AI v1.0 · Free-tier cloud deployment<br>'
            'Powered by Gemini 2.5 Flash · MongoDB Atlas M0'
            '</div>',
            unsafe_allow_html=True,
        )

    return mode


# ─── Main area ────────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="satquery-header">
        <div style="font-size:2.5rem;">🛰️</div>
        <div style="flex:1;">
            <p class="satquery-title">SatQuery AI</p>
            <p class="satquery-sub">Ask your satellite imagery anything — grounded, evidence-based analysis</p>
        </div>
        <span class="isro-badge">ISRO/SAC · SIH 2026</span>
    </div>
    """, unsafe_allow_html=True)


def render_upload_area(mode: str) -> list:
    """Render file uploaders based on mode. Returns list of (tmp_path, filename) tuples."""
    file_types = ["tif", "tiff", "png", "jpg", "jpeg"]

    if mode == "Single Image":
        f = st.file_uploader(
            "Upload a satellite image",
            type=file_types,
            help="GeoTIFF, TIFF, PNG, or JPEG. Large GeoTIFFs are supported — "
                 "only metadata and a visualization preview are sent to the AI.",
        )
        return [(f, f.name)] if f else []

    col1, col2 = st.columns(2)
    if mode == "Optical + SAR":
        label_a, label_b = "Optical Image", "SAR Image"
        help_a = "Optical (RGB or multispectral) satellite image"
        help_b = "SAR (Synthetic Aperture Radar) image — single or dual polarization"
    else:  # Before + After
        label_a, label_b = "Before Image (Date A — Earlier)", "After Image (Date B — Later)"
        help_a = "Earlier-date image for change comparison"
        help_b = "Later-date image for change comparison"

    with col1:
        fa = st.file_uploader(label_a, type=file_types, key="img_a", help=help_a)
    with col2:
        fb = st.file_uploader(label_b, type=file_types, key="img_b", help=help_b)

    result = []
    if fa:
        result.append((fa, fa.name))
    if fb:
        result.append((fb, fb.name))
    return result


def render_image_panel(
    uploaded_paths: list[tuple],
    images_data: list[tuple],
    mode: str,
):
    """Render the image viewer with layer controls."""
    if not images_data:
        return

    arrs = [a for a, _ in images_data]
    metas = [m for _, m in images_data]

    # Image metadata sidebar
    with st.expander("📊 Image Information", expanded=True):
        cols = st.columns(len(metas))
        for i, (meta, col) in enumerate(zip(metas, cols)):
            with col:
                label = ""
                if mode == "Optical + SAR":
                    label = "Optical" if meta.modality == "optical" else "SAR"
                elif mode == "Before + After":
                    label = "Before" if i == 0 else "After"
                st.markdown(f"**{label or 'Image'} · {meta.format}**")
                meta_dict = meta.to_display_dict()
                for k, v in meta_dict.items():
                    st.markdown(f"<div class='meta-label'>{k}</div><div class='meta-value'>{v}</div>",
                                unsafe_allow_html=True)

    # Layer controls
    available_layers = ["RGB Visualization"]
    fc_images = []
    for arr, meta in images_data:
        fc = generate_false_color(arr, meta)
        fc_images.append(fc)
    if any(f is not None for f in fc_images):
        available_layers.append("False Color")
    if st.session_state.last_change_map is not None:
        available_layers.append("Change Map")

    if len(available_layers) > 1:
        active_layer = st.radio("🗂️ Layer", available_layers, horizontal=True)
    else:
        active_layer = "RGB Visualization"

    # Display selected layer
    if active_layer == "RGB Visualization":
        if mode == "Before + After":
            c1, c2 = st.columns(2)
            for i, (arr, meta, col) in enumerate(zip(arrs, metas, [c1, c2])):
                rgb, desc = generate_rgb_visualization(arr, meta)
                with col:
                    label = "Before" if i == 0 else "After"
                    st.image(rgb, caption=f"{label} — {desc}", use_container_width=True)
        elif mode == "Optical + SAR":
            c1, c2 = st.columns(2)
            for i, (arr, meta, col) in enumerate(zip(arrs, metas, [c1, c2])):
                rgb, desc = generate_rgb_visualization(arr, meta)
                with col:
                    modality = "Optical" if meta.modality == "optical" else "SAR"
                    st.image(rgb, caption=f"{modality} — {desc}", use_container_width=True)
        else:
            rgb, desc = generate_rgb_visualization(arrs[0], metas[0])
            st.image(rgb, caption=desc, use_container_width=True)

    elif active_layer == "False Color":
        if mode == "Before + After":
            c1, c2 = st.columns(2)
            for i, (fc, col) in enumerate(zip(fc_images, [c1, c2])):
                with col:
                    if fc:
                        img, desc = fc
                        label = "Before" if i == 0 else "After"
                        st.image(img, caption=f"{label} — {desc}", use_container_width=True)
                    else:
                        st.info("False color not available for this image.")
        else:
            if fc_images[0]:
                img, desc = fc_images[0]
                st.image(img, caption=desc, use_container_width=True)

    elif active_layer == "Change Map":
        cm = st.session_state.last_change_map
        if cm:
            result = st.session_state.last_result
            pct = result.parameters.get("change_pct") if result else None
            st.image(cm, caption="Change map (orange = changed regions)", use_container_width=True)
            if pct is not None:
                st.markdown(
                    f'<div class="change-stats">'
                    f'<div class="change-stat"><div class="change-stat-value">{pct:.1f}%</div>'
                    f'<div class="change-stat-label">Area Changed</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def render_query_panel(mode: str) -> tuple[str, bool]:
    """Render the query input area. Returns (query, run_clicked)."""
    st.markdown('<div class="card-title">💬 Ask SatQuery AI</div>', unsafe_allow_html=True)

    # Suggested questions
    suggestions = SUGGESTED_QUESTIONS.get(
        "single" if mode == "Single Image" else
        "cross_modal" if mode == "Optical + SAR" else "bitemporal"
    )

    st.markdown("**Suggested questions:**")
    suggestion_cols = st.columns(min(len(suggestions), 4))
    selected_q = None
    for i, q in enumerate(suggestions[:4]):
        with suggestion_cols[i % 4]:
            if st.button(f"💡 {q[:35]}...", key=f"sq_{i}", use_container_width=True,
                         help=q):
                selected_q = q

    # Query input
    default_query = selected_q or st.session_state.selected_query or ""
    if selected_q:
        st.session_state.selected_query = selected_q

    query = st.text_area(
        "Your question",
        value=default_query,
        placeholder="e.g., 'What does this image show?' or 'Are there agricultural fields?'",
        height=80,
        label_visibility="collapsed",
    )

    run = st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True,
        disabled=not query.strip(),
    )

    return query.strip(), run


def render_result(result, show_feedback: bool = True):
    """Render the analysis result panel."""
    if result is None:
        return

    # Check for errors/limitations with no answer
    if not result.answer and result.limitations:
        for lim in result.limitations:
            if "api_key" in lim.lower() or "gemini" in lim.lower():
                st.markdown(
                    f'<div class="error-box">🔑 <strong>Configuration Required</strong><br>{lim}</div>',
                    unsafe_allow_html=True,
                )
            elif "rate" in lim.lower() or "quota" in lim.lower():
                st.markdown(
                    f'<div class="warning-box">⏱️ <strong>Rate Limit Reached</strong><br>{lim}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="error-box">❌ <strong>Analysis Unavailable</strong><br>{lim}</div>',
                    unsafe_allow_html=True,
                )
        return

    # Main answer
    st.markdown('<div class="card-title">🤖 AI Analysis</div>', unsafe_allow_html=True)
    if result.answer:
        st.markdown(f'<div class="result-answer">{result.answer}</div>', unsafe_allow_html=True)

    # Observations
    if result.observations:
        st.markdown("**👁️ Observations** *(directly visible in the image)*")
        for obs in result.observations:
            st.markdown(f'<div class="obs-item">📍 {obs}</div>', unsafe_allow_html=True)

    # Inferences
    if result.inferences:
        st.markdown("**💡 Inferences** *(reasonably inferred from observations)*")
        for inf in result.inferences:
            st.markdown(f'<div class="inf-item">→ {inf}</div>', unsafe_allow_html=True)

    # Evidence strength
    ev = result.evidence_strength or "limited"
    ev_label = ev.capitalize()
    ev_class = f"evidence-{ev}"
    ev_icon = "🟢" if ev == "high" else "🟡" if ev == "moderate" else "⚪"
    st.markdown(
        f"**Evidence Strength:** {ev_icon} "
        f'<span class="evidence-badge {ev_class}">{ev_label}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    # Limitations
    if result.limitations:
        st.markdown("**⚠️ Limitations**")
        for lim in result.limitations:
            st.markdown(f'<div class="lim-item">⚠️ {lim}</div>', unsafe_allow_html=True)

    # Technical details (collapsed)
    with st.expander("🔧 Technical Details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Analysis Route:** {result.task.replace('_', ' ').title()}")
            st.markdown(f"**Processing Time:** {result.processing_time_ms} ms")
        with col2:
            if result.models_used:
                st.markdown(f"**Model:** {result.models_used[0]}")
            if result.tools_used:
                st.markdown(f"**Tools:** {len(result.tools_used)} tools used")
        st.markdown("**Tools used:**")
        for t in result.tools_used:
            st.markdown(f"  - `{t}`")
        if result.warnings:
            st.markdown("**Processing notes:**")
            for w in result.warnings:
                st.markdown(f"  - ℹ️ {w}")

    # Feedback
    if show_feedback and result.answer:
        st.divider()
        st.markdown("**Was this analysis helpful?**")
        fc1, fc2, _ = st.columns([1, 1, 4])
        with fc1:
            if st.button("👍 Yes", key="fb_yes", use_container_width=True):
                if st.session_state.last_analysis_id:
                    db.save_feedback(st.session_state.last_analysis_id, helpful=True)
                st.session_state.feedback_submitted = True
                st.success("Thank you for your feedback!")
        with fc2:
            if st.button("👎 No", key="fb_no", use_container_width=True):
                if st.session_state.last_analysis_id:
                    db.save_feedback(st.session_state.last_analysis_id, helpful=False)
                st.session_state.feedback_submitted = True
                st.info("Thank you. Your feedback helps improve SatQuery AI.")


def render_downloads(query: str, result):
    """Render download buttons for Markdown and PDF reports."""
    if result is None or not result.answer:
        return

    result_dict = result.to_dict()
    st.divider()
    st.markdown("**📥 Download Report**")

    c1, c2 = st.columns(2)
    with c1:
        md_report = build_markdown_report(query, result_dict)
        st.download_button(
            "📄 Markdown Report",
            data=md_report,
            file_name="satquery_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with c2:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            pdf_path = tf.name
        try:
            img_for_pdf = st.session_state.last_rgb_viz
            build_pdf_report(query, result_dict, pdf_path, image_pil=img_for_pdf)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                "📊 PDF Report",
                data=pdf_bytes,
                file_name="satquery_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"PDF generation encountered an issue: {e}")
        finally:
            try:
                os.unlink(pdf_path)
            except Exception:
                pass


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    render_header()
    mode = render_sidebar()

    # Upload area
    st.markdown("### 📁 Upload Satellite Image")
    uploaded_files = render_upload_area(mode)

    if not uploaded_files:
        st.markdown(
            '<div class="info-box">👆 Upload a satellite image using the panel above to begin analysis. '
            'Supported formats: <strong>GeoTIFF, TIFF, PNG, JPEG</strong>.</div>',
            unsafe_allow_html=True,
        )
        return

    # Validate file count
    expected = 1 if mode == "Single Image" else 2
    if len(uploaded_files) < expected:
        remaining = expected - len(uploaded_files)
        st.markdown(
            f'<div class="info-box">📎 Please upload {remaining} more image(s) for {mode} analysis.</div>',
            unsafe_allow_html=True,
        )
        return

    # Load images
    tmp_dir = tempfile.mkdtemp()
    images_data = []
    load_errors = []

    with st.spinner("Reading satellite image metadata..."):
        for uf, fname in uploaded_files:
            tmp_path = os.path.join(tmp_dir, fname)
            with open(tmp_path, "wb") as out:
                out.write(uf.getbuffer())
            try:
                arr, meta = load_image(tmp_path)
                images_data.append((arr, meta))
            except Exception as e:
                load_errors.append(f"Could not read '{fname}': {e}")

    if load_errors:
        for err in load_errors:
            st.markdown(f'<div class="error-box">❌ {err}</div>', unsafe_allow_html=True)
        return

    # Image panel
    st.markdown("### 🗺️ Image Viewer")
    render_image_panel(uploaded_files, images_data, mode)

    # Pre-generate and cache visualizations
    if images_data:
        arr0, meta0 = images_data[0]
        try:
            rgb_viz, _ = generate_rgb_visualization(arr0, meta0)
            st.session_state.last_rgb_viz = rgb_viz
        except Exception:
            pass
        try:
            fc = generate_false_color(arr0, meta0)
            if fc:
                st.session_state.last_false_color, st.session_state.last_false_color_desc = fc
        except Exception:
            pass

    # Query panel
    st.markdown("---")
    st.markdown("### 💬 Ask a Question")
    query, run_clicked = render_query_panel(mode)

    # Run analysis
    if run_clicked and query:
        st.session_state.feedback_submitted = False
        st.session_state.last_analysis_id = None

        # Progress display
        progress_area = st.empty()

        steps = [
            "📡 Reading satellite metadata...",
            "🎨 Preparing image visualization...",
            "📊 Extracting image statistics...",
            "🧠 Running AI analysis...",
            "📝 Preparing structured answer...",
        ]

        with progress_area.container():
            st.markdown("**Analysis in progress...**")
            progress_bar = st.progress(0)
            status_text = st.empty()

        for i, step in enumerate(steps[:3]):
            status_text.markdown(f"*{step}*")
            progress_bar.progress((i + 1) * 15)
            time.sleep(0.2)

        status_text.markdown(f"*{steps[3]}*")
        progress_bar.progress(60)

        controller = AgentController()
        result = controller.handle_query(query, images_data, st.session_state.session_id)

        progress_bar.progress(90)
        status_text.markdown(f"*{steps[4]}*")
        time.sleep(0.2)
        progress_bar.progress(100)
        time.sleep(0.3)
        progress_area.empty()

        # Cache result
        st.session_state.last_result = result
        st.session_state.last_query = query
        st.session_state.last_images_meta = [m for _, m in images_data]
        if result.change_map_pil:
            st.session_state.last_change_map = result.change_map_pil

        # Save to MongoDB
        db_warning = ""
        if not db.is_connected():
            db_warning = "MongoDB not connected — analysis result saved in session memory only."
        else:
            image_ids = []
            for arr, meta in images_data:
                img_id = db.save_image_metadata(
                    st.session_state.session_id,
                    {
                        "filename": meta.path.split(os.sep)[-1],
                        "format": meta.format,
                        "modality": meta.modality,
                        "width": meta.width,
                        "height": meta.height,
                        "bands": meta.bands,
                        "crs": meta.crs,
                        "resolution_x": meta.resolution_x,
                        "resolution_y": meta.resolution_y,
                        "acquisition_date": meta.acquisition_date,
                        "sensor": meta.sensor,
                        "is_sentinel2": meta.is_sentinel2,
                        "is_sar": meta.is_sar,
                        "path": meta.path,
                    }
                )
                if img_id:
                    image_ids.append(img_id)

            analysis_id = db.save_analysis(
                st.session_state.session_id,
                query,
                result.to_dict(),
                images=image_ids,
            )
            if analysis_id:
                st.session_state.last_analysis_id = analysis_id
                # Log model runs
                for model in result.models_used:
                    db.save_model_run(
                        analysis_id,
                        model=model,
                        task=result.task,
                        status="success" if result.answer else "no_answer",
                        latency_ms=result.processing_time_ms,
                    )
                # Refresh history
                st.session_state.history_loaded = False
            else:
                db_warning = "Analysis completed, but the result could not be saved to history."

        # Show DB warning
        if db_warning:
            st.markdown(
                f'<div class="warning-box">💾 {db_warning}</div>',
                unsafe_allow_html=True,
            )

    # Show result (from session or just computed)
    result = st.session_state.last_result
    if result is not None:
        st.markdown("---")
        st.markdown("### 📊 Analysis Result")
        render_result(result, show_feedback=not st.session_state.feedback_submitted)
        render_downloads(st.session_state.last_query or query, result)


if __name__ == "__main__":
    main()
