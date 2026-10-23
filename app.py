"""Streamlit UI — 8x8 class quality assessment."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.pipeline import analyze_recording, find_sibling_vtt

load_dotenv()

st.set_page_config(page_title="8x8 Class Quality", page_icon="📚", layout="wide")

QUALITY_COLORS = {
    "Excellent": "#0d9488",
    "Good": "#2563eb",
    "Average": "#d97706",
    "Poor": "#dc2626",
}

st.title("8x8 Class Quality Assessment")
st.caption(
    "Upload a class recording to classify quality (Poor → Excellent) with evidence-based observations. "
    "Uses pretrained Whisper (optional) and an LLM rubric — no custom model training."
)

with st.sidebar:
    st.header("Settings")
    default_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    provider_ix = 0 if default_provider != "openai" else 1
    llm_provider = st.radio(
        "LLM backend",
        options=["ollama", "openai"],
        index=provider_ix,
        format_func=lambda x: "Ollama (open-source, local)" if x == "ollama" else "OpenAI (cloud)",
    )
    if llm_provider == "ollama":
        llm_base_url = st.text_input(
            "Ollama API base URL",
            value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
        llm_model = st.text_input(
            "Ollama model",
            value=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
            help="Run once: `./scripts/setup_ollama.sh` or `ollama pull qwen2.5:7b-instruct`",
        )
        api_key = os.getenv("OLLAMA_API_KEY", "ollama")
    else:
        llm_base_url = os.getenv("OPENAI_BASE_URL") or None
        api_key = st.text_input(
            "OpenAI API key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
        llm_model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    use_vtt_if_uploaded = st.checkbox("Also upload WebVTT (skip Whisper)", value=False)
    use_whisper = st.checkbox("Transcribe with Whisper if no VTT", value=True)
    whisper_model = st.selectbox(
        "Whisper model size",
        ["tiny", "base", "small", "medium"],
        index=2,
        help="Smaller = faster on CPU. 'small' is a good default.",
    )
    st.divider()
    st.markdown(
        "**Privacy:** uploads are processed in a temporary folder and removed after analysis. "
        "Do not commit recordings to git."
    )

col_upload, col_info = st.columns([2, 1])
with col_upload:
    video_file = st.file_uploader("Class recording (MP4)", type=["mp4", "mov", "mkv", "webm"])
    vtt_file = None
    if use_vtt_if_uploaded:
        vtt_file = st.file_uploader("Optional WebVTT transcript", type=["vtt"])

with col_info:
    st.info(
        "Tip: For faster local tests with the sample dataset, upload MP4 + matching `.vtt` "
        "to avoid long CPU transcription."
    )

analyze_btn = st.button("Analyze class", type="primary", disabled=video_file is None)

if analyze_btn and video_file:
    log_box = st.empty()
    logs: list[str] = []

    def on_progress(msg: str) -> None:
        logs.append(msg)
        log_box.code("\n".join(logs[-12:]))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        video_path = tmp / video_file.name
        video_path.write_bytes(video_file.getbuffer())
        vtt_path: Path | None = None
        if vtt_file is not None:
            vtt_path = tmp / vtt_file.name
            vtt_path.write_bytes(vtt_file.getbuffer())

        try:
            with st.spinner("Analyzing…"):
                result = analyze_recording(
                    video_path,
                    vtt_path=vtt_path,
                    whisper_model=whisper_model,
                    use_whisper=use_whisper,
                    api_key=api_key or None,
                    llm_model=llm_model or None,
                    llm_provider=llm_provider,
                    llm_base_url=llm_base_url or None,
                    on_progress=on_progress,
                )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    assessment = result["assessment"]
    quality = assessment.get("quality", "Unknown")
    color = QUALITY_COLORS.get(quality, "#64748b")

    st.markdown(
        f"<h2 style='color:{color}; margin-bottom:0;'>{quality}</h2>"
        f"<p>Confidence: <b>{assessment.get('confidence', '—')}</b> · "
        f"Engine: {assessment.get('engine', '—')} ({assessment.get('model', '—')})</p>",
        unsafe_allow_html=True,
    )
    st.write(assessment.get("summary", ""))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Strengths")
        for s in assessment.get("strengths") or []:
            st.markdown(f"- {s}")
    with c2:
        st.subheader("Improvements")
        for s in assessment.get("improvements") or []:
            st.markdown(f"- {s}")

    st.subheader("Observations")
    for obs in assessment.get("observations") or []:
        if isinstance(obs, dict):
            st.markdown(
                f"**{obs.get('time', '—')}** — {obs.get('note', '')}  \n"
                f"> _{obs.get('quote', '')}_"
            )
        else:
            st.markdown(f"- {obs}")

    with st.expander("Metrics & metadata"):
        st.json({"meta": result["meta"], "features": result["features"]})

    with st.expander("Full assessment JSON"):
        st.json(assessment)

st.divider()
st.markdown("#### Evaluate local sample files (dev)")
dataset_root = Path(__file__).parent / "class recordings -20261022T171511Z-1-001" / "class recordings "
if dataset_root.is_dir():
    mp4s = sorted(dataset_root.glob("*/*.mp4"))
    labels = [p.parent.name.strip() for p in mp4s]
    choice = st.selectbox(
        "Pick a bundled sample (not uploaded — runs on disk)",
        options=range(len(mp4s)),
        format_func=lambda i: f"[{labels[i]}] {mp4s[i].name}",
    )
    if st.button("Run on sample file"):
        p = mp4s[choice]
        vtt = find_sibling_vtt(p)
        logs: list[str] = []
        log_box = st.empty()

        def prog(m: str) -> None:
            logs.append(m)
            log_box.code("\n".join(logs[-12:]))

        with st.spinner("Running sample…"):
            result = analyze_recording(
                p,
                vtt_path=vtt,
                whisper_model=whisper_model,
                use_whisper=not bool(vtt),
                api_key=api_key or None,
                llm_model=llm_model or None,
                llm_provider=llm_provider,
                llm_base_url=llm_base_url if llm_provider == "ollama" else None,
                on_progress=prog,
            )
        st.success(f"Folder label: **{labels[choice]}** → Predicted: **{result['assessment'].get('quality')}**")
        st.json(result["assessment"])
else:
    st.caption("Sample dataset folder not found beside the app (optional for local dev).")
