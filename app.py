import time
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st

from config import EVENTS_PATH
from src.anomaly_detector import PatchAnomalyDetector
from src.artifacts import load_reference_artifacts
from src.edge_simulator import append_event, load_events
from src.feature_extractor import FeatureExtractor

st.set_page_config(page_title="VoltVision", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.stApp { background:#10131a; color:#f5f7fb; } [data-testid="stHeader"] { background:rgba(16,19,26,.95); }
/* Keep the product view free of Streamlit's developer controls. */
#MainMenu, [data-testid="stToolbar"], [data-testid="stHeaderActionElements"], [data-testid="stAppDeployButton"] { display:none !important; }
[data-testid="stMetric"] { background:#191f2a; border:1px solid #293244; border-radius:12px; padding:16px; }
.block-container { max-width:1180px; padding-top:2.4rem; } .brand { font-size:2.2rem; font-weight:700; letter-spacing:-.04em; margin-bottom:.1rem; }
.subtle { color:#a8b3c7; margin-bottom:2rem; } .status-ok { color:#5ee3a1; font-weight:700; } .status-alert { color:#ffb86b; font-weight:700; }
div.stButton > button { background:#4f7cff; color:white; border:0; border-radius:9px; font-weight:650; padding:.55rem 1.1rem; }
div.stButton > button:hover { background:#6990ff; color:white; }</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    """Loads only when an inspection is requested."""
    bank, paths, metadata = load_reference_artifacts()
    detector = PatchAnomalyDetector(metadata["threshold"]).fit(bank, paths)
    return FeatureExtractor(), detector, metadata

st.markdown('<div class="brand">⚡ VoltVision</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Cable visual inspection</div>', unsafe_allow_html=True)
tab_inspect, tab_history = st.tabs(["Inspect cable", "Inspection history"])

with tab_inspect:
    st.subheader("New inspection")
    st.write("Upload a cable image to check whether it should be reviewed.")
    upload = st.file_uploader("Cable image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], label_visibility="collapsed")
    if upload:
        left, right = st.columns([1.05, 1])
        with left: st.image(upload, caption=upload.name, use_container_width=True)
        with right:
            st.write("Ready for inspection")
            if st.button("Inspect image", type="primary", use_container_width=True):
                try:
                    extractor, detector, metadata = load_engine()
                    started = time.perf_counter(); patches, _ = extractor.extract(upload); result = detector.score(patches); latency = (time.perf_counter() - started) * 1000
                    append_event(EVENTS_PATH, "LINE-01", upload.name, result.score, result.decision, latency)
                    st.session_state["inspection"] = (result, metadata, latency, detector.reference_paths)
                except Exception as exc:
                    st.error(f"Inspection is not ready: {exc}")
    inspection = st.session_state.get("inspection")
    if inspection:
        result, metadata, latency, reference_paths = inspection
        if result.decision == "NORMAL": st.markdown('<p class="status-ok">✓ Normal appearance</p>', unsafe_allow_html=True)
        else: st.markdown('<p class="status-alert">● Review required</p>', unsafe_allow_html=True)
        score, cutoff, speed = st.columns(3)
        score.metric("Inspection score", f"{result.score:.3f}"); cutoff.metric("Review threshold", f"{metadata['threshold']:.3f}"); speed.metric("Inspection time", f"{latency:.0f} ms")
        evidence, reference = st.columns(2)
        with evidence:
            fig, ax = plt.subplots(figsize=(5, 3.8)); fig.patch.set_facecolor("#191f2a"); ax.set_facecolor("#191f2a")
            heatmap = ax.imshow(result.heatmap, cmap="magma"); ax.set_title("Areas needing attention", color="white"); ax.set_axis_off(); fig.colorbar(heatmap, ax=ax, label="difference")
            st.pyplot(fig, use_container_width=True)
        with reference:
            st.write("Closest known-normal example")
            if result.closest_index < len(reference_paths) and Path(reference_paths[result.closest_index]).is_file(): st.image(reference_paths[result.closest_index], use_container_width=True)
            else: st.caption("A saved normal reference image is not available.")

with tab_history:
    st.subheader("Inspection history")
    st.caption("Recent image checks recorded by this application.")
    events = load_events(EVENTS_PATH)
    total, normal, review = st.columns(3)
    total.metric("Images checked", len(events)); normal.metric("Normal", int((events["decision"] == "NORMAL").sum()) if len(events) else 0); review.metric("Needs review", int((events["decision"] == "REVIEW REQUIRED").sum()) if len(events) else 0)
    if len(events):
        recent = events.sort_values("timestamp_utc", ascending=False).head(20).copy()
        recent["timestamp_utc"] = recent["timestamp_utc"].str.replace("T", " ").str.replace("+00:00", " UTC")
        recent = recent.rename(columns={"timestamp_utc":"Time", "image_filename":"Image", "anomaly_score":"Score", "decision":"Result", "latency_ms":"Time (ms)"})
        st.dataframe(recent[["Time", "Image", "Score", "Result", "Time (ms)"]], use_container_width=True, hide_index=True)
    else: st.info("No inspections recorded yet.")
