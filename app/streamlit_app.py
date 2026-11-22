from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import LayoutPipeline, load_config
from src.utils import ensure_dir

st.set_page_config(page_title="Room Layout Optimization", layout="wide")


@st.cache_resource(show_spinner=False)
def get_pipeline(algorithm: str, weights: str | None) -> LayoutPipeline:
    config = load_config(
        ROOT / "configs/perception.yaml",
        ROOT / "configs/geometry.yaml",
        ROOT / "configs/scoring.yaml",
        ROOT / "configs/optimizer.yaml",
    )
    if weights:
        config["perception"].setdefault("detector", {})["weights"] = weights
    return LayoutPipeline.from_config(config, algorithm=algorithm)


def main() -> None:
    st.title("Room Layout Optimization")
    st.caption(
        "Detect furniture and structure, lift them into a 2D floor plan, score the layout, "
        "and search for a better arrangement."
    )

    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader("Room photo", type=["jpg", "jpeg", "png", "webp"])
        algorithm = st.selectbox("Optimizer", ("greedy", "annealing", "genetic", "random"))
        scale_mode = st.selectbox("Scale mode", ("auto", "user", "reference", "metric"))
        room_width = room_depth = None
        if scale_mode == "user":
            room_width = st.number_input("Room width (m)", min_value=1.0, value=4.0, step=0.1)
            room_depth = st.number_input("Room depth (m)", min_value=1.0, value=5.0, step=0.1)
        weights = st.text_input(
            "Detector weights (optional)",
            placeholder="yolov8s.pt or runs/detect/.../best.pt",
        ).strip() or None
        optimize = st.checkbox("Run optimizer", value=True)
        run = st.button("Run pipeline", type="primary", use_container_width=True)

    if uploaded is not None:
        st.image(uploaded, caption="Input image", width=420)

    if not run:
        st.info("Upload a room photo and click **Run pipeline**.")
        return
    if uploaded is None:
        st.warning("Please upload an image first.")
        return

    out_dir = ensure_dir(ROOT / "results" / "streamlit" / Path(uploaded.name).stem)
    image_path = out_dir / uploaded.name
    image_path.write_bytes(uploaded.getbuffer())

    pipeline = get_pipeline(algorithm, weights)
    with st.spinner("Running perception, geometry, scoring, and optimization..."):
        try:
            result = pipeline.run(
                image_path,
                output_dir=out_dir,
                room_width_m=room_width,
                room_depth_m=room_depth,
                scale_mode=None if scale_mode == "auto" else scale_mode,
                optimize=optimize,
            )
        except Exception as error:  # noqa: BLE001 - surface to the user
            st.error(f"Pipeline failed: {error}")
            return

    st.success(
        f"Score {result.initial_score.total:.3f} -> {result.best_score.total:.3f} "
        f"({result.improvement:+.3f}), feasible={result.best_score.feasible}"
    )
    reasons = result.floorplan.metadata.get("reliability_reasons", [])
    if reasons:
        st.warning(
            "Low-confidence geometry: " + "; ".join(reasons) + ". Verify the plan and overlay "
            "against the photo before using them."
        )

    columns = st.columns(3)
    columns[0].metric("Detected score", f"{result.initial_score.total:.3f}")
    columns[1].metric("Optimized score", f"{result.best_score.total:.3f}")
    columns[2].metric("Feasible", "yes" if result.best_score.feasible else "no")

    left, right = st.columns(2)
    with left:
        st.subheader("Optimized floor plan")
        st.image(str(out_dir / "floorplan_optimized.png"), use_column_width=True)
    with right:
        st.subheader("Before / after")
        st.image(str(out_dir / "before_after.png"), use_column_width=True)

    overlay = out_dir / "overlay_optimized.png"
    if overlay.exists():
        st.subheader("Optimized layout over the source image")
        if reasons:
            st.caption("Overlay may be misleading: the geometry above failed its reliability check.")
        st.image(str(overlay), use_column_width=True)

    st.subheader("Score breakdown")
    st.json(result.best_score.to_dict())

    st.download_button(
        "Download optimized floor plan (JSON)",
        data=json.dumps(result.best_plan.to_dict(), indent=2),
        file_name="floorplan_optimized.json",
        mime="application/json",
    )
    with st.expander("Warnings and metadata"):
        st.write(result.floorplan.metadata.get("warnings", []))
        st.json(
            {
                "scale_mode": result.floorplan.scale_mode,
                "camera_height": result.floorplan.camera_height,
                "geometry_reliable": result.floorplan.metadata.get("geometry_reliable"),
                "reliability_reasons": result.floorplan.metadata.get("reliability_reasons", []),
                "room_fit": result.floorplan.metadata.get("room_fit"),
                "optimization": result.optimization.to_dict() if result.optimization else None,
            }
        )


if __name__ == "__main__":
    main()
