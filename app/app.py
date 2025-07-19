import tempfile
from pathlib import Path

import streamlit as st
from streamlit_player import st_player

from fuse import predict
from vision_explain import load_frame, generate_heatmap
from deepsafe_sydney.text_explain import explain
from train_vision import DeepFakeModel
import imageio
import whisper


st.title("DeepSafe-Sydney")

uploaded = st.file_uploader("Upload a TikTok-sized MP4", type=["mp4"])

if uploaded:
    tmp_path = Path(tempfile.mkstemp(suffix=".mp4")[1])
    with tmp_path.open("wb") as f:
        f.write(uploaded.read())

    st_player(str(tmp_path), playing=True)

    risk_score = predict(str(tmp_path))
    st.metric("Risk", f"{risk_score:.2%}")

    model = DeepFakeModel.load_from_checkpoint("best.ckpt")
    frame = load_frame(str(tmp_path))
    heatmap = generate_heatmap(model, frame)
    gradcam_path = Path("app/static/gradcam.gif")
    gradcam_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gradcam_path, [heatmap], format="GIF")

    asr = whisper.load_model("tiny")
    text = asr.transcribe(str(tmp_path)).get("text", "")
    words = explain(text)

    col1, col2 = st.columns(2)
    with col1:
        st.image(str(gradcam_path))
    with col2:
        st.write("## Top words")
        for word, weight in words:
            st.write(f"{word}: {weight:.3f}")

    model_card = Path("model_card.pdf")
    if model_card.exists():
        with model_card.open("rb") as f:
            st.download_button("Download model card", f, "model_card.pdf", mime="application/pdf")
