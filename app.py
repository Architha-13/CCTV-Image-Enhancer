import streamlit as st
import numpy as np
from PIL import Image
from sr_engine import upscale_image

st.set_page_config(page_title="CCTV Super Resolution", layout="wide")

st.title("🔍 CCTV Super Resolution x4")
st.markdown("Upload a low-resolution CCTV image to enhance it using your trained Real-ESRGAN model.")

uploaded_file = st.file_uploader("Upload CCTV Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_np, use_container_width=True)

    if st.button("Enhance Image 🚀"):

        with st.spinner("Enhancing image... Please wait"):
            output = upscale_image(image_np)

        with col2:
            st.subheader("Enhanced Image")
            st.image(output, use_container_width=True)

        # Convert result for download (no disk saving needed)
        result_pil = Image.fromarray(output)

        import io
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")
        buffer.seek(0)

        st.download_button(
            label="⬇ Download Enhanced Image",
            data=buffer,
            file_name="enhanced.png",
            mime="image/png"
        )