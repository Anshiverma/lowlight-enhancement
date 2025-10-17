import streamlit as st
from PIL import Image
import numpy as np
import io
from enhancer import enhance_low_light

st.set_page_config(page_title="Low Light Image Enhancer", layout="wide")

st.title("Low Light Image Enhancement App")
st.write("Upload a dark image and get a brighter, enhanced version instantly!")


uploaded = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Original Image", use_container_width=True)

    
    with st.spinner("Enhancing your image... Please wait."):
        enhanced = enhance_low_light(img_np, gamma=1.5, clip_limit=2.0)

    st.success("Enhancement Complete ")

    
    st.image(enhanced, caption="Enhanced Image", use_container_width=True)

   
    enhanced_img = Image.fromarray(enhanced)
    buf = io.BytesIO()
    enhanced_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="⬇ Download Enhanced Image",
        data=byte_im,
        file_name="enhanced.png",
        mime="image/png"
    )

st.markdown("---")
st.markdown("Made with using Streamlit & OpenCV")
