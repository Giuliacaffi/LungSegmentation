import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from monai.networks.nets import UNet
from monai.networks.layers import Norm

PATCH_SIZE = (256, 256)

# ----------------------------------------------------------
# CONFIGURAZIONE PAGINA (solo layout)
# ----------------------------------------------------------
st.set_page_config(
    page_title="Lung Segmentation App",
    layout="wide",
)

# ----------------------------------------------------------
# STILI PERSONALIZZATI (solo colori e testo)
# ----------------------------------------------------------
custom_css = """
<style>
/* Sfondo principale con gradiente molto leggero */
.main {
    background: linear-gradient(135deg, #f4f6fb 0%, #eef7ff 50%, #f7f9fc 100%);
}

/* Contenitore centrale */
.block-container {
    background-color: rgba(255, 255, 255, 0.92);
    border-radius: 16px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
}

/* Titolo principale */
h1 {
    text-align: center;
    font-weight: 700 !important;
    color: #1f2933;
    letter-spacing: 0.03em;
}

/* Testo normale */
p, label, span, div {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
}

/* Sottotitoli (es. st.subheader) */
h2, h3, h4 {
    color: #243b53;
    font-weight: 600;
}

/* Card dei risultati */
.result-card {
    background-color: #ffffff;
    padding: 1.2rem 1.4rem;
    border-radius: 14px;
    border: 1px solid #d8e2f2;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
}

/* Radio e file uploader più “morbidi” */
div[data-baseweb="radio"] > div {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 0.4rem 0.8rem;
    border: 1px solid #e1e7f5;
}

/* Nasconde dettagli file nel file_uploader */
div[data-testid="stFileUploader"] ul {
    display: none !important;
}
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderFileDetails"],
div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"],
div[data-testid="stFileUploader"] span[data-testid="stFileUploaderFileLabel"] {
    display: none !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ----------------------------------------------------------
# 1. LOAD MODEL
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    model = UNet(
        spatial_dims=2,
        in_channels=1,                # grayscale X-ray
        out_channels=1,               # binary mask
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    state_dict = torch.load("best_metric_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ----------------------------------------------------------
# 3. PREPROCESSING
# ----------------------------------------------------------
def preprocess(pil_image: Image.Image):
    pil_image = pil_image.convert("L")  # PNG -> grayscale
    img = np.array(pil_image).astype(np.float32)
    img_resized = cv2.resize(img, PATCH_SIZE, interpolation=cv2.INTER_NEAREST)

    img_resized = np.clip(img_resized, 0, 255)
    img_resized = img_resized / 255.0

    img_resized = np.expand_dims(img_resized, axis=0)   # (1, H, W)
    img_resized = np.expand_dims(img_resized, axis=0)   # (1, 1, H, W)

    x = torch.from_numpy(img_resized).float()
    return x

def postprocess(logits: torch.Tensor):
    """
    logits: torch.Tensor of shape (B, C, H, W)
    C can be 1 (sigmoid) or 2+ (softmax/argmax).
    Returns:
        mask_img: PIL.Image (0-255)
        mask_np:  np.ndarray (H, W) with values {0,1}
    """
    if logits.ndim != 4:
        raise ValueError(f"Expected logits shape (B, C, H, W), got {logits.shape}")

    if logits.shape[1] > 1:
        mask_tensor = torch.argmax(logits, dim=1, keepdim=True).float()
    else:
        mask_tensor = (logits > 0.5).float()

    mask_np = mask_tensor[0, 0].detach().cpu().numpy()
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))

    return mask_img, mask_np

# ----------------------------------------------------------
# 5. STREAMLIT APP
# ----------------------------------------------------------
st.title("Lung Segmentation App")
st.write("Carica una RX torace e segmentiamo i polmoni.")

uploaded_file = st.file_uploader("Carica un PNG RX", type=["png"])

if uploaded_file is not None:
    st.markdown(f"**Visualizzando:** *{uploaded_file.name}*")
    image = Image.open(uploaded_file)

    model = load_model()
    x = preprocess(image)
    st.write("Shape input modello:", x.shape)  # es: torch.Size([1, 1, 256, 256])

    with torch.no_grad():
        y = model(x)

    pred_mask_img, pred_mask_np = postprocess(y)

    overlay = cv2.addWeighted(
        np.array(image.resize(pred_mask_img.size).convert("RGB")),
        0.7,
        cv2.cvtColor(np.array(pred_mask_img), cv2.COLOR_GRAY2RGB),
        0.3,
        0
    )

    # Card per i risultati
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("Risultati della segmentazione")
    colA, colB, colC = st.columns(3)

    with colA:
        st.image(image, caption="Input", use_container_width=True)

    with colB:
        st.image(pred_mask_img, caption="Maschera predetta", use_container_width=True)

    with colC:
        st.image(overlay, caption="Overlay predizione", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    compare_option = st.radio(
        "Vuoi confrontare la predizione con una maschera esistente?",
        ("No", "Sì"),
        index=0
    )

    if compare_option == "Sì":
        gt_file = st.file_uploader(
            "Carica la maschera ground truth (PNG)",
            type=["png"],
            key="gt"
        )

        if gt_file is None:
            st.info("Carica una maschera per vedere il confronto con la predizione.")
        else:
            gt_img = Image.open(gt_file).convert("L")
            gt_img_resized = gt_img.resize(pred_mask_img.size, resample=Image.NEAREST)
            gt_np = np.array(gt_img_resized)

            gt_bin = (gt_np > 0).astype(np.uint8)

            intersection = np.logical_and(pred_mask_np == 1, gt_bin == 1).sum()
            pred_sum = (pred_mask_np == 1).sum()
            gt_sum = (gt_bin == 1).sum()
            dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)

            st.markdown(f"**Dice coefficient (pred vs GT):** {dice:.3f}")

            st.subheader("Confronto predizione vs ground truth")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(gt_img_resized, caption="GT Mask", use_container_width=True)
            with col2:
                st.image(pred_mask_img, caption="Predicted Mask", use_container_width=True)
            with col3:
                combo = np.stack([
                    (gt_bin * 255).astype(np.uint8),       # red
                    (pred_mask_np * 255).astype(np.uint8), # green
                    np.zeros_like(gt_bin, dtype=np.uint8)  # blue
                ], axis=-1)
                st.image(combo, caption="GT (red) vs Pred (green)", use_container_width=True)