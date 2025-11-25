import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage import measure
import maxflow

# ========================== WEBSITE CONFIG ==========================


# ========================== UNET MODEL ==========================
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(1, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.center = CBR(512, 1024)

        self.dec4 = CBR(1024 + 512, 512)
        self.dec3 = CBR(512 + 256, 256)
        self.dec2 = CBR(256 + 128, 128)
        self.dec1 = CBR(128 + 64, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        center = self.center(self.pool(e4))

        d4 = F.interpolate(center, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([d4, e4], 1))

        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], 1))

        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], 1))

        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], 1))

        return torch.sigmoid(self.final(d1))


# ========================== HELPERS ==========================
def resizeimage(img, scale=0.25):
    H, W = img.shape
    newsize = (int(W * scale), int(H * scale))
    return cv2.resize(img, newsize, interpolation=cv2.INTER_AREA)


def unet_prob_map(image_path, model, device, insize=256):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    H, W = img_gray.shape

    inp = cv2.resize(img_gray, (insize, insize)).astype(np.float32) / 255.0
    inp_t = torch.from_numpy(inp[None, None]).to(device)

    with torch.no_grad():
        prob_small = model(inp_t).squeeze().float().cpu().numpy()

    prob = cv2.resize(prob_small, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return img_gray, prob


def grabcut_refine_from_prob(img_gray, prob, iters=5):
    sure_fg = prob > 0.8
    if np.sum(sure_fg) < 10:
        return (prob > 0.5).astype(np.uint8) * 255

    mask = np.full(img_gray.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    mask[prob > 0.8] = cv2.GC_FGD
    mask[prob < 0.2] = cv2.GC_BGD

    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except:
        return (prob > 0.5).astype(np.uint8) * 255

    return np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)


def maxflow_refine_from_prob(img_gray, prob):
    H, W = img_gray.shape
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((H, W))

    eps = 1e-6
    Dfg = -np.log(np.clip(prob, eps, 1.0)).astype(np.float32)
    Dbg = -np.log(np.clip(1.0 - prob, eps, 1.0)).astype(np.float32)

    g.add_grid_tedges(nodeids, Dbg, Dfg)
    g.add_grid_edges(nodeids, weights=1, symmetric=True)

    g.maxflow()
    seg = g.get_grid_segments(nodeids)

    return np.where(seg, 255, 0).astype(np.uint8)


# ==================== CLASSIFIER ====================
@st.cache_resource
def load_classifier():
    return joblib.load('tumor_classifier_model.pkl')


pipe = load_classifier()
CATEGORIES = ['gliomatumor', 'meningiomatumor', 'notumor', 'pituitarytumor']


def extract_features_and_mask(pil_img):
    try:
        image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        image_blur = cv2.medianBlur(image_gray, 5)
        _, otsu_mask = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask_open = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        clean_seed_mask = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))

        contours, _ = cv2.findContours(clean_seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None

        labels = measure.label(clean_seed_mask)
        props = measure.regionprops(labels, intensity_image=image_gray)

        if not props:
            return None, None, None

        tumor_blob = max(props, key=lambda p: p.area)

        area, perimeter = tumor_blob.area, tumor_blob.perimeter
        eccentricity, solidity = tumor_blob.eccentricity, tumor_blob.solidity

        minr, minc, maxr, maxc = tumor_blob.bbox
        crop_gray = image_gray[minr:maxr, minc:maxc]
        crop_mask = labels[minr:maxr, minc:maxc] == tumor_blob.label

        if crop_gray.size < 4 or crop_mask.sum() < 10:
            return None, None, None

        crop = crop_gray.copy()
        crop[~crop_mask] = crop_gray[crop_mask].mean()
        crop_u8 = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        glcm = graycomatrix(
            crop_u8,
            distances=[1, 2, 4],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )

        agg = lambda p: float(graycoprops(glcm, p).mean())

        features = [
            area, perimeter, eccentricity, solidity,
            agg('contrast'), agg('energy'),
            agg('homogeneity'), agg('correlation')
        ]

        return features, clean_seed_mask, image_bgr

    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None, None, None


# ==================== UI CONFIG ====================
BG_COLOR = "#2B2D42"
CARD_COLOR = "#8D99AE"
TEXT_COLOR = "#EDF2F4"

st.set_page_config(page_title="MRI Tumor Detection", page_icon="🧠", layout="wide")

st.markdown(f"""
<style>
.main {{ background-color: {BG_COLOR}; }}
h1, h2, h3, h4, p, span, label {{ color: {TEXT_COLOR} !important; }}
.top-banner {{
    background: {CARD_COLOR};
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 28px;
    font-weight: 700;
}}
.card-box {{
    background: {CARD_COLOR};
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 18px;
}}
.image-box {{
    background: {CARD_COLOR};
    padding: 12px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: 700;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='top-banner'>🧠 MRI Brain Tumor Detection & Segmentation</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Upload MRI Image")
    uploaded = st.file_uploader("Upload MRI", type=['jpg', 'jpeg', 'png'])
    st.markdown("---")
    st.header("Settings")
    display_mode = st.radio("View Mode", ["Show only selected model", "Show all models (comparison)"])
    method_option = st.selectbox(
        "Segmentation Model",
        ["UNet", "GrabCut", "MaxFlow"],
        disabled=(display_mode == "Show all models (comparison)")
    )


# ==================== MAIN LOGIC ====================
if uploaded is not None:
    imgpath = "temp.jpg"
    with open(imgpath, "wb") as f:
        f.write(uploaded.getbuffer())

    device = torch.device('cpu')
    model = UNet().to(device)
    state = torch.load('BRISC_UNet.pth', map_location=device)
    model.load_state_dict(state)
    model.eval()

    img_gray, prob = unet_prob_map(imgpath, model, device)
    imgsmall = resizeimage(img_gray, 0.25)

    unetmask = resizeimage((prob > 0.5).astype(np.uint8) * 255, 0.25)
    grabcut_mask = grabcut_refine_from_prob(img_gray, prob)
    grabcutmask = resizeimage(grabcut_mask, 0.25)
    maxflow_mask = maxflow_refine_from_prob(img_gray, prob)
    maxflowmask = resizeimage(maxflow_mask, 0.25)

    base_rgb = cv2.cvtColor(imgsmall, cv2.COLOR_GRAY2RGB)

    def make_overlay(mask, color):
        mask_rgb = np.zeros_like(base_rgb)
        mask_bin = (mask > 0).astype(np.uint8) * 255
        if color == "red":
            mask_rgb[:, :, 0] = mask_bin
        elif color == "green":
            mask_rgb[:, :, 1] = mask_bin
        elif color == "blue":
            mask_rgb[:, :, 2] = mask_bin
        return cv2.addWeighted(base_rgb, 0.7, mask_rgb, 0.3, 0)

    overlay_unet = make_overlay(unetmask, "red")
    overlay_grabcut = make_overlay(grabcutmask, "green")
    overlay_maxflow = make_overlay(maxflowmask, "blue")
    st.write("")

    st.info("**Classification Result**")

    pil_img = Image.open(imgpath).convert('RGB')
    features, _, _ = extract_features_and_mask(pil_img)

    if features is not None:
        prediction = pipe.predict([features])[0]
        confidence = float(np.max(pipe.predict_proba([features])) * 100)

        st.markdown(f"### Tumor Category: **{prediction.upper()}**")

        with st.expander("**Extracted Features**"):
            st.json({
                "Area": features[0],
                "Perimeter": features[1],
                "Eccentricity": features[2],
                "Solidity": features[3],
                "Contrast": features[4],
                "Energy": features[5],
                "Homogeneity": features[6],
                "Correlation": features[7]
            })
    else:
        st.warning("Could not extract features.")
        prediction = "notumor"

    # If no tumor, show only original
    if prediction == "notumor":
        st.warning("No tumor detected. Only original MRI is displayed.")
        st.markdown("<div class='image-box'>Original MRI</div>", unsafe_allow_html=True)
        st.image(imgsmall, width=600)

    else:
        # Show only selected model
        if display_mode == "Show only selected model":
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='image-box'>Original MRI</div>", unsafe_allow_html=True, width = 550)
                st.write("")
                st.image(imgsmall, width=550)
                
                orig_pil = Image.fromarray(imgsmall)
                orig_pil.save("original_mri.png")
                col1.download_button(
                    label="📥 Download Original",
                    data=open("original_mri.png", "rb"),
                    file_name="Original_MRI.png",
                    mime="image/png"
                )

            with col2:
                st.markdown("<div class='image-box'>Segmentation</div>", unsafe_allow_html=True, width = 550)
                st.write("")

                if method_option == "UNet":
                    st.image(overlay_unet, width=550)
                    img_to_save = overlay_unet
                    filename = "UNet_segmented.png"
                elif method_option == "GrabCut":
                    st.image(overlay_grabcut, width=550)
                    img_to_save = overlay_grabcut
                    filename = "GrabCut_segmented.png"
                else:
                    st.image(overlay_maxflow, width=550)
                    img_to_save = overlay_maxflow
                    filename = "MaxFlow_segmented.png"

                img_pil = Image.fromarray(img_to_save)
                img_pil.save(filename)

                st.download_button(
                    label="📥 Download Image",
                    data=open(filename, "rb"),
                    file_name=filename,
                    mime="image/png"
                )

        else:
            # Compare all models
            st.markdown("<div class='image-box'>Comparison of All Models</div>", unsafe_allow_html=True)
            st.write("")

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            # COL 1 — ORIGINAL
            col1.image(imgsmall, width=550)
            col1.markdown(
                "<p style='text-align:left; font-size:20px; font-weight:600;'>Original MRI</p>",
                unsafe_allow_html=True
            )
            orig_pil = Image.fromarray(imgsmall)
            orig_pil.save("original_mri.png")
            col1.download_button(
                label="📥 Download Original",
                data=open("original_mri.png", "rb"),
                file_name="Original_MRI.png",
                mime="image/png"
            )

            # COL 2 — UNET
            col2.image(overlay_unet, width=550)
            col2.markdown(
                "<p style='text-align:left; font-size:20px; font-weight:600;'>UNet Segmentation</p>",
                unsafe_allow_html=True
            )
            unet_pil = Image.fromarray(overlay_unet)
            unet_pil.save("unet_segmented.png")
            col2.download_button(
                label="📥 Download UNet",
                data=open("unet_segmented.png", "rb"),
                file_name="UNet_Segmented.png",
                mime="image/png"
            )

            # COL 3 — GRABCUT
            col3.image(overlay_grabcut, width=550)
            col3.markdown(
                "<p style='text-align:left; font-size:20px; font-weight:600;'>GrabCut Segmentation</p>",
                unsafe_allow_html=True
            )
            grabcut_pil = Image.fromarray(overlay_grabcut)
            grabcut_pil.save("grabcut_segmented.png")
            col3.download_button(
                label="📥 Download GrabCut",
                data=open("grabcut_segmented.png", "rb"),
                file_name="GrabCut_Segmented.png",
                mime="image/png"
            )

            # COL 4 — MAXFLOW
            col4.image(overlay_maxflow, width=550)
            col4.markdown(
                "<p style='text-align:left; font-size:20px; font-weight:600;'>MaxFlow Segmentation</p>",
                unsafe_allow_html=True
            )
            maxflow_pil = Image.fromarray(overlay_maxflow)
            maxflow_pil.save("maxflow_segmented.png")
            col4.download_button(
                label="📥 Download MaxFlow",
                data=open("maxflow_segmented.png", "rb"),
                file_name="MaxFlow_Segmented.png",
                mime="image/png"
            )

else:
    st.write("")
    st.info("Please upload an MRI image to begin.")

