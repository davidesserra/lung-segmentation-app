import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from monai.networks.nets import UNet
from monai.networks.layers import Norm
# ----------------------------------------------------------
# 1. MODEL ARCHITECTURE (replace with your architecture)
# ----------------------------------------------------------

PATCH_SIZE = (256, 256)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# class UNet(nn.Module):
#     def __init__(self, n_classes=1):
#         super().__init__()

#         self.down1 = DoubleConv(1, 64)
#         self.pool1 = nn.MaxPool2d(2)
#         self.down2 = DoubleConv(64, 128)
#         self.pool2 = nn.MaxPool2d(2)

#         self.bottleneck = DoubleConv(128, 256)

#         self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.conv2 = DoubleConv(256, 128)
#         self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.conv1 = DoubleConv(128, 64)

#         self.out_conv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        bn = self.bottleneck(p2)

        u2 = self.up2(bn)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return torch.sigmoid(self.out_conv(u1))

# ----------------------------------------------------------
# 2. LOAD MODEL
# ----------------------------------------------------------

@st.cache_resource
def load_model():
    model = UNet(
        # Image dimensions
        spatial_dims=2,
        
        # How many input channels?
        in_channels=1,
        
        # How many output channels? (That is, how many classes to recognize?)
        out_channels=2,
        
        # Feature maps at each level of the net
        channels=(16, 32, 64, 128, 256),
        
        # Downsampling factors at each level
        strides=(2, 2, 2, 2),

        # Number of residual blocks
        num_res_units=2,

        # Normalization to be used  after convolutions
        norm=Norm.BATCH,
    )
    state_dict = torch.load("lung_segmenter.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ----------------------------------------------------------
# 3. PREPROCESSING
# ----------------------------------------------------------


def preprocess(pil_image: Image.Image):
    # 1) Forza a 1 canale (equivalente a prendere il primo canale in Kaggle)
    pil_image = pil_image.convert("L")  # PNG -> grayscale

    # 2) PIL -> numpy float32, shape (H, W)
    img = np.array(pil_image).astype(np.float32)

    # 3) Resize a PATCH_SIZE (come Resized + ResizeWithPadOrCropd per l'immagine)
    img_resized = cv2.resize(img, PATCH_SIZE, interpolation=cv2.INTER_NEAREST)

    # 4) Normalizza [0, 255] -> [0, 1] (come ScaleIntensityRanged)
    img_resized = np.clip(img_resized, 0, 255)
    img_resized = img_resized / 255.0

    # 5) Aggiungi dimensione canale: (H, W) -> (1, H, W)
    img_resized = np.expand_dims(img_resized, axis=0)

    # 6) Aggiungi batch: (1, H, W) -> (1, 1, H, W)
    img_resized = np.expand_dims(img_resized, axis=0)

    # 7) Numpy -> torch tensor
    x = torch.from_numpy(img_resized).float()

    return x


# ----------------------------------------------------------
# 4. POSTPROCESSING
# ----------------------------------------------------------

# def postprocess(mask_tensor):
#     mask = mask_tensor.detach().cpu().numpy()

#     if mask.ndim == 4:
#         mask = mask[0, 0, ...]
#     elif mask.ndim == 3:
#         mask = mask[0, ...]
#     if mask.ndim == 3 and mask.shape[-1] == 1:
#         mask = mask[..., 0]

#     mask = (mask > 0.5).astype(np.uint8) * 255

#     return Image.fromarray(mask)

def postprocess(logits: torch.Tensor):
    """
    logits: torch.Tensor of shape (B, C, H, W)
    C can be 1 (sigmoid) or 2+ (softmax/argmax).
    Returns:
        mask_img: PIL.Image (0-255)
        mask_np:  np.ndarray (H, W) with values {0,1}
    """

    # Move to CPU numpy
    if logits.ndim != 4:
        raise ValueError(f"Expected logits shape (B, C, H, W), got {logits.shape}")

    # If multi-class (C>1), use argmax to get class labels
    if logits.shape[1] > 1:
        # (B, 2, H, W) -> (B, 1, H, W) with classes {0,1}
        mask_tensor = torch.argmax(logits, dim=1, keepdim=True).float()
    else:
        # Binary case: apply threshold 0.5
        mask_tensor = (logits > 0.5).float()

    # Take first sample of batch
    mask_np = mask_tensor[0, 0].detach().cpu().numpy()  # (H, W), values {0,1}

    # Convert to 0-255 uint8 image
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))

    return mask_img, mask_np



# ----------------------------------------------------------
# 5. STREAMLIT APP
# ----------------------------------------------------------

# st.title("Lung Segmentation App")
# st.write("Carica una RX torace e segmentiamo i polmoni.")

# uploaded_file = st.file_uploader("Carica un PNG RX", type=["png"])

# # Optional: ground-truth mask
# gt_file = st.file_uploader("Carica la maschera ground truth (opzionale, PNG)", type=["png"], key="gt")

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Input", use_container_width=True)

#     model = load_model()
#     x = preprocess(image)
#     st.write("Shape input modello:", x.shape)  # es: torch.Size([1, 1, 256, 256])

#     with torch.no_grad():
#         y = model(x)

#     # Prediction mask (PIL + numpy 0/1)
#     pred_mask_img, pred_mask_np = postprocess(y)
#     st.image(pred_mask_img, caption="Maschera predetta", use_container_width=True)

#     # Overlay prediction on image
#     overlay = cv2.addWeighted(
#         np.array(image.resize(pred_mask_img.size).convert("RGB")),
#         0.7,
#         cv2.cvtColor(np.array(pred_mask_img), cv2.COLOR_GRAY2RGB),
#         0.3,
#         0
#     )
#     st.image(overlay, caption="Overlay predizione", use_container_width=True)

#     # If GT mask is provided, compare
#     if gt_file is not None:
#         gt_img = Image.open(gt_file).convert("L")
#         # Resize GT to match prediction size
#         gt_img_resized = gt_img.resize(pred_mask_img.size, resample=Image.NEAREST)
#         gt_np = np.array(gt_img_resized)

#         # Binarize GT: anything > 0 becomes 1
#         gt_bin = (gt_np > 0).astype(np.uint8)

#         # Compute Dice
#         intersection = np.logical_and(pred_mask_np == 1, gt_bin == 1).sum()
#         pred_sum = (pred_mask_np == 1).sum()
#         gt_sum = (gt_bin == 1).sum()
#         dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)

#         st.markdown(f"**Dice coefficient (pred vs GT):** `{dice:.3f}`")

#         # Show GT, prediction, and overlay
#         st.subheader("Confronto predizione vs ground truth")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.image(gt_img_resized, caption="GT Mask", use_container_width=True)
#         with col2:
#             st.image(pred_mask_img, caption="Predicted Mask", use_container_width=True)
#         with col3:
#             combo = np.stack([
#                 (gt_bin * 255).astype(np.uint8),       # red
#                 (pred_mask_np * 255).astype(np.uint8), # green
#                 np.zeros_like(gt_bin, dtype=np.uint8)  # blue
#             ], axis=-1)
#             st.image(combo, caption="GT (red) vs Pred (green)", use_container_width=True)
# ----------------------------------------------------------
# 5. STREAMLIT APP
# ----------------------------------------------------------

st.title("Lung Segmentation App")
st.write("Carica una RX torace e segmentiamo i polmoni.")

hide_file_details = """
    <style>
    /* Nasconde qualunque lista di file caricati dentro il file_uploader */
    div[data-testid="stFileUploader"] ul {
        display: none !important;
    }

    /* Nasconde eventuali dettagli file in altre versioni */
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderFileDetails"],
    div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"],
    div[data-testid="stFileUploader"] span[data-testid="stFileUploaderFileLabel"] {
        display: none !important;
    }
    </style>
"""
st.markdown(hide_file_details, unsafe_allow_html=True)




uploaded_file = st.file_uploader("Carica un PNG RX", type=["png"])

if uploaded_file is not None:
    st.markdown(f"**Visualizzando:** *{uploaded_file.name}*")
    image = Image.open(uploaded_file)

    # Compute prediction once (we'll use it in both branches)
    model = load_model()
    x = preprocess(image)
    st.write("Shape input modello:", x.shape)  # es: torch.Size([1, 1, 256, 256])

    with torch.no_grad():
        y = model(x)

    # Prediction mask (PIL + numpy 0/1)
    pred_mask_img, pred_mask_np = postprocess(y)

    # Overlay prediction on image (calcolata una volta sola)
    overlay = cv2.addWeighted(
        np.array(image.resize(pred_mask_img.size).convert("RGB")),
        0.7,
        cv2.cvtColor(np.array(pred_mask_img), cv2.COLOR_GRAY2RGB),
        0.3,
        0
    )

    # 🔹 Prima riga: Input – Pred – Overlay
    st.subheader("Risultati della segmentazione")
    colA, colB, colC = st.columns(3)

    with colA:
        st.image(image, caption="Input", use_container_width=True)

    with colB:
        st.image(pred_mask_img, caption="Maschera predetta", use_container_width=True)

    with colC:
        st.image(overlay, caption="Overlay predizione", use_container_width=True)

    # 🔹 Chiedi se vuole confrontare con una maschera GT
    compare_option = st.radio(
        "Vuoi confrontare la predizione con una maschera esistente?",
        ("No", "Sì"),
        index=0
    )

    if compare_option == "No":
        pass

    else:
        # ✅ Ask for GT mask, and only then show everything together
        gt_file = st.file_uploader(
            "Carica la maschera ground truth (PNG)",
            type=["png"],
            key="gt"
        )

        if gt_file is None:
            st.info("Carica una maschera per vedere il confronto con la predizione.")
        else:
            gt_img = Image.open(gt_file).convert("L")
            # Resize GT to match prediction size
            gt_img_resized = gt_img.resize(pred_mask_img.size, resample=Image.NEAREST)
            gt_np = np.array(gt_img_resized)

            # Binarize GT: anything > 0 becomes 1
            gt_bin = (gt_np > 0).astype(np.uint8)

            # Compute Dice
            intersection = np.logical_and(pred_mask_np == 1, gt_bin == 1).sum()
            pred_sum = (pred_mask_np == 1).sum()
            gt_sum = (gt_bin == 1).sum()
            dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)

            st.markdown(f"**Dice coefficient (pred vs GT):** `{dice:.3f}`")

            # Show GT, prediction, and combined visualization
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
