import os
import urllib.request
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision import transforms
from PIL import Image
import numpy as np

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Cat Breed Detector 🐱",
    layout="centered"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_URL = "https://huggingface.co/nabielherdiana/cat-breed-efficientnetv2/resolve/main/efficientnet_v2s_aug-model.pth"
MODEL_PATH = "efficientnet_v2s_aug-model.pth"

NUM_CLASSES = 20

# ===============================
# CLASS LABELS
# ===============================
IDX_TO_CLASS = {
    0: "Abyssinian",
    1: "American Shorthair",
    2: "Bengal",
    3: "Birman",
    4: "British Shorthair",
    5: "Domestic Long Hair",
    6: "Domestic Shorthair",
    7: "Exotic Shorthair",
    8: "Himalayan",
    9: "Maine Coon",
    10: "Norwegian Forest",
    11: "Oriental Short Hair",
    12: "Persian",
    13: "Ragdoll",
    14: "Russian Blue",
    15: "Scottish Fold",
    16: "Siamese",
    17: "Sphynx",
    18: "Turkish Angora",
    19: "Turkish Van"
}

# ===============================
# IMAGE TRANSFORM
# ===============================
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading model from Hugging Face...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    model = nn.Sequential(
        model,
        nn.LogSoftmax(dim=1)
    )

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model

model = load_model()

# ===============================
# STREAMLIT UI
# ===============================
st.title("🐱 Cat Breed Detector")
st.markdown(
    """
Upload a cat image and the model will predict **Top-3 most likely cat breeds**
using a **fine-tuned EfficientNetV2-S** model.
"""
)

uploaded_file = st.file_uploader(
    "Upload a cat image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.exp(output)[0]
        top_probs, top_idxs = torch.topk(probs, 3)

    st.subheader("🔮 Prediction Result")

    for i, (idx, prob) in enumerate(zip(top_idxs, top_probs), start=1):
        breed = IDX_TO_CLASS[idx.item()]
        st.write(f"**{i}. {breed}** — {prob.item()*100:.2f}%")

    st.progress(float(top_probs[0]))
