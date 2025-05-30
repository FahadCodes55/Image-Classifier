import torch
import torchvision.transforms as transforms
from torchvision import models
import streamlit as st
from PIL import Image
import json
import requests

# Load ImageNet class labels
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
labels = json.loads(requests.get(LABELS_URL).text)

def get_label(idx):
    return labels[str(idx)][1]  # Extract human-readable label

# Load the pre-trained MobileNetV2 model
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Classify the uploaded image
def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        with torch.no_grad():
            output = model(processed_image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top3_prob, top3_idx = torch.topk(probabilities, 3)

        return [(get_label(idx.item()), prob.item()) for idx, prob in zip(top3_idx, top3_prob)]

    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return []

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="📸", layout="centered")
    st.title("AI Image Classifier")
    st.write("Upload an image, and AI will tell you what’s in it!")

    model = load_model()
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Classify Image"):
            with st.spinner("Analyzing Image..."):
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Predictions")
                    for label, score in predictions:
                        st.write(f"**{label}** : {score:.2f}")

if __name__ == "__main__":
    main()