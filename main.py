import streamlit as st
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw
import torch

st.set_page_config(page_title="YOLOs Object Detection", layout="centered")
st.title("🔍 YOLOs Object Detection with Bounding Boxes")

# Load YOLOs model and processor
@st.cache_resource
def load_model():
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    return model, processor

model, image_processor = load_model()

# Upload image
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🕵️ Detecting objects...")

    # Prepare image for model
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Get results
    target_size = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=target_size
    )[0]

    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}"
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), label_text, fill="red")

    # Show image with detections
    st.image(image, caption="🎯 Detected Objects", use_column_width=True)

    # Display results as text
    st.markdown("### 📝 Detection Results")
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        st.write(f"✅ **{model.config.id2label[label.item()]}** with confidence **{round(score.item(), 2)}** at {box.tolist()}")
