import streamlit as st
import torch
from class_extract import dictionary
from PIL import Image
from model import MyModel,common_transforms
from pathlib import Path
root_dir = Path(__file__).parent
model_pred = MyModel()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_pred.load_state_dict(torch.load(root_dir / "model_weights.pth"))
model_pred.to(device=device)
st.title("🐒 Monkey Species Classifier")
st.write("Upload an image of a monkey and get its predicted species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    image_tensor = common_transforms(image).to(device=device)
    image_tensor = image_tensor.unsqueeze(0)
    output = model_pred(image_tensor)
    output = torch.argmax(torch.softmax(output,dim=1),dim=1).item()
    out = "Monkey Species :"+dictionary[output][1]
    st.write(out)
