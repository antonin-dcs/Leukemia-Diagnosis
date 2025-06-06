#Interface_Cellule 
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from reseau_8 import Network  # adapte ce nom si nécessaire

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Charger le modèle et les poids
model = Network().to(device)
model.load_state_dict(torch.load("HEMAI.pth", map_location=device))
model.eval()

# Définir les transformations identiques à l'entraînement
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

import os

if os.path.exists("HEMAI.pth"):
    print("✅ Modèle déjà entraîné, saute l'entraînement.")
else:
    # Entraîne ici
    torch.save(model.state_dict(), "HEMAI.pth")

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return "Saine" if predicted.item() == 1 else "Cancéreuse"

import gradio as gr

def gradio_predict(image):
    # Convertir le tableau numpy en image PIL
    image_pil = Image.fromarray(np.uint8(image))
    image_pil.save("temp.jpg")
    result = predict("temp.jpg")
    return result
interface = gr.Interface(fn=gradio_predict, inputs="image", outputs="text")
interface.launch()
