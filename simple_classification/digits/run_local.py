import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import build_model
import spaces
import numpy as np

device = "cpu"
model = build_model()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@spaces.GPU
def predict(img):
    if img is None:
        return {}
    
    img = img.convert("L")

    # check if image has a white background. If yes, invert it
    arr = np.array(img)
    if arr.mean() > 127:
        img = Image.eval(img, lambda x: 255 - x) # inversion happens with 255 - x

    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        # probs = F.softmax(logits, dim=1)[0]
        predictions = logits.argmax(dim=1)
    return 
    return {str(i): float(probs[i]) for i in range(10)}

predict("./mnist_pngs/0_label7.png")