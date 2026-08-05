import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import build_model
import spaces
import numpy as np
from matplotlib import pyplot
import cv2

np.set_printoptions(threshold=np.inf, linewidth=200)
torch.set_printoptions(linewidth=200)

device = "cpu"
model = build_model()
# model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.load_state_dict(torch.load("artifacts/mnist_cnn.pth", map_location=device, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def normalize_to_mnist_style(img):
    # # Accept PIL Image or numpy array
    # if isinstance(img, Image.Image):
    #     img = np.array(img)

    img = img.astype(np.float32)

    # 1. Contrast-stretch so max value hits 255
    if img.max() > 0:
        img = img * (255.0 / img.max())

    # 2. Threshold out near-zero noise, then dilate to thicken strokes
    _, mask = cv2.threshold(img, 20, 255, cv2.THRESH_TOZERO)
    kernel = np.ones((2, 2), np.uint8)
    thickened = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    # 3. (Optional but matches MNIST) re-center by center of mass
    return Image.fromarray(thickened)

def process_image(img):
    img = img.convert("L") # converts a color image into a 8-bit pixels, grayscale image
    arr = np.array(img)

    # check if image has a white background. If yes, invert it
    if arr.mean() > 127:
        img = Image.eval(img, lambda x: 255 - x) # inversion happens with 255 - x
        arr = np.array(img)

    # 1. Threshold to find "digit" pixels vs background
    threshold = 50
    mask = arr > threshold

    # 2. Find bounding box of the digit
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # 3. Crop to that bounding box
    digit = arr[y0:y1, x0:x1]
    # print(digit)

    # 4. Resize so the longest side is 20px, preserving aspect ratio
    digit_img = Image.fromarray(digit)
    h, w = digit.shape
    scale = 20 / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    digit_img = digit_img.resize((new_w, new_h))

    # 5. Paste onto a 28x28 black canvas, centered
    canvas = Image.new("L", (28, 28), color=0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas.paste(digit_img, (paste_x, paste_y))

    img = canvas
    img = normalize_to_mnist_style(np.array(img))
    return img

@spaces.GPU
def predict(img):
    if img is None:
        return {}
    
    np.set_printoptions(threshold=np.inf, linewidth=200)
    torch.set_printoptions(linewidth=200)

    img = process_image(img)
    print(np.array(img))
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        # predictions = logits.argmax(dim=1)
        # return 
    return {str(i): float(probs[i]) for i in range(10)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", image_mode="RGB"),
    outputs=gr.Label(num_top_classes=3),
    title="Kishy's Digit Classifier",
    description="Upload or draw a digit and get a prediction."
)

demo.launch()