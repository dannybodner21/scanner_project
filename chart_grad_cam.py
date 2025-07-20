import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Configuration
MODEL_PATH = 'chart_model.pth'
IMAGE_PATH = 'chart_dataset/ETHUSDT_202506111230_neutral.png'
IMAGE_SIZE = 224
LABELS = ['bearish', 'bullish', 'neutral']

# Transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Hook for gradients
grads = []
activations = []

def save_grad(grad):
    grads.append(grad)

# Register forward and backward hooks
def register_hooks(layer):
    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_grad)
    layer.register_forward_hook(forward_hook)

register_hooks(model.layer4[1].conv2)

# Load image
img = Image.open(IMAGE_PATH).convert('RGB')
x = transform(img).unsqueeze(0)

# Forward pass
output = model(x)
pred = output.argmax(dim=1).item()
print(f"Predicted: {LABELS[pred]}")

# Backward pass
model.zero_grad()
class_loss = output[0, pred]
class_loss.backward()

# Get grad-CAM
gradients = grads[0][0].detach().numpy()
features = activations[0][0].detach().numpy()
weights = gradients.mean(axis=(1, 2))
cam = np.zeros(features.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * features[i]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
cam -= cam.min()
cam /= cam.max()

# Overlay heatmap
img_np = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
heatmap = (cam * 255).astype(np.uint8)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

# Display
plt.imshow(overlay)
plt.title(f"Prediction: {LABELS[pred]}")
plt.axis('off')
plt.show()