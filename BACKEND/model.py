import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image 
from pathlib import Path


model = models.densenet121(weights=None)
for param in model.parameters():
    param.requires_grad = False

num_filter = model.classifier.in_features
model.classifier = nn.Linear(num_filter,2)

Base_dir = Path(__file__).resolve().parent.parent
state_dict = torch.load((Base_dir/"MODEL"/"best_model_epoch.pth"),map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

def inference(img_file):
    transform = transforms.Compose([
      transforms.Resize(224),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.486,0.455,0.406],[0.228,0.225,0.224])
    ])
    img_inp = transform(Image.open(img_file).convert("RGB")).unsqueeze(0)
    output = model(img_inp)
    probabilities = torch.softmax(output,dim=1)
    pred_class = torch.argmax(probabilities,dim=1)
    conf = probabilities[0][pred_class]
    return pred_class.item(), conf.item()