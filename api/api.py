import base64
from ast import literal_eval

import cv2
import torch
import numpy as np
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from torchvision import transforms

from api.utils import Config, ImageInput, download_weights, load_classes
from model import Cifar100Model

import warnings
warnings.filterwarnings("ignore")

api = FastAPI(title="CIFAR 100",
              version='0.1.0'
              )
origins = ["*"]
api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)

config = Config.load_config()['api']
resized_size = 224
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((resized_size, resized_size)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

])
device = torch.device('cuda' if config['cuda'] else 'cpu')

model = Cifar100Model().load_from_checkpoint(download_weights(config['weights']), map_location=device)
model.to(device)
model.eval()

classes = load_classes()


@api.post('/classify')
def classify(img_input: ImageInput):
    img_object = base64.b64decode(img_input.base64_img)
    image = cv2.imdecode(np.fromstring(img_object, np.uint8), cv2.IMREAD_COLOR)

    # print(image.shape)
    input_img = transform(image)
    input_img = torch.unsqueeze(input_img, dim=0)
    input_img.to(device)

    logits = model(input_img)
    preds = torch.softmax(logits, dim=-1)
    preds = torch.flatten(preds)
    class_id = torch.argmax(preds, dim=-1)

    prob = preds[class_id].item()

    return {'class': classes[class_id], 'prob': prob}

