import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

from torchvision import models

import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RESIZE_SIZE = 224, 224
norm_mean, norm_std = [0.4704, 0.4669, 0.3898], [0.2037, 0.2002, 0.2051]


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = norm_std * inp + norm_mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def classify(model, data_path, image_file_path):

    classes_labels = pd.read_csv(os.path.join(data_path, 'birds latin names.csv'))

    trs = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(RESIZE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    image = torchvision.io.read_image(image_file_path)
    image = trs(image)

    inputs = image.unsqueeze(dim=0).to(device)
    model.eval()

    res = []
    with torch.no_grad():
        outputs = model(inputs)

        probs = nn.functional.softmax(outputs.data, dim=1)
        top_probs = torch.topk(probs, 3)
        for pred_labels, values, input_image in zip(top_probs.indices, top_probs.values, inputs):
            for pred_label, pred_value in zip(pred_labels, values):
                res.append(f"{pred_value.item()*100.0:.3f}% {classes_labels.iloc[pred_label.item()]['class']}")

    return res

def create_resnet18(pretrained):
    resnet18 = models.resnet18(pretrained=pretrained)

    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 400)
    return resnet18.to(device)

def load_trained_resnet18(epoch: int):
    resnet18 = create_resnet18(pretrained=False)

    resnet_11_model = torch.load(f'ResNet18-{epoch}.pth', map_location ='cpu')

    resnet18.load_state_dict(resnet_11_model['model_state_dict'])
    resnet18.eval()
    return resnet18

