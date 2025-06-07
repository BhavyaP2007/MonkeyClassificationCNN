import torch
from torchvision import datasets,transforms
import pandas as pd
from pathlib import Path
df = pd.read_csv("C:/Users/bpmch/OneDrive/Desktop/python/pytorch/monkey_classification/10_Monkey_Species/monkey_labels.txt")
# print(df.head())
data_transform = transforms.Compose(
    [
        transforms.Resize(size=(224,224),antialias=True),
        transforms.ToTensor()
    ]
)
train_data = datasets.ImageFolder(
    root=Path("C:/Users/bpmch/OneDrive/Desktop/python/pytorch/monkey_classification/10_Monkey_Species/training/training"),
    transform=data_transform
)
r_mean = 0
g_mean = 0
b_mean = 0
r_std = 0
g_std = 0
b_std = 0
for tuple_img in train_data:
    i = tuple_img[0]
    r = i[0]
    g = i[1]
    b = i[2]
    r_mean += r.mean()
    g_mean += g.mean()
    b_mean += b.mean()
    r_std += r.std()
    g_std += g.std()
    b_std += b.std()
means = [r_mean,g_mean,b_mean]
stds = [r_std,g_std,b_std]  