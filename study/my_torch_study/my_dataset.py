import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import pandas as pd
from torchvision.io import read_image

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download=True,
    transform=ToTensor()
)

#iterating and visualizing the dataset
label_map = {
    0: "T-Shirt",
    1: "Trouse",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols*rows+1):
    sample_idx = torch.randint(len(training_data), size=(1, )).item()
    img, labe = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label_map[labe])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap = "gray")
plt.show()

def test_custom():
    class CustomImageDataset(Dataset):
        def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
            self.image_labels = pd.read_csv(annotations_file)
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform
        def __len__(self):
            return len(self.image_labels)
        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.image_labels.iloc[idx, 0])
            image = read_image(img_path)
            label = self.image_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label