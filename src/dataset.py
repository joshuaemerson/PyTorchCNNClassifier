import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import show_fashion_mnist_image


base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '../data')

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

fashion_trainset = datasets.FashionMNIST(root=data_path, train=True, transform=train_transform, download=True)
fashion_testset = datasets.FashionMNIST(root=data_path, train=False, transform=test_transform, download=True)

# Check random image to ensure they were loaded properly and augmentations were applied correctly
train_image, train_label = fashion_trainset[1500]
show_fashion_mnist_image(train_image, train_label, train=True)
test_image, test_label = fashion_testset[1000]
show_fashion_mnist_image(test_image, test_label, train=False)

# Set up the Data Loaders for the Train and Test Datasets
train_loader = DataLoader(fashion_trainset, batch_size=32, shuffle=True)
test_loader = DataLoader(fashion_testset, batch_size=32, shuffle=False)



