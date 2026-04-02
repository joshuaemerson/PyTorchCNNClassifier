import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

# Select the appropriate device to load data and params to for training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_fashion_mnist_image(image, label, train=True):
    """
    Displays a single FashionMNIST image with its class label and dataset split.

    Args:
        image: The image to display, as a PIL image or tensor.
        label (int): The class index of the image (0-9).
        train (bool): Indicates which split the image came from. Used for the
                      plot title only — does not load data. Default is True.
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    split = 'Train' if train else 'Test'
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'[{split}] {class_names[label]}')
    plt.axis('off')
    plt.show()

def get_data_loaders(train_set, test_set, batch_size):
    """Creates and returns training, validation, and test data loaders.

    Splits the training set 80/20 into training and validation subsets,
    then wraps all three splits in DataLoaders.

    Args:
        train_set: The full training dataset to be split into train and validation subsets.
        test_set: The test dataset.
        batch_size (int): The number of samples per batch in the data loaders.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    # Split the train_set into a train and validation set
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = random_split(train_set, [train_size, val_size])

    # Set up the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader