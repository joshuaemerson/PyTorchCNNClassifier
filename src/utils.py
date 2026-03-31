import matplotlib.pyplot as plt
import torchvision

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