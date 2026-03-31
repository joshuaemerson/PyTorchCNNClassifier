import torch
import torch.nn as nn

# Select the appropriate device to load data and params to for training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

# Define a Flexible CNN Class to be used for image classification
class FlexibleCNN(nn.Module):
    """
    A customizable convolutional neural network (CNN) for image classification.
    It dynamically constructs convolutional blocks based on provided hyperparameters
    such as the number of layers, filter sizes, kernel sizes, and dropout rates.
    """

    def __init__(
        self, n_layers, n_filters, kernel_sizes, dropout_rate, fc_size, num_classes=2
    ):
        """
        Initializes the FlexibleCNN.

        Args:
            n_layers (int): Number of convolutional layers.
            n_filters (list): Number of filters for each convolutional layer.
            kernel_sizes (list): Kernel sizes for each convolutional layer.
            dropout_rate (float): Dropout rate for regularization.
            fc_size (int): Number of units in the fully connected layer.
            num_classes (int): Number of output classes.
        """
        super(FlexibleCNN, self).__init__()

        self.num_classes = num_classes
        
        self.features = nn.ModuleList()
        in_channels = 1  # Grayscale input images
        
        for i in range(n_layers): 
            # Create convolutional layer with dynamic parameters
            
            # Extract the number of filters and kernel size for the current layer, from n_filters and kernel_sizes
            out_channels = n_filters[i] 
            kernel_size = kernel_sizes[i] 
            
            
            padding = (kernel_size - 1) // 2 

            # Create a convolutional block, by using a `nn.Sequential` container to group layers together
            block = nn.Sequential(
                # Add a Convolutional layer `Conv2d` with parameters: `in_channels`, `out_channels`, `kernel_size`, and `padding`
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
                # Add a Batch normalization layer `BatchNorm2d` with `num_features` as `out_channels` 
                nn.BatchNorm2d(num_features=out_channels),
                # Add a ReLU activation
                nn.ReLU(), 
                # Add a MaxPool2d layer `MaxPool2d`, with `kernel_size=2` and `stride=2`
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
                
            # Append the convolutional block to the features ModuleList
            self.features.append(block)

            # Update in_channels for the next layer (the input channels for the next layer is the output channels of the current layer)
            in_channels = out_channels
            

        self.dropout_rate = dropout_rate
        self.fc_size = fc_size        
                
        # Classifier will be initialized after calculating flattened size
        self.classifier = None  
        self._flattened_size = None 
        

    def _create_classifier(self, flattened_size):
        """
        Creates the fully connected classifier part of the model based on the flattened feature size.

        Args:
            flattened_size (int): Size of the flattened feature maps.
        """

        # Create the classifier using a Sequential container
        self.classifier = nn.Sequential(
            # Add a dropout layer with the dropout rate defined at initialization
            nn.Dropout(p=self.dropout_rate), 
            # Add a fully connected layer `Linear` with `in_features=flattened_size` and `out_features` as `fc_size`
            nn.Linear(flattened_size, self.fc_size),
            # Activation function
            nn.ReLU(), 
            # # Another dropout layer
            nn.Dropout(p=self.dropout_rate), 
            # Add the final fully connected layer with `in_features` as `fc_size` and `out_features` as `num_classes`
            nn.Linear(self.fc_size, self.num_classes),
        )

    def forward(self, x):
        """
        Defines the forward pass of the FlexibleCNN.

        Args:
            x (torch.Tensor): Input tensor (batch of images).

        Returns:
            torch.Tensor: Output tensor (classification scores).
        """
        # Apply convolutional feature extraction layers
        for layer in self.features:
            x = layer(x)

        # Flatten the output x for the classifier (start_dim=1 to keep the batch dimension)
        x = torch.flatten(x, start_dim=1)

        # Dynamically create classifier if it doesn't exist
        if self.classifier is None:
            # Get the size of the flattened feature maps from the x tensor
            self._flattened_size = x.shape[1]

            # Create the classifier with the `_flattened_size` 
            self._create_classifier(self._flattened_size)
            
            # Extract the device from the input tensor
            device = x.device
            
            # Move the classifier to the same device as the input tensor, to ensure compatibility with GPU/CPU
            if self.classifier is not None:
                self.classifier.to(device)

        # Classification
        return self.classifier(x)