import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from .dataset import train_loader, val_loader
from .model import FlexibleCNN

def training_epoch(model, train_loader, optimizer, loss_function, device, num_epochs, emty_cache=True):
    """Performs a single training epoch for a given model.

    Args:
        model: The neural network model to be trained.
        train_loader: The DataLoader for the training data.
        optimizer: The optimization algorithm to update model weights.
        loss_function: The loss function used to evaluate model performance.
        device: The device (e.g., 'cuda' or 'cpu') to run the training on.
        num_epochs: The total number of epochs for training.
        emty_cache: A boolean flag to empty the CUDA cache after each batch.    
    Returns:
        The average loss for the epoch.
    """
    # Set the model to training mode
    model.train()

    # Initialize the total loss for the epoch
    running_loss = 0.0

    # Iterate over the batches of data in the training loader
    for images, labels in train_loader:
        # Move images and labels to the specified device
        images = images.to(device)
        labels = labels.to(device)

        # Perform a forward pass to get the model's predictions
        outputs = model(images)
        # Calculate the loss between the predictions and true labels
        loss = loss_function(outputs, labels)

        # Clear the gradients from the previous iteration
        optimizer.zero_grad()
        # Perform backpropagation to compute gradients
        loss.backward()
        # Update the model's weights using the optimizer
        optimizer.step()

        # Add the current batch's loss to the running total
        running_loss += loss.item()

        # Optionally, clear memory to prevent out-of-memory errors
        if emty_cache:
            del images, labels, outputs, loss
            torch.cuda.empty_cache()

    # Calculate the average loss for the entire epoch
    epoch_loss = running_loss / len(train_loader)
    # Return the calculated average epoch loss
    return epoch_loss

def evaluate_model(model, val_loader, device):
    """
    Evaluates the performance of a model on a validation dataset.

    Args:
        model: The model to be evaluated.
        val_loader: DataLoader for the validation dataset.
        device: The device (e.g., 'cpu' or 'cuda') to run the evaluation on.
    Returns:
        The accuracy of the model on the validation dataset.
    """
    # Set the model to evaluation mode
    model.eval()
    # Initialize variables for tracking correct predictions and total samples
    correct, total = 0, 0
    # Disable gradient calculations for inference
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # Get the predicted class with the highest probability
            _, predicted = torch.max(outputs, 1)
            # Update the total number of samples
            total += labels.size(0)
            # Update the count of correctly predicted samples
            correct += (predicted == labels).sum().item()

            # Release memory to prevent memory leaks
            del inputs, labels, outputs
            # Clear the GPU cache
            torch.cuda.empty_cache()

    # Calculate the accuracy
    accuracy = correct / total
    return accuracy

def design_search_space(trial):
    """
    Design the search space for hyperparameter optimization of the FlexibleCNN model.
    This function uses Optuna to suggest hyperparameters for the CNN architecture and training process.
    Args:
        trial (optuna.Trial): An Optuna trial object used to suggest hyperparameters.
    Returns:
        dict: A dictionary containing the suggested hyperparameters.    
    """
    # CNN Architecture Hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_filters = [ 
        trial.suggest_int(f"n_filters_layer{i}", 8, 64, step=8) for i in range(n_layers)
    ] 
    kernel_sizes = [ 
        trial.suggest_int(f"kernel_size_layer{i}", 3, 5, step=2) for i in range(n_layers)
    ] 
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    fc_size = trial.suggest_int("fc_size", 64, 512, step=64)

    
    # Training Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    
    return {
        "n_layers": n_layers,
        "n_filters": n_filters,
        "kernel_sizes": kernel_sizes,
        "dropout_rate": dropout_rate,
        "fc_size": fc_size,
        "learning_rate": learning_rate,
    }

def objective_function(trial, device, dataset_path, n_epochs=4, test=False):
    """
    Objective function for Optuna to optimize the hyperparameters of the FlexibleCNN model.
    Args:
        trial (optuna.Trial): An Optuna trial object used to suggest hyperparameters.
        n_epochs (int): Number of epochs for training the model.
        silent (bool): If True, suppresses output during training and evaluation.
        test (bool): If True, extracts attributes from the trial for evaluation purposes.
    Returns:
        float: The accuracy of the model on the validation set.
    """
    params = design_search_space(trial)

    # Define the model using the FlexibleCNN class with the parameters from the trial
    model = FlexibleCNN(
        n_layers = params["n_layers"],
        n_filters = params["n_filters"],
        kernel_sizes = params["kernel_sizes"],
        dropout_rate = params["dropout_rate"],
        fc_size = params["fc_size"],
        num_classes=10
    ) 
    
    # Initialize the dynamic classifier layer by passing a dummy input through the model
    # This ensures all parameters are instantiated before the optimizer is defined
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    model = model.to(device)
    model(dummy_input)
        
    # Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    loss_fcn = nn.CrossEntropyLoss()

    # Training the model
    model = model.to(device)
    
    # Training
    for epoch in range(n_epochs):
        _ = training_epoch(
            model,
            train_loader,
            optimizer,
            loss_fcn,
            device,
            n_epochs,
        )

    # Evaluation

    accuracy = evaluate_model(model, val_loader, device)
    return accuracy

