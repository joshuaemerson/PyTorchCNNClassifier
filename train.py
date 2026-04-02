import os
import optuna
import torch.nn as nn
import torch.optim as optim
from src.utils import DEVICE
from src.model import FlexibleCNN
from src.dataset import train_loader, val_loader, test_loader
from src.trainer import objective_function, training_epoch, evaluate_model

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '../data')

def main(n_epochs=3, n_trials=3):
    # Create an optuna study to maximize validation accuracy
    study = optuna.create_study(direction='maximize')

    # Optimize the hyperparams in the objective function for the defined number of epochs and trials
    study.optimize(lambda trial: objective_function(trial, n_epochs=n_epochs, device=DEVICE, dataset_path=data_path), n_trials=n_trials)

    # Check the best trial

    best_trial = study.best_trial
    best_params = study.best_params
    print(f"Best trial: {best_trial.number}")
    print(f"Best params: {best_params}")

    # Define a new model using the FlexibleCNN class with the parameters from the best trial
    n_layers = best_params['n_layers']
    n_filters = [best_params[f'n_filters_layer{i}'] for i in range(n_layers)]
    kernel_sizes = [best_params[f'kernel_size_layer{i}'] for i in range(n_layers)]

    best_model = FlexibleCNN(
        n_layers = n_layers,
        n_filters = n_filters,
        kernel_sizes = kernel_sizes,
        dropout_rate = best_params["dropout_rate"],
        fc_size = best_params["fc_size"],
        num_classes=10
    ) 

    # Optimizer and Loss Function
    optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
    loss_fcn = nn.CrossEntropyLoss()

    # Train the best model
    _ = training_epoch(
            best_model,
            train_loader,
            optimizer,
            loss_fcn,
            DEVICE,
            n_epochs,
        )

    # Evaluate the model on the test set (see how the model generalizes to new data)
    accuracy = evaluate_model(best_model, test_loader, DEVICE)
    return accuracy


if __name__ == '__main__':
    accuracy = main()
    print(f'Accuracy on Test Set = {accuracy}')
