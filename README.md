# PyTorchCNNClassifier

A convolutional neural network (CNN) classifier built with PyTorch for classifying images from the Fashion-MNIST dataset. The architecture is flexible and automatically tuned using Optuna, a hyperparameter optimization framework.

---

## What It Does

The classifier trains a CNN on the Fashion-MNIST dataset, which consists of 70,000 grayscale images across 10 clothing categories:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

Rather than using a fixed architecture, the project uses **Optuna** to search for the best combination of hyperparameters — including the number of convolutional layers, filter sizes, kernel sizes, dropout rate, fully connected layer size, and learning rate. Once the best trial is found, the model is retrained with those parameters and evaluated on the held-out test set.

---

## Project Structure

```
PyTorchCNNClassifier/
├── train.py              # Entry point — runs the Optuna study and final training
├── requirements.txt      # Python dependencies
└── src/
    ├── model.py          # FlexibleCNN architecture
    ├── trainer.py        # Training loop, evaluation, and Optuna objective function
    ├── dataset.py        # Data loading and preprocessing
    └── utils.py          # Shared utilities (e.g. device detection)
```

---

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- Optuna
- matplotlib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/joshuaemerson/PyTorchCNNClassifier.git
cd PyTorchCNNClassifier
```

### 2. Set up a virtual environment (recommended)

```bash
python -m venv myenv
source myenv/bin/activate        # macOS/Linux
myenv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Run training

```bash
python train.py
```

The script will:

1. Download the Fashion-MNIST dataset automatically on first run
2. Run an Optuna hyperparameter search across the configured number of trials
3. Retrain the best model found on the full training set
4. Print the final accuracy on the test set

### Configuring the run

The `main()` function in `train.py` accepts two parameters you can adjust directly:

```python
def main(n_epochs=3, n_trials=3):
```

- `n_epochs` — number of training epochs per Optuna trial
- `n_trials` — number of hyperparameter configurations Optuna will try

Increasing both will improve results at the cost of longer training time.

---

## Achieving ~80% Accuracy

With the default settings of 3 epochs and 3 trials, results will vary. To reach approximately 80% accuracy on the test set, the following configuration is recommended:

- `n_epochs = 10` — more epochs give the model time to converge
- `n_trials = 20` — more trials give Optuna a better chance of finding strong hyperparameters

Optuna searches over:

- Number of convolutional layers (1 to 3)
- Number of filters per layer
- Kernel size per layer
- Dropout rate
- Fully connected layer size
- Learning rate

The best results tend to come from architectures with 2 to 3 convolutional layers, moderate dropout (around 0.3), and a learning rate in the range of 0.0005 to 0.005. Optuna will naturally converge toward these regions given enough trials.

The dataset itself is relatively small (28x28 grayscale images), so very deep architectures are not needed and can actually hurt performance by reducing spatial dimensions too aggressively before the fully connected layers.

---

## Notes

- The Fashion-MNIST data is downloaded automatically via `torchvision` and saved to a `data/` directory relative to the project root.
- Training runs on GPU automatically if one is available, otherwise falls back to CPU.
- The Optuna study is held in memory and is not persisted between runs. Each run starts a fresh study.
