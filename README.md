Recurrent neural networks for system identification with quadratic constraints. In this package we evaluate the use of generalized sector conditions and how they can be used in system identification.

## Environment variables
```
DATASET_DIRECTORY # The folder contains the subfolders train, test and validation
RESULT_DIRECTORY # Directory in which artifacts will be stored 
CONFIG_DIRECTORY # Directory of configuration files, currently it requires a base.json file
```

## Configuration
```
{
    "epochs": "Number of training epochs",
    "eps": "Epsilon value for numerical stability",
    "dt": "Time step size",
    "optimizer": {
        "name": "Name of the optimizer (e.g., adam, sgd)",
        "learning_rate": "Learning rate for the optimizer"
    },
    "nz": "Size of the latent space",
    "batch_size": "Number of samples per batch",
    "window": "Size of the time window for input sequences",
    "loss_function": "Loss function to be used (e.g., mse, cross_entropy)",
    "horizons": {
        "training": "Number of time steps for training horizon",
        "testing": "Number of time steps for testing horizon"
    },
    "input_names": [
        "List of input variable names"
    ],
    "output_names": [
        "List of output variable names"
    ]
}
```

## Usage
For training a model you run `train_model.py` from the `scripts` directory
```
python scripts/train_model.py --model_name <MODEL_NAME> # e.g. tanh, dzn, dznGen
```
For validation
```
python scripts/evaluate_model.py --model_name <MODEL_NAME> # e.g. tanh, dzn, dznGen
```
## tikzplotlib
In `utils/plot.py` we use the package [tikzplotlib](https://pypi.org/project/tikzplotlib/) which is outdated and does not work with most recent `matplotlib` package. You can simply remove `tikzplotlib` and its corresponding calls in `plot.py`, then you will only get `*.png` images.