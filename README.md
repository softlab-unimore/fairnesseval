from random import random

# Fairness Evaluation and Testing Repository

Automated decision-making systems can potentially introduce biases, raising ethical concerns. This has led to the
development of numerous bias mitigation techniques.
However, the selection of a fairness-aware model for a specific dataset often involves a process of trial and error, as
it is not always feasible to predict in advance whether the mitigation measures provided by the model will meet the
user's requirements, or what impact these measures will have on other model metrics such as accuracy and run time.

Existing fairness toolkits lack a comprehensive benchmarking framework. To bridge this gap, we present FairnessEval, a
framework specifically designed to evaluate fairness in Machine Learning models. FairnessEval streamlines dataset
preparation,
fairness evaluation, and result presentation, while also offering customization options.
In this demonstration, we highlight the functionality of FairnessEval in the selection and validation of fairness-aware
models.
We compare various approaches and simulate deployment scenarios to showcase FairnessEval effectiveness.

## fairnesseval DEMO

[Here](https://github.com/softlab-unimore/fairnesseval/blob/main/notebooks/DEMO_fairnesseval.ipynb)
you can find a demo notebook with working examples.

You can interact with the notebook and run the library with your experiments.

## fairnesseval API Quick Start

[Here](https://github.com/softlab-unimore/fairnesseval/blob/main/notebooks/fairnesseval_Quick_Start.ipynb)
you can find a quick start guide to the fairnesseval API with working examples.

## Experiment parameters

| Parameter                   | Description                                                                                                                                                                                                                                                                          |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `experiment_id`             | ID of the experiment to run. Required.                                                                                                                                                                                                                                               |
| `dataset_name`              | List of dataset names. Required.                                                                                                                                                                                                                                                     |
| `model_name`                | List of model names. Required.                                                                                                                                                                                                                                                       |
| `results_path RESULTS_PATH` | Path to save results.                                                                                                                                                                                                                                                                |
| `train_fractions`           | List of fractions to be used for training.                                                                                                                                                                                                                                           |
| `random_seeds`              | List of random seeds to use. All random seeds set are related to this random seed. For each random_seed a new train_test split is done.                                                                                                                                              |
| `metrics`                   | Metric set to be used for evaluation. Available metric set names are `default`, `conversion_to_binary_sensitive_attribute`. To use custom metrics add a new key to `metrics_code_map` in `fairnesseval.metrics.py`.                                                                  |
| `preprocessing`             | Preprocessing function to be used. Available preprocessing functions are `conversion_to_binary_sensitive_attribute`, `binary_split_by_mean_y`, `default`. To add a new preprocessing function add a new key to `preprocessing_function_map` in `fairnesseval.utils_prepare_data.py`. |
| `split_strategy`            | Splitting strategy. Available split strategies are `stratified_train_test_split`, `StratifiedKFold`.                                                                                                                                                                                 |
| `train_test_fold`           | List of `train_test_fold` to run with k-fold.                                                                                                                                                                                                                                        |
| `model_params`              | Dict with key, value pairs of model hyper parameter names (key) and list of values to be iterated (values). When multiple list of parameters are specified the cross product is used to generate all the combinations to test.                                                       |
| `debug`                     | Debug mode if set, the program will stop at the first exception.                                                                                                                                                                                                                     |

This table provides a clear and concise overview of the parameters and their descriptions.

[//]: # (TODO define synthetic generations. explain how to use it. Automatically find and load it.)

## Installation

Clone the repository, navigate to the root directory, and run the following command:

```bash
 pip install -e .
```

## Adding a New Custom Model

To add a new custom model to the FairnessEval framework, follow these steps:

### 1. Define the Model Wrapper

Create a new wrapper class for your custom model in the `wrappers` module. This class should implement the necessary
methods for fitting and predicting with your model.
The wrapper must receive the `random_state` parameter.

```python
# In fairnesseval/models/wrappers.py
class CustomModelWrapper:
    def __init__(self, random_state, **kwargs):  # Random state is required here
        # Initialize your model with any required parameters
        self.model = YourCustomModel(
            [random_state, ] ** kwargs)  # Random state is NOT required here, but it is recommended to set it. 

    def fit(self, X, y):  # Or fit(self, X, y, sensitive_features) if your model require sensitive features
        # Fit the model to the data
        pass

    def predict(self, X):  # Or predict(self, X, sensitive_features): if your model requires sensitive features
        # Predict using the model
        pass
```

### 2. Register the Model

Add your custom model to the `additional_models_dict` in the `wrapper.py` file.
This dictionary maps the model name to the corresponding wrapper class.

```python
# In fairnesseval/models/models.py
additional_models_dict = {
    'most_frequent': partial(sklearn.dummy.DummyClassifier, strategy="most_frequent"),
    'LogisticRegression': sklearn.linear_model.LogisticRegression,
    'CustomModel': CustomModelWrapper,  # <-- Add your custom model here
}
```

### 3. Use the Custom Model

You can now use your custom model in the experiment configuration by specifying its name (`CustomModel`) in the
`model_names` list.

```python
experiment_conf = {
    'experiment_id': 'custom_experiment',
    'dataset_names': ['your_dataset'],
    'model_names': 'CustomModel',
    'random_seed': 42,
    'model_params': {'param1': value1, 'param2': value2},  # Add any hyperparameters for your model here
    'train_fractions': [0.8],
    'results_path': 'path/to/results',
    'params': ['--debug']
}
```

By following these steps, you can integrate a new custom model into the FairnessEval framework and use it for your
experiments.

# Adding a new Dataset

Adding a New Dataset
To add a new dataset to the FairnessEval framework, follow these steps:

1. Prepare the Dataset

Ensure your dataset is in CSV format. The second last column should be the target variable, and the last column should
be the sensitive attribute.

2. Save the Dataset

Save the CSV file in the datasets folder. Use the naming convention '[dataset_name].csv' for the file name.

3. Load the Dataset

The framework will automatically detect the dataset based on the file name and structure.

4. Use the Dataset

You can now use your new dataset in the experiment configuration by specifying its name in the dataset_names list 
(include the '.csv' extension in the name).

```python
experiment_conf = {
    'experiment_id': 'new_dataset_experiment',
    'dataset_names': ['new_dataset.csv'],  # <-- Add your new dataset here
    'model_names': ['LogisticRegression'],
    'random_seed': 42,
    'model_params': {'C': [0.1, 1, 10]},
    'train_fractions': [0.8],
    'results_path': 'path/to/results',
    'params': ['--debug']
}
```


By following these
steps, you can integrate a new dataset into the FairnessEval framework and use it for your experiments.
