# Fairness Evaluation and Testing Repository

Automated decision-making systems can potentially introduce biases, raising ethical concerns. This has led to the
development of numerous bias mitigation techniques.
However, the selection of a fairness-aware model for a specific dataset often involves a process of trial and error, as
it is not always feasible to predict in advance whether the mitigation measures provided by the model will meet the
user's requirements, or what impact these measures will have on other model metrics such as accuracy and run time.

Existing fairness toolkits lack a comprehensive benchmarking framework. To bridge this gap, we present FairnessEval, a
framework specifically designed to evaluate fairness in Machine Learning models. FairnessEval streamlines dataset preparation,
fairness evaluation, and result presentation, while also offering customization options.
In this demonstration, we highlight the functionality of FairnessEval in the selection and validation of fairness-aware models.
We compare various approaches and simulate deployment scenarios to showcase FairnessEval effectiveness.


## fairnesseval DEMO
[Here](https://github.com/softlab-unimore/fairnesseval/blob/main/notebooks/DEMO%20fairnesseval.ipynb)
you can find a demo notebook with working examples.

You can interact with the notebook and run the library with your experiments.

## fairnesseval API Quick Start

[Here](https://github.com/softlab-unimore/fairnesseval/blob/main/notebooks/fairnesseval%20Quick%20Start.ipynb)
you can find a quick start guide to the fairnesseval API with working examples.


## Experiment parameters

| Parameter | Description                                                                                                                                                                                                                                                                          |
| --- |---|
| `experiment_id` | ID of the experiment to run. Required.                                                                                                                                                                                                                                               |
| `dataset_name` | List of dataset names. Required.                                                                                                                                                                                                                                                     |
| `model_name` | List of model names. Required.                                                                                                                                                                                                                                                                |
| `results_path RESULTS_PATH` | Path to save results.                                                                                                                                                                                                                                                                |
| `train_fractions` | List of fractions to be used for training.                                                                                                                                                                                                                                           |
| `random_seeds` | List of random seeds to use. All random seeds set are related to this random seed. For each random_seed a new train_test split is done.                                                                                                                                              |
| `metrics` | Metric set to be used for evaluation. Available metric set names are `default`, `conversion_to_binary_sensitive_attribute`. To use custom metrics add a new key to `metrics_code_map` in `fairnesseval.metrics.py`.                                                                  |
| `preprocessing` | Preprocessing function to be used. Available preprocessing functions are `conversion_to_binary_sensitive_attribute`, `binary_split_by_mean_y`, `default`. To add a new preprocessing function add a new key to `preprocessing_function_map` in `fairnesseval.utils_prepare_data.py`. |
| `split_strategy` | Splitting strategy. Available split strategies are `stratified_train_test_split`, `StratifiedKFold`.                                                                                                                                                                                 |
| `train_test_fold` | List of `train_test_fold` to run with k-fold.                                                                                                                                                                                                                                        |
| `model_params` | Dict with key, value pairs of model hyper parameter names (key) and list of values to be iterated (values). When multiple list of parameters are specified the cross product is used to generate all the combinations to test.                                                       |
| `debug` | Debug mode if set, the program will stop at the first exception.                                                                                                                                                                                                                     |

This table provides a clear and concise overview of the parameters and their descriptions.

[//]: # (TODO define synthetic generations. explain how to use it. Automatically find and load it.)
