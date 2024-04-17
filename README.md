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

Installation
```bash
pip install git+https://github.com/softlab-unimore/fairnesseval@main
```

# faireval API
This tool provides two interfaces for running fairness experiments on your data.

**1. Python Interface**
You can define the experiment settings in the form of a Python dictionary and use one of the following Python functions to run experiments:
    
1. `fairnesseval.run.launch_experiment_by_id` let you define and organize your experiments in a python module (default at `fairnesseval.experiment_definitions`). Then you will need to call this function by specifying only the id of the experiment you want to run. **This is the reccommended interface.**
    
2. `fairnesseval.run.launch_experiment_by_config` let you run an experiment by passing the dictionary of parameters of your experiment in input.

**2. Command Line Interface**
Alternatively, you can use the command line interface of `fairnesseval.run` to specify the experiment settings using traditional CLI parameters.

[//]: # (TODO define synthetic generations. explain how to use it. Automatically find and load it.)

## 1 Python Interface

To launch an experiment you can run Python script that read experiment parameters from a module (default at `fairnesseval.experiment_definitions`).

Loading experiment definitions is more powerful and flexible, it allows to:

*   launch multiple experiments in a row.
*   specify multiple datasets.
*   specify multiple models.
*   configurations are more organized and readable.

Define your experiment in a file. (You can find example of experiment configuration in `fairnesseval.experiment_definitions`).

Eg.: Create `exp_def.py` and define an experiment.
```python
import itertools
import json

import pandas as pd

RANDOM_SEEDs_RESTRICTED_V1 = [1]

TRAIN_FRACTIONS_SMALLER_DATASETS_v1 = [0.063, 0.251, 1.]
TRAIN_FRACTIONS_v1 = [0.001, 0.004, 0.016, 0.063, 0.251, 1]  # np.geomspace(0.001,1,7) np.linspace(0.001,1,7)

experiment_definitions = [
    {
        'experiment_id': 'new_experiment',
        'dataset_names': ('adult_sigmod',),
        'model_names': ('LogisticRegression',),
        'random_seeds': RANDOM_SEEDs_RESTRICTED_V1,
        'results_path': './demo_results'
    }
]

```

Copy the path to the experiment configuration file just defined.

In my case on colab: `/content/exp_def.py`

Then run the experiment in Colab
```python
import fairnesseval as fe
fe.run.launch_experiment_by_id('new_experiment', '/content/exp_def.py')
```



or save the following code in a .py file to run the experiments.


```python
# FILE runner.py
import fairnesseval as fe

if __name__ == "__main__":
    conf_todo = [
        "new_experiment",
        # ... (list of configurations to be executed)
    ]
    for x in conf_todo:
        fe.run.launch_experiment_by_id(x, '/content/exp_def.py')

```

Then launch the python script:
```bash
python -m runner
```

Otherwise you can use `launch_experiment_by_config`.
E.g.:
```python
fe.run.launch_experiment_by_config(
    {
        'experiment_id': 'new_experiment',
        'dataset_names': ('adult_sigmod',),
        'model_names': ('LogisticRegression',),
        'random_seeds': [1],
        'results_path': './demo_results'
    }
    )
```


## CLI interface
The equivalent CLI call to run the experiment defined before is:

```bash
cd fairnesseval/src/fairnesseval
!python -m run --dataset_name adult_sigmod --model_name LogisticRegression --experiment_id new_experiment --random_seeds 1 --results_path /content/demo_results
```



List of available models can be found in the `models.wrappers` module.
It is possible to define new models by importing the model class in the `models.wrappers` module and add the name and class as
key, value pair in the `additional_models_dict` dictionary.

```python
from example_model import ExampleModel

additional_models_dict = {
    # model_name: model_class,
    'example_model': ExampleModel
}

```

[//]: # ()

[//]: # (# Scalable Fairlearn)

[//]: # ()

[//]: # ()

[//]: # (## Example Runs)

[//]: # ()

[//]: # (#### Synth)

[//]: # (```)

[//]: # (time stdbuf -oL python run.py synth hybrids --eps=0.05 -n=10000 -f=3 -t=0.5 -t0=0.3 -t1=0.6 -v=1 --test_ratio=0.3 --sample_seeds=0,1,2,3,4,5,6,7,8,9 --train_fractions=0.016 --grid-fraction=0.5)

[//]: # (```)

[//]: # ()

[//]: # (```)

[//]: # (time stdbuf -oL python run.py synth hybrids --eps=0.05 -n=1000000 -f=3 -t=0.5 -t0=0.3 -t1=0.6 -v=1 --test_ratio=0.3 --sample_seeds=0,1,2,3,4,5,6,7,8,9 --train_fractions=0.016 --grid-fraction=0.5)

[//]: # (```)

[//]: # ()

[//]: # (##### Unmitigated)

[//]: # (```)

[//]: # (time stdbuf -oL python run.py synth unmitigated -n=10000 -f=3 -t=0.5 -t0=0.3 -t1=0.6 -v=1 --test_ratio=0.3)

[//]: # (```)

[//]: # ()

[//]: # (##### Fairlearn)

[//]: # (```)

[//]: # (time stdbuf -oL python run.py synth fairlearn --eps=0.05 -f=3 -t=0.5 -t0=0.3 -t1=0.6 -v=1 --test_ratio=0.3 -n=10000)

[//]: # (```)

[//]: # (```)

[//]: # (time stdbuf -oL python run.py synth fairlearn --eps=0.05 -f=3 -t=0.5 -t0=0.3 -t1=0.6 -v=1 --test_ratio=0.3 -n=1000000)

[//]: # (```)

[//]: # ()

[//]: # ()

[//]: # (#### Adult)

[//]: # (```)

[//]: # (time stdbuf -oL python run.py adult unmitigated)

[//]: # (```)

[//]: # ()

[//]: # ()

[//]: # (```)

[//]: # (time stdbuf -oL python run.py adult fairlearn --eps=0.05)

[//]: # (```)

[//]: # (```)

[//]: # (time stdbuf -oL python run.py adult hybrids --eps=0.05 --sample_seeds=0,1,2,3,4,5,6,7,8,9 --train_fractions=0.001,0.004,0.016,0.063,0.251,1 --grid-fraction=0.5)

[//]: # (```)

[//]: # ()

[//]: # ()

[//]: # ()

[//]: # ()

[//]: # (## TODOs)

[//]: # ()

[//]: # (### Complete Hybrid Method)

[//]: # (* Single hybrid method that gets the best of all hybrid methods we have)

[//]: # (* Show that it works on both train and test data)

[//]: # ()

[//]: # (### Scaling experiments)

[//]: # (* Show running time savings when dataset is very large &#40;use synthetic data&#41;)

[//]: # (* Also try logistic regression on large image dataset)

[//]: # ()

[//]: # (### Multiple datasets)

[//]: # (* Show it works on three datasets)

[//]: # (* Try logistic regression on large image dataset)

[//]: # ()

[//]: # (### Increasing number of attributes)

[//]: # (* Decide if we can do that experiment...)

[//]: # ()

[//]: # (### Other things)

[//]: # (* How to subsample for the scalability plot to ensure + and - points are treated equally &#40;stratified data sampling?&#41;)
