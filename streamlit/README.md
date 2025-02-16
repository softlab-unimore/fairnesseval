# Fairnesseval Demonstration

A demonstration web application for FairnessEval is available.  
To run the demonstration, navigate to the `streamlit` folders and run the streamlit server:

```bash
cd fairnesseval
cd streamlit 
streamlit run Fairnesseval.py
```

## DEMO input example

Here I will show you how to use the demo by providing input examples for the different models and datasets.

### Example 1: Exponentiated Gradient with Logistic Regression on Adult dataset

- Dataset: 'adult', 'compas', 'german',
- Model: `expgrad`
- Model parameters:

```
{'eps': 0.005,
 'base_model_code': ['lr'],
 'constraint_code': 'dp',
 'base_model_grid_params': {'C': [0.1]}}
```
- Train Fractions: `[0.25]`
- Train Test fold: `0`


### Example 2: N models on M datasets
If models have similar parameters, you can use the same parameters for all models. 
Not specified exclusive parameters of each model will be set at their default value.


- Dataset: adult, compas, german
- Model: Calmon, Feld, ThresholdOptimizer, ZafarDI, ZafarEO
- Model parameters:

```
{'base_model_code': 'lr', 'base_model_grid_params': {'C': [0.1]}}
- ```

- Train Fractions: `[0.063, 0.251, 1]`
- Train Test fold: `[0, 1]`
- Default values for others.

## Debugging

If something goes wrong during an experiment, you can copy the experiment definition displayed in the console
at the beginning of the experiment launch and debug it in the python environment to debug the issue.
For instance: you can copy-paste the experiment definition similarly to the comment in
`fairnesseval/experiment_routine.py` and run that module in the python environment.
(`python -m fairnesseval.experiment_routine`).

This is what you may put in the main function of `experiment_routine.py` should look like:

```python
from fairnesseval import run

if __name__ == "__main__":
    run.launch_experiment_by_config({"experiment_id": "demo.05",
                                     "dataset_names": ["adult"],
                                     "model_names": ["Calmon"],
                                     "model_params": {"base_model_code": "lr",
                                                      "base_model_grid_params": {"C": [0.1, 1]}},
                                     "results_path": "path/to/results/CHANGE_ME",
                                     "params": ["--debug"]})
```

## Other experiment examples in code format used in the demo video

In [experiment_definitions.py](..%2Fsrc%2Ffairnesseval%2Fexperiment_definitions.py) you can find example experiment
definitions, these include the experiments used in the demonstration video and in the demo article.

```python
sigmod_datasets = ['adult', 'compas', 'german', ]
TRAIN_FRACTIONS_SMALLER_DATASETS_v1 = [0.063, 0.251, 1.]
BASE_EPS_V1 = [0.005]
demo_examples = [
    {
        'experiment_id': 'live.demo.0',
        'dataset_names': ['adult', 'compas', 'german', ],
        'model_names': ['expgrad'],
        'model_params':
            {'eps': 0.005,
             'base_model_code': ['lr'],
             'constraint_code': 'dp',
             'base_model_grid_params': {'C': [0.1, 1]}},
        'train_fractions': [0.063, 0.251, 1.],
        'random_seeds': [0],
    },

    {
        'experiment_id': 'live.demo.1',
        'dataset_names': ['adult', 'compas', 'german', ],
        'model_names': ['ThresholdOptimizer', 'Feld', 'Calmon', 'ZafarDI', 'ZafarEO', ],
        'model_params': {'base_model_code': 'lr',
                         'base_model_grid_params': {'C': [0.1, 1]}},
        'train_fractions': [0.25],
        'random_seeds': [0],
    },

    {'experiment_id': 'demo.D.0r',
     'dataset_names': ['adult', 'compas', 'german', ],
     'model_names': ['expgrad'],
     'model_params': {'base_model_code': 'lr',
                      'constraint_code': 'dp',
                      'eps': [0.005],
                      'base_model_grid_params': {'C': [0.1, 1]}},
     'random_seeds': [0],
     'train_fractions': [0.063, 0.251, 1.],
     },
    {'experiment_id': 'demo.D.1r',
     'dataset_names': ['adult', 'compas', 'german', ],
     'model_names': ['LogisticRegression'],
     'random_seeds': [0],
     'train_fractions': [0.063, 0.251, 1.],
     },

    {'experiment_id': 'demo.A.1r',
     'dataset_names': ['adult'],
     'model_names': ['ThresholdOptimizer', 'Feld', 'ZafarDI', ],
     'model_params': {'base_model_code': ['lr'],
                      'constraint_code': 'dp',
                      'base_model_grid_params': {'C': [0.1, 1]}},
     'random_seeds': [0],
     },
]


```

