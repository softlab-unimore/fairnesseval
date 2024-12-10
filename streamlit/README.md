# Fairnesseval Demonstration

A demonstration web application for FairnessEval is available.  
To run the demonstration, navigate to the `streamlit` folders and run the streamlit server:

```bash
cd fairnesseval
cd streamlit 
streamlit run Farnesseval.py
```

## Example Experiment

In [experiment_definitions.py](..%2Fsrc%2Ffairnesseval%2Fexperiment_definitions.py) you can find example experiment
definitions, these include the experiments used in the demonstration video and in the demo article.

###

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

