RANDOM_SEEDs_RESTRICTED_V1 = [1]

TRAIN_FRACTIONS_SMALLER_DATASETS_v1 = [0.063, 0.251, 1.]
TRAIN_FRACTIONS_v1 = [0.001, 0.004, 0.016, 0.063, 0.251, 1]  # np.geomspace(0.001,1,7) np.linspace(0.001,1,7)

experiment_definitions = [
    {
        'experiment_id': 'demo.0.test',
        'dataset_names': ('adult_sigmod',),
        'model_names': ('LogisticRegression',),
        'random_seeds': RANDOM_SEEDs_RESTRICTED_V1,
        'results_path': './demo_results'
    },
    {
        "experiment_id": "demo.2.test",
        "dataset_names": ["compas", "german"],
        "model_names": ["LogisticRegression"],
        "random_seeds": RANDOM_SEEDs_RESTRICTED_V1,
        "train_fractions": TRAIN_FRACTIONS_SMALLER_DATASETS_v1,
        "results_path": "demo_results",
        "params": ["--debug"],
    },
    {
        'experiment_id': 'demo.1.test',
        'dataset_names': ['adult_sigmod_no_SA', 'compas_no_SA'],
        'model_names': ['fairlearn'],
        'random_seeds': [1],
        'model_params': {'eps': [0.005, 0.15],
                         'constraint_code': 'dp',
                         'base_model_code': 'lr',
                         'eta0': [2.0],
                         'run_linprog_step': [False],
                         'max_iter': [25, 50],
                         'base_model_grid_params': dict(
                             solver=[  # 'newton-cg', 'liblinear',
                                 'lbfgs'],
                             penalty=['l2'
                                      ],
                             C=[0.01,  1, 1000],
                             # [10, 1.0, 0.1, 0.05, 0.01],
                             max_iter=[200],
                         )
                         },
        'results_path': 'demo_results',
        'train_test_fold': [0],
        'params': ['--debug']
    },
    {
        "experiment_id": "demo.x.test",
        'dataset_names': ['adult_sigmod'],
        'model_names': ['LogisticRegression'],
        'random_seeds': RANDOM_SEEDs_RESTRICTED_V1,
    },

]
