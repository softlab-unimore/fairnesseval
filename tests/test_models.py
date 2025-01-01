import copy
import itertools
import shutil
import unittest
import os

import joblib

from fairnesseval.experiment_definitions import sigmod_datasets
from fairnesseval.graphic.utils_results_data import load_results_experiment_id
from fairnesseval.run import ExperimentRun, launch_experiment_by_config
from fairnesseval.utils_experiment_parameters import ACS_SELECTED_DATASETS


class TestModels(unittest.TestCase):

    def setUp(self):
        self.base_config = {
            'results_path': os.path.join(os.path.dirname(__file__), 'test_results_to_delete'),
        }

    def tearDown(self):
        # Clean up the test results directory after the test
        try:
            if os.path.exists(self.base_config['results_path']):
                shutil.rmtree(self.base_config['results_path'])
        except:
            # try to remove the directory again
            try:
                shutil.rmtree(self.base_config['results_path'])
            except:
                # print(f"Could not remove directory {self.experiment_config['results_path']}")
                raise Exception(f"Could not remove directory {self.base_config['results_path']}")

    def test_ThresholdOptimizer(self):
        config = self.base_config | {
            'experiment_id': 'threshold_optimizer_test',
            'dataset_names': sigmod_datasets,
            'model_names': ['ThresholdOptimizer'],
            'model_params': {'base_model_code': 'lr',
                             'constraints': 'demographic_parity',
                             'base_model_grid_params': {'C': [0.1, 1]}},
            'random_seeds': [1],
            'train_test_fold': [0],
        }
        self.check_results_pre_post(config)

    def test_Calmon(self):
        config = self.base_config | {
            'experiment_id': 'Calmon_test',
            'dataset_names': sigmod_datasets,
            'model_names': ['Calmon'],
            'model_params': {'base_model_code': 'lr',
                             'base_model_grid_params': {'C': [0.1, 1]}},
            'random_seeds': [1],
            'train_test_fold': [0],
        }
        self.check_results_pre_post(config)

    def test_ZafarDI(self):
        config = self.base_config | {
            'experiment_id': 'ZafarDI',
            'dataset_names': sigmod_datasets,
            'model_names': ['ZafarDI'],
            'model_params': {'base_model_code': 'lr',
                             'base_model_grid_params': {'C': [0.1, 1]}},
            'random_seeds': [1],
            'train_test_fold': [0],
        }
        self.check_results_pre_post(config)

    def test_ZafarEO(self):
        config = self.base_config | {
            'experiment_id': 'ZafarEO',
            'dataset_names': sigmod_datasets,
            'model_names': ['ZafarEO'],
            'model_params': {'base_model_code': 'lr',
                             'base_model_grid_params': {'C': [0.1, 1]}},
            'random_seeds': [1],
            'train_test_fold': [0],
        }
        self.check_results_pre_post(config)

    def test_Feld(self):
        config = self.base_config | {
            'experiment_id': 'Feld',
            'dataset_names': sigmod_datasets,
            'model_names': ['Feld'],
            'model_params': {'base_model_code': 'lr',
                             'base_model_grid_params': {'C': [0.1, 1]}},
            'random_seeds': [1],
            'train_test_fold': [0],
        }
        self.check_results_pre_post(config)

    def test_Pleiss(self):
        config = self.base_config | {
            'experiment_id': 'Pleiss',
            'dataset_names': sigmod_datasets,
            'model_names': ['Pleiss'],
            'model_params': {'base_model_code': 'lr',
                             'base_model_grid_params': {'C': [0.1, 1]}},
            'random_seeds': [1],
            'train_test_fold': [0],
        }
        self.check_results_pre_post(config)

    # def test_expgrad(self): # extended to test base_model_grid_params
    #     config = self.base_config | {
    #         'experiment_id': 'expgrad',
    #         'dataset_names': sigmod_datasets,
    #         'model_names': ['expgrad'],
    #         'model_params': {'eps': [0.005, 0.15],
    #                          'constraint_code': 'dp',
    #                          'base_model_code': 'lr',
    #                          'base_model_grid_params': {'C': [0.1, 1]}},
    #         'random_seeds': [1],
    #         'train_test_fold': [0],
    #     }
    #     self.check_results_pre_post(config)
    #     res_df = load_results_experiment_id(experiment_code_list=[config['experiment_id']],
    #                                         results_path=config['results_path'])
    #     # eps has unique values that are the same as the values in the config
    #     self.assertEqual(res_df['eps'].unique().tolist(), config['model_params']['eps'])


    def check_results_pre_post(self, config):
        launch_experiment_by_config(copy.deepcopy(config))
        base_model = config['model_params']['base_model_code']
        res_path = os.path.join(config['results_path'], config['experiment_id'])
        self.assertTrue(os.path.exists(os.path.join(res_path, 'run.log')))
        for dataset in config['dataset_names']:
            self.assertTrue(
                os.path.exists(os.path.join(res_path, f"{config['experiment_id']}_{dataset}_{base_model}.csv")))

    def test_expgrad_and_base_model_grid_params(self):
        config = self.base_config | {
            'experiment_id': 'test_exp_base_model_grid_params',
            'dataset_names': ['adult',  'german', 'compas',  ],
            'model_names': ['expgrad'],
            'model_params': {'eps': [0.005, 0.15],
                             'constraint_code': 'dp',
                             'base_model_code': 'lr',
                             'base_model_grid_params': {'C': [0.1, 1]}},
            'random_seeds': [100],
            'train_test_fold': [0,1,2],
            'train_fractions': [0.063, 0.251, 1.0],
        }
        self.check_results_pre_post(config)
        base_model_code, random_seed = config['model_params']['base_model_code'], 0 # todo random_seed is not used for grid search.
        res_path = os.path.join(config['results_path'], )
        for dataset_str, turn_frac in itertools.product(config['dataset_names'], config['train_fractions']):
            grid_search_path = os.path.join(config['results_path'], 'tuned_models', dataset_str,
                                            f'grid_search_{base_model_code}_rnd{random_seed}_frac{turn_frac}.pkl')
            gs = joblib.load(grid_search_path)
            self.assertEqual(gs.param_grid, config['model_params']['base_model_grid_params'])

    def test_ACS(self):
        config = self.base_config | {
            'experiment_id': 'threshold_optimizer_ACS_test',
            'dataset_names': ACS_SELECTED_DATASETS,
            'model_names': ['ThresholdOptimizer'],
            'model_params': {'base_model_code': 'lr',
                             'constraints': 'demographic_parity',
                             'base_model_grid_params': {'C': [0.1, 1]}},
            'train_fractions': [.25],
            'random_seeds': [1],
            'train_test_fold': [0],
        }
        self.check_results_pre_post(config)


if __name__ == '__main__':
    unittest.main()
