import shutil
import unittest
import os

from fairnesseval.experiment_definitions import sigmod_datasets
from fairnesseval.graphic.utils_results_data import load_results_experiment_id
from fairnesseval.run import ExperimentRun, launch_experiment_by_config


class TestModels(unittest.TestCase):

    def setUp(self):
        self.base_config = {
            'results_path': os.path.join(os.path.dirname(__file__), 'test_results_to_delete'),
        }

    def tearDown(self):
        # Clean up the test results directory after the test
        try:
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
        self.check_results_pre_post_pre_post(config)

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
        self.check_results_pre_post_pre_post(config)

    def test_expgrad(self):
        config = self.base_config | {
            'experiment_id': 'expgrad',
            'dataset_names': sigmod_datasets,
            'model_names': ['expgrad'],
            'model_params': {'eps': [0.005, 0.15],
                             'constraint_code': 'dp',
                             'base_model_code': 'lr',
                             'base_model_grid_params': {'C': [0.1, 1]}},
            'random_seeds': [1],
            'train_test_fold': [0],
        }
        self.check_results_pre_post(config)
        res_df = load_results_experiment_id(experiment_code_list=[config['experiment_id']],
                                            results_path=config['results_path'])
        # eps has unique values that are the same as the values in the config
        self.assertEqual(res_df['eps'].unique().tolist(), config['model_params']['eps'])


    def check_results_pre_post(self, config):
        launch_experiment_by_config(config)
        base_model = config['model_params']['base_model_code']
        res_path = os.path.join(config['results_path'], config['experiment_id'])
        self.assertTrue(os.path.exists(os.path.join(res_path, 'run.log')))
        for dataset in sigmod_datasets:
            self.assertTrue(
                os.path.exists(os.path.join(res_path, f"{config['experiment_id']}_{dataset}_{base_model}.csv")))


if __name__ == '__main__':
    unittest.main()
