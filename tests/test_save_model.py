import shutil
import unittest
import os

import joblib

from fairnesseval.run import ExperimentRun, launch_experiment_by_config

class TestSaveModels(unittest.TestCase):

    def setUp(self):
        # Set up the necessary parameters for the test
        self.experiment_config = {
            'experiment_id': 'demo.0.test',
            'dataset_names': ('adult_sigmod',),
            'model_names': ('LogisticRegression',),
            'random_seeds': [1],
            'train_test_fold': [0],
            'results_path': './test_results_you_can_delete',
            'params': ['save_models']
        }

    def tearDown(self):
        # Clean up the test results directory after the test
        try:
            shutil.rmtree(self.experiment_config['results_path'])
        except:
            # try to remove the directory again
            try:
                shutil.rmtree(self.experiment_config['results_path'])
            except:
                # print(f"Could not remove directory {self.experiment_config['results_path']}")
                raise Exception(f"Could not remove directory {self.experiment_config['results_path']}")

    def test_save_model(self):
        # Launch the experiment using the configuration
        launch_experiment_by_config(self.experiment_config)

        # Check if the model file was created
        model_filename = 'LogisticRegression_adult_sigmod_1_0.pkl'
        model_filepath = os.path.join(self.experiment_config['results_path'], self.experiment_config['experiment_id'], 'models', model_filename)
        self.assertTrue(os.path.exists(model_filepath), f"Model file {model_filepath} was not created.")

        # Optionally, you can also check if the saved model can be loaded
        loaded_model = joblib.load(model_filepath)
        self.assertIsNotNone(loaded_model, "Saved model could not be loaded.")

if __name__ == '__main__':
    unittest.main()