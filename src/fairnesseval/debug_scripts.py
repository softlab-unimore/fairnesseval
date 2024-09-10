import os

import joblib

from fairnesseval import utils_experiment_parameters
from fairnesseval.run import ExperimentRun
from fairnesseval.utils_experiment_parameters import ACS_SELECTED_DATASETS

if __name__ == '__main__':
    # load finetuned configurations in a dictionary for each dataset.
    exp_run = ExperimentRun()

    # exp_run.dataset_str = 'ACSEmployment'
    # exp_run.base_result_dir = utils_experiment_parameters.FAIR2_SAVE_PATH.replace('fairlearn-2','fairlearn-3')
    # x = exp_run.load_best_params(base_model_code='lr', fraction=1.)
    # dataset_names = ACS_SELECTED_DATASETS

    directory = os.path.join(utils_experiment_parameters.FAIR2_SAVE_PATH,'ACSEmployment', 'tuned_models')
    fraction = float(1)
    base_model_code = 'lr'
    random_seed = 0
    # os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'2_grid_search_{base_model_code}_rnd{random_seed}_frac{fraction}.pkl')
    grid_clf = joblib.load(path)
    best_conf = grid_clf.best_params_

    ciao = 'ciao'

# rlp_df_filtered_v2.query(
#         '(dataset_name=="ACSEmployment" & run_linprog_step == False)')[
#             ['run_linprog_step', 'max_iter', 'dataset_name', 'eta0', 'constraint_code']].apply(
#             lambda x: '_'.join(x.astype(str)), axis=1)

# filter