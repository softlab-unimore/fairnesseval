import streamlit as st
import streamlit_scrollable_textbox as stx
import os
from fairnesseval import utils_experiment_parameters as exp_params, utils_prepare_data
from fairnesseval.models.models import all_available_models
from fairnesseval.run import launch_experiment_by_config
from fairnesseval.utils_general import get_project_root
from redirect import stdout, stderr, stdouterr

from components.open_folder import open_folder

# if 'log_area' not in st.session_state:
#     st.session_state['log_area'] = ''
#
# js = f"""
# <script>
#     function scroll(dummy_var_to_force_repeat_execution){{
#         var textAreas = parent.document.getElementsByClassName('st-key-log_area');
#         for (let index = 0; index < textAreas.length; index++) {{
#             textAreas[index].scrollTop = textAreas[index].scrollHeight;
#         }}
#     }}
#     scroll({len(st.session_state.log_area)})
# </script>
# """


def stpage01_exp_definition_and_execution():
    results_path = os.path.join(get_project_root(), 'streamlit', 'demo_results')
    # find an available experiment ID. Cycle 'demo.{i:02d}' until a free one is found.
    i = 0
    while os.path.exists(os.path.join(results_path, f'demo.{i:02d}')):
        i += 1

    # List of datasets and models
    all_datasets = exp_params.ACS_dataset_names + ['adult', 'compas', 'german']
    # scan folder ../datasets for other csv datasets and add them to the list
    datasets_path = os.path.join(get_project_root(), 'datasets')
    for file in os.listdir(datasets_path):
        if file.endswith('.csv'):
            all_datasets.append(file)

    model_list = all_available_models
    model_list = [model for model in model_list if 'fairlearn' not in model]

    def eval_none(x):
        if not x.strip():
            return None
        try:
            return eval(x)
        except Exception as e:
            print(f"Error evaluating input {x}: {str(e)}")
            return None

    # Streamlit title
    st.title("Experiment definition and execution")

    # Dataset selection
    selected_datasets = st.multiselect(
        'Datasets',
        sorted(all_datasets),
        # default=[sorted(all_datasets)[-3]]
    )

    # Model selection
    selected_models = st.multiselect(
        'Models',
        sorted(model_list),
        # default=[sorted(model_list)[0]]
    )

    # Model parameters input
    model_parameters = st.text_area(
        'Model parameters: enter parameters as key-value pairs. These values will be used as parameters of the model. (Optional)',
        placeholder='{"param1": value1, "param2": value2}')

    # Train fractions input
    train_fractions = st.text_input(
        'Train fractions: enter training fractions (e.g., [0.016, 0.063, 0.251, 1]). (Optional)', '')

    # Random seed input
    random_seed = st.text_input('Random Seed: enter a value or a list of values (e.g. [41,42,23]) (default: 0)', '0')

    # split_strategy
    available_split_strategies = utils_prepare_data.split_strategy_map.keys()
    split_strategy = st.selectbox('Split strategy (default: StratifiedKFold with 3 folds.)', available_split_strategies, index=1,
                                  help='Choose the split strategy. Default is StratifiedKFold with 3 folds. ')

    # train_test_fold
    if split_strategy == 'StratifiedKFold':
        train_test_fold = st.text_input('Train test fold (for StratifiedKFold, K=3): enter a value or a list of values (e.g. [0,1])',
                                        '[0, 1, 2]', help='Choose the train test fold. To run a single fold, enter a single value.'
                                                          'eg.: [0]')

    # Experiment ID
    experimentID = st.text_input('Experiment ID: enter the name of the experiment (Required)', f'demo.{i:02d}')

    # Button to run the experiment
    if st.button('Run Experiment'):
        # Check if experiment ID exists in the demo_results folder

        experiment_path = os.path.join(results_path, experimentID)

        if os.path.exists(experiment_path):
            st.warning(f"Experiment ID '{experimentID}' already exists. Please choose a different name.")
        else:
            log_area = st.empty()
            with stdouterr(to=log_area, format='code', buffer_separator='\n', max_buffer=10000):

                experiment_conf = {
                    'experiment_id': experimentID,
                    'dataset_names': selected_datasets,
                    'model_names': selected_models,
                    'random_seed': eval_none(random_seed),
                    'model_params': eval_none(model_parameters),
                    'train_fractions': eval_none(train_fractions),
                    'results_path': results_path,
                    'split_strategy': split_strategy,
                    'train_test_fold': eval_none(train_test_fold),
                    'params': ['--debug']  # Placeholder for other parameters
                }
                experiment_conf = {k: v for k, v in experiment_conf.items() if v is not None}
                try:
                    # Execute the experiment
                    launch_experiment_by_config(experiment_conf)
                    st.success("Experiment successfully completed!")
                    st.markdown( # this should scroll the log to bottom but it isn't working...
                        """<script>document.querySelector('#' + document.querySelector('div[data-baseweb="scrollable-container"]').getAttribute('id')).scrollTop = document.querySelector('  # ' + document.querySelector('div[data-baseweb=scrollable-container]').getAttribute('id')).scrollHeight;</script>""",
                        unsafe_allow_html=True)
                    # open_folder('Open results folder', results_path) # not working
                except Exception as e:
                    print(f"Error during experiment execution: {str(e)}")


if __name__ == '__main__':
    stpage01_exp_definition_and_execution()
