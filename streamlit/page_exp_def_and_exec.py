import streamlit as st
import os
from fairnesseval import utils_experiment_parameters as exp_params
from fairnesseval.run import launch_experiment_by_config
from fairnesseval.utils_general import get_project_root
from redirect import stdout, stderr, stdouterr


def stpage01_exp_definition_and_execution():
    # List of datasets and models
    all_datasets = exp_params.ACS_dataset_names + ['adult', 'compas', 'german']
    # scan folder ../datasets for other csv datasets and add them to the list
    datasets_path = os.path.join(get_project_root(), 'datasets')
    for file in os.listdir(datasets_path):
        if file.endswith('.csv'):
            all_datasets.append(file)

    model_list = [
        'LogisticRegression',
        'Expgrad',
        'ThresholdOptimizer',
        'ZafarDI',
        'ZafarEO',
        'Feld'
    ]

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
        default=[sorted(all_datasets)[-3]]
    )

    # Model selection
    selected_models = st.multiselect(
        'Models',
        sorted(model_list),
        default=[sorted(model_list)[2]]
    )

    # Model parameters input
    model_parameters = st.text_area(
        'Model parameters: enter parameters as key-value pairs. These values will be used as parameters of the model. (Optional)',
        placeholder='{"param1": value1, "param2": value2}')

    # Train fractions input
    train_fractions = st.text_input(
        'Train fractions: enter training fractions (e.g., [0.016, 0.063, 0.251, 1]). (Optional)', '')

    # Random seed input
    random_seed = st.text_input('Random Seed: enter a value or a list of values (e.g. [41,42,23]) (Required)', '')

    # Experiment ID
    experimentID = st.text_input('Experiment ID: enter the name of the experiment (Required)', '')

    # Button to run the experiment
    if st.button('Run Experiment'):
        # Check if experiment ID exists in the demo_results folder
        results_path = '../streamlit/demo_results'
        experiment_path = os.path.join(results_path, experimentID)

        if os.path.exists(experiment_path):
            st.warning(f"Experiment ID '{experimentID}' already exists. Please choose a different name.")
        else:
            log_area = st.empty()
            with stdouterr(to=log_area, format='code', max_buffer=5000, buffer_separator='\n'):
                experiment_conf = {
                    'experiment_id': experimentID,
                    'dataset_names': selected_datasets,
                    'model_names': selected_models,
                    'random_seed': eval_none(random_seed),
                    'model_params': eval_none(model_parameters),
                    'train_fractions': eval_none(train_fractions),
                    'results_path': results_path,
                    'params': ['--debug']  # Placeholder for other parameters
                }

                experiment_conf = {k: v for k, v in experiment_conf.items() if v}

                try:

                    # Execute the experiment
                    launch_experiment_by_config(experiment_conf)

                    st.success("Experiment successfully completed!")
                except Exception as e:
                    print(f"Error during experiment execution: {str(e)}")
