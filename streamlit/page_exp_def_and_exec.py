import streamlit as st
import logging
import os
from fairnesseval import utils_experiment_parameters as exp_params
from fairnesseval.run import launch_experiment_by_config
from redirect import stdout, stderr, stdouterr


def pagina1():
    # List of datasets and models
    all_datasets = exp_params.ACS_dataset_names + ['adult', 'compas', 'german']
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
            logger.error(f"Error evaluating input {x}: {str(e)}")
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
    model_parameters = st.text_area('Model parameters: enter parameters as key-value pairs (e.g., {"param1": value1, "param2": value2}). These values will be used as parameters of the model. (Optional)', '')

    # Train fractions input
    train_fractions = st.text_input('Train fractions: enter training fractions (e.g., [0.016, 0.063, 0.251, 1]). (Optional)', '')

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
            # Create an empty area to display logs below the button
            log_area = st.empty()

            # Set up the logger to use the stdout redirection for Streamlit
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            # Clear any existing handlers
            if logger.hasHandlers():
                logger.handlers.clear()

            # Use the redirect class to capture stdout logs in the Streamlit log area
            with stdouterr(to=log_area, format='code', max_buffer=5000, buffer_separator='\n'):
                # Experiment configuration
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

                # Log the configuration before cleaning it up (formatted with new lines)

               # for key, value in experiment_conf.items():
                #    logger.info(f"{key}: {value}")

                # Remove empty values from the configuration
                experiment_conf = {k: v for k, v in experiment_conf.items() if v}

                # Log the final cleaned-up configuration (formatted with new lines)

              #  for key, value in experiment_conf.items():
               #     logger.info(f"{key}: {value}")

                # Attempt to run the experiment
                try:

                    # Execute the experiment
                    launch_experiment_by_config(experiment_conf)

                    st.success("Experiment successfully completed!")
                except Exception as e:
                    logger.error(f"Error during experiment execution: {str(e)}")

# Run the function pagina1
#pagina1()