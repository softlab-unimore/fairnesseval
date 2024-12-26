import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from fairnesseval import utils_experiment_parameters as exp_params
from fairnesseval import utils_prepare_data
from fairnesseval.utils_general import get_project_root
from fairnesseval.utils_prepare_data import eda_for_fair
from redirect import stdouterr


def stpage_eda():
    # Main Streamlit application
    st.title("Dataset Selector")

    all_datasets = exp_params.ACS_dataset_names + ['adult', 'compas', 'german']
    # scan folder ../datasets for other csv datasets and add them to the list
    datasets_path = os.path.join(get_project_root(), 'datasets')
    for file in os.listdir(datasets_path):
        if file.endswith('.csv'):
            all_datasets.append(file)

    # Dataset selection
    dataset_name = st.selectbox(
        'Select datasets',
        sorted(all_datasets),
        # default=[sorted(all_datasets)[-3]]  # Default to the last three datasets
    )
    if st.button("Analyse Dataset"):
        log_area = st.empty()
        with stdouterr(to=log_area, format='code', buffer_separator='\n'):
            try:
                # Load the dataset
                _ = utils_prepare_data.get_dataset(dataset_name)
            except Exception as e:
                st.error(f"Error during dataset generation: {str(e)}")
            st.success("Dataset loaded successfully!")

        log_area.empty()
        X, y, A = _[:3]
        dataset = pd.concat([X, y, A], axis=1)
        # remove duplicated columns
        dataset = dataset.loc[:, ~dataset.columns.duplicated()]
        mem_usage = X.memory_usage().sum() / (2 ** (10 * 3))
        t_dict = {
            'dataset_name': dataset_name,
            'size': X.shape[0],
            'columns': X.shape[1],
            'sensitive_attr': A.name,
            'sensitive_attr_nunique': A.nunique(),
            'target_col': y.name,
            'sensitive_attr_unique_values': A.unique(),
            'mem_usage(GB)': mem_usage
        }
        st.write(f"### Memory Usage and Other Details:")
        st.json(t_dict)

        # Display the description of the dataset
        st.write(f"### Description of {dataset_name}")
        st.dataframe(dataset.describe())

        res = eda_for_fair(X, y, A)
        for key, value in res.items():
            st.write(f"### Stats of target over sensitive feature: {key}")
            st.dataframe(value)



        # # Create and display a pairplot
        # st.write(f"### Pairplot of {dataset_name}")
        # sns.pairplot(dataset)
        # plt.title(f"Pairplot of {dataset_name}")
        # st.pyplot(plt)
