# prova4.py
import os

import streamlit as st
import matplotlib.pyplot as plt
from fairnesseval.graphic import utils_results_data
from fairnesseval.graphic.graphic_demoB import plot_function_B
from fairnesseval.utils_general import get_project_root
from components.style import st_normal

# Directory for results
res_dir = os.path.join(get_project_root(), 'streamlit','demo_results')



# Upload data
def load_results(experiment_codes):
    if len(experiment_codes) > 0:
        try:
            results_df = utils_results_data.load_results_experiment_id(experiment_codes, res_dir)
            return results_df
        except Exception as e:
            st.error(f"Error during download data: {str(e)}")
            return None
    else:
        return None

# Function for generating graph
def generate_graph(chart_name, experiment_code_list, model_list, x_axis, y_axis_list, grouping_col, dataset_list):
    params = {
        'chart_name': chart_name,
        'experiment_code_list': experiment_code_list,
        'model_list': model_list,
        'x_axis': x_axis,
        'y_axis_list': [y_axis_list],
        'grouping_col': grouping_col,
        'dataset_list': [dataset_list]
    }

    # Check avaiable parameters
    for key, value in params.items():
        if key in ['grouping_col']:
            continue
        if not value or (isinstance(value, list) and len(value) < 1):
            st.warning(f"Missing parameters: {key}")
            return

    # Graphic generation
    try:
        print(params)
        fig = plot_function_B(**params, res_path=res_dir, single_plot=False, show=False)
        st.pyplot(fig)  # Mostra il grafico con Streamlit
    except Exception as e:
        st.error(f"Error during graphic generation: {str(e)}")

# Definition for page 2 for menu
def stpage02_single_plot():
    with st_normal():
        # title
        st.title("Presentation Single Dataset")

        # Select graphic's name
        chart_name = st.text_input('Graphic name', 'demo.A')

        # Seelect avaiable experiment
        experiment_list = sorted(utils_results_data.available_experiment_results(res_dir))
        selected_experiments = st.multiselect('Experiment', experiment_list)

        # Loading data
        if selected_experiments:
            results_df = load_results(selected_experiments)
            if results_df is not None:
                # Select model
                model_list = results_df['model_code'].unique().tolist()
                selected_models = st.multiselect('Models', model_list, default=model_list)

                # Select attributes for axis X and Y
                columns = results_df.columns.tolist()
                x_axis = st.selectbox('X-Attributes', sorted(columns))
                y_axis = st.selectbox('Y-Attributes', sorted(columns))

                # Select dataset
                datasets = results_df['dataset_name'].unique().tolist()
                selected_dataset = st.selectbox('Select Dataset', datasets)

                # Seelect grouping attribute (optionale)
                grouping_col = st.selectbox('Grouping attribute (Optional)', [None] + sorted(columns))

                # Button to generate graphic
                if st.button('Generate graphic'):
                    generate_graph(chart_name, selected_experiments, selected_models, x_axis, y_axis, grouping_col,
                                   selected_dataset)
            else:
                st.warning("No experiment available with the data you have selected.")
        else:
            st.warning("Select one experiment.")