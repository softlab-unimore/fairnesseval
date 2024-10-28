# prova5_1.py
import streamlit as st
import matplotlib.pyplot as plt
from fairnesseval.graphic import utils_results_data
from fairnesseval.graphic.graphic_demoB import plot_function_B

# Directory for results
res_dir = '../streamlit/demo_results'

# Load results
def load_results(experiment_codes):
    if len(experiment_codes) > 0:
        try:
            results_df = utils_results_data.load_results_experiment_id(experiment_codes, res_dir)
            return results_df
        except Exception as e:
            st.error(f"Error during data downloading: {str(e)}")
            return None
    return None

# Function for generating graph
def generate_graph(chart_name, experiment_code_list, model_list, x_axis, y_axis_list, grouping_col, available_columns):
    # Controlla se le colonne selezionate sono presenti nel DataFrame
    missing_columns = [col for col in [x_axis] + y_axis_list if col not in available_columns]
    if missing_columns:
        st.error(f"Missing columns in DataFrame: {missing_columns}")
        return

    params = {
        'chart_name': chart_name,
        'experiment_code_list': experiment_code_list,
        'model_list': model_list,
        'x_axis': x_axis,
        'y_axis_list': y_axis_list,
        'grouping_col': grouping_col,
    }

    #Genereting graph
    try:
        fig = plot_function_B(**params, res_path=res_dir, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during graphic generation: {str(e)}")

# Function for menu
def pagina3():
    # Titolo
    st.title("Presentation Multiple Plots")

    # Select graphic's name
    chart_name = st.text_input('Graphic name', 'demo.B')

    # Select avaiable experiment
    experiment_list = sorted(utils_results_data.available_experiment_results(res_dir))
    selected_experiments = st.multiselect('Experiment', experiment_list)

    # Load data
    if selected_experiments:
        results_df = load_results(selected_experiments)
        if results_df is not None:
            # Select models to show
            model_list = results_df['model_code'].unique().tolist()
            selected_models = st.multiselect('Models', model_list, default=model_list)

            # Select attribute for  X  Y
            columns = results_df.columns.tolist()
            x_axis = st.selectbox('X-Attributes', sorted(columns))
            y_axis = st.multiselect('Y-Attributes', sorted(columns))

            # Select attribute to group (optional)
            grouping_col = st.selectbox('Grouping attributes (Optional)', [None] + sorted(columns))

            # Botton to generate graphic
            if st.button('Generate graphic'):
                generate_graph(chart_name, selected_experiments, selected_models, x_axis, y_axis, grouping_col, columns)
        else:
            st.warning("No experiment available with the data you have selected")
    else:
        st.warning("Select one experiment.")