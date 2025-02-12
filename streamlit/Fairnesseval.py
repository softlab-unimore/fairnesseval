# main.py
import base64

import streamlit as st
from page_exp_def_and_exec import stpage01_exp_definition_and_execution
from page_single_plot import stpage02_single_plot
from page_multiple_plot import stpage03_multiplot
from stpage_synthetic_datasets import stpage05_synthetic_generator
from stpage_eda import stpage_eda
from upload_dataset import st_dataset_upload

st.set_page_config(layout="wide")

# Title
st.title("Fairnesseval")

# Sidebar
st.sidebar.title("Menu")
pagina_selezionata = st.sidebar.radio(
    "Go to:",
    ["Welcome page",
     "Synthetic dataset generator",
     "Upload dataset",
     "EDA",
     "Experiment definition and execution",
     "Presentation single dataset",
     "Presentation multiple plots",

     ]
)

if pagina_selezionata == "Welcome page":
    # Title
    st.write("# Welcome to Fairnesseval!")

    st.image("architecture.svg", caption="Fairnesseval architecture", use_container_width=True)

    st.markdown("""
<div style="text-align: justify; line-height: 1.6;">
Automated decision-making systems can potentially introduce biases, raising ethical concerns. 
This has led to the development of numerous bias mitigation techniques. 
However, the selection of a fairness-aware model for a specific dataset often involves a process of trial and error, 
as it is not always feasible to predict in advance whether the mitigation measures provided by the model will meet the user's requirements, 
or what impact these measures will have on other model metrics such as accuracy and run time. 
Existing fairness toolkits lack a comprehensive benchmarking framework. To bridge this gap, we present FairnessEval, 
a framework specifically designed to evaluate fairness in Machine Learning models. FairnessEval streamlines dataset preparation, 
fairness evaluation, and result presentation, while also offering customization options. In this demonstration, 
we highlight the functionality of FairnessEval in the selection and validation of fairness-aware models. 
We compare various approaches and simulate deployment scenarios to showcase FairnessEval effectiveness.
</div>
""", unsafe_allow_html=True)

elif pagina_selezionata == "Synthetic dataset generator":
    stpage05_synthetic_generator()
elif pagina_selezionata == "Upload dataset":
    st_dataset_upload()
elif pagina_selezionata == "Experiment definition and execution":
    stpage01_exp_definition_and_execution()
elif pagina_selezionata == "Presentation single dataset":
    stpage02_single_plot()
elif pagina_selezionata == "Presentation multiple plots":
    stpage03_multiplot()
elif pagina_selezionata == "EDA":
    stpage_eda()
