# main.py
import streamlit as st
from page_exp_def_and_exec import pagina1  #MODIFICARE
from page_single_plot import pagina2 #MODIFICARE
from page_multiple_plot import pagina3 #MODIFICARE


# Title
st.title("Fairnesseval")



# Sidebar
st.sidebar.title("Menu")
pagina_selezionata = st.sidebar.radio(
    "Go to:",
    ["Welcome page", "Experiment definition and execution", "Presentation single dataset", "Presentation multiple plots"]
)



if pagina_selezionata == "Welcome page":
    # Title
    st.write("# Welcome to Fairnesseval!")

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

elif pagina_selezionata == "Experiment definition and execution":
    pagina1()
elif pagina_selezionata == "Presentation single dataset":
    pagina2()
elif pagina_selezionata == "Presentation multiple plots":
    pagina3()