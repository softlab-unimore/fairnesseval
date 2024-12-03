import streamlit as st


def page05():
    st.title("Dataset Generator")

    with st.form(key='dataset_generator_form'):
        n_samples = st.number_input('Number of samples:', min_value=1, value=int(1e6))
        n_features = st.number_input('Number of features:', min_value=1, value=5)

        group_values = st.text_input('Sensitive attribute values (space-separated):', '0 1')
        group_values = list(map(int, group_values.split()))

        group_probabilities = st.text_input('Probabilities of each sensitive attribute value (space-separated):', '0.55 0.45')
        group_probabilities = list(map(float, group_probabilities.split()))

        group_target_probabilities = st.text_input('Target probabilities for each sensitive attribute value (space-separated):', '0.5 0.3')
        group_target_probabilities = list(map(float, group_target_probabilities.split()))

        neg_to_pos_target_flip_prob = st.text_input('Neg to pos target flip probabilities (space-separated):', '0.15 0.1')
        neg_to_pos_target_flip_prob = list(map(float, neg_to_pos_target_flip_prob.split()))

        pos_to_neg_target_flip_prob = st.text_input('Pos to neg target flip probabilities (space-separated):', '0.1 0.15')
        pos_to_neg_target_flip_prob = list(map(float, pos_to_neg_target_flip_prob.split()))

        eps = st.text_input('Epsilon for feature randomization (space-separated):', '0.0 0.1 0.2 0.3 0.4')
        eps = list(map(float, eps.split()))

        random_seed = st.number_input('Random seed:', min_value=0, value=42)

        output_path = st.text_input('Output path for the generated dataset:', './datasets/default_dataset.csv')

        submit_button = st.form_submit_button("Generate Dataset")

        if submit_button:
            # df = generate_dataset(n_samples, n_features, group_values, group_probabilities,
            #                       group_target_probabilities, neg_to_pos_target_flip_prob,
            #                       pos_to_neg_target_flip_prob, eps, random_seed, output_path)
            st.write("Dataset generated successfully!")
            # st.dataframe(df.head())  # Display first few rows of the generated dataframe

if __name__ == "__main__":
    page05()