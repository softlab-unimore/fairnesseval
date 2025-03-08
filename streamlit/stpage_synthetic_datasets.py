import os.path

from streamlit_vertical_slider import vertical_slider

import streamlit as st
from fairnesseval.run import to_arg
from fairnesseval.synthetyc_dataset_generator import generate_and_save_synthetic_dataset
from fairnesseval.utils_general import get_project_root
from redirect import stdouterr


def create_callback(cur_key):
    def adjust_group_prob():
        num_groups = st.session_state['num_groups']
        group_probabilities = []
        key_list = []
        for i in range(num_groups):
            prob = st.session_state.get(f'prob_{i}', 0)
            group_probabilities.append(prob)
            key_list.append(f'prob_{i}')

        # for each iteration, adjust the probabilities as max between remaining probability and 0
        remaining_prob = 1 - st.session_state.get(cur_key, 0)
        for i, key in enumerate(key_list):
            if key == cur_key:
                continue
            if group_probabilities[i] >= remaining_prob:
                group_probabilities[i] = remaining_prob
                st.session_state[f'prob_{i}'] = remaining_prob
                remaining_prob = 0
            else:
                remaining_prob -= group_probabilities[i]
        key_list.remove(cur_key)
        if remaining_prob > 0:
            key = key_list[-1]
            st.session_state[key] += remaining_prob

    return adjust_group_prob


def stpage05_synthetic_generator():
    st.title("Dataset Generator")

    # with st.form(key='dataset_generator_form', clear_on_submit=False):
    n_samples = st.number_input('Number of samples:', min_value=1, value=int(1e6),
                                help='Number of samples in the dataset.')
    n_features = st.number_input('Number of features:', min_value=1, value=5, )

    # describe the epsilon for each feature
    st.markdown("Epsilon for feature randomization", help='''Epsilon for feature randomization. Each feature, denoted as 𝑋𝑖 𝑗 ,
                                                 is derived from 𝑌𝑖 with a probability of 1/2 + eps𝑗 , and its complement(1 −𝑌𝑖 )
                                                  with the remaining probability. 
            A smaller value of epsilon suggests a more challenging task for the classifier,
            thereby encouraging a thorough examination of fairness across different subsets of the data.''')
    eps_list = []
    cols = st.columns(n_features)
    for i, col in enumerate(cols):
        with col:
            tval = vertical_slider(
                label=f'f{i}',
                key=f'eps{i}',
                height=100,
                step=0.05,
                default_value=0,
                min_value=0,
                max_value=.5,
                # value_always_visible=True,
            )
            eps_list.append(tval)
    # eps = st.text_input('Epsilon for feature randomization (space-separated):', '0.0 0.1 0.2 0.3 0.4',
    #                     help='''Epsilon for feature randomization. Each feature, denoted as 𝑋𝑖 𝑗 ,
    #                                              is derived from 𝑌𝑖 with a probability of 1/2 + eps𝑗 , and its complement(1 −𝑌𝑖 )
    #                                               with the remaining probability.''')
    # eps = list(map(float, eps.split()))

    # group_values = st.text_input('Sensitive attribute values (space-separated: int):', '0 1', )
    # group_values = list(map(int, group_values.split()))
    n_group_values = st.number_input('Number of sensitive attribute values:', min_value=2, value=2)
    group_values = range(n_group_values)

    # group_probabilities = st.text_input('Probabilities of each sensitive attribute value (space-separated):',
    #                                     '0.55 0.45')
    # group_probabilities = list(map(float, group_probabilities.split()))
    #
    # group_target_probabilities = st.text_input(
    #     'Target probabilities for each sensitive attribute value (space-separated):', '0.5 0.3')
    # group_target_probabilities = list(map(float, group_target_probabilities.split()))
    #

    if group_values:
        num_groups = len(group_values)
        st.session_state['num_groups'] = num_groups

        # Create horizontal columns for group probabilities
        cols = st.columns(4)  # Create two columns for better layout

        with cols[0]:
            st.write("Group Probabilities")
        with cols[1]:
            st.write("Group Target Probabilities")
        with cols[2]:
            st.write("Negative to Positive Target Flip Probabilities")
        with cols[3]:
            st.write("Positive to Negative Target Flip Probabilities")

        cols = st.columns(4)  # Create two columns for better layout
        with cols[0]:
            group_probabilities = []

            for i in range(num_groups):
                if f'prob_{i}' not in st.session_state:
                    st.session_state[f'prob_{i}'] = (1 / num_groups)
                prob = st.slider(f'{group_values[i]}', 0.0, 1.0, value=1 / num_groups,
                                 key=f'prob_{i}', on_change= create_callback(f'prob_{i}'))  # Individual sliders
                group_probabilities.append(prob)
        with cols[1]:
            group_target_probabilities = []
            for i in range(num_groups):
                target_prob = st.slider(f'{group_values[i]}', 0.0, 1.0, 0.5,
                                        key=f'target_prob_{i}')  # Individual sliders
                group_target_probabilities.append(target_prob)
        with cols[2]:
            neg_to_pos_target_flip_prob = []
            for i in range(num_groups):
                neg_flip_prob = st.slider(f'{group_values[i]}', 0.0, 1.0, 0.15,
                                          key=f'neg_to_pos_flip_{i}',
                                          help='Probabilities of flipping the outcome for each sensitive attribute value from Negative '
                                               'to Positive. '
                                               f'Group {group_values[i]}')  # Individual sliders
                neg_to_pos_target_flip_prob.append(neg_flip_prob)

        with cols[3]:
            pos_to_neg_target_flip_prob = []
            for i in range(num_groups):
                pos_flip_prob = st.slider(f'{group_values[i]}', 0.0, 1.0, 0.15,
                                          key=f'pos_to_neg_flip_{i}',
                                          help='Probabilities of flipping the outcome for each sensitive attribute value from Positive '
                                               'to Negative. '
                                               f'Group {group_values[i]}'
                                          )  # Individual sliders
                pos_to_neg_target_flip_prob.append(pos_flip_prob)

    # neg_to_pos_target_flip_prob = st.text_input('Neg to pos target flip probabilities (space-separated):',
    #                                             '0.15 0.1',
    #                                             help='Probabilities of flipping the outcome for each sensitive attribute value from Negative '
    #                                                  'to Positive.')
    # neg_to_pos_target_flip_prob = list(map(float, neg_to_pos_target_flip_prob.split()))
    #
    # pos_to_neg_target_flip_prob = st.text_input('Pos to neg target flip probabilities (space-separated):',
    #                                             '0.1 0.15',
    #                                             help='Probabilities of flipping the outcome for each sensitive attribute value from Positive '
    #                                                  'to Negative.')
    # pos_to_neg_target_flip_prob = list(map(float, pos_to_neg_target_flip_prob.split()))

    random_seed = st.number_input('Random seed:', min_value=0, value=42)

    output_filename = st.text_input('Output name for the generated dataset:', 'demo_synth.csv')
    output_path = get_project_root()
    output_path = os.path.join(output_path, 'datasets', output_filename)

    # submit_button = st.form_submit_button("Generate Dataset")

    log_area = st.empty()

    # if submit_button:
    if st.button('Generate Dataset'):
        args_list = to_arg([], {
            'n_samples': n_samples,
            'n_features': n_features,
            'group_values': group_values,
            'group_probabilities': group_probabilities,
            'group_target_probabilities': group_target_probabilities,
            'neg_to_pos_target_flip_prob': neg_to_pos_target_flip_prob,
            'pos_to_neg_target_flip_prob': pos_to_neg_target_flip_prob,
            'eps': eps_list,
            'random_seed': random_seed,
            'output_path': output_path,
        }, )

        with stdouterr(to=log_area, format='code', max_buffer=5000, buffer_separator='\n'):
            try:
                generate_and_save_synthetic_dataset(args_list)
            except Exception as e:
                st.error(f"Error during dataset generation: {str(e)}")
            st.success("Dataset generated successfully!")
            st.stop()

        # st.dataframe(df.head())  # Display first few rows of the generated dataframe


if __name__ == "__main__":
    stpage05_synthetic_generator()
