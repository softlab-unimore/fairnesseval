import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from fairnesseval.run import to_arg
import fairnesseval
from fairnesseval.utils_general import get_project_root


class DatasetGenerator:
    def __init__(self, arguments):
        self.args = arguments

    def generate_dataset(self):
        rnd = np.random.RandomState(self.args.random_seed)
        # Generate an array of sensitive attribute values
        if sum(self.args.group_probabilities) != 1:
            raise ValueError('Sum of group_probabilities should be equal to 1')

        sensitive_attributes = rnd.choice(self.args.group_values, size=self.args.n_samples,
                                          p=self.args.group_probabilities)
        # Generate outcomes based on sensitive attribute values
        group_probabilities = dict(zip(self.args.group_values, self.args.group_target_probabilities))
        outcomes = rnd.rand(self.args.n_samples)
        for group in self.args.group_values:
            group_mask = sensitive_attributes == group
            outcomes[group_mask] = np.where(outcomes[group_mask] < group_probabilities[group], 1, 0)

        # Flip outcomes based on sensitive attribute values
        flipped_outcomes = outcomes.copy()
        flip_rnd_value = rnd.rand(self.args.n_samples)
        neg_mask = outcomes == 0
        pos_mask = ~neg_mask
        for group in self.args.group_values:
            group_mask = sensitive_attributes == group

            neg_threshold = self.args.neg_to_pos_target_flip_prob[group]
            flipped_outcomes[group_mask & neg_mask & (flip_rnd_value < neg_threshold)] = 1

            pos_threshold = self.args.pos_to_neg_target_flip_prob[group]
            flipped_outcomes[group_mask & pos_mask & (flip_rnd_value < pos_threshold)] = 0

        # Generate additional features
        # copy outcome n_feature times in features array
        features = np.tile(flipped_outcomes, (self.args.n_features, 1)).T
        rnd_value = rnd.rand(self.args.n_samples, self.args.n_features)
        # copy eps n_outcomes times in eps array
        eps = np.tile(self.args.eps, (self.args.n_samples, 1))
        equal_mask = rnd_value < (
                0.5 + eps)  # derived from Y_i with a probability of 1/2 + eps_i, and its complement(1 −Y_i ) with the remaining probability
        complement_mask = ~equal_mask
        features[complement_mask] = 1 - features[complement_mask]

        # Create DataFrame
        data = np.column_stack((features, flipped_outcomes, sensitive_attributes))
        columns = [f'Feature_{i + 1}' for i in range(self.args.n_features)] + ['Outcome'] + ['Sensitive Attribute']
        # dtypes int
        df = pd.DataFrame(data, columns=columns, dtype=int)

        return df


def get_parser():
    parser = argparse.ArgumentParser(description='Dataset Generator')
    parser.add_argument('--n_samples', type=int, default=int(1e6), help='Number of samples in the dataset.')
    parser.add_argument('--n_features', type=int, default=5, help='Number of features in the dataset.')
    parser.add_argument('--group_values', type=int, nargs='+', default=[0, 1], help='Sensitive attribute values.')
    # # List of fraction of each sensitive attribute value in the dataset
    parser.add_argument('--group_probabilities', type=float, nargs='+', default=[0.55, 0.45],
                        help='Probabilities of each sensitive attribute value. [expected to be a list with the same '
                             'length of group_values]')
    parser.add_argument('--group_target_probabilities', type=float, nargs='+', default=[0.5, 0.3],
                        help='Probabilities of positive outcome for each sensitive attribute value. [expected to be a '
                             'list with the same length of group_values]')

    parser.add_argument('--neg_to_pos_target_flip_prob', type=float, nargs='+', default=[0.15, 0.1],
                        help='Probabilities of flipping the outcome for each sensitive attribute value from Negative '
                             'to Positive. [expected to be a list with the same length of group_values]')
    parser.add_argument('--pos_to_neg_target_flip_prob', type=float, nargs='+', default=[0.1, 0.15],
                        help='Probabilities of flipping the outcome for each sensitive attribute value from Positive '
                             'to Negative. [expected to be a list with the same length of group_values]')
    parser.add_argument('--eps', type=float, nargs='+', default=np.arange(5) / 10,
                        help='''Epsilon for feature randomization. Each feature, denoted as 𝑋𝑖 𝑗 ,
                             is derived from 𝑌𝑖 with a probability of 1/2 + eps𝑗 , and its complement(1 −𝑌𝑖 )
                              with the remaining probability. [expected to be a list of length n_features]''')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--output_path', type=str, default='./datasets/default_dataset.csv',
                        help='Output path for the generated dataset.')
    return parser


def generate_and_save_synthetic_dataset(args_list=None):
    parser = get_parser()
    if args_list is not None:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))



    generator = DatasetGenerator(args)
    dataset = generator.generate_dataset()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    dataset.to_csv(args.output_path, index=False)
    # save configurations in a json file
    config_path = os.path.join(os.path.dirname(args.output_path),
                               f'{os.path.basename(args.output_path).split(".")[0]}_config.json')
    json.dump(vars(args), open(config_path, 'w'), indent=4)
    print(f'Dataset saved in {args.output_path}')
    print(f'Configurations saved in {config_path}')



if __name__ == '__main__':
    original_argv = sys.argv.copy()
    kwargs = {'n_samples': int(1e5),
              'n_features': 5,
              'group_values': [0, 1],
              'group_probabilities': [0.55, 0.45],
              'group_target_probabilities': [0.5, 0.3],
              'neg_to_pos_target_flip_prob': [0.15, 0.1],
              'pos_to_neg_target_flip_prob': [0.1, 0.15],
              'eps': [-.2, -.1, 0, .1, .2],  # np.arange(5)/10 - 0.2
              'random_seed': 42,
              'output_path': os.path.join(get_project_root(), 'datasets', 'synth_1e5_dataset.csv')
              }
    generate_and_save_synthetic_dataset(to_arg([], kwargs))

    sys.argv = original_argv
