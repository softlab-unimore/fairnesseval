import argparse
import os
import sys

import numpy as np
import pandas as pd

from fairnesseval.run import to_arg
import fairnesseval

class DatasetGenerator:
    def __init__(self, arguments):
        self.args = arguments

    def generate_dataset(self):
        rnd = np.random.RandomState(self.args.random_seed)
        # Generate an array of sensitive attribute values
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
        equal_mask = rnd_value < (0.5 + eps) # derived from Y_i with a probability of 1/2 + eps_i, and its complement(1 −Y_i ) with the remaining probability
        complement_mask = ~equal_mask
        features[complement_mask] = 1 - features[complement_mask]

        # Create DataFrame
        data = np.column_stack((sensitive_attributes, flipped_outcomes, features))
        columns = ['Sensitive Attribute'] + ['Outcome'] + [f'Feature_{i + 1}' for i in range(self.args.n_features)]
        df = pd.DataFrame(data, columns=columns)

        return df


def main():
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
    # random seed
    args = parser.parse_args()
    generator = DatasetGenerator(args)
    dataset = generator.generate_dataset()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    dataset.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    original_argv = sys.argv.copy()
    kwargs = {'n_samples': int(1e8),
              'n_features': 5,
              'group_values': [0, 1],
              'group_probabilities': [0.55, 0.45],
              'group_target_probabilities': [0.5, 0.3],
              'neg_to_pos_target_flip_prob': [0.15, 0.1],
              'pos_to_neg_target_flip_prob': [0.1, 0.15],
              'eps': [-.2, -.1, 0, .1, .2], # np.arange(5)/10 - 0.2
              'random_seed': 42,
              'output_path': fairnesseval.utils_experiment_parameters.FAIR2_SAVE_PATH + '/datasets/synth_1e9_dataset.csv',
              }
    sys.argv = to_arg([], kwargs, original_argv)

    main()
