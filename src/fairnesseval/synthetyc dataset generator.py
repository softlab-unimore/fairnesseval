import argparse
import numpy as np
import pandas as pd


class DatasetGenerator:
    def __init__(self, n_samples=1000000, n_features=32, eps=0.1):
        self.n_samples = n_samples
        self.n_features = n_features
        self.eps = eps

    def generate_dataset(self):
        # Generate sensitive attribute values
        sensitive_attributes = np.random.choice(['m', 'w', 'nb'], size=self.n_samples)

        # Generate outcomes based on sensitive attribute values
        probabilities = {'m': 0.7, 'w': 0.65, 'nb': 0.6}
        outcomes = np.random.rand(self.n_samples)
        for i, attr in enumerate(sensitive_attributes):
            outcomes[i] = 1 if np.random.rand() < probabilities[attr] else 0

        # Adjust outcomes to introduce fairness issues
        for i, attr in enumerate(sensitive_attributes):
            if outcomes[i] == 0:
                outcomes[i] = 1 if np.random.rand() < 0.2 else 0
            elif outcomes[i] == 1:
                outcomes[i] = 0 if np.random.rand() < 0.1 else 1

        # Generate additional features
        features = np.random.rand(self.n_samples, self.n_features)
        for j in range(self.n_features):
            features[:, j] = np.random.rand(self.n_samples) < (0.5 + self.eps)

        # Create DataFrame
        data = np.column_stack((sensitive_attributes, outcomes, features))
        columns = ['Sensitive Attribute'] + ['Outcome'] + [f'Feature_{i + 1}' for i in range(self.n_features)]
        df = pd.DataFrame(data, columns=columns)

        return df


def main(args):
    generator = DatasetGenerator(args.n_samples, args.n_features, args.eps)
    dataset = generator.generate_dataset()
    dataset.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Generator')
    parser.add_argument('--n_samples', type=int, default=1000000, help='Number of samples')
    parser.add_argument('--n_features', type=int, default=32, help='Number of features')
    # # List of fraction of each sensitive attribute value in the dataset
    # parser.add_argument('--group_probabilities', type=float, nargs='+',
    parser.add_argument('--group_positive_probabilities', type=float, nargs='+',
                        help='Probabilities of a positive outcome for each sensitive attribute value')
    parser.add_argument('--eps', type=float, default=0.1, help='Epsilon for feature randomization')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for the generated dataset')
    args = parser.parse_args()
    main(args)

from argparse import ArgumentParser

arg_parser = ArgumentParser()

arg_parser.add_argument('--groups')
arg_parser.add_argument('--group_prob')
arg_parser.add_argument('--y_prob')
arg_parser.add_argument('--switch_pos')
arg_parser.add_argument('--switch_neg')
