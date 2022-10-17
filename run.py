import sys
import json
import os
import socket
from argparse import ArgumentParser
from datetime import datetime
from warnings import simplefilter

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from baselines import run_unmitigated, run_fairlearn_full
from run_hybrids import run_hybrids
from synthetic_data import get_synthetic_data, data_split
from utils import load_data


def main(*args, **kwargs):
    simplefilter(action='ignore', category=FutureWarning)
    arg_parser = ArgumentParser()

    arg_parser.add_argument("dataset")
    arg_parser.add_argument("method")

    # For Fairlearn and Hybrids
    arg_parser.add_argument("--eps", type=float)

    # For synthetic data
    arg_parser.add_argument("-n", "--num-data-points", type=int)
    arg_parser.add_argument("-f", "--num-features", type=int)
    arg_parser.add_argument("-t", "--type-ratio", type=float)
    arg_parser.add_argument("-t0", "--t0-ratio", type=float)
    arg_parser.add_argument("-t1", "--t1-ratio", type=float)
    arg_parser.add_argument("-v", "--data-random-variation", type=int)
    arg_parser.add_argument("--test-ratio", type=float)

    # For hybrid methods
    arg_parser.add_argument("--sample-variations")
    arg_parser.add_argument("--exp-fractions")
    arg_parser.add_argument("--grid-fractions")

    args = arg_parser.parse_args()

    ####

    host_name = socket.gethostname()
    if "." in host_name:
        host_name = host_name.split(".")[-1]

    dataset = args.dataset
    method = args.method
    eps = args.eps

    if dataset == "adult":
        dataset_str = f"adult"
        X, y, A = load_data()

    elif dataset == "synth":
        num_data_pts = args.num_data_points
        num_features = args.num_features
        type_ratio = args.type_ratio
        t0_ratio = args.t0_ratio
        t1_ratio = args.t1_ratio
        random_variation = args.data_random_variation
        test_ratio = args.test_ratio
        dataset_str = f"synth_n{num_data_pts}_f{num_features}_t{type_ratio}_t0{t0_ratio}_t1{t1_ratio}_tr{test_ratio}_v{random_variation}"

        print(f"Generating synth data "
              f"(n={num_data_pts}, f={num_features}, t={type_ratio}, t0={t0_ratio}, t1={t1_ratio}, "
              f"v={random_variation})...")
        All = get_synthetic_data(
            num_data_pts=num_data_pts,
            num_features=num_features,
            type_ratio=type_ratio,
            t0_ratio=t0_ratio,
            t1_ratio=t1_ratio,
            random_seed=random_variation + 40)
        X, y, A = All.iloc[:, :-2], All.iloc[:, -2], All.iloc[:, -1]
        # assert 0 < test_ratio < 1
        # print(f"Splitting train/test with test_ratio={test_ratio}")
        # X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all = data_split(All, test_ratio)

    else:
        raise ValueError(dataset)

    results = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    to_stratify = A.astype(str) + y.astype(str)
    for train_test_fold, (train_index, test_index) in enumerate(skf.split(X, to_stratify)):
        datasets_divided = []
        for turn_index in [train_index, test_index]:
            for turn_df in [X, y, A]:
                datasets_divided.append(turn_df.iloc[turn_index])
        if method == "hybrids":
            # Subsampling process
            # num_samples = 1  # 10  # 20
            # num_fractions = 6  # 20
            # fractions = np.logspace(-3, 0, num=num_fractions)
            # fractions = [0.004]
            sample_variations = [int(x) for x in args.sample_variations.split(",")]
            exp_fractions = [float(x) for x in args.exp_fractions.split(",")]
            grid_fractions = [float(x) for x in args.grid_fractions.split(",")]
            # assert 0 <= grid_fraction <= 1
            method_str = f"hybrids"
            if len(grid_fractions) > 1:
                method_str += f'_g{grid_fractions}'
            if len(exp_fractions) > 1:
                method_str += f'_e{exp_fractions}'
            method_str += f'_eps{eps}'

            print(f"Running Hybrids with sample variations {sample_variations} and fractions {exp_fractions}, "
                  f"and grid-fraction={grid_fractions}...\n")
            turn_results = run_hybrids(*datasets_divided, eps=eps, sample_indices=sample_variations,
                                       fractions=exp_fractions, grid_fractions=grid_fractions,
                                       train_test_fold=train_test_fold)

        elif method == "unmitigated":
            turn_results = run_unmitigated(*datasets_divided, train_test_fold=train_test_fold)
            method_str = f"unmitigated"

        elif method == "fairlearn":
            turn_results = run_fairlearn_full(*datasets_divided, eps, train_test_fold=train_test_fold)
            method_str = f"fairlearn_e{eps}"
        else:
            raise ValueError(method)
        results += turn_results
    results_df = pd.DataFrame(results)
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    res_dir = f'results/{host_name}/{dataset_str}/'

    os.makedirs(res_dir, exist_ok=True)
    for prefix in [f'{current_time_str}', f'last']:
        path = os.path.join(res_dir, f'{prefix}_{method_str}.csv')
        results_df.to_csv(path, index=False)
        print(f'Saving results in: {path}')


if __name__ == "__main__":
    main()
