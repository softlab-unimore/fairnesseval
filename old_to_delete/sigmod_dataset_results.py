import os

import pandas as pd

from .graphic_utility import PlotUtility
from .utils_results_data import load_results, filter_results, get_info

if __name__ == '__main__':
    save = True
    show = False
    df_list = []

    datasets = [
        'compas',
        'german',
        'adult'
    ]

    dataset_results_path = os.path.join("../results", "fairlearn-2")
    for dataset_name in datasets:
        dirs_df = load_results(dataset_results_path, dataset_name, read_files=True)
        df_list.append(dirs_df)
    all_dirs_df = pd.concat(df_list)

    random_seed = 2
    train_test_seed = 1
    all_model_df = filter_results(all_dirs_df, conf=dict(
        exp_fraction='VARY',
        random_seed=str(random_seed),
        train_test_split=str(train_test_seed),
    ))
    all_model_df.query('phase == "evaluation"')
    df = all_model_df.query('phase == "evaluation"').groupby(['model_code', 'frac', 'dataset_name', 'eps'],
                                                             as_index=False).agg('mean')
    df = df[df['frac'].isin([0.251, 1])]
    df = df[df['model_code'].isin(PlotUtility.to_plot_models)]
    pl_util = PlotUtility()


    host_name, current_time_str = get_info()

    for dataset_name in datasets:
        dataset_plot_path = os.path.join(pl_util.base_plot_dir, host_name, dataset_name)
        os.makedirs(dataset_plot_path, exist_ok=True)
        turn_df = df.query(f'dataset_name == "{dataset_name}"').loc[:, [
                        'model_code', 'frac', 'eps',
                        # 'dataset_name',  'random_seed','iterations', 'total_train_size', 'total_test_size', 'exp_frac',
                        # 'time', 'train_test_fold', 'sample_seed', 'grid_frac',
                        # 'test_error',

                        'test_accuracy', 'test_precision','test_recall', 'test_f1',
                        'test_di', 'test_violation', 'test_TPRB', 'test_TNRB',
                        # 'train_error',
                        'train_accuracy', 'train_precision', 'train_recall','train_f1',
                        'train_di', 'train_violation', 'train_TPRB', 'train_TNRB',
                                                                       ]]

        turn_df.to_csv(os.path.join(dataset_plot_path, f'performance-single-split-rs{random_seed}-tts{train_test_seed}.csv'))

    # all_model_df.query('phase == "evaluation"').groupby(['model_name', 'frac', 'dataset_name']).size
    # all_model_df.query('phase == "evaluation"').sort_values(['model_name', 'frac', 'dataset_name'])
    all_results_df = pd.concat(all_dirs_df['df'].values)
