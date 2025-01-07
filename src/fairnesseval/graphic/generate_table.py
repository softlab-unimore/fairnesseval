import itertools
import os
import pandas as pd
from fairnesseval.experiment_routine import new_exp_done
from fairnesseval.graphic import utils_results_data, graphic_utility
from fairnesseval.graphic.style_utility import replace_words
from fairnesseval.graphic.utils_results_data import prepare_for_plot
from fairnesseval.graphic.graphic_utility import plot_all_df_subplots, PlotUtility
from fairnesseval.utils_experiment_parameters import DEFAULT_RESULTS_PATH
from fairnesseval.utils_general import get_project_root

if __name__ == '__main__':

    experiment_code_list = new_exp_done
    # remove exp containinng rlp_F
    experiment_code_list = [exp for exp in experiment_code_list if 'rlp_F' not in exp]
    rlp_false_conf_list = [
        'rlp_F_1.0N',
        'rlp_F_1.1N',
        'rlp_F_1.12N',
    ]

    dataset_results_path = DEFAULT_RESULTS_PATH

    base_plot_dir = os.path.join(DEFAULT_RESULTS_PATH, 'plots')
    results_df = utils_results_data.load_results_experiment_id(experiment_code_list, dataset_results_path)

    rlp_df = utils_results_data.load_results_experiment_id(rlp_false_conf_list, dataset_results_path)
    model_code = 'RLP=' + rlp_df['run_linprog_step'].map({True: 'T', False: 'F'}) + ' max_iter=' + rlp_df[
        'max_iter'].astype(str)
    rlp_df['model_code'] = model_code
    rlp_df = utils_results_data.best_gap_filter_on_eta0(rlp_df)
    rlp_df_filtered_v2 = rlp_df[rlp_df['max_iter'].isin([50])]

    all_df = pd.concat([results_df, rlp_df_filtered_v2])

    restricted_v2 = ['unconstrained', 'Calmon', 'Feld', 'ZafarDI', 'ZafarEO', 'ThresholdOptimizer']
    sort_map = {name: i for i, name in enumerate(restricted_v2)}
    all_df = all_df.assign(model_sort=all_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, True, True]).drop(columns=['model_sort'])
    # all_df.loc[all_df['model_code'].str.contains('unconstrained'), 'eps'].unique()
    all_df = all_df[all_df['phase'] != 'evaluation']
    # drop nan cols
    all_df = all_df.dropna(axis=1, how='all')

    aggregated_results = prepare_for_plot(all_df, grouping_col=['eps', 'subsample'])
    table_save_path = PlotUtility.get_base_path_static(base_plot_dir, 'table', '')
    table_save_path = os.path.join(table_save_path, 'table_results.csv')
    os.makedirs(os.path.dirname(table_save_path), exist_ok=True)
    # find cols with nan
    cols_with_nan = aggregated_results.columns[aggregated_results.isna().any()].tolist()
    for col, dtype in aggregated_results.dtypes.items():
        if dtype == 'object':
            aggregated_results[col] = aggregated_results[col].astype(str).apply(replace_words)

    aggregated_results.columns = [replace_words(' '.join(x.replace('_', ' ').split())) for x in
                                  aggregated_results.columns]
    cols_to_delete = ['best gap', 'best iter', 'eps', 'eta0', 'last iter', 'max iter', 'n oracle calls',
                      'n oracle calls dummy returned', 'random seed', 'subsample', ]
    cols_to_delete_expanded = [f'{replace_words(col)} {suffix}' for col in cols_to_delete for suffix in
                               ['mean', 'std', 'error']]
    aggregated_results = aggregated_results.drop(columns=cols_to_delete_expanded)
    aggregated_results.to_csv(table_save_path, index=False)
    aggregated_results.to_csv(os.path.join(get_project_root(), 'table_results.csv'), index=False)
    # try to read the table
    # aggregated_results = pd.read_csv(table_save_path)

    aggregated_results = prepare_for_plot(all_df, grouping_col=['eps', 'subsample'], return_multi_index=True)
    # split columns names by '__' and create a 2 level index
    # aggregated_results.columns = pd.MultiIndex.from_tuples([x.split('__') for x in aggregated_results.columns])
    # aggregated_results.columns = pd.MultiIndex.from_frame(
    #     aggregated_results.columns.to_frame().fillna('')
    # )
    aggregated_results = aggregated_results.reset_index()
    # replace '_' with ' ' in the columns
    for i, l in enumerate(aggregated_results.columns.levels):
        aggregated_results.columns = aggregated_results.columns.set_levels(
            l.fillna('').str.replace('_', ' ').str.strip(), level=i)
    # apply replace_words to the columns of type str or object
    for col, dtype in aggregated_results.dtypes.items():
        if dtype == 'object':
            aggregated_results[col] = aggregated_results[col].astype(str).apply(replace_words)

    index_cols = ['dataset name', 'model code', 'base model code', 'constraint code', 'eps', 'subsample']
    aggregated_results = aggregated_results.set_index([(x, '') for x in index_cols])
    # remove columns with error as 2nd level
    aggregated_results = aggregated_results.loc[:, aggregated_results.columns.get_level_values(1) != 'std']
    # remo columns as 1st level .['best gap','best iter','eps mean','eta0 mean','last iter','max iter mean' ,'n oracle calls'
    # 'n oracle calls dummy returned','random seed mean','subsample mean',]

    aggregated_results = aggregated_results.drop(columns=cols_to_delete, level=0)

    # aggregated_results.to_latex(buf=table_save_path.replace('.csv', '.tex'),
    #                             # only 3 decimal places
    #                             float_format="%.3f",
    #
    #                             )

    # aggregated_results.to_latex(clines="skip-last;data", )
    #

    cols = aggregated_results.columns.levels[0].copy()
    # remove the following words from the columns
    to_remove = ['train_', 'test_', '_mean']
    for word in to_remove:
        cols = cols.str.replace(word, '')
    sorted(set(cols.tolist()))
    print(f"Saved table to {table_save_path}")
