import ast
import itertools
import logging
import os
import re
import socket
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import sem, t

from fairnesseval import utils_experiment_parameters
from fairnesseval.run import params_initials_map

seed_columns = ['random_seed', 'train_test_fold', 'train_test_seed']
cols_to_synch = ['dataset_name', 'base_model_code', 'constraint_code', 'eps', 'train_test_seed', 'random_seed',
                 'train_test_fold', ]
cols_to_index = ['dataset_name', 'base_model_code', 'constraint_code', 'eps', 'model_code', 'method', 'exp_frac',
                 'grid_frac']

time_columns = ['metrics_time', 'time', 'grid_oracle_times']
numerical_cols = ['time', 'train_error', 'train_accuracy', 'test_accuracy',
                  'train_violation', 'train_di', 'train_TPRB', 'train_TNRB', 'train_f1',
                  'train_precision', 'train_recall', 'test_error', 'test_violation',
                  'test_di', 'test_TPRB', 'test_TNRB', 'test_f1', 'test_precision',
                  'test_recall',  # 'total_train_size', 'total_test_size',
                  'n_oracle_calls_', 'n_oracle_calls_dummy_returned_', ]
non_numeric_cols = ['best_iter_', 'best_gap_', 'last_iter_',
                    'oracle_execution_times_', 'metrics_time', 'grid_oracle_times',
                    'model_code',
                    'moving_param']

suffix_attr_map = {
    'eps': 'eps',
    'exp': 'exp_frac',
    'gri': 'grid_frac',
}

constraint_code_to_name = {'dp': 'DemographicParity', 'eo': 'EqualizedOdds'}


def get_numerical_cols(df):
    num_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
    return num_cols


def add_sigmod_metric(df):
    for split in ['train', 'test']:
        df[split + '_di'] = pd.concat([df[split + '_di'], 1 / df[split + '_di']], axis=1).min(axis=1)
        df[split + '_accuracy'] = 1 - df[split + '_error']
        df[split + '_TPRB'] = 1 - df[split + '_TPRB']
        df[split + '_TNRB'] = 1 - df[split + '_TNRB']
    return df


def calculate_movign_param(path, df: pd.DataFrame):
    cols_to_check = suffix_attr_map.values()
    if np.intersect1d(list(cols_to_check), df.columns).shape[0] < len(cols_to_check):
        df['frac'] = 1
        df['model_code'] = df['model_name']
        return df
    suffix = ''
    if path is not None:
        for key, name in suffix_attr_map.items():
            if f'_{key}[' in path:
                suffix += f'{key}'

    if suffix == '':
        for key, name in suffix_attr_map.items():
            if df[name].nunique() > 1:
                suffix = f'{key}'
                break
    df['moving_param'] = suffix
    df['model_code'] = df['model_name'] + '_' + df['moving_param']
    # np.where(df['moving_param'] != '', '_' + df['moving_param'], '')

    for key, col in suffix_attr_map.items():
        mask = df['moving_param'] == key
        df.loc[mask, 'frac'] = df.loc[mask, col]
    # fix_expgrad_times(df)
    return df


def take_max_for_grid_search(df):
    # Take max of oracle calls time for grid search
    grid_mask = df['phase'] == 'grid_frac'
    grid_time_series = df[grid_mask]['grid_oracle_times'].apply(
        lambda x: np.array(ast.literal_eval(x)).max())
    df.loc[grid_mask, 'time'] = grid_time_series
    return df


def prepare_data(df):
    df = df.reset_index(drop=True)
    if 'dataset' in df.columns:
        df['dataset_name'] = df['dataset'].str.replace('_aif360', '').str.replace('_sigmod', '')
    df = add_sigmod_metric(df)
    if 'model_code' not in df.columns:
        df.rename(columns={'model_name': 'model_code'}, inplace=True)

    if 'method' in df.columns:
        expgrad_mask = df['method'] == 'hybrids'
        hybrid = df[expgrad_mask].copy()
        non_hybrid = df[~expgrad_mask]
        if not hybrid.empty:
            hybrid['frac'].fillna(1, inplace=True)
            # hybrid = calculate_movign_param(None, hybrid)
            if not df[df['phase'] == 'grid_frac'].empty:
                hybrid = take_max_for_grid_search(hybrid)
                hybrid['eps'] = pd.to_numeric(hybrid['eps'], errors='coerce')
                models_with_gridsearch = hybrid['model_code'].isin(
                    hybrid.query('phase == "grid_frac"')['model_code'].unique())
                hybrid.loc[~models_with_gridsearch, 'grid_frac'] = 0
        df = pd.concat([hybrid, non_hybrid])
    return df



def add_missing_columns(df):
    if 'train_test_seed' not in df.columns:
        df['train_test_seed'] = 0
    return df


def select_set_expgrad_time(df: pd.DataFrame) -> pd.DataFrame:
    sub_mask = df['model_name'].str.contains('sub')
    lp_off_mask = df['model_name'].str.contains('LP_off')
    no_h7_mask = ~(df['model_name'].str.contains('hybrid_7'))
    for lp_mask in [lp_off_mask, ~lp_off_mask]:
        if lp_mask.any():
            turn_mask = ~sub_mask & lp_mask & no_h7_mask  # not sub is static sampling
            expgrad_time = df[turn_mask & (df['model_name'].str.contains('expgrad_frac'))]['time'].values[0]
            df.loc[turn_mask, 'time'] = expgrad_time

            turn_mask = sub_mask & lp_mask  # sub is adaptive sampling --> take h7 that is expgrad adaptive
            expgrad_time = df[lp_mask & (df['model_name'].str.contains('hybrid_7'))]['time'].values[0]
            df.loc[turn_mask, 'time'] = expgrad_time
    return df


def filter_results(dirs_df, conf: dict = {}):
    for key, value in conf.items():
        if not isinstance(value, list):
            value = [value]
        if key in dirs_df.columns:
            dirs_df = dirs_df[dirs_df[key].isin(value)]
        else:
            if key == 'states':
                pass
            else:
                assert False, f'\'{key}\' is not a valid key for filter. values available are: {dirs_df.columns.values}'
    if dirs_df.empty:
        return dirs_df
    df = pd.concat(dirs_df['df'].values)
    df = add_sigmod_metric(df)
    return df.reset_index(drop=True)


def read_experiment_configuration(path):
    config = dict(dir=path)
    model = re.findall(r'(?P<model>^[a-zA-Z]+)_', path)[0]
    g = re.findall(r'(?P<name>[a-z]+)\((?P<value>[a-zA-Z0-9\.]+)\)\_?', path)
    config = {params_initials_map[x[0]]: x[1] for x in g}
    config['model'] = model
    return config


def load_results(dataset_path, dataset_name, prefix='last', read_files=False):
    base_dir = os.path.join(dataset_path, dataset_name)
    path_list = pd.Series([x for x in os.scandir(base_dir) if x.is_dir() and x.name != 'tuned_models'])
    if read_files:
        path_list = pd.Series([x for x in os.scandir(base_dir) if x.is_file()])

    config_list = []
    for turn_dir in path_list:
        config = read_experiment_configuration(turn_dir.name)
        config['dataset_name'] = dataset_name
        if read_files:
            df = pd.read_csv(turn_dir)
            df = add_missing_columns(df)
            df = calculate_movign_param(turn_dir.path, df)
        else:
            df = load_results_single_directory(turn_dir.path, prefix=prefix)

        models_with_gridsearch = df.query('phase == "grid_frac"')['model_code'].unique()
        df.loc[~df['model_code'].isin(models_with_gridsearch), 'grid_frac'] = 0
        if 'grid_fractions' in config.keys() and config['grid_fractions'] == '1.0':
            # mask = ~df['model_code'].str.contains('|'.join(['expgrad_fracs', 'hybrid_7', 'unconstrained_']))
            mask = df['model_code'].isin(models_with_gridsearch)
            df.loc[mask, 'model_code'] = df.loc[mask, 'model_code'].str.replace('_gf_1', '') + '_gf_1'

        for key, value in config.items():
            if key not in df.columns:
                df[key] = value
        config['df'] = df
        config_list.append(config)
    return pd.DataFrame(config_list).fillna({'constraint_code': 'dp', 'train_test_split': '0'}).fillna('')


def load_results_single_directory(base_dir, prefix='last'):
    files = pd.Series([x.name for x in os.scandir(base_dir) if x.is_file()])
    if files.shape[0] == 0:
        print(f'empty directory {base_dir}')
    filesToScan = files[files.str.startswith(prefix)]
    # name_df = last_files.str.extract(r'^(last)_([^_]*)_?(.*)\.(.*)$', expand=True)
    # name_df.rename(columns={0: 'last', 1: 'model', 2: 'params', 3: 'extension'}, inplace=True)
    df_list = []
    for turn_file in filesToScan:
        full_path = os.path.join(base_dir, turn_file)
        df_list.append(pd.read_csv(full_path))
    all_df = pd.concat(df_list)
    all_df = add_missing_columns(all_df)
    all_df = calculate_movign_param(base_dir, all_df)

    if 'rls(False)' in base_dir:
        to_drop = all_df[all_df['model_name'].isin(['unconstrained', 'unconstrained_frac'])].index
        all_df.drop(index=to_drop, inplace=True)

    return all_df


def fix_expgrad_times(df: pd.DataFrame) -> pd.DataFrame:
    expgrad_phase_mask = df['phase'] == "expgrad_fracs"
    expgrad_df = df[expgrad_phase_mask]
    to_group_cols = np.intersect1d(seed_columns + ['frac'], expgrad_df.columns).tolist()
    expgrad_df = expgrad_df.groupby(to_group_cols).apply(select_set_expgrad_time)
    df[expgrad_phase_mask] = expgrad_df


def aggregate_phase_time(df):
    turn_time_columns = list(set(df.columns[df.columns.str.contains('time')].tolist() + time_columns))
    cols_to_group = np.setdiff1d(df.columns, turn_time_columns + ['phase']).tolist()
    results_df = df.groupby(cols_to_group,
                            as_index=False, dropna=False, sort=False)[turn_time_columns].sum(min_count=1)
    return results_df


def get_info():
    host_name = socket.gethostname()
    if "." in host_name:
        host_name = host_name.split(".")[-1]

    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return host_name, current_time_str


def get_last_results_datetime(base_dir):
    files = pd.Series(os.listdir(base_dir))
    name_df = files.str.extract(r'^(\d{4}-\d{2}-\d{2})_((?:\d{2}-{0,1}){3})_(.*)\.(.*)$', expand=True)
    name_df.rename(columns={0: 'date', 1: 'time', 2: 'model', 3: 'extension'}, inplace=True)
    name_df['full_name'] = files
    name_df = name_df.query('extension == "csv"')
    last_files = name_df.sort_values(['date', 'time'], ascending=False).groupby('model').head(1)
    df_dict = {model_name: pd.read_csv(os.path.join(base_dir, turn_name))
               for turn_name, model_name in (last_files[['full_name', 'model']].values)}
    all_model_df = pd.concat(df_dict.values())
    return all_model_df


def get_confidence_error(data, confidence: float = 0.95):
    if data is None:
        return [np.nan, np.nan, np.nan]

    a = np.asarray(data).astype(float)
    n = len(a)
    m, se = np.nanmean(a, 0), sem(a, nan_policy="omit", ddof=1)
    t_value = t.ppf((1.0 + confidence) / 2., n - 1)
    h1 = m - se * t_value
    h2 = m + se * t_value

    return (h2 - h1) /2


def mean_confidence_interval(data, confidence: float = 0.95):
    """
    Args:
        data:
        confidence:

    Returns:
        mean and confidence limit values for the given data and confidence
    """
    if data is None:
        return [np.nan, np.nan, np.nan]

    a = np.asarray(data).astype(float)
    n = len(a)
    m, se = np.nanmean(a, 0), sem(a, nan_policy="omit", ddof=1)
    t_value = t.ppf((1.0 + confidence) / 2., n - 1)
    h1 = m - se * t_value
    h2 = m + se * t_value
    return np.array([m, h1, h2])


def add_combined_stats(df, alphas=[.05, .5, .95]):
    not_combined_df = df.loc[df['model_name'] != "combined"]
    cols_to_group = ['eps', 'frac', 'random_seed', 'grid_frac', 'n']
    cols_to_group = np.intersect1d(cols_to_group, df.columns).tolist()
    combo_stat_list = []
    for alpha in alphas:
        turn_f = partial(get_combined_groupby, alpha=alpha)
        combined_stats = df.groupby(cols_to_group, as_index=False).apply(turn_f)
        combo_stat_list.append(combined_stats.copy())
    df = pd.concat(combo_stat_list + [not_combined_df]).drop_duplicates().reset_index(drop=True)
    return df


def get_combined_hybrid(train_err_hybrids, train_vio_hybrids, alpha):
    # alpha = importance of error vs. violation
    n = len(train_err_hybrids)
    if len(train_vio_hybrids) != n:
        raise Exception()
    scores = [
        alpha * train_err_hybrids[i] + (1 - alpha) * train_vio_hybrids[i]
        for i in range(n)
    ]
    best_index = scores.index(min(scores))
    return best_index


def get_combined_groupby(x, alpha=0.5):
    hybrid_res = x[x['model_name'].str.startswith('hybrid')]
    composed_metric = hybrid_res['train_violation'] * (1 - alpha) + hybrid_res['train_error'] * alpha
    combo_res = hybrid_res.loc[composed_metric.idxmin()]
    comb_df = x[x['model_name'] == 'combined']
    for col in np.setdiff1d(comb_df.columns,
                            ['eps', 'frac', 'model_name', 'time', 'phase', 'random_seed', 'grid_frac', 'n']):
        comb_df[col] = combo_res[col]
    comb_df['alpha'] = alpha
    return comb_df

def available_experiment_results(results_path):
    if not os.path.exists(results_path):
        print(f"The path {results_path} does not exist.")
        return []

    # List all directories in the path
    experiments = [name for name in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, name))]

    return experiments

def load_results_experiment_id(experiment_code_list, dataset_results_path):
    df_list = []
    for experiment_code in experiment_code_list:
        cur_dir = None
        for host_dir in os.scandir(dataset_results_path):
            tmp_dir = os.path.join(host_dir, experiment_code)
            if os.path.isdir(host_dir) and os.path.exists(tmp_dir):
                cur_dir = tmp_dir
                break
        if cur_dir is None or not os.path.exists(cur_dir):
            cur_dir = os.path.join(dataset_results_path, experiment_code)
        if cur_dir is None or not os.path.exists(cur_dir):
            logging.warning(f'{tmp_dir} does not exists. Skipped.')
            continue

        for filepath in os.scandir(cur_dir):
            if filepath.name.endswith('.csv'):
                df = pd.read_csv(filepath)

                current_config = utils_experiment_parameters.get_config_by_id(experiment_code)
                df = prepare_data(df)

                if current_config is not None and 'grid_fractions' in current_config.keys() and current_config['grid_fractions'] == [1]:
                    models_with_gridsearch = df.query('phase == "grid_frac"')['model_code'].unique()
                    mask = df['model_code'].isin(models_with_gridsearch) & (df['grid_frac'] == 1)
                    df.loc[mask, 'model_code'] = df.loc[mask, 'model_code'].str.replace('_gf_1', '') + '_gf_1'

                df.loc[df['model_code'].str.contains('unconstrained'), ['n_oracle_calls_',
                                                                        'n_oracle_calls_dummy_returned_',
                                                                        'oracle_execution_times_', 'grid_oracle_times',
                                                                        'last_iter_', 'best_gap_', 'best_iter_',
                                                                        'grid_frac'
                                                                        'time_oracles','eps']] = np.nan

                df_list.append(df)
    all_df = pd.concat(df_list)
    mask = all_df['model_code'].str.contains('unconstrained')
    if 'exp_frac' in all_df.columns:
        all_df.loc[mask, 'exp_frac'] = all_df.loc[mask, 'exp_frac'].fillna(1)
    all_df.columns = all_df.columns.str.replace('violation', 'DemographicParity')
    return all_df


def get_error(df):
    ci = mean_confidence_interval(df)
    err = (ci[2] - ci[1]) / 2
    return pd.Series({'error': err})


column_rename_map_before_plot = {'train_violation': 'train_DemographicParity',
                                 'test_violation': 'test_DemographicParity',
                                 }


def add_threshold(df):
    cols = ['dataset_name', 'base_model_code', 'constraint_code', 'eps']
    unique_values_dict = {}
    for col in cols:
        unique_values_dict[col] = df[col].unique().tolist()

    row_list = []
    for values in itertools.product(*unique_values_dict.values()):
        df_row = dict(zip(cols, values))
        df_row['model_code'] = 'Threshold'
        constraint_code = values[-2]
        eps = values[-1]
        constraint_name = constraint_code_to_name[constraint_code]

        for phase in ['train', 'test']:
            df_row[f'{phase}_{constraint_name}_mean'] = eps
        row_list.append(df_row.copy())
    return pd.concat([pd.DataFrame(row_list), df])


def align_seeds(df):
    df_list = []
    for key, group_df in df.groupby(['dataset_name', 'base_model_code', 'constraint_code', ], sort=False):
        df_list.append(
            group_df.groupby(seed_columns, sort=False).filter(lambda x: x.shape[0] >= x['model_code'].nunique()))
    return pd.concat(df_list)


def prepare_for_plot(df, grouping_col=None):
    # df = align_seeds(df)
    time_aggregated_df = aggregate_phase_time(df)
    groupby_col_list = cols_to_index
    if grouping_col is not None:
        groupby_col_list += [grouping_col]
    groupby_col_list = np.intersect1d(groupby_col_list, time_aggregated_df.columns).tolist()
    time_aggregated_df.columns = time_aggregated_df.columns.str.replace('violation', 'DemographicParity')
    time_aggregated_df = time_aggregated_df.rename(columns=column_rename_map_before_plot)
    new_numerical_cols = list(set(get_numerical_cols(time_aggregated_df) + groupby_col_list))
    # convert to numeric all columns from new_numerical_cols that are not already numeric
    time_aggregated_df[new_numerical_cols] = time_aggregated_df[new_numerical_cols].apply(pd.to_numeric, errors='ignore')
    # # all datasets
    #     # Check available combinations
    # df[['base_model_code', 'constraint_code', 'dataset_name','exp_grid_ratio','exp_frac']].astype('str').apply(lambda x: '_'.join(x.astype(str)), axis=1).value_counts()
    # df[['random_seed','train_test_seed','train_test_fold']].astype('str').apply(lambda x: '_'.join(x.astype(str)), axis=1).value_counts()
    # todo check seed values, filter older seeds.
    grouped_data = time_aggregated_df.groupby(groupby_col_list, as_index=True, dropna=False, sort=False)[
        new_numerical_cols]
    mean_error_df = grouped_data.agg(['mean', ('error', get_error)])
    mean_error_df.loc[:, (slice(None), 'error')] = mean_error_df.loc[:, (slice(None), 'error')].fillna(0)
    mean_error_df.columns = mean_error_df.columns.map('_'.join).str.strip('_')
    size_series = grouped_data.size()
    size_series.name = 'size'
    mean_error_df = mean_error_df.join(size_series).reset_index()
    return mean_error_df


def best_gap_filter_on_eta0(all_df):
    f = lambda x: x[x['eta0'] == x.loc[x['best_gap_'].idxmin(), 'eta0']]
    all_df = all_df.groupby(cols_to_synch + ['max_iter']).apply(f).reset_index(drop=True)
    return all_df
