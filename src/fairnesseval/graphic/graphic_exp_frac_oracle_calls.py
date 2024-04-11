import ast
import itertools
import os

import pandas as pd

from graphic import utils_results_data
from graphic.graphic_utility import select_oracle_call_time, PlotUtility, plot_all_df_subplots, \
    extract_expgrad_oracle_time
from graphic.utils_results_data import prepare_for_plot

if __name__ == '__main__':
    save = True
    show = False

    experiment_code_list = set(list([
        # 'sigmod_h_exp_1.0',       # not replicated
        # 'sigmod_h_exp_2.0',       # not replicated
        # 'sigmod_h_exp_3.0',       # not replicated
        # 's_h_exp_EO_1.0',         # not replicated
        # 'acs_h_gs1_1.0', # different random seeds applied
        # 'acs_h_gsSR_1.1', # not replicated

        # 'acs_h_gsSR_2.0', # not replicated
        # 'acs_h_gsSR_2.1', # not replicated

        's_h_exp_1.0r',  # DP    |   lr, lgbm    |   german, compas | gs1
        's_h_exp_EO_1.0r',  # EO    |   lr, lgbm    |   german, compas | gs1
        's_h_exp_2.0r',  # DP    |   lr, lgbm    |   german, compas | sqrt
        's_h_exp_EO_2.0r',  # EO    |   lr, lgbm    |   german, compas | sqrt

        'acs_h_gs1_1.0r',  # DP  |   lr, lgbm    |   adult ACSPublic | gs1
        'acs_h_gs1_EO_1.0r',  # EO  |   lr, lgbm    |   adult ACSPublic | gs1
        'acs_h_gsSR_1.0r',  # DP  |   lr, lgbm    |   adult ACSPublic | sqrt
        'acs_h_gsSR_EO_1.0r',  # EO  |   lr, lgbm    |   adult ACSPublic | sqrt

    ]))

    employment_conf = [

        'acsE_h_gs1_1.0r',  # DP |   lr  |   Employment replicated
        'acsE_h_gs1_1.1r',  # DP | lgbm  |   Employment seed 0
        'acsE_h_gs1_1.2r',  # DP | lgbm  |   Employment seed 1

        'acsE_h_gs1_EO_1.0r',  # EO |   lr  |   Employment replicated
        'acsE_h_gs1_EO_1.1r'  # EO | lgbm  |   Employment replicated

        # 'acs_h_gs1_1.1',        # DP | lgbm  |   Employment not replicated
        # 'acs_h_gs1_EO_2.0',     # EO | lgbm  |   Employment not replicated
        # 'acsE_h_gs1_1.0',     # DP |   lr  |   Employment not replicated
        # 'acs_h_gs1_EO_1.0',   # EO |   lr  |   Employment not replicated
    ]
    employment_sqrt_conf = [

        'acsE_h_gsSR_1.0r',  # DP |lr,lgbm|   Employment replicated          | sqrt
        'acsE_h_gsSR_1.1r',  # EO |lr,lgbm|   Employment replicated          | sqrt

        # 'acs_h_gsSR_1.0',       # DP | lr   |   bigger not replicated     | sqrt
        # 'acsE_h_gsSR_1.1',      # DP | lgbm |   Employment not replicated | sqrt
        # 'acs_h_gsSR_2.0',  # EO | lr   |   bigger not replicated     | sqrt
        # 'acs_h_gsSR_2.1',  # EO | lgbm |   Employment not replicated | sqrt
    ]

    sqrt_conf = [x for x in experiment_code_list if 'SR' in x] + ['s_h_exp_2.0r', 's_h_exp_EO_2.0r']
    gf_1_conf = set(experiment_code_list) - set(sqrt_conf)

    grid_chart_models = [
        # 'expgrad_fracs',
        # 'hybrid_3_gf_1',
        # 'hybrid_5',
        # 'hybrid_3',
        # 'hybrid_3_gf_1',
        'hybrid_7',
        # 'sub_hybrid_5',
        'sub_hybrid_3_gf_1',
        '',
    ]
    grid_sqrt_models = [
        'hybrid_3',
        'hybrid_3_gf_1',
        'sub_hybrid_3',  # sqrt
        'sub_hybrid_3_gf_1',
    ]
    exp_frac_models = [
        'unconstrained_frac',
        'hybrid_5',
        'hybrid_7',
        # 'unconstrained',

    ]

    sort_map = {name: i for i, name in enumerate(exp_frac_models)}

    dataset_results_path = os.path.join("results")

    all_df = utils_results_data.load_results_experiment_id(gf_1_conf, dataset_results_path).query(
        'dataset_name != "ACSEmployment"')

    employment_df = utils_results_data.load_results_experiment_id(employment_conf, dataset_results_path)
    employment_df = employment_df.query('dataset_name == "ACSEmployment"')
    # employment_df = employment_df.query(
    #     'train_test_fold == 0 and random_seed == 0 and train_test_seed == 0')  # remove replications in ACSEmployment
    # employment_df = employment_df[employment_df['model_code'].isin(grid_chart_models)]
    # employment_df.query('model_name == "hybrid_5" & base_model_code == "lr" & constraint_code== "eo" & dataset_name == "ACSEmployment"')

    all_df = pd.concat([all_df, employment_df]).reset_index(drop=True)

    all_df = all_df.assign(model_sort=all_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, True, True]).drop(columns=['model_sort'])
    all_df.loc[all_df['model_code'].str.contains('unconstrained'), 'eps'] = pd.NA

    extract_expgrad_oracle_time(all_df, new_col_name='time_oracles', cols_to_select='all')
    all_df['avg_time_oracles'] = all_df['time_oracles'] / all_df['n_oracle_calls_']

    y_axis_list_short = ['time', 'test_error', 'test_violation']
    y_axis_list_long = y_axis_list_short + ['train_error', 'train_violation', 'avg_time_oracles', 'n_oracle_calls_',
                                            'time_oracles', 'ratio_fit_total']
    all_df = all_df.join(all_df[all_df['phase'] == 'expgrad_fracs']['oracle_execution_times_'].agg(
        lambda x: pd.DataFrame(ast.literal_eval(x)).agg(sum)))
    all_df['ratio_fit_total'] = all_df['fit'] / all_df['time']
    # select lgbm base_model_code  one of the larger data sets: in ACSEmployment, ACSPublicCoverage
    x = all_df.query('exp_frac == 1 and model_code == "hybrid_7"')
    ab = prepare_for_plot(x, 'exp_frac')
    ab[['base_model_code', 'constraint_code', 'dataset_name', 'ratio_fit_total_mean', 'ratio_fit_total_error', 'eps',
        'exp_frac', 'method', 'model_code', 'eps_mean',
        'wts-con_mean', 'wts-con_error', 'time_mean', 'time_error',
        'train_error_mean', 'train_error_error', 'fit_mean', 'fit_error', 'wts-obj_mean', 'wts-obj_error',
        'train_DemographicParity_mean', 'train_DemographicParity_error',
        'red-Y-W_mean', 'red-Y-W_error', 'exp_frac_mean', 'exp_frac_error', 'n_oracle_calls__mean',
        'n_oracle_calls__error',
        'avg_time_oracles_mean', 'avg_time_oracles_error', ]]
    y_lim_map = {'test_error': None, 'test_violation': (-0.01, 0.2), 'train_violation': (-0.01, 0.2),
                 'test_EqualizedOdds': (-0.01, 0.2)}

    for y_axis_list, suffix in [(y_axis_list_long, ''), (y_axis_list_short, '_v2'), ]:
        y_lim_list = [y_lim_map.get(x, None) for x in y_axis_list]
        plot_all_df_subplots(all_df[all_df['phase'] != 'evaluation'], model_list=exp_frac_models,
                             chart_name='exp_frac' + suffix, grouping_col='exp_frac',
                             save=save, show=show, sharex=False, increasing_marker_size=False,
                             sharey=False, xlog=True,
                             ylim_list=y_lim_list,
                             axis_to_plot=list(itertools.product(['exp_frac'], y_axis_list)))

    gf_1_df = all_df
    """
     Loading sqrt only when needed. Avoid multiple version of same configs (eg. hybrid_5)
     loading and selecting only models with sqrt from sqrt experiments results.
    """
    sqrt_df = utils_results_data.load_results_experiment_id(sqrt_conf, dataset_results_path)
    sqrt_df = sqrt_df[sqrt_df['model_code'].isin(grid_sqrt_models)]
    employment_sqrt_df = utils_results_data.load_results_experiment_id(employment_sqrt_conf, dataset_results_path)
    employment_sqrt_df = employment_sqrt_df[employment_sqrt_df['model_code'].isin(grid_sqrt_models)]
    employment_sqrt_df = employment_sqrt_df.query('dataset_name == "ACSEmployment"')
    # ' and train_test_fold == 0 and random_seed == 0 and train_test_seed == 0')

    # gf_1_df = gf_1_df[gf_1_df['model_code'].isin(['sub_hybrid_3_gf_1'])] # todo delete
    all_df = pd.concat([sqrt_df, gf_1_df, employment_sqrt_df]).reset_index(drop=True)

    all_df = select_oracle_call_time(all_df)
    all_df = all_df.assign(model_sort=all_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, True, True]).drop(columns=['model_sort'])

    pl_util = PlotUtility(save=save, show=show)

    x_axis_list = ['time_oracles']
    y_axis_list_short = ['_'.join(x) for x in itertools.product(['test'], ['error', 'violation'])]
    y_axis_list_long = y_axis_list_short + ['train_violation', 'train_error', 'n_oracle_calls_']
    # y_axis_list = ['test_error','train_violation']

    for y_axis_list, suffix in [(y_axis_list_long, ''), (y_axis_list_short, '_v2'), ]:
        y_lim_list = [y_lim_map.get(x, None) for x in y_axis_list]
        plot_all_df_subplots(all_df, model_list=exp_frac_models + grid_sqrt_models,
                             chart_name='oracle_calls' + suffix,
                             grouping_col='exp_frac',
                             save=save, show=show, sharex=False, increasing_marker_size=True, xlog=True,
                             ylim_list=y_lim_list,
                             sharey=False, axis_to_plot=list(itertools.product(x_axis_list, y_axis_list)))

    # plot_by_df(pl_util, all_df, grid_chart_models, model_set_name='oracle_calls',
    #            grouping_col='exp_frac')
