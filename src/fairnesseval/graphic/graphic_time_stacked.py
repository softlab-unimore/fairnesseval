import os

import pandas as pd

from graphic import utils_results_data
from graphic.graphic_utility import plot_routine_other

if __name__ == '__main__':
    save = True
    show = False

    experiment_code_list = set(list([
        's_h_exp_1.0r',
        's_h_exp_EO_1.0r',

        'acs_h_gs1_1.0r',
        'acs_h_gs1_EO_1.0r',

    ]))

    employment_conf = [
        'acsE_h_gs1_1.0',
        'acs_h_gs1_EO_1.0',
        'acs_h_gs1_EO_2.0',
        'acs_h_gs1_1.1',
    ]
    employment_sqrt_conf = [
        'acs_h_gsSR_1.0',
        'acs_h_gsSR_2.0',
        'acsE_h_gsSR_1.1',
        'acs_h_gsSR_2.1',
    ]

    sqrt_conf = [
        'acs_h_gsSR_1.0r',
        'acs_h_gsSR_EO_1.0r',
        's_h_exp_2.0r',
        's_h_exp_EO_2.0r']
    gf_1_conf = set(experiment_code_list) - set(sqrt_conf)

    exp_frac_models = [
        # 'expgrad_fracs_exp',
        # 'hybrid_3_exp_gf_1',
        # 'hybrid_5_exp',
        # 'hybrid_3_exp',
        # 'hybrid_3_exp_gf_1',
        'hybrid_7_exp',
        # 'sub_hybrid_5_exp',
        # 'sub_hybrid_3_exp_gf_1',
    ]

    sort_map = {name: i for i, name in enumerate(exp_frac_models)}

    dataset_results_path = os.path.join("results", "fairlearn-2")

    all_df = utils_results_data.load_results_experiment_id(gf_1_conf, dataset_results_path)

    employment_df = utils_results_data.load_results_experiment_id(employment_conf, dataset_results_path)
    employment_df = employment_df.query('dataset_name == "ACSEmployment"')
    employment_df = employment_df.query(
        'train_test_fold == 0 and random_seed == 0 and train_test_seed == 0')  # remove replications in ACSEmployment

    all_df = pd.concat([all_df, employment_df]).reset_index(drop=True)

    all_df = all_df.assign(model_sort=all_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, True, True]).drop(columns=['model_sort'])
    for dataset_name, df in all_df.groupby('dataset_name'):
        plot_routine_other(df, dataset_name, save=save,show=show)
    # todo check generated images and their locations

    # extract_expgrad_oracle_time(all_df, new_col_name='time_oracles', cols_to_select='all')
    # y_axis_list_short = ['time', 'test_error', 'test_violation']
    # y_axis_list_long = y_axis_list_short + ['train_error', 'train_violation', 'n_oracle_calls_', 'time_oracles']
    # for y_axis_list, suffix in [(y_axis_list_short, '_v2'), (y_axis_list_long, '')]:
    #     continue
    #     plot_all_df_subplots(all_df, model_list=exp_frac_models,
    #                          model_set_name='exp_frac' + suffix, grouping_col='exp_frac',
    #                          save=save, show=show, sharex=False, increasing_marker_size=False,
    #                          ylim_list=[None, None, (0, 0.06)],
    #                          sharey='row', xlog=True,
    #                          axis_to_plot=list(itertools.product(['exp_frac'], y_axis_list)))
    #
    # gf_1_df = all_df
    # """
    #  Loading sqrt only when needed. Avoid multiple version of same configs (eg. hybrid_5_exp)
    #  loading and selecting only models with sqrt from sqrt experiments results.
    # """
    # sqrt_df = utils_results_data.load_results_experiment_id(sqrt_conf, dataset_results_path)
    # sqrt_df = sqrt_df[sqrt_df['model_code'].isin(grid_sqrt_models)]
    # employment_sqrt_df = utils_results_data.load_results_experiment_id(employment_sqrt_conf, dataset_results_path)
    # employment_sqrt_df = employment_sqrt_df[employment_sqrt_df['model_code'].isin(grid_sqrt_models)]
    # employment_sqrt_df = employment_sqrt_df.query(
    #     'dataset_name == "ACSEmployment" and train_test_fold == 0 and random_seed == 0 and train_test_seed == 0')
    #
    # # gf_1_df = gf_1_df[gf_1_df['model_code'].isin(['sub_hybrid_3_exp_gf_1'])] # todo delete
    # all_df = pd.concat([sqrt_df, gf_1_df, employment_sqrt_df]).reset_index(drop=True)
    #
    # all_df = select_oracle_call_time(all_df)
    # all_df = all_df.assign(model_sort=all_df['model_code'].map(sort_map)).sort_values(
    #     ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
    #     ascending=[True, False, True, True]).drop(columns=['model_sort'])
    #
    # pl_util = PlotUtility(save=save, show=show)
    # # Align random seed values --> check sqrt time is at least equal to adaptive or greater.
    #
    # x_axis_list = ['time_oracles']
    # y_axis_list_short = ['_'.join(x) for x in itertools.product(['test'], ['error', 'violation'])]
    # y_axis_list_long = y_axis_list_short + ['train_violation', 'train_error', 'n_oracle_calls_']
    # # y_axis_list = ['test_error','train_violation']
    # y_lim_map = {'test_error': None, 'test_violation': (0, 0.1), 'train_violation': (0, 0.1)}
    # for y_axis_list, suffix in [(y_axis_list_long, ''), (y_axis_list_short, '_v2'), ]:
    #     y_lim_list = [y_lim_map.get(x, None) for x in y_axis_list]
    #     plot_all_df_subplots(all_df, model_list=exp_frac_models + grid_sqrt_models,
    #                          model_set_name='oracle_calls' + suffix,
    #                          grouping_col='exp_frac',
    #                          save=save, show=show, sharex=False, increasing_marker_size=True, xlog=True,
    #                          # ylim_list=y_lim_list,
    #                          sharey='row', axis_to_plot=list(itertools.product(x_axis_list, y_axis_list)))
    #
    # # plot_by_df(pl_util, all_df, grid_chart_models, model_set_name='oracle_calls',
    # #            grouping_col='exp_frac')
