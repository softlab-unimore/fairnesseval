import itertools
import os

import pandas as pd
from tensorflow.python.data.experimental import unique
from tensorflow.python.ops.gen_experimental_dataset_ops import unique_dataset

from fairnesseval.graphic import utils_results_data, graphic_utility
from fairnesseval.graphic.style_utility import StyleUtility
from fairnesseval.graphic.utils_results_data import prepare_for_plot
from fairnesseval.graphic.graphic_utility import plot_all_df_subplots, PlotUtility
from fairnesseval.utils_experiment_parameters import DEFAULT_RESULTS_PATH

if __name__ == '__main__':
    save = True
    show = False

    experiment_code_list = [
        # "s_h_1.0", old not aligned seeds
        # 's_h_EO_1.0',
        's_h_1.0r',
        's_h_EO_1.0r',
        'acs_h_eps_1.0r',  # PublicC
        'acs_eps_EO_1.0r',  # PublicC

        'acsE_eps_EO_1.0r',
        'acsE_h_eps_1.0r',
        'acs_h_eps_1.2r',
        # "acs_h_eps_1.LGBM0",  # Employment + PublicC lgbm DP
        # "acs_h_eps_1.E0",  # Employment LR DP
        # 'acs_eps_EO_1.1',  # Employment LR EO
        # 'acs_eps_EO_2.1',  # Employment lgbm EO

        # "s_c_1.0r",
        's_c_1.0N',

        # "s_zDI_1.1",
        # 's_zDI_1.2',
        # 's_zDI_1.22',
        # 's_zEO_1.1',
        's_zDI_1.1N',
        's_zDI_1.2N',
        's_zEO_1.1N',

        # 's_tr_1.0r',
        # "s_tr_1.1r",
        # 's_tr_2.0r',
        # 's_tr_2.1r',
        's_tr_1.0N',
        's_tr_2.0N',

        # 's_f_1.0r',
        # 's_f_1.1r',
        's_f_1.0N',
        's_f_1.1N',  # BIG


        's_KearnsPleiss_1.0N',  # small
        's_k_1.1N', # BIG

        'most_frequent_sig.0r',

    ]
    rlp_false_conf_list = [
        'f_eta0_eps.1',  # small | eps2 | LR | DP | eta1
        'f_eta0_eps.2',  # small | eps2 | LR | EO | eta1
        # 'f_eta0_eps.3P',      # PUB   | eps2 | LR | DP | eta1
        # 'f_eta0_eps.4P',      # PUB   | eps2 | LR | EO | eta1
        # 'f_eta0_eps.3E',      # EMP   | eps2 | LR | DP | eta1
        # 'f_eta0_eps.4E',      # EMP   | eps2 | LR | EO | eta1
        # 'f_eta0_eps.4.2E',    # EMP,g | eps2 | LR | EO | eta2
        # 'f_eta0_eps.3.2E',    # EMP,g | eps2 | LR | DP | eta2
        'rlp_F_ACS.1',  # P+E   | eps2 | LR | DP EO | eta3

    ]

    dataset_results_path = DEFAULT_RESULTS_PATH
    base_plot_dir = os.path.join(DEFAULT_RESULTS_PATH, 'plots')
    results_df = utils_results_data.load_results_experiment_id(experiment_code_list, dataset_results_path)
    #fillna of constraint code with dp
    results_df.loc[results_df['model_code'] == "ZafarEO", 'constraint_code'] = 'eo'
    results_df['constraint_code'] = results_df['constraint_code'].fillna('dp')

    additional_df = []
    for model_to_duplicate in ['Feld', 'Calmon', 'Kearns', 'Pleiss']:
        tdf = results_df.query('model_code == @model_to_duplicate').copy()
        tdf['constraint_code'] = 'eo'
        additional_df.append(tdf)
    results_df = pd.concat([results_df] + additional_df)
    results_df = results_df[~((results_df['model_code'] == "unconstrained") & (results_df['exp_frac'] == 0.251))]

    # Check number of replication
    # all_df.loc[all_df['model_code'] == 'ZafarDI', ['train_test_fold', 'random_seed', 'train_test_seed']].apply(lambda x: '_'.join(x.astype(str)), 1).nunique()
    # a = utils_results_data.load_results_experiment_id(['acs_h_gs1_1.0'], dataset_results_path)
    # a[a['dataset_name'].str.startswith('ACS')][['random_seed','train_test_fold', 'train_test_seed']].apply(lambda x: '_'.join(x.astype(str)),axis=1).unique() # value_counts()
    # a.query('dataset_name == "ACSEmployment"')[np.intersect1d(x.columns, utils_results_data.cols_to_aggregate)].apply(lambda x: '_'.join(x.astype(str)), axis=1).unique().tolist()

    rlp_df = utils_results_data.load_results_experiment_id(rlp_false_conf_list, dataset_results_path)
    model_code = 'RLP=' + rlp_df['run_linprog_step'].map({True: 'T', False: 'F'}) + ' max_iter=' + rlp_df[
        'max_iter'].astype(str)
    rlp_df['model_code'] = model_code
    rlp_df = utils_results_data.best_gap_filter_on_eta0(rlp_df)

    grouping_col = 'eps'
    x_axis_list = ['eps']

    rlp_df_filtered_v2 = rlp_df[rlp_df['max_iter'].isin([50])]

    # rlp_df_filtered_v2 = rlp_df_filtered_v2.query(
    #     '~(dataset_name=="ACSEmployment" & max_iter == 50 & run_linprog_step == False)')
    # rlp_df_filtered_v2 = rlp_df_filtered_v2.query('~(dataset_name!="ACSEmployment" & max_iter == 100)')
    # tmp_filter = rlp_df_filtered_v2['dataset_name'] == "ACSEmployment"
    # rlp_df_filtered_v2.loc[tmp_filter, 'model_code'] = rlp_df_filtered_v2.loc[tmp_filter, 'model_code'].str.replace(
    #     ' max_iter=100', '')
    # join as string the fields ['run_linprog_step', 'max_iter', 'dataset_name', 'eta0', 'constraint_code']
    # rlp_df_filtered_v2
    # # filter acsEmployment

    all_df = pd.concat([results_df, rlp_df_filtered_v2])
    # replace adult with adult_sigmod check unique dataset names
    unique_dataset_names = all_df['dataset_name'].unique()
    all_df['dataset_name'] = all_df['dataset_name'].replace('adult', 'adult_sigmod')
    restricted_v2 = ['unconstrained', 'Calmon', 'Feld', 'ZafarDI', 'ZafarEO', 'ThresholdOptimizer', 'Kearns', 'Pleiss'] + \
                    rlp_df_filtered_v2['model_code'].unique().tolist() + ['hybrid_7']+ ['EXPGRAD++']
    sort_map = {name: i for i, name in enumerate(restricted_v2)}
    all_df = all_df.assign(model_sort=all_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, True, True]).drop(columns=['model_sort'])
    # all_df.loc[all_df['model_code'].str.contains('unconstrained'), 'eps'].unique()
    all_df = all_df[all_df['phase'] != 'evaluation']
    #replace hybryd_7 with EXPGRAD++
    all_df['model_code'] = all_df['model_code'].replace('hybrid_7', 'EXPGRAD++')
    # version v3
    plot_all_df_subplots(all_df, model_list=restricted_v2, chart_name='eps_v3', grouping_col='eps',
                         save=save, show=show, sharex=False, sharey=False,
                         axis_to_plot=[['test_violation', 'test_error'],
                                       ['test_violation', 'time'],
                                       ],
                         params=dict(no_errorbar=True)
                         )
    # version v4
    plot_all_df_subplots(all_df, model_list=restricted_v2, chart_name='eps_v4', grouping_col='eps',
                         save=save, show=show, sharex=False, sharey=False,
                         axis_to_plot=[['test_violation', 'test_error'],
                                       ['test_violation', 'time'],
                                       ],
                         )

    # Version v1 nad v2
    rlp_df_filtered = rlp_df[rlp_df['max_iter'].isin([5, 10, 50, 100])]
    all_df = pd.concat([results_df, rlp_df_filtered])
    restricted_v1 = (['unconstrained', 'Calmon', 'Feld', 'ZafarDI', 'ZafarEO', 'ThresholdOptimizer'] + \
                     rlp_df_filtered['model_code'].unique().tolist() + ['hybrid_7'])
    # todo add most_frequent

    sort_map = {name: i for i, name in enumerate(restricted_v1)}
    all_df = all_df.assign(model_sort=all_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, False, True]).drop(columns=['model_sort'])
    all_df.loc[all_df['model_code'].str.contains('unconstrained'), 'eps'] = pd.NA

    y_axis_list_short = ['time'] + ['_'.join(x) for x in itertools.product(['test'], ['error', 'violation'])]
    y_axis_list_long = y_axis_list_short + ['train_error', 'train_violation']
    for y_axis_list, suffix in [(y_axis_list_short, '_v2'), (y_axis_list_long, '')]:
        plot_all_df_subplots(all_df, model_list=restricted_v1, chart_name='eps' + suffix, grouping_col='eps',
                             save=save, show=show,
                             axis_to_plot=list(itertools.product(x_axis_list, y_axis_list)))
