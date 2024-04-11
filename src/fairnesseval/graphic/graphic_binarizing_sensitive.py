import itertools
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from graphic import utils_results_data
from graphic.style_utility import StyleUtility
from graphic.utils_results_data import prepare_for_plot
from graphic_utility import plot_all_df_subplots, PlotUtility
import graphic_utility

if __name__ == '__main__':
    save = True
    show = False

    experiment_code_list = [

        'acsER_binB1.0r',
        'acsER_binB1.1r',
        'acsER_binB2.0r',
        # 'acsER_bin2.0r',
        # 'acsER_bin2.1r',
        # 'acsER_bin3.0r',
        # 'acsER_bin4.0r',
        # 'acsER_binB2.1r',
    ]

    dataset_results_path = os.path.join("results")
    base_plot_dir = os.path.join('results', 'plots')
    all_df = utils_results_data.load_results_experiment_id(experiment_code_list, dataset_results_path)
    all_df.columns = [x.replace('violation', 'DemographicParity') for x in all_df.columns]
    # all_df = all_df.rename(
    #     columns={f'{pre}_{metric}': f'{pre}_{metric} binary' for pre in ['test', 'train'] for metric in
    #              ['error', 'DemographicParity', 'EqualizedOdds']})
    multivalued_sensitive_df = utils_results_data.load_results_experiment_id([
        # 'acsER_bin2.2r',
        # 'acsER_bin3.1r',
        'acsER_binB1.0Mr',
        'acsER_binB1.1Mr',
        'acsER_binB2.0Mr',
    ], dataset_results_path)
    multivalued_sensitive_df.columns = [x.replace('violation', 'DemographicParity') for x in
                                        multivalued_sensitive_df.columns]
    multivalued_sensitive_df = multivalued_sensitive_df.rename(
        columns={f'{pre}_{metric}': f'{pre}_{metric} Multi' for pre in ['test', 'train'] for metric in
                 ['error', 'DemographicParity', 'EqualizedOdds']})
    new_cols = [f'{pre}_{metric} Multi' for pre in ['test', 'train'] for metric in
                ['error', 'EqualizedOdds', 'DemographicParity']]
    to_synch = utils_results_data.cols_to_index + utils_results_data.seed_columns + ['phase']
    all_df = all_df.merge(
        multivalued_sensitive_df[to_synch + new_cols],
        on=to_synch, how='outer')

    restricted = []
    # restricted = [x.replace('_exp', '_eps') for x in restricted]
    restricted += [#'UNMITIGATED',
                   'Feld', 'ZafarDI', 'ThresholdOptimizer', 'EXPGRAD=adaptive GS=No LP=Yes',
                   'hybrid_7', ]
    all_df['model_code'] = all_df['model_code'].str.replace('UNMITIGATED full', 'UNMITIGATED')

    sort_map = {name: i for i, name in enumerate(restricted)}

    all_df = all_df.assign(model_sort=all_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, True, True]).drop(columns=['model_sort'])
    all_df.loc[all_df['model_code'].str.contains('unconstrained'), 'eps'] = pd.NA

    grouping_col = 'eps'
    x_axis_list = ['eps']
    y_axis_list_long = ['_'.join(x) for x in
                        itertools.product(['train'], [  # 'error',
                            'DemographicParity', 'DemographicParity_orig',
                            'DemographicParity Multi'])]
    y_axis_list_short = ['_'.join(x) for x in
                         itertools.product(['test'], [
                             # 'DemographicParity',  # binary over binary
                             'DemographicParity_orig',  # binary over multi-valued
                             'DemographicParity Multi',  # multi over multi-valued
                         ])]

    tranformed_df_list = []
    model_list = []
    # create a new set of models starting from models code but appending '_orig' and using the columns suffixed with '_orig'
    for conf in ['_orig', ' Multi']:
        all_df_tranformed = all_df.copy()
        turn_cols = [x for x in all_df.columns if conf in x]
        all_df_tranformed['model_code'] = all_df_tranformed['model_code'] + conf
        for col in turn_cols:
            all_df_tranformed[col.replace(conf, '')] = all_df_tranformed[col]
        # all_df_tranformed['model_code'] = all_df_tranformed['model_code'].str.replace('hybrid_7_orig','EXPGRAD=adaptive GS=No LP=Yes ORIG')
        all_df_tranformed['model_code'] = all_df_tranformed['model_code'].map(
            {'hybrid_7' + conf: 'EXPGRAD=adaptive GS=No LP=Yes' + conf,
             'unconstrained' + conf: 'UNMITIGATED full' + conf})
        tranformed_df_list.append(all_df_tranformed)
        model_list += [x + conf for x in restricted]
    conf = ' binary'
    all_df['model_code'] = all_df['model_code'] + conf
    all_df['model_code'] = all_df['model_code'].map(
        {'hybrid_7' + conf: 'EXPGRAD=adaptive GS=No LP=Yes' + conf,
         'unconstrained' + conf: 'UNMITIGATED' + conf}).fillna(all_df['model_code'])
    model_list += [x + conf for x in restricted]
    # model_list = (restricted + orig_model_list + [x.replace('_orig', ' Multi') for x in orig_model_list])
    to_plot_df = pd.concat(tranformed_df_list + [all_df])
    model_list = to_plot_df['model_code'].unique()
    # plot_all_df_subplots(to_plot_df,
    #                      model_list=model_list, chart_name='bin_v3',
    #                      grouping_col='eps',
    #                      save=save, show=show, sharex=False, sharey=False,
    #                      axis_to_plot=[['train_violation', 'train_error'], ],
    #                      )

    model_list = list(restricted)
    all_df['model_code'] = all_df['model_code'].str[:-7]
    # exclude ZafarDI when constraint_code is eo
    all_df = all_df[~((all_df['model_code'] == 'ZafarDI') & (all_df['constraint_code'] == 'eo'))]
    mean_error_df = prepare_for_plot(all_df[all_df['model_code'].isin(model_list)], grouping_col)
    mean_error_df['model_code'] = mean_error_df['model_code'].map(StyleUtility.get_label)
    mean_error_df['model_code'] = mean_error_df['model_code'].str.replace('EXPGRAD=adaptive GS=No LP=Yes',
                                                                          '     EXPGRAD++')
    mean_error_df.columns = mean_error_df.columns.str.replace('_mean', '')

    y_axis_map = {
        '_orig': ' binarized training',
        'Multi': 'multi-valued training'}
    for key, value in y_axis_map.items():
        mean_error_df.columns = mean_error_df.columns.str.replace(key, value)
    y_bin_map = {f'{phase}_{cc}{agg}': f'{phase}_{cc} binary{agg}' for phase in ['test', 'train'] for cc in
                 ['DemographicParity', 'EqualizedOdds'] for agg in ['', '_mean', '_error']}
    mean_error_df = mean_error_df.rename(columns=y_bin_map)
    pl_util = PlotUtility(save=save, show=show, suffix='', annotate_mode='all')
    for (y_axis_list, suffix) in [(y_axis_list_short, '_v2'), (y_axis_list_long, '')]:
        fig, axes = plt.subplots(1, 2, figsize=np.array([6.4*2.1, 4.8]) / 1.8)
        for ((t_constraint, cc), ax) in zip([('DemographicParity', 'dp'), ('EqualizedOdds', 'eo')], axes):

            # plot_all_df_subplots(all_df, model_list=restricted, chart_name='eps' + suffix, grouping_col='eps',
            #                      save=save, show=show,
            #                      axis_to_plot=list(itertools.product(x_axis_list, y_axis_list)),
            #                      custom_add_graphic_object='bar')
            y_axis_list = [x.replace('DemographicParity', t_constraint) for x in y_axis_list]
            for key, value in y_axis_map.items():
                y_axis_list = [x.replace(key, value) for x in y_axis_list]
            y_axis_list = [y_bin_map.get(x, x) for x in y_axis_list]

            graphic_utility.bar_plot_function_by_model(df=mean_error_df.query(f'constraint_code == "{cc}"'), ax=ax,
                                                       fig=fig,
                                                       y_axis_list=y_axis_list)
        if show:
            fig.show()
        pl_util.save_figure(additional_dir_path='all_df', name=f'binary' + suffix, fig=fig)


        # plt.figure(figsize=(10, 6))
        # axes = all_df.boxplot(column=y_axis_list, by=['model_code'], rot=90, figsize=(10, 10))
