import itertools
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from graphic import utils_results_data, graphic_utility
from graphic.style_utility import StyleUtility
from graphic.utils_results_data import prepare_for_plot
from graphic_utility import plot_all_df_subplots, PlotUtility, plot_demo_subplots

if __name__ == '__main__':
    save = True
    show = False

    experiment_code_list = [
        'demo.A.1r',
        ]

    dataset_results_path = os.path.join("results")
    base_plot_dir = os.path.join('results', 'plots')
    results_df = utils_results_data.load_results_experiment_id(experiment_code_list, dataset_results_path)
    #filter dataset_name == adult_sigmod
    results_df = results_df[results_df['dataset_name'] == 'adult_sigmod']
    #replace hybrid_7 model_code with EXPGRAD
    results_df['model_code'] = results_df['model_code'].replace('hybrid_7', 'EXPGRAD')
    model_list = ['unconstrained', 'EXPGRAD', 'ThresholdOptimizer', 'Calmon', 'Feld', 'ZafarDI', 'ZafarEO', ]
    sort_map = {name: i for i, name in enumerate(model_list)}
    all_df = results_df.assign(model_sort=results_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, False, True]).drop(columns=['model_sort'])

    pl_util = PlotUtility(save=save, show=show, suffix='')
    def legend_hook(ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        return by_label
    pl_util.params['legend_hook'] = None
    plot_demo_subplots(all_df, model_list=model_list, chart_name='A', save=save, show=show,
                       axis_to_plot=[['train_DemographicParity', 'train_error', ], ['test_DemographicParity', 'test_error'], ['test_DemographicParity','time']],
                       sharex=False, result_path_name='demo',
                       pl_util=pl_util, single_plot=False)

    mean_error_df = prepare_for_plot(all_df[all_df['model_code'].isin(model_list)])
    mean_error_df['model_code'] = mean_error_df['model_code'].map(StyleUtility.get_label)
    mean_error_df.columns = mean_error_df.columns.str.replace('_mean', '')

    pl_util = PlotUtility(save=save, show=show, suffix='')
    fig, axes = plt.subplots(1, 1, figsize=np.array([6.4 * 2.1, 4.8]) / 1.8)
    pl_util = PlotUtility(save=save, show=show, suffix='', annotate_mode='all')
    y_axis_list = ['time']
    graphic_utility.bar_plot_function_by_model(df=mean_error_df, ax=axes,
                                               fig=fig,
                                               y_axis_list=y_axis_list)
    axes.set_yscale("log")
    if show:
        fig.show()
    pl_util.save_figure(additional_dir_path='demo', name=f'time', fig=fig)