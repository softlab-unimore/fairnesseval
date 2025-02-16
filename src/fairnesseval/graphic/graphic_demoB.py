import os

import numpy as np

from fairnesseval import utils_results_data
from fairnesseval.graphic.graphic_utility import plot_demo_subplots, PlotUtility, plot_all_df_subplots
from fairnesseval.utils_experiment_parameters import DEMO_SAVE_PATH
from fairnesseval.utils_general import get_project_root


def plot_function_B(chart_name, experiment_code_list, model_list, x_axis, y_axis_list, grouping_col=None,
                    res_path='./demo_results', save=True, show=False, single_plot=True, dataset_list=None,
                    base_plot_dir=None, **kwargs):
    if base_plot_dir is None:
        base_plot_dir = os.path.join(get_project_root(), 'streamlit', 'demo_plots')
    dataset_results_path = os.path.join(res_path)
    results_df = utils_results_data.load_results_experiment_id(experiment_code_list, dataset_results_path)
    if dataset_list is not None:
        results_df = results_df[results_df['dataset_name'].isin(dataset_list)]
    sort_map = {name: i for i, name in enumerate(model_list)}
    all_df = results_df.assign(model_sort=results_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'model_sort'],
        ascending=[True, True]).drop(columns=['model_sort'])

    return plot_demo_subplots(all_df, model_list=model_list, chart_name=chart_name, save=save, show=show,
                              axis_to_plot=[[x_axis, y_axis] for y_axis in y_axis_list],
                              sharex=True,
                              use_subplots=single_plot, grouping_col=grouping_col,
                              pl_params=dict(base_plot_dir=base_plot_dir), **kwargs)


if __name__ == '__main__':
    dataset_results_path = DEMO_SAVE_PATH
    base_plot_dir = os.path.join(dataset_results_path.replace('demo_results', 'demo_plots'))
    plot_function_B(**{"chart_name": "demo.B", "experiment_code_list": ["demo.D.0r", "demo.D.1r"],
                       "model_list": ["fairlearn", "LogisticRegression"], "x_axis": "train_fractions",
                       "y_axis_list": ["time", "test_error", "test_DemographicParity"],
                       "grouping_col": "train_fractions", "res_path": dataset_results_path, "show": False,
                       }, params={"figsize": [3.15, 2.0999999999999996]}
                    )

    params = {'chart_name': 'demo.A', 'experiment_code_list': ('demo.default.test',),
              'model_list': ('LogisticRegression',), 'x_axis': 'train_error', 'y_axis_list': ['train_EqualizedOdds'],
              'grouping_col': None, 'dataset_list': ['adult']}

    # plot_function_B(**params, res_path=dataset_results_path, single_plot=False, show=True, save=False)

    save = True
    show = True

    experiment_code_list = [
        'demo.D.0r',
        'demo.D.1r',
        # 'demo.C.1r',
    ]

    results_df = utils_results_data.load_results_experiment_id(experiment_code_list, dataset_results_path)
    # results_df['model_code'] = results_df['model_code'].replace('hybrid_7', 'EXPGRAD++')
    # results_df['model_code'] = results_df['model_code'].replace('expgrad', 'EXPGRAD')
    model_list = [
        'LogisticRegression',
        'expgrad',
        # 'ThresholdOptimizer', 'Calmon', 'Feld', 'ZafarDI', 'ZafarEO',
    ]
    sort_map = {name: i for i, name in enumerate(model_list)}
    all_df = results_df.assign(model_sort=results_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, False, True]).drop(columns=['model_sort'])

    x_axis = 'train_fractions'
    pl_util = PlotUtility(save=save, show=show, suffix='', base_plot_dir=base_plot_dir)
    plot_demo_subplots(all_df, model_list=model_list, chart_name='D', save=save, show=show,
                       axis_to_plot=[[x_axis, y_axis] for y_axis in ['time', 'test_error', 'test_DemographicParity', ]],
                       sharex=True, grouping_col=x_axis, pl_util=pl_util,
                       params=dict(figsize=np.array([4.5, 3]) * 0.7))
