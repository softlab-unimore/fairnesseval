import os

import numpy as np

from graphic import utils_results_data
from graphic_utility import plot_demo_subplots



def plot_function_B(chart_name, experiment_code_list, model_list, x_axis, y_axis_list, grouping_col=None, res_path='./demo_results', save=True, show=False, single_plot=True, dataset_list=None):
    dataset_results_path = os.path.join(res_path)
    base_plot_dir = os.path.join(res_path, 'plots')
    results_df = utils_results_data.load_results_experiment_id(experiment_code_list, dataset_results_path)
    if dataset_list is not None:
        results_df = results_df[results_df['dataset_name'].isin(dataset_list)]
    sort_map = {name: i for i, name in enumerate(model_list)}
    all_df = results_df.assign(model_sort=results_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'model_sort'],
        ascending=[True, True]).drop(columns=['model_sort'])

    return plot_demo_subplots(all_df, model_list=model_list, chart_name=chart_name, save=save, show=show,
                        axis_to_plot=[[x_axis, y_axis] for y_axis in y_axis_list],
                        sharex=True, result_path_name=dataset_results_path,
                        single_plot=single_plot, grouping_col=grouping_col, params=dict(figsize=np.array([16,9])*.60))



if __name__ == '__main__':
    params = {'chart_name': 'demo.A', 'experiment_code_list': ('demo.A.1', 'demo.A.2', 'demo.A.3', 'demo.A.4', 'demo.A.5'), 'model_list': ('LogisticRegression', 'ZafarDI', 'Feld', 'ThresholdOptimizer', 'fairlearn'), 'x_axis': 'train_error', 'y_axis_list': ['train_DemographicParity'], 'grouping_col': None, 'dataset_list': ['compas']}

    plot_function_B(**params, res_path='./demo_results', single_plot=False)

    save = True
    show = False

    experiment_code_list = [
        'demo.D.0r',
        #'demo.C.1r',
        ]

    dataset_results_path = os.path.join("results")
    base_plot_dir = os.path.join('results', 'plots')
    results_df = utils_results_data.load_results_experiment_id(experiment_code_list, dataset_results_path)
    results_df['model_code'] = results_df['model_code'].replace('hybrid_7', 'EXPGRAD')
    results_df['model_code'] = results_df['model_code'].replace('fairlearn', 'EXPGRAD')
    model_list = [
        # 'unconstrained',
        'EXPGRAD',
        #'ThresholdOptimizer', 'Calmon', 'Feld', 'ZafarDI', 'ZafarEO',
        ]
    sort_map = {name: i for i, name in enumerate(model_list)}
    all_df = results_df.assign(model_sort=results_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, False, True]).drop(columns=['model_sort'])

    x_axis = 'train_fractions'
    plot_demo_subplots(all_df, model_list=model_list, chart_name='D', save=save, show=show,
                       axis_to_plot=[[x_axis, y_axis ] for y_axis in ['test_error', 'test_DemographicParity','time']],
                       sharex=False, result_path_name='demo',
                        single_plot=True, grouping_col=x_axis)
