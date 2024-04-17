import ast
import itertools
import logging
from copy import deepcopy

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

from fairnesseval.graphic import utils_results_data
from fairnesseval.graphic.style_utility import StyleUtility, replace_words, replace_words_in_list
from fairnesseval.graphic.utils_results_data import get_info, get_confidence_error, mean_confidence_interval, \
    aggregate_phase_time, filter_results, seed_columns, prepare_for_plot, constraint_code_to_name
import matplotlib as mpl

from fairnesseval.utils_general import intersection_sorted, difference_sorted

sns.set()  # for plot styling
# sns.set(rc={'figure.figsize':(8,6)})
# sns.set_context('notebook')
# sns.set_style('whitegrid')
sns.set_style("ticks")

plt.rcParams.update({'font.size': 16, "figure.dpi": 200, 'savefig.dpi': 300,
                     # 'figure.figsize': (16 * 2 / 3, 9 * 2 / 3)
                     })
plt.rcParams['figure.constrained_layout.use'] = True
# make matplotlib pdf-s text readable by illustrator
plt.rcParams['pdf.fonttype'] = 42
sns.set_context(rc={"legend.fontsize": 7})

restricted_list = [
    'expgrad_fracs_exp',
    'expgrad_fracs_LP_off_exp',
    'hybrid_5_exp',
    # 'hybrid_5_U_exp',
    # # 'hybrid_1_exp',
    # # 'hybrid_2_exp',
    # 'hybrid_3_exp',
    # 'hybrid_3_U_exp',
    # # 'hybrid_4_exp',
    # 'hybrid_6_U_exp',
    'hybrid_6_exp',

    'hybrid_7_exp',  # ExpGradSub
    'hybrid_7_LP_off_exp',
    # 'sub_hybrid_3_exp',
    # 'sub_hybrid_4_exp',
    # 'sub_hybrid_5_U_exp',
    # 'sub_hybrid_5_exp',
    # 'sub_hybrid_6_U_exp',
    'sub_hybrid_6_exp',
    'unconstrained_frac_exp',
]


class PlotUtility():
    to_plot_models = [
        # 'expgrad_fracs_gri',
        # 'hybrid_5_gri',
        # 'hybrid_5_U_gri',
        # 'hybrid_1_gri',
        # 'hybrid_2_gri',
        # 'hybrid_3_gri',
        # 'hybrid_3_U_gri',
        # 'hybrid_4_gri',
        # 'hybrid_6_gri',
        # 'hybrid_6_U_gri',
        'expgrad_fracs_exp',
        'expgrad_fracs_LP_off_exp',
        'hybrid_5_exp',
        # 'hybrid_5_U_exp',
        # 'hybrid_1_exp',
        # 'hybrid_2_exp',
        'hybrid_3_exp',
        # 'hybrid_3_U_exp',
        # 'hybrid_4_exp',
        # 'hybrid_6_U_exp',
        'hybrid_6_exp',
        'hybrid_6_exp_gf_1',
        'hybrid_7_exp',
        'hybrid_7_LP_off_exp',
        'unconstrained_exp',
        # 'fairlearn_full',
        # 'unmitigated',

        ## exp subsample models
        # 'sub_hybrid_1_exp',
        # 'sub_hybrid_2_exp',
        # 'sub_hybrid_3_U_exp',
        'sub_hybrid_3_exp',
        # 'sub_hybrid_4_exp',
        # 'sub_hybrid_5_U_exp',
        # 'sub_hybrid_5_exp',
        # 'sub_hybrid_6_U_exp',
        'sub_hybrid_6_exp',
        'unconstrained_frac_exp',

        'sub_hybrid_6_exp_gf_1',

        ## eps models
        # 'expgrad_fracs_eps',
        # 'hybrid_1_eps',
        # 'hybrid_2_eps',
        # 'hybrid_3_U_eps',
        # 'hybrid_3_eps',
        # 'hybrid_4_eps',
        # 'hybrid_5_U_eps',
        # 'hybrid_5_eps',
        # 'hybrid_6_U_eps',
        # 'hybrid_6_eps',
        # 'fairlearn_full_eps',

    ]  # + other_models # todo remove other models commented

    color_list = mpl.colormaps['tab20'].colors

    # sns.color_palette("hls", len(self.to_plot_models))
    # color_list = list(mcolors.TABLEAU_COLORS.keys())
    def __init__(self, save: bool = True, show: bool = True, suffix: str = '',
                 base_plot_dir=os.path.join('results', 'plots'), annotate_mode='all',
                 custom_add_graphic_object=None):
        '''

        :param save: bool. whether to save the chart
        :param show: bool. whether show the plot in runtime
        :param suffix: str to add at saving directory. Identify feature/config of thecharts
        :param base_plot_dir:
        '''
        self.show = show
        self.suffix = suffix
        self.save_flag = save
        self.base_plot_dir = base_plot_dir
        self.annotate_mode = annotate_mode
        self.params = dict(no_errorbar=False)
        if custom_add_graphic_object == 'bar':
            self.custom_add_graphic_object = self.add_bar
        else:
            self.custom_add_graphic_object = None

    def _start_plot(self):
        plt.close('all')
        self.fig = plt.figure()
        self.ax = plt.subplot()

    def _end_plot(self, x_axis, y_axis, title):

        self.ax.set_ylabel(StyleUtility.replace_words(y_axis))
        self.ax.set_xlabel(StyleUtility.replace_words(x_axis))

        # self.ax.set_title(StyleUtility.replace_words(f'{title} - {x_axis} v.s. {y_axis}'))
        if x_axis == 'time':
            self.ax.set_xscale("log")

        if y_axis == 'time':
            self.ax.set_yscale("log")
            # ylabel = self.ax.get_ylabel() # todo delete
        self.ax.legend()
        if self.show:
            self.fig.show()

    def plot(self, all_model_df, dataset_name, x_axis='frac', y_axis='time', groupby_col='frac'):
        self._start_plot()
        self.cols_to_aggregate = np.intersect1d(seed_columns, all_model_df.columns).tolist()
        self.groupby_col = groupby_col

        all_model_df = all_model_df[all_model_df['model_code'].isin(self.to_plot_models)]
        time_aggregated_df = aggregate_phase_time(all_model_df)
        time_aggregated_df[self.groupby_col].fillna(1, inplace=True)
        self.x_values = time_aggregated_df[self.groupby_col].unique()
        self.n_points = len(self.x_values)
        to_iter = time_aggregated_df[time_aggregated_df['model_code'].isin(self.to_plot_models)].groupby(['model_code'],
                                                                                                         dropna=False)
        self.n_models = len(self.to_plot_models)
        for model_code, turn_df in to_iter:
            self.curr_index = self.to_plot_models.index(model_code)
            errorbar_params = self.get_line_params(self.curr_index, model_code)
            errorbar_params.update(StyleUtility.get_style(model_code))
            if x_axis == 'frac':
                x_offset = (((self.curr_index / self.n_models) - 0.5) * 20 / 100) + 1
            else:
                x_offset = 1
            self.add_plot(self.ax, turn_df, x_axis, y_axis, errorbar_params=errorbar_params, x_offset_relative=x_offset)

        self._end_plot(x_axis, y_axis, title=f'{dataset_name} {x_axis} vs {y_axis}')

    def add_plot(self, ax, turn_df, x_axis, y_axis, errorbar_params, x_offset_relative=1, ):
        agg_x_axis = self.groupby_col  # if x_axis == 'time' else x_axis
        turn_data = turn_df.pivot(index=self.cols_to_aggregate, columns=agg_x_axis, values=y_axis)
        ci = mean_confidence_interval(turn_data)
        yerr = (ci[2] - ci[1]) / 2
        y_values = ci[0]
        zorder = 10 if len(y_values) == 1 else None
        if x_axis != 'frac':
            time_data = turn_df.pivot(index=self.cols_to_aggregate, columns=agg_x_axis, values=x_axis)
            ci_x = mean_confidence_interval(time_data)
            xerr = (ci_x[2] - ci_x[1]) / 2
            x_values = ci_x[0]
        else:
            xerr = None
            x_values = turn_data.columns
        # if label not in ['UNMITIGATED full']:
        ax.errorbar(x_values * x_offset_relative, y_values, xerr=xerr, yerr=yerr, zorder=zorder, **errorbar_params)
        label = errorbar_params['label']
        color = errorbar_params['color']
        if label in ['UNMITIGATED full', 'EXPGRAD=static GS=off LP=off', 'UNMITIGATED=static'] + ['ThresholdOptimizer']:
            if label in ['UNMITIGATED full']:
                y_values = [y_values.mean()]
            elif label in ['UNMITIGATED=static']:
                y_values = [y_values[-1]]
            ax.axhline(y_values[-1], linestyle="-.", color=color, zorder=10,
                       linewidth=self.linewidth)  # y_values[-1] > min(y_values) and 'error' in y_axis and 'un' not in label

        # ax.fill_between(x_values, ci[1], ci[2], color=color, alpha=0.3)
        # if len(y_values) == 1:
        #     ax.plot(self.x_values, np.repeat(y_values, self.n_points), "-.", color=color, zorder=10, label=label)
        # else:
        #     ax.plot(x_values, y_values, color=color, label=label, marker="x", linestyle='--', markersize=self.markersize)

    def add_line_errorbar(self, value_dict, grouping_values, model_code, index, n_lines, label_suffix=''):
        if pd.isna(list(value_dict.values())).all():
            return
        if self.params.get('no_errorbar'):
            value_dict['xerr'], value_dict['yerr'] = None, None
        x, xerr, y, yerr = value_dict.values()

        line_params = self.get_line_params(index, model_code=model_code)
        markers_params = self.get_marker_params(index, total=n_lines, grouping_values=grouping_values,
                                                model_code=model_code)
        label_params = self.get_all_params(model_code)
        label_params['label'] += label_suffix

        if len(set(y)) == 1 or pd.isna(y).all():
            if pd.isna(x).all():
                for key in ['fmt', 'elinewidth']:
                    try:
                        line_params.pop(key)
                    except:
                        pass
                label_params.pop('marker')
                # label_params.pop('elinewidth')
                self.ax.axhline(y[-1], **(line_params | label_params))
                return
            else:
                # self.ax.axhline(y[-1], zorder=10, **line_params)
                pass

        markers_params['markersize'] = markers_params.pop('s') ** .5
        params = markers_params | label_params | line_params
        # if len(set(x)) > 1:
        #     params.pop('linewidth')
        self.ax.errorbar(**value_dict, **params)

    def add_bar(self, value_dict, grouping_values, model_code, index, n_lines, label_suffix=''):
        if pd.isna(list(value_dict.values())).all():
            return
        x, xerr, y, yerr = value_dict.values()

        # markers_params = self.get_marker_params(index, total=n_lines, grouping_values=grouping_values,
        #                                         model_code=model_code)
        # label_params = self.get_all_params(model_code)

        bar_params = self.get_line_params(index, model_code=model_code)
        bar_params['label'] += label_suffix

        assert len(set(y)) == 1

        width = 0.8 / n_lines
        offset = width * index
        rects = self.ax.bar(x=offset, height=y, width=width, yerr=yerr, **bar_params)
        self.ax.bar_label(rects, padding=3)

    def add_multiple_lines(self, df, grouping_col, model_list, increasing_marker_size, annotate_col=None):
        n_lines = len(model_list)
        df_groups = df.groupby('model_code', sort=False)
        # df['model_code'].drop_duplicates().apply(StyleUtility.get_label).tolist() # to get missing labels
        for i, (model_code, turn_df) in enumerate(df_groups):
            if grouping_col is not None:
                turn_df = turn_df.sort_values(grouping_col)
            index = list(model_list).index(model_code)
            value_dict = turn_df[['x', 'xerr', 'y', 'yerr']].to_dict(orient='list')
            grouping_values = None
            if increasing_marker_size and grouping_col is not None:
                grouping_values = turn_df[grouping_col]
            if self.custom_add_graphic_object is None:
                self.add_line_errorbar(value_dict, grouping_values=grouping_values, model_code=model_code,
                                       index=index, n_lines=n_lines)
            else:
                self.custom_add_graphic_object(value_dict, grouping_values=grouping_values, model_code=model_code,
                                               index=index, n_lines=n_lines)
            if annotate_col is not None:
                annotate_values = turn_df[annotate_col]
                self.add_annotation(value_dict['x'], value_dict['y'], annotate_values, model_code=model_code)

    def add_annotation(self, x, y, annotate_values, annotation_fontize=10, model_code=None):
        to_annotate = self.custom_annotation_hook(x, y, annotate_values, model_code=model_code, col=self.col,
                                                  row=self.row)
        if self.annotate_mode == 'first-last':
            to_iter = [to_annotate[0], to_annotate[-1]]
        elif self.annotate_mode == 'all':
            to_iter = to_annotate
        elif self.annotate_mode == 'last':
            to_iter = [to_annotate[-1]]
        elif self.annotate_mode == 'first':
            to_iter = [to_annotate[0]]

        for kwargs in to_iter:
            self.ax.annotate(**kwargs,
                             fontsize=annotation_fontize)

    @staticmethod
    def custom_annotation_hook(x, y, annotate_values, model_code=None, **kwargs):
        return [dict(text=f'{tt: >3g}', xy=(tx, ty), xycoords='data') for tx, ty, tt in zip(x, y, annotate_values)]

    def get_color(self, index):
        return self.color_list[index % len(self.color_list)]

    def get_marker(self, index, total=None):
        if total is None:
            total = len(self.to_plot_models)
        rot = Affine2D().rotate_deg(index / total * 120)  # rotation for markers
        return MarkerStyle('1', 'left', rot)

    def get_line_params(self, index, model_code=None):
        return {key: value for key, value in StyleUtility.get_style(model_code).items() if
                key in StyleUtility.line_keys}

    def get_marker_params(self, index, total, grouping_values=None, model_code=None):
        return {key: value for key, value in StyleUtility.get_style(model_code).items() if
                key in StyleUtility.marker_keys}

    def get_bar_params(self, index, model_code=None):
        return {key: value for key, value in StyleUtility.get_style(model_code).items() if
                key in StyleUtility.bar_keys}

    def get_all_params(self, model_code, index=None, total=None):
        return {key: value for key, value in StyleUtility.get_style(model_code).items() if
                key in StyleUtility.label_keys}
        # tmp_dict = self.get_line_params(index)
        # tmp_dict.update(label=self.get_label(model_code=model_code),
        #                 marker=self.get_marker(index, total), fmt='--', markersize=self.markersize)
        # return tmp_dict

    def save_figure(self, additional_dir_path, name, fig=None):
        if self.save_flag:
            if fig is None:
                fig = self.fig
            self.save_figure_static(self.base_plot_dir, additional_dir_path, name, fig, suffix=self.suffix)

    @staticmethod
    def save_figure_static(base_dir, additional_dir_path, name, fig, suffix='', svg=False):
        host_name, current_time_str = get_info()
        dir_path = PlotUtility.get_base_path_static(base_dir, additional_dir_path, suffix=suffix)
        for t_dir in [dir_path]:
            for t_name in [
                # f'{current_time_str}_{name}',
                f'{name}']:
                t_full_path = os.path.join(t_dir, t_name)
                os.makedirs(t_dir, exist_ok=True)
                fig.savefig(t_full_path + '.pdf', bbox_inches="tight")
                if svg:
                    t_full_path_svg = os.path.join(t_dir + '_svg', t_name)
                    os.makedirs(t_dir + '_svg', exist_ok=True)
                    fig.savefig(t_full_path_svg + '.svg', format='svg', bbox_inches="tight")

    def get_base_path(self, additional_dir_path, suffix=''):
        return self.get_base_path_static(self.base_plot_dir, additional_dir_path, suffix=suffix)

    @staticmethod
    def get_base_path_static(base_plot_dir, additional_dir_path, suffix='', ):
        host_name, current_time_str = get_info()
        return os.path.join(base_plot_dir, host_name + suffix, additional_dir_path)

    def apply_plot_function_and_save(self, df, additional_dir_path, plot_function, name, fig=None, ax=None, **kwargs):
        plt.close('all')
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        plot_function(df, ax=ax, fig=fig, **kwargs)
        # ax.set_title(StyleUtility.replace_words(name))
        if self.show:
            plt.show()
        self.save_figure(additional_dir_path=additional_dir_path, name=name, fig=fig)


def time_stacked_by_phase(df, ax, fig: plt.figure, name_col='label'):
    fig.set_figheight(8)
    fig.set_figwidth(20)
    old = plt.rcParams.get('savefig.dpi')
    plt.rcParams.update({'savefig.dpi': 400})

    df = df.query('model_code == "hybrid_7_exp"')
    extracted_oracle_times = df['oracle_execution_times_'].agg(lambda x: pd.DataFrame(ast.literal_eval(x)).agg(sum))
    df = df.join(extracted_oracle_times)
    to_index = extracted_oracle_times.columns.tolist()
    to_plot = df.groupby(['frac', 'phase'], as_index=False).agg(
        {x: ['mean', ('error', get_confidence_error)] for x in to_index})
    to_plot.columns = to_plot.columns.map('_'.join).str.strip('_')
    to_plot = to_plot.query('phase == "expgrad_fracs"')
    yerr = to_plot.loc[:, [x + '_error' for x in to_index]]
    to_plot.plot.bar(x='frac', stacked=True, y=[x + '_mean' for x in to_index], yerr=yerr.values.T, rot=45, ax=ax)

    xticklabels = ax.xaxis.get_ticklabels()
    for label in xticklabels:
        label.set_ha('right')
    plt.rcParams.update({'savefig.dpi': old})


def bar_plot_function_by_dataset(df, ax, fig: plt.figure, name_col='label', y_axis_list=None):
    to_plot = df.pivot_table(values=['time_mean', 'time_error'], index=['dataset_name'], columns=['model_code'], )
    yerr = to_plot.loc[:, (['time_error'], slice(None))]
    yerr.columns = yerr.columns.get_level_values(1)
    tmp = to_plot.loc[:, (['time_mean'], slice(None))]
    tmp.columns = tmp.columns.get_level_values(1)
    tmp.plot.bar(logy=True, yerr=yerr, rot=0, ax=ax, fontsize=8, )


def bar_plot_function_by_model(df, ax, fig: plt.figure, name_col='model_code', y_axis_list=None):
    orig_y_axis_list = pd.Series(y_axis_list)
    y_axis_list = pd.Series(replace_words_in_list(orig_y_axis_list))

    # ylabel = ' '.join(y_axis_list[0].split(' ')[:5])
    # y_axis_list = pd.Series([' '.join(x.split(' ')[5:]) for x in y_axis_list])

    df = df.set_index(name_col)
    index = df.index.array
    # index[1::2] = '\n' + index[1::2]
    df.index = index
    df = df[pd.concat([orig_y_axis_list, orig_y_axis_list + '_error'])]

    df = df.rename(columns=dict(zip(orig_y_axis_list, y_axis_list)))
    df = df.rename(columns=dict(zip(orig_y_axis_list + '_error', y_axis_list + '_error')))
    yerr = df[y_axis_list + '_error']
    yerr.columns = y_axis_list

    df[yerr.columns].plot.bar(yerr=yerr, rot=0, fontsize=10, ax=ax)
    legend = ax.get_legend()
    labels = (x.get_text() for x in legend.get_texts())
    ax.get_legend().remove()
    fig.legend(legend.legendHandles, labels,
               ncol=min(7, len(y_axis_list)),
               loc='upper center',
               bbox_to_anchor=(0.5, 0.0),
               bbox_transform=fig.transFigure,
               fontsize=10,
               )
    ax.set_xlabel('')
    # ax.set_ylabel(ylabel)


def phase_time_vs_frac(df, ax, fig, y_log=True):
    to_plot = df.groupby(['frac', 'phase']).agg({'time': ['mean', ('error', get_confidence_error)]}).unstack('phase')
    yerr = to_plot.loc[:, ('time', 'error', slice(None))]
    to_plot.plot(y=('time', 'mean'), yerr=yerr.values.T, rot=0, ax=ax, ylabel=StyleUtility.replace_words('time'))
    if y_log:
        ax.set_yscale("log")


def plot_metrics_time(df, ax, fig):
    to_plot = df.query('phase == "evaluation"').copy().reset_index(drop=True)
    convert_df = lambda x: pd.DataFrame(ast.literal_eval(x)).set_index('metric').T
    metric_times = pd.concat(to_plot['metrics_time'].apply(convert_df).values).reset_index()
    cols = ['frac']
    train_cols = list(metric_times.columns[metric_times.columns.str.startswith('train')])
    all_df = pd.concat([to_plot[cols], metric_times[train_cols]], 1)
    # all_df.boxplot(column=train_cols, ax=ax, rot=45);

    to_plot = all_df.groupby(cols).agg(['mean', ('error', get_confidence_error)])
    yerr = to_plot.loc[:, (slice(None), 'error')]
    to_plot.loc[:, (slice(None), 'mean')].plot(yerr=yerr.values.T, rot=0, ax=ax,
                                               ylabel=StyleUtility.replace_words('time'))


def plot_routine_performance_violation(all_model_df, dataset_name, save=True, show=True, suffix='', ):
    missed_conf = np.setdiff1d(all_model_df['model_code'].unique(),
                               list(StyleUtility.map_df.index.values)).tolist()
    assert len(missed_conf) == 0, missed_conf

    pl_util = PlotUtility(save=save, show=show, suffix=suffix)

    original_list = deepcopy(pl_util.to_plot_models)
    model_set_list = [(original_list, ''),
                      (restricted_list, 'restricted'),
                      (['unconstrained_frac_exp'], 'unconstrained'),
                      ]
    for model_list, set_name in model_set_list:
        plot_cycle(all_model_df, dataset_name, model_list, set_name, pl_util)


def plot_cycle(all_model_df, dataset_name, model_list, set_name, pl_util):
    to_iter = list(itertools.product(['train', 'test'], ['error', 'violation', 'di'],
                                     [('time', all_model_df), ('frac', all_model_df)]
                                     ))
    set_name += '_' if set_name != '' else ''
    for phase, metric_name, (x_axis, turn_df) in to_iter:
        y_axis = f'{phase}_{metric_name}'
        # plt.close('all')
        if set_name != '' and x_axis == 'frac':
            continue
        pl_util.to_plot_models = model_list
        pl_util.plot(turn_df, dataset_name=dataset_name, y_axis=y_axis, x_axis=x_axis)
        pl_util.save_figure(additional_dir_path=dataset_name,
                            name=f'{set_name}{x_axis}_vs_{y_axis}')


def plot_routine_other(all_model_df, dataset_name, save=True, show=True, suffix='', annotate_mode='all'):
    pl_util = PlotUtility(save=save, show=show, suffix='', annotate_mode=annotate_mode)
    df = all_model_df
    df = df[df['model_code'].isin(pl_util.to_plot_models)]
    df.loc[:, 'label'] = df['model_code'].apply(StyleUtility.get_label)
    # df.loc[:, 'model_code'] = PlotUtility.map_df.loc[df['model_code'], 'label'].values
    # split_name_value = re.compile("(?P<name>[a-zA-Z\_]+)\=(?P<value>[a-zA-Z]+)")
    # model_code_map = {}
    # for name in df['model_code'].unique():
    #     model_code_map[name] = ' '.join([f'{x[0][0]}={x[1][0]}' for x in split_name_value.findall(name)])
    # df['model_code'] = df['model_code'].map(model_code_map)
    for name, plot_f in [
        ['time_stacked_by_phase', time_stacked_by_phase],
        ['metrics_time_vs_frac', plot_metrics_time],
        ['phase_time_vs_frac', phase_time_vs_frac],
    ]:
        pl_util.apply_plot_function_and_save(df=df, additional_dir_path=name, plot_function=plot_f, name=dataset_name)

    # pl_util.plot(all_model_df, x_axis='frac', y_axis='time', dataset_name=dataset_name)
    # if save is True:
    #     pl_util.save_figure(additional_dir_path=name, name=dataset_name)

    ### train_error_vs_eps
    # pl_util = PlotUtility(show=show)
    # phase, metric_name, x_axis = 'train', 'error', 'eps'
    # y_axis = f'{phase}_{metric_name}'
    # y_axis = 'time'
    # pl_util.to_plot_models = ['fairlearn_full_eps', 'expgrad_fracs_eps']
    # pl_util.plot(eps_df, y_axis=y_axis, x_axis='eps')
    # if save is True:
    #     pl_util.save(base_plot_dir, dataset_name=dataset_name, name=f'{y_axis}_vs_{x_axis}')


def rename_columns_to_plot(df, x_axis, y_axis):
    for key, column in {'x': x_axis, 'y': y_axis}.items():
        for (suffix, sub_col) in {'': 'mean', 'err': 'error'}.items():
            assert f'{column}_{sub_col}' in df.columns, f'{column}_{sub_col} not in df.columns, check column names.'
            df[f'{key}{suffix}'] = df[f'{column}_{sub_col}']
    return df


def plot_all_df_subplots(all_df, model_list, chart_name, save, show=True, grouping_col=None, axis_to_plot=None,
                         sharex=True,
                         sharey='row', result_path_name='all_df', single_chart=False, xlog=False,
                         increasing_marker_size=False,
                         subplots_by_col='dataset_name', subplots=True,
                         ylim_list=None, add_threshold=False, annotate_mode='all', annotate_col=None,
                         custom_add_graphic_object=None, pl_util=None, params={}):
    """

    :param all_df:
    :param model_list: List of model codes to plot
    :param chart_name:
    :param save:
    :param show:
    :param grouping_col: Column to group by the results in order to compare the metrics when varying the parameter of a model.
                E.g.: 'fraction' for comparing perfomance of the model when varying the fraction of the dataset used for training.
    :param axis_to_plot:
    :param sharex:
    :param sharey:
    :param result_path_name:
    :param single_chart:
    :param xlog:
    :param increasing_marker_size:
    :param subplots_by_col:
    :param subplots:
    :param ylim_list:
    :param add_threshold:
    :param annotate_mode:
    :param annotate_col:
    :param custom_add_graphic_object:
    :param pl_util:
    :param params:
    :return:
    """
    if annotate_col is not None:
        annotate_col += '_mean'
    if chart_name != '':
        chart_name += '_'
    # filtered_df = utils_results_data.prepare_data(all_df)
    model_list = list(model_list)

    mean_error_df = prepare_for_plot(all_df[all_df['model_code'].isin(model_list)], grouping_col)
    if add_threshold:
        mean_error_df = utils_results_data.add_threshold(mean_error_df)
        model_list = ['Threshold'] + model_list
    # mean_error_df = mean_error_df[mean_error_df['model_code']]
    if pl_util is None:
        pl_util = PlotUtility(save=save, show=show, suffix='', annotate_mode=annotate_mode,
                              custom_add_graphic_object=custom_add_graphic_object)
    if params is not None:
        pl_util.params = params
    if axis_to_plot is None:
        axis_to_plot = [[grouping_col, 'time'],
                        [grouping_col, 'test_error'],
                        [grouping_col, 'test_violation'],
                        [grouping_col, 'train_error'],
                        [grouping_col, 'train_violation'],
                        ]
        assert grouping_col is not None, 'grouping_col must be defined if axis_to_plot is not defined.'

    for keys, df_to_plot in mean_error_df.groupby(['base_model_code', 'constraint_code'], sort=False):
        base_model_code, constraint_code = keys
        # replace 'violation' with turn constraint name
        turn_axis_list = [[x.replace('violation', constraint_code_to_name[constraint_code]) for x in pair] for pair in
                          axis_to_plot]
        if subplots:
            pl_util.show = False
            figsize = np.array([14.4 / 5 * df_to_plot[subplots_by_col].nunique(), 2.4 * len(list(turn_axis_list))])
            if len(list(turn_axis_list)) == 1:
                figsize = np.array([5, 5])
            fig, axes_array = plt.subplots(nrows=len(turn_axis_list), ncols=df_to_plot[subplots_by_col].nunique(),
                                           sharex=sharex,
                                           sharey=sharey, figsize=figsize,
                                           tight_layout=True)  # todo fix sharey not showing multiple axis labels
            axes_array = np.array(axes_array).reshape(len(turn_axis_list), -1)
            pl_util.fig = fig
        for row, (x_axis, y_axis) in enumerate(turn_axis_list):
            pl_util.row = row
            if 'violation' in y_axis:
                y_axis = y_axis.replace('violation', constraint_code_to_name[constraint_code])
            check_axis_validity(df_to_plot, x_axis, y_axis)
            df_to_plot = rename_columns_to_plot(df_to_plot, x_axis, y_axis)

            col_to_use = ['x', 'xerr', 'y', 'yerr', 'model_code'] + ([grouping_col] if grouping_col is not None else [])

            if annotate_col is not None:
                col_to_use += [annotate_col]
            df_subplot = df_to_plot.groupby([subplots_by_col], sort=False, )[col_to_use]

            for col, (subplot_value, turn_df) in enumerate(df_subplot):
                pl_util.col = col
                if subplots:
                    pl_util.ax = axes_array[row, col]

                    pl_util.add_multiple_lines(turn_df, grouping_col, model_list, increasing_marker_size, annotate_col)
                    pl_util._end_plot(x_axis, y_axis, f'{subplot_value}')
                    pl_util.ax.set_title(StyleUtility.replace_words(f'{subplot_value}'))
                    pl_util.ax.get_legend().remove()
                    if xlog:
                        pl_util.ax.set_xscale("log")
                    if sharey is False:
                        pl_util.ax.yaxis.set_tick_params(which='both', labelleft=True)

                else:
                    pl_util._start_plot()
                    pl_util.add_multiple_lines(df_to_plot, grouping_col, model_list, increasing_marker_size=True,
                                               annotate_col=annotate_col)
                    pl_util._end_plot(x_axis, y_axis,
                                      title=f'{constraint_code} - {subplot_value} - {base_model_code} - VARY {grouping_col}')
                    name = f'all_{base_model_code}_{constraint_code}_VARY_{grouping_col}_{x_axis}_vs_{y_axis}'
                    pl_util.save_figure(additional_dir_path=subplot_value, name=name)

        if subplots:
            if sharey == 'row':
                if ylim_list is not None:
                    for r, lim in enumerate(ylim_list):
                        if lim is not None:
                            for tax in axes_array[r, :]:
                                tax.set_ylim(lim)
            if params.get('same_scale_ylim_row', False):
                same_scale_ylim_row(axes_array)
            tmp_dict = [ax.get_legend_handles_labels() for ax in axes_array.flat[::-1]]
            handles, labels = max(tmp_dict, key=lambda x: len(x[1]))
            if len(labels) != len(model_list):
                logging.warning('Some model are not displayed.')

            if pl_util.params.get('legend_hook', None) is not None:
                pl_util.params['legend_hook'](fig, handles, labels)
            else:
                fig.legend(handles, labels, ncol=min(7, len(labels)),
                           loc='upper center',
                           bbox_to_anchor=(0.5, 0.0),
                           bbox_transform=fig.transFigure,
                           # borderaxespad=-2,
                           fontsize=10,
                           )
            for ax in axes_array[1:].flat:
                ax.set_title('')
            # Remove axis labels from inner plots
            for ax in axes_array[:, 1:].flat:
                ax.set_ylabel('')
                # ax.label_outer()
            xlabels = []

            if sharex:
                for ax in axes_array[:, :].flat:
                    xlabels.append(ax.get_xlabel())
                    ax.set_xlabel('')
                # for ax in axes_array[:-1, :].flat:
                #     ax.xaxis.set_ticklabels([])
                fig.tight_layout()
                if np.unique(xlabels).size == 1:
                    fig.supxlabel(xlabels[0].replace('\n', ' '), y=0, va='baseline', fontsize='small')

            # pl_util.fig.suptitle(StyleUtility.replace_words(f'{base_model_code} - {constraint_code}'))
            # fig.subplots_adjust(bottom=0.1)

            if show:
                fig.show()
            pl_util.save_figure(additional_dir_path=result_path_name,
                                name=f'{chart_name}all_{base_model_code}_{constraint_code}_VARY_{grouping_col}_subplots')
    pl_util.show = show

    dir_path = pl_util.get_base_path(additional_dir_path=result_path_name)
    mean_error_df.to_csv(os.path.join(dir_path, f'{chart_name}all_VARY_{grouping_col}_metrics_mean_error.csv'))
    # if single_chart:
    #     plot_all_df_single_chart(pl_util, grouping_col, mean_error_df, model_set_name)
    return mean_error_df


def same_scale_ylim_row(axes_array):
    for axes_row in axes_array:
        ylim_list = [ax.get_ylim() for ax in axes_row]
        max_diff = max(ymax - ymin for ymin, ymax in ylim_list)
        for ax, (ymin, ymax) in zip(axes_row, ylim_list):
            center = np.mean((ymin, ymax))
            new_ymin = center - max_diff / 2.0
            new_ymax = new_ymin + max_diff
            ax.set_ylim(new_ymin, new_ymax)


def plot_all_df_single_chart(pl_util, grouping_col, filtered_df, model_set_name='',
                             additional_dir_path=os.path.join('all_df'),
                             ):
    if model_set_name != '':
        model_set_name += '_'
    additional_dir_path = os.path.join(additional_dir_path, 'single_chart')
    mean_error_df = prepare_for_plot(filtered_df, grouping_col)
    for keys, df_to_plot in mean_error_df.groupby(['base_model_code', 'constraint_code']):
        base_model_code, constraint_code = keys
        for x_axis, y_axis in [
            ['time', 'test_error'],
            ['time', 'test_violation'],
            ['time', 'test_di', ],
            ['test_violation', 'test_error'],
            ['test_di', 'test_error'],
        ]:
            pl_util._start_plot()
            df_to_plot = rename_columns_to_plot(df_to_plot, x_axis, y_axis)
            df_groups = df_to_plot.groupby(['dataset_name', 'model_code'])[
                ['x', 'xerr', 'y', 'yerr', grouping_col]]
            n_lines = len(df_groups)
            for i, (key, value) in enumerate(df_groups):
                dataset_name, model_code = key
                value = value.sort_values(grouping_col)
                value_dict = value[['x', 'xerr', 'y', 'yerr']].to_dict(orient='list')
                grouping_values = value[grouping_col]
                pl_util.add_line_errorbar(value_dict, grouping_values, model_code=model_code,
                                          label_suffix=f' | {dataset_name}', index=i, n_lines=n_lines)
                pl_util.add_annotation(value_dict['x'], value_dict['y'], grouping_values)
            pl_util._end_plot(x_axis, y_axis, title=f'{base_model_code} - VARY {grouping_col}')
            name = f'{model_set_name}all_{base_model_code}_{constraint_code}_VARY_{grouping_col}_{x_axis}_vs_{y_axis}'
            pl_util.save_figure(additional_dir_path=additional_dir_path,
                                name=name)


def select_oracle_call_time(results_df, name_time_oracles_col='time_oracles'):
    df = results_df[results_df['phase'].isin(['expgrad_fracs', 'grid_frac'])].copy()
    # Take max of oracle calls time for grid search
    grid_mask = df['phase'] == 'grid_frac'
    grid_time_series = df[grid_mask]['grid_oracle_times'].apply(
        lambda x: np.array(ast.literal_eval(x)).max())

    df.loc[grid_mask, name_time_oracles_col] = grid_time_series
    # Take sum of oracle calls time for expgrad
    extract_expgrad_oracle_time(df, new_col_name=name_time_oracles_col)
    return df


def extract_expgrad_oracle_time(df, new_col_name='time', cols_to_select=['fit_sum']):
    df[new_col_name] = np.nan
    exp_mask = df['phase'] == 'expgrad_fracs'
    exp_time_df = df[exp_mask]['oracle_execution_times_'].agg(
        lambda x: pd.DataFrame(ast.literal_eval(x)).sum())
    exp_time_df.columns += '_sum'
    if cols_to_select == 'all':
        cols_to_select = exp_time_df.columns
    df.loc[exp_mask, new_col_name] = exp_time_df[cols_to_select].sum(1)


def plot_by_df(pl_util: PlotUtility, all_df, model_list, model_set_name, grouping_col,
               x_axis_list=['time'],
               y_axis_list=['_'.join(x) for x in
                            itertools.product(['train', 'test'], ['error', 'violation'])],
               ):
    if model_set_name != '':
        pl_util.suffix = '_' + model_set_name

    mean_error_df = prepare_for_plot(all_df, grouping_col)
    mean_error_df = mean_error_df[mean_error_df['model_code'].isin(model_list)]

    grouped = mean_error_df.groupby(['base_model_code', 'dataset_name', 'constraint_code'], sort=False)
    for key, turn_df in grouped:
        base_model_code, dataset_name, constraint_code = key
        to_iter = list(itertools.product(y_axis_list,
                                         x_axis_list
                                         ))
        for y_axis, x_axis in to_iter:
            if 'violation' in y_axis:
                y_axis = y_axis.replace('violation', constraint_code_to_name[constraint_code])
            # to_plot_models = [x.replace('_exp', '_eps') for x in to_plot_models]
            if model_set_name != '' and x_axis == 'frac':
                continue
            check_axis_validity(turn_df, x_axis, y_axis)

            df_to_plot = rename_columns_to_plot(turn_df, x_axis, y_axis)
            pl_util._start_plot()
            pl_util.add_multiple_lines(df_to_plot, grouping_col, model_list, increasing_marker_size=True)
            pl_util._end_plot(x_axis, y_axis,
                              title=f'{constraint_code} - {dataset_name} - {base_model_code} - VARY {grouping_col}')
            name = f'all_{base_model_code}_{constraint_code}_VARY_{grouping_col}_{x_axis}_vs_{y_axis}'
            pl_util.save_figure(additional_dir_path=dataset_name, name=name)


def check_axis_validity(df, x_axis, y_axis):
    if x_axis not in df.columns and x_axis + '_mean' not in df.columns:
        raise ValueError(f'{x_axis} is not a valid x_axis')
    if y_axis == x_axis:
        raise ValueError(f'{x_axis} and {y_axis} are not a valid x_axis, y_axis combination.')


def plot_demo_subplots(all_df, model_list, chart_name, save, show=False, grouping_col=None, axis_to_plot=None,
                       sharex=True,
                       sharey='row', result_path_name='all_df', xlog=False,
                       increasing_marker_size=False,
                       horizontal_subplots_by_col='dataset_name', single_plot=True,
                       ylim_list=None, annotate_mode='all', annotate_col=None,
                       custom_add_graphic_object=None, pl_util=None, params={}):
    if annotate_col is not None:
        annotate_col += '_mean'
    name_suffix = f' - VARY {grouping_col}' if grouping_col is not None else ''
    # filtered_df = utils_results_data.prepare_data(all_df)
    model_list = list(model_list)

    df_to_plot = prepare_for_plot(all_df[all_df['model_code'].isin(model_list)], grouping_col)
    # mean_error_df = mean_error_df[mean_error_df['model_code']]
    if pl_util is None:
        pl_util = PlotUtility(save=save, show=show, suffix='', annotate_mode=annotate_mode,
                              custom_add_graphic_object=custom_add_graphic_object)
    if single_plot:
        pl_util.show = False
        if params.get('figsize', None) is not None:
            figsize = params['figsize']
        else:
            figsize = np.array(
                [14.4 / 5 * df_to_plot[horizontal_subplots_by_col].nunique(), 2.4 * len(list(axis_to_plot))])
        # fig = plt.figure()
        #
        # fig = plt.gcf()
        # fig.clear()
        # axes_array = fig.add_subplot(
        fig, axes_array= plt.subplots(
            nrows=len(axis_to_plot), ncols=df_to_plot[horizontal_subplots_by_col].nunique(),
            sharex=sharex,
            sharey=sharey, figsize=figsize,
            tight_layout=True)  # todo fix sharey not showing multiple axis labels
        axes_array = np.array(axes_array).reshape(len(axis_to_plot), -1)
        pl_util.fig = fig
    for row, (x_axis, y_axis) in enumerate(axis_to_plot):
        pl_util.row = row
        check_axis_validity(df_to_plot, x_axis, y_axis)
        df_to_plot = rename_columns_to_plot(df_to_plot, x_axis, y_axis)

        col_to_use = ['x', 'xerr', 'y', 'yerr', 'model_code'] + ([grouping_col] if grouping_col is not None else [])

        if annotate_col is not None:
            col_to_use += [annotate_col]
        df_subplot = df_to_plot.groupby([horizontal_subplots_by_col], sort=False, )[col_to_use]

        for col, (subplot_value, turn_df) in enumerate(df_subplot):
            pl_util.col = col
            if single_plot:
                pl_util.ax = axes_array[row, col]

                pl_util.add_multiple_lines(turn_df, grouping_col, model_list, increasing_marker_size, annotate_col)
                pl_util._end_plot(x_axis, y_axis, f'{subplot_value}')
                pl_util.ax.set_title(StyleUtility.replace_words(f'{subplot_value}'))
                pl_util.ax.get_legend().remove()
                if xlog:
                    pl_util.ax.set_xscale("log")
                if sharey is False:
                    pl_util.ax.yaxis.set_tick_params(which='both', labelleft=True)

            else:
                pl_util._start_plot()
                pl_util.add_multiple_lines(df_to_plot, grouping_col, model_list, increasing_marker_size=True,
                                           annotate_col=annotate_col)
                pl_util._end_plot(x_axis, y_axis,
                                  title=f'{subplot_value}{name_suffix}')
                name = f'{chart_name}_{name_suffix}_{x_axis}_vs_{y_axis}'
                pl_util.save_figure(additional_dir_path=result_path_name, name=name)

    if single_plot:
        if sharey == 'row':
            if ylim_list is not None:
                for r, lim in enumerate(ylim_list):
                    if lim is not None:
                        for tax in axes_array[r, :]:
                            tax.set_ylim(lim)
        if params.get('same_scale_ylim_row', False):
            same_scale_ylim_row(axes_array)
        tmp_dict = [ax.get_legend_handles_labels() for ax in axes_array.flat[::-1]]
        handles, labels = max(tmp_dict, key=lambda x: len(x[1]))
        if len(labels) != len(model_list):
            logging.warning('Some model are not displayed.')

        if pl_util.params.get('legend_hook', None) is not None:
            pl_util.params['legend_hook'](fig, handles, labels)
        else:
            fig.legend(handles, labels, ncol=min(7, len(labels)),
                       loc='upper center',
                       bbox_to_anchor=(0.5, 0.0),
                       bbox_transform=fig.transFigure,
                       # borderaxespad=-2,
                       fontsize=10,
                       )
        for ax in axes_array[1:].flat:
            ax.set_title('')
        # Remove axis labels from inner plots
        for ax in axes_array[:, 1:].flat:
            ax.set_ylabel('')
            # ax.label_outer()
        xlabels = []

        if sharex:
            for ax in axes_array[:, :].flat:
                xlabels.append(ax.get_xlabel())
                ax.set_xlabel('')
            # for ax in axes_array[:-1, :].flat:
            #     ax.xaxis.set_ticklabels([])
            fig.tight_layout()
            if np.unique(xlabels).size == 1:
                fig.supxlabel(xlabels[0].replace('\n', ' '), y=0, va='baseline', fontsize='medium')

        # pl_util.fig.suptitle(StyleUtility.replace_words(f'{base_model_code} - {constraint_code}'))
        # fig.subplots_adjust(bottom=0.1)

        if show:
            fig.show()
        pl_util.save_figure(additional_dir_path=result_path_name,
                            name=f'{chart_name}_VARY_{grouping_col}_subplots')
    # replace 'violation' with turn constraint name

    pl_util.show = show

    dir_path = pl_util.get_base_path(additional_dir_path=result_path_name)

    df_to_plot = df_to_plot.drop(columns=['x', 'xerr', 'y', 'yerr'])

    first_columns = intersection_sorted(['model_code', 'dataset_name', 'train_fractions'], df_to_plot.columns)
    df_to_plot = df_to_plot[first_columns + difference_sorted(df_to_plot.columns, first_columns)]
    os.makedirs(os.path.join(dir_path), exist_ok=True)
    df_to_plot.to_csv(os.path.join(dir_path, f'{chart_name}_VARY_{grouping_col}_metrics_mean_error.csv'))
    # if single_chart:
    #     plot_all_df_single_chart(pl_util, grouping_col, mean_error_df, model_set_name)
    return pl_util.fig


