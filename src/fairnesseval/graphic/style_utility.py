import itertools
from copy import deepcopy

import pandas as pd
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

rename_word_dict = {'lr': 'Logistic Regression',
                    'lgbm': 'LightGBM',
                    'dp': 'Demographic Parity',
                    'eo': 'Equalized Odds',
                    'unmitigated': 'Unmitigated',
                    'ACSEmployment': 'ACS Employment',
                    'ACSPublicCoverage': 'ACS Public Coverage',
                    'adult': 'Adult',
                    'adult_sigmod': 'Adult',
                    'compas': 'COMPAS',
                    'german': 'German',
                    'time': 'Training time [s]',
                    'test_error': 'Test Error',
                    'test_DemographicParity': 'Demographic parity\ndifference on test data',
                    'test_EqualizedOdds': 'Equalized odds\ndifference on test data',
                    'train_error': 'Train Error',
                    'train_DemographicParity': 'Demographic parity\ndifference on train data',
                    'train_EqualizedOdds': 'Equalized odds\ndifference on train data',
                    'eps': 'Epsilon value (eps)',
                    'EXPGRAD=adaptive': 'EXPGRAD++ (adaptive sampling)',
                    'exp_frac': r'Sampling fraction $\rho$',
                    'RLP=False': 'EXPGRAD',
                    'RLP=F': 'EXPGRAD',
                    'GS=No': '',  # remove?
                    'LP=Yes': '',  # remove?
                    'LP=No': '',  # remove?
                    'max_iter=50': '',  # remove?
                    'full': '',  # remove?
                    'ZafarDI': 'ZAFAR DI',
                    'ZafarEO': 'ZAFAR EO',
                    'Feld': 'FELD',
                    'Calmon': 'CALMON',
                    'ThresholdOptimizer': 'HARDT',
                    'binary_mean': 'binary',
                    'multivalued_mean': 'multivalued',
                    'UNMITIGATED=static': 'UNMITIGATED (static sampling)',
                    'EXPGRAD=static': 'EXPGRAD++ (static sampling)',
                    'fairlearn_full_naive_sample': 'Expgrad static sampling',
                    'fairlearn_full_smart_sample': 'Expgrad adaptive sampling',
                    'train_fractions': 'Train fraction',
                    'expgrad': 'EXPGRAD',
                    # 'train_EqualizedOdds mean':'',
                    # 'train_EqualizedOdds_orig mean',
                    # 'train EqualizedOdds Multi mean',
                    }


# function to rename words in sentences by first splitting the sentence into words and then joining the words after using a dictionary to replace the words
def replace_words(sentence):
    """
    Replace words in a sentence using a dictionary
    :param sentence: sentence to replace words
    :return: sentence with words replaced
    """
    words = sentence.split()
    for i in range(len(words)):
        if words[i] in rename_word_dict.keys():
            words[i] = rename_word_dict[words[i]]
    return ' '.join(words).strip()


def replace_words_in_list(sentence_list):
    return [replace_words(x) for x in sentence_list]


def generate_map_df():
    values_dict = {}
    model_names = ['hybrid_1', 'hybrid_2', 'hybrid_3', 'hybrid_4', 'hybrid_5', 'hybrid_6']
    unconstrained = [True, False]
    active_sampling = [True, False]
    run_linprog = [True, False]
    grid_mode = ['sqrt', 'gf_1']
    to_iter = itertools.product(model_names, unconstrained, active_sampling, grid_mode)

    for t_varing in ['_exp', '_gri', '_eps', '', ' binary']:
        for t_run_lp in run_linprog:
            for t_model_name, t_unconstrained, t_active_sampling, t_grid_mode in deepcopy(to_iter):
                name = 'sub_' if t_active_sampling else ''
                name += t_model_name + ('_U' if t_unconstrained else '')
                name += '_LP_off' if t_run_lp else ''
                name += t_varing
                if t_grid_mode == 'gf_1':
                    name += f'_gf_1'
                    t_grid_mode = '1'

                label = f'EXPGRAD=' + ('adaptive' if t_active_sampling else 'static')
                label += ' GS=' + (
                    t_grid_mode if any('_' + x in t_model_name for x in ['1', '2', '3', '4', '6']) else 'No')
                label += ' LP=Yes'
                label += ' +U' if t_unconstrained else ''
                if 'hybrid_6' in t_model_name:
                    label += ' *e&g'
                label += ' run_linprog=F' if t_run_lp else ''
                values_dict[name] = label.strip()

            rlp_name = "_LP_off" if t_run_lp else ""
            rlp_label = ' run_linprog=F' if t_run_lp else ''
            values_dict['hybrid_7' + rlp_name + t_varing] = 'EXPGRAD=adaptive GS=No LP=Yes' + rlp_label
            values_dict['expgrad_fracs' + rlp_name + t_varing] = 'EXPGRAD=static GS=No LP=No' + rlp_label
        values_dict['unconstrained' + t_varing] = 'UNMITIGATED full'
        values_dict['unconstrained_frac' + t_varing] = 'UNMITIGATED=static'

    return pd.DataFrame.from_dict({key: replace_words(value) for key, value in values_dict.items()}, orient='index',
                                  columns=['label'])


linewidth = 1.5  # 0.5
markersize = 30  # 8
rlp_false_markersize = markersize ** .5
base_config = dict(linewidth=linewidth, elinewidth=linewidth / 4,
                   marker=MarkerStyle('1', 'left', 0), s=markersize, markevery=1,  # mew=linewidth / 2,
                   )
unmitigated_markersize = markersize ** 0.9


class StyleUtility:
    linewidth = linewidth
    markersize = markersize

    common_keys = ['color', 'alpha', 'zorder']
    line_keys = common_keys + ['linestyle', 'linewidth', 'elinewidth']
    marker_keys = common_keys + ['marker', 's']
    bar_keys = common_keys + ['label']
    label_keys = common_keys + ['marker', 'label']

    map_df = generate_map_df()
    other_models = ['ThresholdOptimizer', 'Calmon', 'ZafarDI', 'Feld']
    map_df = pd.concat([map_df,
                        pd.DataFrame.from_dict({x: replace_words(x) for x in other_models},
                                               orient='index', columns=['label'])
                        ])

    graphic_style_map = {
        'fairlearn': {'color': 'tab:blue', 'marker': 'o', 'linestyle': '-'},
        'fairlearn_full_smart_sample max_iter=20': {'color': 'tab:green', 'marker': '.', 'linestyle': '-.'},
        'fairlearn_full_naive_sample max_iter=20': {'color': 'tab:purple', 'marker': '.', 'linestyle': 'dotted'},
        'fairlearn_full_smart_sample max_iter=50': {'color': 'tab:olive', 'marker': '.', 'linestyle': '-.'},
        'fairlearn_full_naive_sample max_iter=50': {'color': 'tab:cyan', 'marker': '.', 'linestyle': 'dotted'},
        'EXPGRAD++': {'color': 'tab:blue', 'marker': '.', 'linestyle': '--'},
        'EXPGRAD=adaptive GS=No LP=Yes': {'color': 'tab:blue', 'marker': '.', 'linestyle': '--'},
        'EXPGRAD=adaptive GS=No LP=Yes binary': {'color': 'tab:blue', 'marker': '.', 'linestyle': '--'},
        'EXPGRAD=static GS=No LP=Yes': {'color': 'tab:orange', 'marker': '.', 'linestyle': '-.'},
        'EXPGRAD=static GS=No LP=Yes binary': {'color': 'tab:orange', 'marker': '.', 'linestyle': '-.'},
        'EXPGRAD=adaptive GS=No LP=Yes_orig': {'color': 'tab:blue', 'marker': MarkerStyle('o', fillstyle='none'),
                                               'linestyle': '-.', 's': markersize * 3},
        'EXPGRAD=adaptive GS=No LP=Yes Multi': {'color': 'tab:blue',
                                                'marker': MarkerStyle('o', fillstyle='top',
                                                                      transform=Affine2D().rotate_deg(45)),
                                                'linestyle': '-.',
                                                's': markersize * 3},

        'EXPGRAD=adaptive GS=sqrt LP=Yes': {'color': 'tab:brown', 'marker': '.', 'linestyle': '-.'},
        'EXPGRAD=adaptive GS=1 LP=Yes': {'color': 'tab:red', 'marker': '.', 'linestyle': '-.'},
        'EXPGRAD=static GS=1 LP=Yes': {'color': 'tab:purple', 'marker': '.', 'linestyle': '-.'},
        'EXPGRAD=static GS=sqrt LP=Yes': {'color': 'tab:green', 'marker': '.', 'linestyle': '-.'},
        'LogisticRegression': {'color': 'tab:brown', 'marker': 'X', 'linestyle': '-',
                               # 's': unmitigated_markersize**1.6,
                               },
        'UNMITIGATED full': {'color': 'tab:brown', 'marker': 'X', 'linestyle': '-', 's': unmitigated_markersize,
                             'zorder': 3},
        'UNMITIGATED full binary': {'color': 'tab:brown', 'marker': 'X', 'linestyle': '-', 's': unmitigated_markersize,
                                    'zorder': 3},
        'UNMITIGATED full_orig': {'color': 'tab:brown', 'marker': MarkerStyle('X', fillstyle='none'), 'linestyle': '-',
                                  's': markersize * 3, 'zorder': 3},
        'UNMITIGATED full Multi': {'color': 'tab:brown',
                                   'marker': MarkerStyle('X', fillstyle='top', transform=Affine2D().rotate_deg(45)),
                                   'linestyle': '-', 's': markersize * 3},
        'UNMITIGATED=static': {'color': 'tab:brown', 'marker': 'X', 'linestyle': '-.', 's': unmitigated_markersize,
                               'zorder': 3},
        'UNMITIGATED=static binary': {'color': 'tab:brown', 'marker': 'X', 'linestyle': '-.',
                                      's': unmitigated_markersize},

        'Threshold': {'color': 'black', 'marker': 'P', 'linestyle': 'solid', 'linewidth': linewidth},

        'RLP=F': {'color': 'tab:orange', 'marker': 's', 'linestyle': '-.', 's': rlp_false_markersize},
        'RLP=F max_iter=5': {'color': 'tab:brown', 'marker': 'v', 'linestyle': '-.', 's': rlp_false_markersize},
        'RLP=F max_iter=10': {'color': 'tab:gray', 'marker': '^', 'linestyle': '-.', 's': rlp_false_markersize},
        'RLP=F max_iter=20': {'color': 'tab:orange', 'marker': '<', 'linestyle': '-.', 's': rlp_false_markersize},
        'RLP=F max_iter=50': {'color': 'tab:pink', 'marker': '>', 'linestyle': '-.', 's': rlp_false_markersize * 1.5},
        'RLP=F max_iter=100': {'color': 'tab:cyan', 'marker': 'd', 'linestyle': '-.', 's': rlp_false_markersize},

        'ThresholdOptimizer': {'color': 'tab:green', 'marker': '*', 'linestyle': '-.', 's': markersize * 1.25},
        'ThresholdOptimizer binary': {'color': 'tab:green', 'marker': '*', 'linestyle': '-.'},
        'ThresholdOptimizer_orig': {'color': 'tab:green', 'marker': '*', 'linestyle': '-.', 'fillstyle': 'none'},

        'Calmon': {'color': 'tab:red', 'marker': '^', 'linestyle': '--'},
        'ZafarDI': {'color': 'tab:purple', 'marker': 's', 'linestyle': (0, (1, 1)), 'linewidth': linewidth},
        'ZafarDI binary': {'color': 'tab:purple', 'marker': 's', 'linestyle': (0, (1, 1)),
                           'linewidth': linewidth},
        'ZafarDI_orig': {'color': 'tab:purple', 'marker': MarkerStyle('s', fillstyle='none'), 'linestyle': (0, (1, 1)),
                         'linewidth': linewidth, },
        'ZafarEO': {'color': 'tab:purple', 'marker': 's', 'linestyle': (0, (1, 1)), 'linewidth': linewidth},
        'ZafarEO binary': {'color': 'tab:purple', 'marker': 's', 'linestyle': (0, (1, 1)),
                           'linewidth': linewidth},
        'Feld': {'color': 'tab:orange', 'marker': 'D', 'linestyle': (5, (10, 3)), 'alpha': 0.75, 's': markersize ** .7},
        'Feld binary': {'color': 'tab:orange', 'marker': 'D', 'linestyle': (5, (10, 3))},
        'Feld_orig': {'color': 'tab:orange', 'marker': MarkerStyle('D', fillstyle='none'), 'linestyle': (5, (10, 3))},
        'default': {'color': 'tab:red', 'marker': 's', 'linestyle': '--'},

    }
    graphic_style_map['expgrad'] = graphic_style_map['fairlearn']
    graphic_style_map = {replace_words(key): dict(base_config, label=replace_words(key), **value) for key, value in
                         graphic_style_map.items()}

    @staticmethod
    def get_label(model_code):
        if model_code in StyleUtility.map_df.index:
            return StyleUtility.map_df.loc[model_code, 'label']
        else:
            return model_code

    @staticmethod
    def get_style(key, index=None):
        label = replace_words(StyleUtility.get_label(key))
        if label not in StyleUtility.graphic_style_map.keys():
            print(f'key {label} not in graphic_style_map')
            return StyleUtility.graphic_style_map['default']
        return StyleUtility.graphic_style_map[label]

    @staticmethod
    def replace_words(sentence):
        return replace_words(sentence)
