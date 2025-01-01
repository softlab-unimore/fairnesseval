import inspect
from functools import partial

import numpy as np
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, \
    equalized_odds_ratio, make_derived_metric, true_positive_rate, true_negative_rate, false_positive_rate, \
    false_negative_rate

from fairnesseval import utils_prepare_data
from fairlearn.reductions import DemographicParity, ErrorRate, EqualizedOdds
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score


def divide_non_0(a, b):
    res = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    res[a == b] = 1
    return res.item() if res.shape == () else res


def get_metric_function(metric_f):
    def f(X, Y, S, y_pred):
        return metric_f(y_true=Y, y_pred=y_pred >= .5, zero_division=0)

    return f


def convert_metric_to_use_original_sensitive(metric_f):
    params = inspect.signature(metric_f).parameters.keys()
    if 'predict_method' in params:
        def f(X, Y, S, predict_method):
            data_values = utils_prepare_data.DataValuesSingleton()
            s_orig = data_values.get_current_original_sensitive_attr()
            return metric_f(X, Y, s_orig, predict_method=predict_method)
    elif 'y_pred' in params:
        def f(X, Y, S, y_pred):
            data_values = utils_prepare_data.DataValuesSingleton()
            s_orig = data_values.get_current_original_sensitive_attr()
            return metric_f(X, Y, s_orig, y_pred=y_pred)
    else:
        raise ValueError('metric function must have either predict_method or y_pred as parameter')
    return f


def getViolation(X, Y, S, predict_method):
    disparity_moment = DemographicParity()
    disparity_moment.load_data(X, Y, sensitive_features=S)
    return disparity_moment.gamma(predict_method).max()


def getEO(X, Y, S, predict_method):
    eo = EqualizedOdds()
    eo.load_data(X, Y, sensitive_features=S)
    return eo.gamma(predict_method).max()


def getError(X, Y, S, predict_method):
    error = ErrorRate()
    error.load_data(X, Y, sensitive_features=S)
    return error.gamma(predict_method)[0]


def di(X, Y, S, y_pred):
    y_pred = y_pred >= .5
    s_values = np.unique(S)
    s_values.sort()
    group_0_mask = S == s_values[0]
    group_1_mask = S == s_values[1]
    PrY1_S0 = np.sum(y_pred[group_0_mask.ravel()] == 1) / np.sum(group_0_mask)
    PrY1_S1 = np.sum(y_pred[group_1_mask.ravel()] == 1) / np.sum(group_1_mask)
    disparate_impact = divide_non_0(PrY1_S0, PrY1_S1)
    return disparate_impact


def trueRateBalance(X, Y, S, y_pred):
    y_pred = y_pred >= .5
    s_values = np.unique(S)
    s_values.sort()
    mask_0 = (S == s_values[0]).ravel()
    mask_1 = (S == s_values[1]).ravel()
    results = {}
    for turn_mask, group in zip([mask_1, mask_0], [1, 0]):
        try:
            TN, FP, FN, TP = confusion_matrix(Y[turn_mask], y_pred[turn_mask] == 1).ravel()
            results[f'TPR_{group}'] = TP / (TP + FN)
            results[f'TNR_{group}'] = TN / (FP + TN)
        except Exception as e:
            results[f'TPR_{group}'] = np.nan
            results[f'TNR_{group}'] = np.nan
    return results


def TPRB(X, Y, S, y_pred):
    rates_dict = trueRateBalance(X, Y, S, y_pred)
    return np.abs(rates_dict['TPR_1'] - rates_dict['TPR_0'])  # TPRB


def TNRB(X, Y, S, y_pred):
    rates_dict = trueRateBalance(X, Y, S, y_pred)
    return np.abs(rates_dict['TNR_1'] - rates_dict['TNR_0'])  # TNRB


default_metrics_dict = {'error': getError,
                        'violation': getViolation,
                        'EqualizedOdds': getEO,
                        'di': di,
                        'TPRB': TPRB,
                        'TNRB': TNRB,
                        'f1': get_metric_function(f1_score),
                        'precision': get_metric_function(precision_score),
                        'recall': get_metric_function(recall_score),
                        }

tpr_diff = make_derived_metric(metric=true_positive_rate, transform="difference")
tnr_diff = make_derived_metric(metric=true_negative_rate, transform="difference")
fpr_diff = make_derived_metric(metric=false_positive_rate, transform="difference")
fnr_diff = make_derived_metric(metric=false_negative_rate, transform="difference")
# tpr_diff(Y,  y_pred, sensitive_features=S ,method="between_groups")
# TPRB(X, Y, S, y_pred)

default_metrics_dict_v1 = {'error': getError,
                           'violation': getViolation,
                           'EqualizedOdds': getEO,
                           # 'di': di,
                           # 'TPRB': TPRB,
                           # 'TNRB': TNRB,
                           'f1': get_metric_function(f1_score),
                           'precision': get_metric_function(precision_score),
                           'recall': get_metric_function(recall_score),

                           # todo add individual discrimination and tprb.
                           'demographic_parity_difference': partial(demographic_parity_difference, method='to_overall'),
                           'demographic_parity_ratio': partial(demographic_parity_ratio, method='to_overall'),
                           'equalized_odds_difference': partial(equalized_odds_difference, method='to_overall'),
                           'equalized_odds_ratio': partial(equalized_odds_ratio, method='to_overall'),
                           'dp_diff_bg': partial(demographic_parity_difference, method='between_groups'),
                           'dp_ratio_bg': partial(demographic_parity_ratio, method='between_groups'),
                           'eo_diff_bg': partial(equalized_odds_difference, method='between_groups'),
                           'eo_ratio_bg': partial(equalized_odds_ratio, method='between_groups'),
                           'tprb': partial(tpr_diff, method="to_overall"),
                           'tnrb': partial(tnr_diff, method="to_overall"),
                           'tprb_bg': partial(tpr_diff, method="between_groups"),
                           'tnrb_bg': partial(tnr_diff, method="between_groups"),
                           'fpr_diff': partial(fpr_diff, method="to_overall"),
                           'fnr_diff': partial(fnr_diff, method="to_overall"),
                           'fpr_diff_bg': partial(fpr_diff, method="between_groups"),
                           'fnr_diff_bg': partial(fnr_diff, method="between_groups"),

                           }

metrics_code_map = {
    'default': default_metrics_dict,
    'default_v1': default_metrics_dict_v1,
    'conversion_to_binary_sensitive_attribute': default_metrics_dict | {
        'violation_orig': convert_metric_to_use_original_sensitive(getViolation),
        'EqualizedOdds_orig': convert_metric_to_use_original_sensitive(getEO),
        'di_orig': convert_metric_to_use_original_sensitive(di),
        'TPRB_orig': convert_metric_to_use_original_sensitive(TPRB),
        'TNRB_orig': convert_metric_to_use_original_sensitive(TNRB),
    }
}


# Metrics function may follow one of these 2 interfaces.
# f(X, Y, S, predict_method)
# or
# f(X, Y, S, y_pred)
# if the metric function takes f(y_true,y_pred) parameters only
# then you may simply wrap the function with get_metric_function


def get_metrics_dict(metrics_code):
    if metrics_code not in metrics_code_map.keys():
        raise ValueError(f'metric {metrics_code} is not supported')
    return metrics_code_map.get(metrics_code)
