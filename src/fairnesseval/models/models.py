import logging
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV

from fairnesseval.models import wrappers


def get_model_parameter_grid(base_model_code=None):
    if base_model_code is None or base_model_code == 'lr':
        # Unmitigated LogRes
        return {'solver': [  # 'newton-cg',
            'lbfgs',
            'liblinear',
            'saga'
        ],
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.005, 0.001, 0.1, 1, 10, 100, 1000],
            # [10, 1.0, 0.1, 0.05, 0.01],
            'max_iter': [200],
        }
    elif base_model_code == 'gbm':
        return dict(n_estimators=[10, 100, 500],
                    learning_rate=[0.001, 0.01, 0.1],
                    subsample=[0.5, 0.7, 1.0],
                    max_depth=[3, 7, 9])
    elif base_model_code == 'lgbm':
        return dict(
            l2_regularization=[10, 0.1, 0.01],
            learning_rate=[0.001, 0.01, 0.1],
            max_depth=[3, 7, 9])
    else:
        assert False, f'available model codes are:{["lr", "gbm", "lgbm"]}'


def get_base_model(base_model_code, random_seed=0):
    if base_model_code is None:
        return None
    if base_model_code == 'lr':
        # Unmitigated LogRes
        model = LogisticRegression(solver='liblinear', fit_intercept=True, random_state=random_seed)
    elif base_model_code == 'gbm':
        model = GradientBoostingClassifier(random_state=random_seed)
    elif base_model_code == 'lgbm':
        model = HistGradientBoostingClassifier(random_state=random_seed)
    else:
        assert False, f'available model codes are:{["lr", "gbm", "lgbm"]}'
    return model


def finetune_model(base_model_code, X, y, random_seed=0):
    base_model = get_base_model(base_model_code=base_model_code, random_seed=random_seed)
    parameters = get_model_parameter_grid(base_model_code=base_model_code)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=random_seed)
    clf = GridSearchCV(base_model, parameters, cv=cv, n_jobs=1, scoring=['f1', 'accuracy'], refit='f1')
    clf.fit(X, y)
    return clf


model_list = ['hybrids', 'unmitigated', 'fairlearn', 'ThresholdOptimizer', 'MetaFairClassifier',
                  'AdversarialDebiasing', 'Kearns', 'Calmon', 'ZafarDI', 'Hardt', 'fairlearn_full', 'ZafarEO',
                  'Feld', 'expgrad']

def get_model(method_str, random_state=42, **kwargs):
    param_dict = dict(method_str=method_str, random_state=random_state, )

    methods_name_dict = {x: x for x in model_list}
    if method_str == methods_name_dict['ThresholdOptimizer']:
        estimator = kwargs.pop('base_model', None)
        model = wrappers.ThresholdOptimizerWrapper(
            estimator=estimator,
            objective="accuracy_score",
            prefit=False,
            predict_method='predict_proba',
            random_state=random_state, **kwargs)
        if kwargs.get('eps') is not None:
            logging.warning(f"eps has no effect with {method_str} methos")
    elif method_str == methods_name_dict['AdversarialDebiasing']:
        pass
        # privileged_groups, unprivileged_groups = utils_prepare_data.find_privileged_unprivileged(**datasets)
        # sess = tf.Session()
        # # Learn parameters with debias set to True
        # model = AdversarialDebiasing(privileged_groups=privileged_groups,
        #                              unprivileged_groups=unprivileged_groups,
        #                              scope_name='debiased_classifier',
        #                              debias=True,
        #                              sess=sess)
    elif method_str == methods_name_dict['Kearns']:
        model = wrappers.Kearns(**param_dict, **kwargs)
    elif method_str == methods_name_dict['Calmon']:
        model = wrappers.CalmonWrapper(**param_dict, **kwargs)
    elif method_str == methods_name_dict['Feld']:
        model = wrappers.FeldWrapper(**param_dict, **kwargs)
    elif method_str == methods_name_dict['ZafarDI']:
        model = wrappers.ZafarDI(**param_dict, **kwargs)
    elif method_str == methods_name_dict['ZafarEO']:
        model = wrappers.ZafarEO(**param_dict, **kwargs)
    elif method_str == methods_name_dict['Hardt']:
        model = wrappers.Hardt(**param_dict, **kwargs)
    elif method_str in [methods_name_dict['fairlearn_full'], methods_name_dict['fairlearn'], methods_name_dict['expgrad']]:
        model = wrappers.ExponentiatedGradientPmf(**param_dict, **kwargs)
    else:
        try:
            model = wrappers.create_wrapper(**param_dict, **kwargs)
        except Exception as e:
            raise e
            raise ValueError(
                f'the method specified ({method_str}) is not allowed. Valid options are {methods_name_dict.values()}')
    return model
