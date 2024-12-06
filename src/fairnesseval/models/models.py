import inspect
import pickle
from functools import partial
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from fairnesseval.models import wrappers
from fairnesseval.models.wrappers import AdversarialDebiasingWrapper


additional_models_dict = {
    'most_frequent': partial(sklearn.dummy.DummyClassifier, strategy="most_frequent"),
    'LogisticRegression': sklearn.linear_model.LogisticRegression,
    'adversarial_debiasing': AdversarialDebiasingWrapper,
}


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


def finetune_model(base_model_code, X, y, random_seed=0, params_grid=None):
    base_model = get_base_model(base_model_code=base_model_code, random_seed=random_seed)
    if params_grid is None:
        params_grid = get_model_parameter_grid(base_model_code=base_model_code)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=random_seed)
    clf = GridSearchCV(base_model, params_grid, cv=cv, n_jobs=1, scoring=['f1', 'accuracy'], refit='f1')
    clf.fit(X, y)
    return clf


methods_name_dict = {
    'fairlearn_full': wrappers.ExponentiatedGradientPmf,
    'fairlearn': wrappers.ExponentiatedGradientPmf,
    'expgrad': wrappers.ExponentiatedGradientPmf,
    'Kearns': wrappers.Kearns,
    'Calmon': wrappers.CalmonWrapper,
    'Feld': wrappers.FeldWrapper,
    'ZafarDI': wrappers.ZafarDI,
    'ZafarEO': wrappers.ZafarEO,
    'Hardt': wrappers.Hardt,
    'ThresholdOptimizer': wrappers.ThresholdOptimizerWrapper,
}

class PersonalizedWrapper:
    def __init__(self, method_str, random_state=42, datasets=None, **kwargs):
        self.method_str = method_str
        model_class = methods_name_dict.get(method_str)
        if 'datasets' in inspect.signature(model_class.__init__).parameters:
            kwargs['datasets'] = datasets
        self.model = model_class(random_state=random_state, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = pickle.dumps(self.model)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = pickle.loads(state['model'])


def create_wrapper(method_str, random_state=42, datasets=None, **kwargs):
    model_class = additional_models_dict.get(method_str, None)
    if model_class is None:
        raise ValueError(
            f"Model {method_str} not found in available models."
            f" Available models are {list(additional_models_dict.keys()) + list(methods_name_dict.keys())}")

    params = inspect.signature(model_class.fit).parameters.keys()
    if 'sensitive_features' in params:
        def fit(self, X, y, sensitive_features):
            self.model.fit(X, y, sensitive_features)
            return self
    else:
        def fit(self, X, y, sensitive_features):
            self.model.fit(X, y)
            return self
    PersonalizedWrapper.fit = fit

    params = inspect.signature(model_class.predict).parameters.keys()
    if 'sensitive_features' in params:
        def predict(self, X, sensitive_features):
            return self.model.predict(X, sensitive_features=sensitive_features)
    else:
        def predict(self, X):
            return self.model.predict(X)
    PersonalizedWrapper.predict = predict

    return PersonalizedWrapper(method_str, random_state, datasets, **kwargs)


def get_model(method_str, random_state=42, **kwargs):
    kwargs = dict(method_str=method_str, random_state=random_state, ) | kwargs
    if method_str in methods_name_dict:
        model = methods_name_dict[method_str](**kwargs)
    else:
        try:
            model = create_wrapper(**kwargs)
        except Exception as e:
            raise e
    return model
