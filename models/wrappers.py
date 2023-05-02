import copy
from random import seed

import numpy as np
from aif360.algorithms.inprocessing import GerryFairClassifier
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions \
    import get_distortion_adult, get_distortion_german, get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from fairlearn.postprocessing import ThresholdOptimizer

import fair_classification.loss_funcs
import fair_classification.utils
import utils_experiment as ut_exp
import utils_prepare_data


class GeneralAifModel():
    def __init__(self, datasets):
        self.aif_dataset = copy.deepcopy(datasets[3])


def replace_values_aif360_dataset(X, y, sensitive_features, aif360_dataset):
    aif360_dataset = aif360_dataset.copy()
    y = y if y is not None else np.zeros_like(sensitive_features)
    aif360_dataset.features = X
    sensitive_features = sensitive_features.reshape(-1, 1)
    y = y.reshape(-1, 1)
    # if aif360_dataset.__class__.__name__ == 'GermanDataset':
    #     y[y == 0] = 2 #reconvert to 1,2 scale of GermanDataset
    aif360_dataset.labels = y
    aif360_dataset.protected_attributes = sensitive_features
    aif360_dataset.instance_names = np.arange(X.shape[0])
    return aif360_dataset


class CalmonWrapper(GeneralAifModel):
    def __init__(self, method_str, base_model, constrain_name, eps, random_state, datasets):
        super().__init__(datasets)
        X, y, A, aif_dataset = datasets
        self.op = OptimPreproc(OptTools, self.get_option(aif_dataset),
                               # unprivileged_groups=datasets[3]['unprivileged_groups'],
                               # privileged_groups=datasets[3]['privileged_groups'],
                               seed=random_state)
        self.base_model = base_model
        self.method_str = method_str

    def fit(self, X, y, sensitive_features):
        aif_dataset = replace_values_aif360_dataset(X, y, sensitive_features, self.aif_dataset)
        self.op = self.op.fit(aif_dataset)
        dataset_transf_train = self.op.transform(aif_dataset, transform_Y=True)
        dataset_transf_train = aif_dataset.align_datasets(dataset_transf_train)
        train = dataset_transf_train.features
        self.base_model.fit(train, dataset_transf_train.labels, )

    def predict(self, X, sensitive_features):
        aif_dataset = replace_values_aif360_dataset(X, None, sensitive_features, self.aif_dataset)
        # (self.aif360_dataset.convert_to_dataframe()[0].iloc[:,:-1].values == aif_dataset.convert_to_dataframe()[0].iloc[:,:-1].values).all()
        df_transformed = self.op.transform(aif_dataset, transform_Y=True)
        df_transformed = aif_dataset.align_datasets(df_transformed)
        X = df_transformed.features
        return self.base_model.predict(X)

    def get_option(self, aif_dataset):
        base_conf = {"epsilon": 0.05,
                     "clist": [0.99, 1.99, 2.99],
                     "dlist": [.1, 0.05, 0]}

        key = aif_dataset.__class__.__name__
        if key == ut_exp.sigmod_dataset_map['adult_sigmod']:
            base_conf.update(distortion_fun=get_distortion_adult)
        elif key == ut_exp.sigmod_dataset_map['compas']:
            base_conf.update(distortion_fun=get_distortion_compas)
        elif key == ut_exp.sigmod_dataset_map['german']:
            base_conf.update(distortion_fun=get_distortion_german)
        else:
            raise ValueError(f'{key} not found in available dataset configurations')
        return base_conf


class Hardt(GeneralAifModel):
    def __init__(self, method_str, base_model, constrain_name, eps, random_state, datasets):
        super().__init__(datasets)
        X, y, A, aif_dataset = datasets
        self.base_model = base_model
        self.postprocess_model = EqOddsPostprocessing(
            privileged_groups=[{aif_dataset.protected_attribute_names[0]: aif_dataset.privileged_protected_attributes}],
            unprivileged_groups=[
                {aif_dataset.protected_attribute_names[0]: aif_dataset.unprivileged_protected_attributes}],
            seed=random_state)

    def fit(self, X, y, sensitive_features):
        aif_dataset = replace_values_aif360_dataset(X, y, sensitive_features, self.aif_dataset)
        self.base_model.fit(X, y)
        y_pred = self.base_model.predict(X)
        aif_dataset_pred = aif_dataset.copy()
        aif_dataset_pred.labels = y_pred
        self.postprocess_model.fit(dataset_true=aif_dataset, dataset_pred=aif_dataset_pred)

    def predict(self, X, sensitive_features):
        y_pred = self.base_model.predict(X)
        aif_dataset = replace_values_aif360_dataset(X, y_pred, sensitive_features, self.aif_dataset)
        aif_corrected = self.postprocess_model.predict(aif_dataset)
        return aif_corrected.labels


class ZafarDI:
    def __init__(self, method_str, base_model, constrain_name, eps, random_state, datasets):
        seed(random_state)  # set the random seed so that the random permutations can be reproduced again
        np.random.seed(random_state)
        X, y, A, aif_dataset = datasets

        """ Classify such that we optimize for accuracy while achieving perfect fairness """

        self.fit_params = dict(loss_function=fair_classification.loss_funcs._logistic_loss,
                               apply_fairness_constraints=1,  # optimize accuracy subject to fairness constraints
                               apply_accuracy_constraint=0,
                               sep_constraint=0,
                               sensitive_attrs_to_cov_thresh={aif_dataset.protected_attribute_names[0]: 0},
                               sensitive_attrs=aif_dataset.protected_attribute_names,
                               gamma=None)
        key = aif_dataset.__class__.__name__
        if key == ut_exp.sigmod_dataset_map['adult_sigmod']:
            pass
        elif key == ut_exp.sigmod_dataset_map['compas']:
            pass
        elif key == ut_exp.sigmod_dataset_map['german']:
            pass
        else:
            raise ValueError(f'{key} not found in available dataset configurations')

    def fit(self, X, y, sensitive_features):
        self.w = fair_classification.utils.train_model(X, y,
                                                       {self.fit_params['sensitive_attrs'][0]: sensitive_features},
                                                       **self.fit_params)
        return self

    def predict(self, X):
        return 1 - np.sign(np.dot(X, self.w))


class Kearns(GeneralAifModel):
    # todo to complete. Not working
    def __init__(self, method_str, base_model, constrain_name, eps, random_state, datasets):
        super().__init__(datasets)
        X, y, A, aif_dataset = datasets
        self.base_model = base_model
        self.init_kearns(aif_dataset)
        self.method_str = method_str
        self.threshold = 0.5

    def init_kearns(self, aif_dataset):
        base_conf = dict(
            max_iters=100,
            C=100,
            printflag=True,
            gamma=.005,
            fairness_def='FP',
            heatmapflag=False
        )
        self.fit_params = dict(early_termination=True)
        self.predict_params = dict(threshold=0.5)
        key = aif_dataset.__class__.__name__
        if key == ut_exp.sigmod_dataset_map['adult_sigmod']:
            base_conf['fairness_def'] = 'FN'
            self.predict_params['threshold'] = 0.5
        elif key == ut_exp.sigmod_dataset_map['compas']:
            self.predict_params['threshold'] = 0.9898
        elif key == ut_exp.sigmod_dataset_map['german']:
            self.predict_params['threshold'] = 0.98
        else:
            raise ValueError(f'{key} not found in available dataset configurations')
        base_conf['predictor'] = self.base_model
        self.conf = base_conf
        self.kearns = GerryFairClassifier(**self.conf)

    def fit(self, X, y, sensitive_features):
        aif_dataset = replace_values_aif360_dataset(X, y, sensitive_features, self.aif_dataset)
        self.kearns.fit(aif_dataset, **self.fit_params)
        # self.base_model.fit(X,y)
        return self

    def predict(self, dataset, threshold=None):
        # return super().predict(dataset, **self.predict_params).labels
        pass


class ThresholdOptimizerWrapper(ThresholdOptimizer):
    def __init__(self, *args, random_state=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = random_state

    def fit(self, X, y, sensitive_features):
        return super().fit(X, y, sensitive_features=sensitive_features)

    def predict(self, X, sensitive_features):
        return super().predict(X, sensitive_features=sensitive_features, random_state=self.random_state)