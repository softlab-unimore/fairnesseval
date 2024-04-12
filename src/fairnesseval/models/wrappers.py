import copy
import inspect
from random import seed

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

from aif360.algorithms.inprocessing import GerryFairClassifier
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions \
    import get_distortion_adult, get_distortion_german, get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.datasets import StandardDataset

from fairlearn.postprocessing import ThresholdOptimizer

import fair_classification.utils
import fair_classification.funcs_disp_mist
import fair_classification.loss_funcs
from fairnesseval import utils_prepare_data
from fairnesseval import utils_experiment_parameters as ut_exp
from fairlearn.reductions import ExponentiatedGradient
from functools import partial


class GeneralAifModel():
    def __init__(self, datasets):
        # def __init__(self, method_str, base_model, constraint_code, eps, random_state, datasets, **kwargs):
        # __init__(self, method_str, base_model, constraint_code, eps, random_state, datasets):
        # __init__(self, method_str, base_model, constraint_code, eps, random_state, datasets):

        if len(datasets) >= 4:
            self.aif_dataset = copy.deepcopy(datasets[3])
        else:
            self.aif_dataset = GeneralAifModel.get_aif_dataset(datasets=datasets)

    @staticmethod
    def get_aif_dataset(datasets):
        X, Y, A = datasets[:3]
        priviliged_class = [Y.groupby(A).mean().sort_values(ascending=False).index.tolist()[0]]
        if Y.name is None:
            Y.name = 'label'
        protected_name = A.name
        return StandardDataset(df=pd.concat([X, Y, A], axis=1),
                               label_name=Y.name,
                               favorable_classes=[1],
                               protected_attribute_names=[protected_name],
                               privileged_classes=[priviliged_class],
                               instance_weights_name=None,
                               categorical_features=[],
                               features_to_keep=X.columns.tolist() + [Y.name, protected_name],
                               na_values=[np.nan],
                               metadata=dict(label_maps=[{1: 1, 0: 0}],
                                             protected_attribute_maps=[{x: x for x in A.unique()}]
                                             ),
                               custom_preprocessing=None)

    def fit(self, X, y, sensitive_features):
        pass

    # predict(self, X, sensitive_features):

    def predict(self, X):
        pass


def replace_values_aif360_dataset(X, y, sensitive_features, aif360_dataset):
    aif360_dataset = aif360_dataset.copy()
    y = y if y is not None else np.zeros_like(sensitive_features)
    aif360_dataset.features = pd.concat([X, sensitive_features], axis=1)
    sensitive_features = np.array(sensitive_features).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    # if aif360_dataset.__class__.__name__ == 'GermanDataset':
    #     y[y == 0] = 2 #reconvert to 1,2 scale of GermanDataset
    aif360_dataset.labels = y
    aif360_dataset.protected_attributes = sensitive_features
    aif360_dataset.instance_names = np.arange(X.shape[0])
    return aif360_dataset


class CalmonWrapper(GeneralAifModel):
    def __init__(self, method_str, base_model, datasets, eps=None, constraint_code=None, random_state=None):
        super().__init__(datasets)
        X, y, A = datasets[:3]
        self.op = OptimPreproc(OptTools, self.get_option(self.aif_dataset),
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


class FeldWrapper(GeneralAifModel):
    def __init__(self, method_str, base_model, datasets, eps=None, constraint_code=None, random_state=None):
        super().__init__(datasets)
        X, y, A = datasets[:3]
        self.preprocess_model = DisparateImpactRemover(sensitive_attribute=A.name)
        self.base_model = base_model
        self.method_str = method_str

    def fit(self, X, y, sensitive_features):
        aif_dataset = replace_values_aif360_dataset(X, y, sensitive_features, self.aif_dataset)
        self.sensitive_attribute = aif_dataset.protected_attribute_names[0]
        features = aif_dataset.features.to_numpy().tolist()
        index = aif_dataset.feature_names.index(self.sensitive_attribute)
        self.repairer = self.preprocess_model.Repairer(features, index, self.preprocess_model.repair_level, False)

        repaired_ds = self.transform(aif_dataset)
        train = repaired_ds.features
        self.base_model.fit(train, repaired_ds.labels)

    def transform(self, aif_dataset):
        # Code took from original aif360 code and modified to save fitted model
        features = aif_dataset.features.to_numpy().tolist()
        index = aif_dataset.feature_names.index(self.sensitive_attribute)
        repaired_ds = aif_dataset.copy()
        repaired_features = self.repairer.repair(features)
        repaired_ds.features = np.array(repaired_features, dtype=np.float64)
        # protected attribute shouldn't change
        repaired_ds.features[:, index] = repaired_ds.protected_attributes[:,
                                         repaired_ds.protected_attribute_names.index(self.sensitive_attribute)]
        return repaired_ds

    def predict(self, X, sensitive_features):
        aif_dataset = replace_values_aif360_dataset(X, None, sensitive_features, self.aif_dataset)
        # (self.aif360_dataset.convert_to_dataframe()[0].iloc[:,:-1].values == aif_dataset.convert_to_dataframe()[0].iloc[:,:-1].values).all()
        df_transformed = self.transform(aif_dataset)
        X = df_transformed.features
        return self.base_model.predict(X)


class Hardt(GeneralAifModel):
    def __init__(self, random_state, method_str=None, base_model=None, datasets=None, eps=None, constraint_code=None, ):
        super().__init__(datasets)
        self.method_str = method_str
        X, y, A = datasets
        self.base_model = base_model
        self.postprocess_model = EqOddsPostprocessing(
            privileged_groups=[
                {self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.privileged_protected_attributes}],
            unprivileged_groups=[
                {self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.unprivileged_protected_attributes}],
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
    def __init__(self, method_str, datasets, base_model=None, eps=None, constraint_code=None, random_state=None):
        seed(random_state)  # set the random seed so that the random permutations can be reproduced again
        np.random.seed(random_state)
        X, y, A = datasets[:3]
        self.method_str = method_str

        """ Classify such that we optimize for fairness subject to a certain loss in accuracy """
        params = dict(
            apply_fairness_constraints=0,
            # flag for fairness constraint is set back to 0 since we want to apply the accuracy constraint now
            apply_accuracy_constraint=1,  # now, we want to optimize fairness subject to accuracy constraints
            # sep_constraint=1,
            # # set the separate constraint flag to one, since in addition to accuracy constrains, we also want no misclassifications for certain points (details in demo README.md)
            # gamma=1000.0,
            sep_constraint=0,
            gamma=0.001,
            # gamma controls how much loss in accuracy we are willing to incur to achieve fairness -- increase gamme to allow more loss in accuracy
        )

        def log_loss_sklearn(w, X, y, return_arr=None):
            return sklearn.metrics.log_loss(y_true=y, y_pred=np.sign(np.dot(X, w)), normalize=return_arr)

        self.fit_params = dict(loss_function=fair_classification.loss_funcs._logistic_loss,  # log_loss_sklearn,
                               sensitive_attrs_to_cov_thresh={A.name: 0},
                               sensitive_attrs=[A.name],
                               **params
                               )

    def fit(self, X, y, sensitive_features):
        self.w = fair_classification.utils.train_model(X.values, y * 2 - 1,
                                                       {self.fit_params['sensitive_attrs'][0]: sensitive_features},
                                                       **self.fit_params)
        return self

    def predict(self, X):
        y_pred = np.dot(X.values, self.w)
        y_pred = np.where(y_pred > 0, 1, 0)
        return y_pred


class ZafarEO:
    def __init__(self, method_str, base_model, datasets, eps=None, constraint_code=None, random_state=None):
        seed(random_state)  # set the random seed so that the random permutations can be reproduced again
        np.random.seed(random_state)
        X, y, A = datasets[:3]  # todo fix name of A to race
        self.method_str = method_str
        """ Now classify such that we optimize for accuracy while achieving perfect fairness """
        # sensitive_attrs_to_cov_thresh = {A.name: {group: {0: 0, 1: 0} for group in A.unique()}}  # zero covariance threshold, means try to get the fairest solution
        sensitive_attrs_to_cov_thresh = {A.name: {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}, 2: {0: 0,
                                                                                        1: 0}}}  # zero covariance threshold, means try to get the fairest solution

        cons_params = dict(
            cons_type=1,  # see cons_type in fair_classification.funcs_disp_mist.get_constraint_list_cov line 198
            tau=5.0,
            mu=1.2,
            sensitive_attrs_to_cov_thresh=sensitive_attrs_to_cov_thresh,

        )
        self.sensitive_attrs = [A.name]

        self.fit_params = dict(loss_function="logreg",  # log_loss_sklearn,

                               EPS=1e-6,
                               cons_params=cons_params,
                               )

    def fit(self, X, y, sensitive_features):
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.w = fair_classification.funcs_disp_mist.train_model_disp_mist(X, y * 2 - 1,
                                                                           {self.sensitive_attrs[
                                                                                0]: sensitive_features},
                                                                           **self.fit_params)
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        y_pred = np.dot(X, self.w)
        y_pred = np.where(y_pred > 0, 1, 0)
        return y_pred


class Kearns(GeneralAifModel):
    # todo to complete. Not working
    def __init__(self, method_str, base_model, datasets, eps=None, constraint_code=None, random_state=None):
        super().__init__(datasets)
        X, y, A = datasets
        self.base_model = base_model
        self.init_kearns()
        self.method_str = method_str
        self.threshold = 0.5

    def init_kearns(self, ):
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
        key = self.aif_dataset.__class__.__name__
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
    def __init__(self, *args, random_state=0, datasets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = random_state

    def fit(self, X, y, sensitive_features):
        return super().fit(X, y, sensitive_features=sensitive_features)

    def predict(self, X, sensitive_features):
        return super().predict(X, sensitive_features=sensitive_features, random_state=self.random_state)


class ExponentiatedGradientPmf(ExponentiatedGradient):
    def __init__(self, base_model, constraint_code, eps, random_state, run_linprog_step=None, eta0=None,
                 method_str='fairlearn_full', datasets=None, **kwargs):
        self.method_str = method_str
        if eta0 is not None:
            kwargs['eta0'] = eta0
        if run_linprog_step is not None:
            kwargs['run_linprog_step'] = run_linprog_step
        if 'constraint_code' in kwargs:
            kwargs.pop('constraint_code')
        constraint = utils_prepare_data.get_constraint(constraint_code=constraint_code, eps=eps)
        super(ExponentiatedGradientPmf, self).__init__(base_model, constraints=copy.deepcopy(constraint), eps=eps,
                                                       nu=1e-6, random_state=random_state, **kwargs)

    def fit(self, X, y, sensitive_features, **kwargs):
        return super().fit(X, y, sensitive_features=sensitive_features, **kwargs)

    def predict(self, X, random_state=None):
        return self._pmf_predict(X)[:, 1]

    def get_stats_dict(self):
        res_dict = {}
        for key in ['best_iter_', 'best_gap_',
                    # 'weights_', '_hs',  'predictors_', 'lambda_vecs_',
                    'last_iter_', 'n_oracle_calls_',
                    'n_oracle_calls_dummy_returned_', 'oracle_execution_times_', ]:
            res_dict[key] = getattr(self, key)
        return res_dict


additional_models_dict = {
    'most_frequent': partial(sklearn.dummy.DummyClassifier, strategy="most_frequent"),
    'LogisticRegression': sklearn.linear_model.LogisticRegression,
}


def create_wrapper(method_str, random_state=42, datasets=None,  **kwargs):
    model_class = additional_models_dict.get(method_str)

    class PersonalizedWrapper:
        def __init__(self, method_str, random_state=42, **kwargs):
            self.method_str = method_str
            self.model = model_class(random_state=random_state, **kwargs)

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

    return PersonalizedWrapper(method_str, random_state, **kwargs)
