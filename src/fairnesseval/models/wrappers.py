import copy
import inspect
from random import seed
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

from aif360.algorithms.inprocessing import GerryFairClassifier, AdversarialDebiasing
from aif360.algorithms.postprocessing import EqOddsPostprocessing, CalibratedEqOddsPostprocessing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions \
    import get_distortion_adult, get_distortion_german, get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.datasets import StandardDataset, BinaryLabelDataset

from fairlearn.postprocessing import ThresholdOptimizer

import fair_classification.utils
import fair_classification.funcs_disp_mist
import fair_classification.loss_funcs
from fairnesseval import utils_prepare_data
from fairnesseval import utils_experiment_parameters as ut_exp
from fairlearn.reductions import ExponentiatedGradient
from functools import partial
import tensorflow as tf

from fairnesseval.utils_general import LoggerSingleton


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

    def predict(self, X):
        pass


def replace_values_aif360_dataset(X, y, sensitive_features, aif360_dataset):
    aif360_dataset = aif360_dataset.copy()
    y = y if y is not None else np.zeros_like(sensitive_features)
    aif360_dataset.features = pd.concat([X, sensitive_features], axis=1)
    if isinstance(sensitive_features, pd.Series):
        sf_name = sensitive_features.name
    else:
        sf_name = sensitive_features.columns
    aif360_dataset.features_names = X.columns + sf_name
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
        u = np.unique(A)
        self.pam = [dict(zip(u, u))]
        u = np.unique(y)
        self.lm = [dict(zip(u, u))]

        self.label_name = y.name

        self.op = OptimPreproc(OptTools, optim_options=self.get_option(self.aif_dataset),
                               privileged_groups=[{self.aif_dataset.protected_attribute_names[0]:
                                                       self.aif_dataset.privileged_protected_attributes[0]}],
                               unprivileged_groups=[{self.aif_dataset.protected_attribute_names[0]:
                                                         self.aif_dataset.unprivileged_protected_attributes[0]}],
                               seed=random_state)
        self.base_model = base_model
        self.method_str = method_str

    def fit(self, X, y, sensitive_features):
        # X, self.selected_columns = utils_prepare_data.convert_floats_to_categorical(X, y, self.base_model)
        aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name],
                                         protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        aif_dataset.metadata['protected_attribute_maps'] = self.pam
        aif_dataset.metadata['label_maps'] = self.lm

        self.op = self.op.fit(aif_dataset)
        dataset_transf_train = self.op.transform(aif_dataset, transform_Y=True)
        dataset_transf_train = aif_dataset.align_datasets(dataset_transf_train)
        train = dataset_transf_train.features[:, :-1]
        self.base_model.fit(train, dataset_transf_train.labels.ravel())

    def predict(self, X, sensitive_features):
        # X, _ = utils_prepare_data.convert_floats_to_categorical(X, None, self.base_model,
        #                                                         selected_columns=self.selected_columns)
        df = pd.concat([X, sensitive_features], axis=1)
        df[self.label_name] = 0

        aif_dataset = BinaryLabelDataset(df=df, label_names=[self.label_name],
                                         protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        aif_dataset.metadata['protected_attribute_maps'] = self.pam
        aif_dataset.metadata['label_maps'] = self.lm

        df_transformed = self.op.transform(aif_dataset, transform_Y=True)
        df_transformed = aif_dataset.align_datasets(df_transformed)
        X = df_transformed.features[:, :-1]
        return self.base_model.predict(X)

    def get_option(self, aif_dataset):

        def get_distortion(vold, vnew):
            total_cost = 0.0
            for k in vold:
                if k in vnew:
                    try:
                        total_cost += abs(vnew[k] - vold[k])
                    except Exception:
                        if vnew[k] != vold[k]:
                            total_cost += 1

            return total_cost

        base_conf = {"epsilon": 0.05,
                     "clist": [0.99, 1.99, 2.99],
                     "dlist": [.1, 0.05, 0],
                     "distortion_fun": get_distortion}
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


# class Hardt(GeneralAifModel):
#     def __init__(self, random_state, method_str=None, base_model=None, datasets=None, eps=None, constraint_code=None, ):
#         super().__init__(datasets)
#         self.method_str = method_str
#         X, y, A = datasets[:3]
#         self.base_model = base_model
#         self.postprocess_model = EqOddsPostprocessing(
#             privileged_groups=[
#                 {self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.privileged_protected_attributes}],
#             unprivileged_groups=[
#                 {self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.unprivileged_protected_attributes}],
#             seed=random_state)
#
#     def fit(self, X, y, sensitive_features):
#         aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name],
#                                          protected_attribute_names=[sensitive_features.name])
#         self.base_model.fit(X, y)
#         y_pred = self.base_model.predict(X)
#         aif_dataset_pred = aif_dataset.copy()
#         aif_dataset_pred.labels = y_pred
#         self.postprocess_model.fit(dataset_true=aif_dataset, dataset_pred=aif_dataset_pred)
#
#     def predict(self, X, sensitive_features):
#         y_pred = self.base_model.predict(X)
#         aif_dataset = replace_values_aif360_dataset(X, y_pred, sensitive_features, self.aif_dataset)
#         aif_corrected = self.postprocess_model.predict(aif_dataset)
#         return aif_corrected.labels


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
        X, y, A = datasets[:3]
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


class PleissWrapper(GeneralAifModel):
    def __init__(self, method_str, base_model, datasets, random_state=None):
        super().__init__(datasets)
        self.postprocessor = CalibratedEqOddsPostprocessing(
            privileged_groups=[
                {self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.privileged_protected_attributes[0]}],
            unprivileged_groups=[
                {self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.unprivileged_protected_attributes[0]}],
            seed=random_state)
        self.base_model = base_model
        self.method_str = method_str

    def fit(self, X, y, sensitive_features):
        # aif_dataset = replace_values_aif360_dataset(X, y, sensitive_features, self.aif_dataset)
        aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name],
                                         protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)

        self.base_model.fit(X, y)
        y_pred = self.base_model.predict(X).reshape(-1, 1)
        aif_dataset_pred = aif_dataset.copy()
        aif_dataset_pred.labels = y_pred
        self.postprocessor.fit(dataset_true=aif_dataset, dataset_pred=aif_dataset_pred)

    def predict(self, X, sensitive_features):
        y_pred = self.base_model.predict(X)

        df = pd.concat([X, sensitive_features], axis=1)
        df['label'] = y_pred

        aif_dataset = BinaryLabelDataset(df=df, label_names=['label'],
                                         protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)

        aif_corrected = self.postprocessor.predict(aif_dataset)
        return aif_corrected.labels


class Kearns(GeneralAifModel):
    def __init__(self, method_str, base_model, datasets, random_state=None):
        super().__init__(datasets)
        X, y, A = datasets[:3]
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

        self.conf = base_conf
        self.kearns = GerryFairClassifier(**self.conf)

    def fit(self, X, y, sensitive_features):
        aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name],
                                         protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        self.kearns.fit(aif_dataset, **self.fit_params)
        return self

    def predict(self, X, sensitive_features):
        df = pd.concat([X, sensitive_features], axis=1)
        df['label'] = 0
        aif_dataset = BinaryLabelDataset(df=df, label_names=['label'],
                                         protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)

        df_transformed = self.kearns.predict(aif_dataset)
        return df_transformed.labels


class ThresholdOptimizerWrapper(ThresholdOptimizer):
    def __init__(self, method_str, random_state=0, datasets=None, **kwargs):
        self.method_str = method_str
        estimator = kwargs.pop('base_model', None)
        kwargs = dict(objective="accuracy_score",
                      prefit=False,
                      predict_method='predict_proba',
                      estimator=estimator) | kwargs
        if kwargs.get('eps', None):
            logger = LoggerSingleton()
            logger.warning(f"eps has no effect with {method_str} methos")
        constraint_code_to_name = {'dp': 'demographic_parity',
                                   'eo': 'equalized_odds'}
        if 'constraint_code' in kwargs:
            cc = kwargs.pop('constraint_code')
            kwargs['constraints'] = constraint_code_to_name.get(cc, cc)
        super().__init__(**kwargs)
        self.random_state = random_state

    def fit(self, X, y, sensitive_features):
        return super().fit(X, y, sensitive_features=sensitive_features)

    def predict(self, X, sensitive_features):
        return super().predict(X, sensitive_features=sensitive_features, random_state=self.random_state)


class ExponentiatedGradientPmf(ExponentiatedGradient):
    def __init__(self, base_model, eps, random_state, run_linprog_step=None, eta0=None,
                 method_str='fairlearn_full', datasets=None, constraint_code=None, constraint=None, **kwargs):
        self.method_str = method_str
        self.random_state = random_state
        if eta0 is not None:
            kwargs['eta0'] = eta0
        if run_linprog_step is not None:
            kwargs['run_linprog_step'] = run_linprog_step
        if 'constraint_code' in kwargs:
            kwargs.pop('constraint_code')
        if constraint is None:
            if constraint_code is None:
                assert False, 'constraint_code or constraint should be provided'
            constraint = utils_prepare_data.get_constraint(constraint_code=constraint_code, eps=eps)
        if 'nu' not in kwargs:
            kwargs['nu'] = 1e-6
        self.original_kwargs = kwargs.copy()

        super(ExponentiatedGradientPmf, self).__init__(base_model, constraints=copy.deepcopy(constraint), eps=eps,
                                                       **kwargs)

    def fit(self, X, y, sensitive_features, **kwargs):
        if self.subsample is not None and self.subsample < 1:
            self.subsample = int(X.shape[0] * self.subsample)
        elif self.subsample == 1:
            self.subsample = None
        if hasattr(X, 'values'):
            X = X.values
        return super(ExponentiatedGradientPmf, self).fit(X, y, sensitive_features=sensitive_features, **kwargs)

    def predict_proba(self, X):
        X = X.values if hasattr(X, 'values') else X
        return self._pmf_predict(X)[:, 1]

    def predict(self, X, threshold=0.5):
        X = X.values if hasattr(X, 'values') else X
        return (self._pmf_predict(X)[:, 1] > threshold).astype(int)

    def get_stats_dict(self):
        res_dict = {}
        for key in ['best_iter_', 'best_gap_',
                    # 'weights_', '_hs',  'predictors_', 'lambda_vecs_',
                    'last_iter_', 'n_oracle_calls_',
                    'n_oracle_calls_dummy_returned_', 'oracle_execution_times_', ]:
            res_dict[key] = getattr(self, key)
        return res_dict


# class AdversarialDebiasingWrapper():
#     def __init__(self, random_state, **kwargs):  # Random state is required here
#         # Initialize your model with any required parameters
#         self.model = AdversarialDebiasing(random_state, **kwargs)  # Random state is NOT required here, but it is recommended to set it.
#
#     def fit(self, X, y):  # Or fit(self, X, y, sensitive_features) if your model require sensitive features
#         # Fit the model to the data
#         self.model.fit(X, y)
#
#     def predict(self, X):  # Or predict(self, X, sensitive_features): if your model requires sensitive features
#         # Predict using the model
#         self.model.predict_proba(X)


class AdversarialDebiasingWrapper(GeneralAifModel):
    def __init__(self, datasets, random_state=None):
        super().__init__(datasets)
        tf.compat.v1.reset_default_graph()
        self.sess = tf.compat.v1.Session()
        self.model = AdversarialDebiasing(
            privileged_groups=[
                {self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.privileged_protected_attributes[0]}],
            unprivileged_groups=[
                {self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.unprivileged_protected_attributes[0]}],
            scope_name=f'adversarial_debiasing_{random_state}',
            sess=self.sess
        )

    def fit(self, X, y, sensitive_features):
        # aif_dataset = replace_values_aif360_dataset(X, y, sensitive_features, self.aif_dataset)
        aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name],
                                         protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        self.model = self.model.fit(aif_dataset)

    def predict(self, X, sensitive_features):
        df = pd.concat([X, sensitive_features], axis=1)
        df['label'] = 0
        aif_dataset = BinaryLabelDataset(df=df, label_names=['label'],
                                         protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)

        df_transformed = self.model.predict(aif_dataset)
        return df_transformed.labels

    def __del__(self):
        """Ensure the TensorFlow session is closed when the object is deleted."""
        if self.sess is not None:
            self.sess.close()
            print("TensorFlow session closed.")

def only_unmitigated(method_str, base_model, datasets, random_state=None):
    base_model.set_params(random_state=random_state)
    return base_model