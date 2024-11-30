import copy
from datetime import datetime
import inspect
from random import seed

import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from scipy.optimize import minimize

from aif360.algorithms.inprocessing import GerryFairClassifier, PrejudiceRemover, AdversarialDebiasing
from aif360.algorithms.postprocessing import EqOddsPostprocessing, CalibratedEqOddsPostprocessing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions \
    import get_distortion_adult, get_distortion_german, get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing import LFR, Reweighing
from aif360.datasets import StandardDataset, BinaryLabelDataset

from fairlearn.postprocessing import ThresholdOptimizer

import fair_classification.utils
import fair_classification.funcs_disp_mist
import fair_classification.loss_funcs
from fairnesseval import utils_prepare_data
from fairnesseval import utils_experiment_parameters as ut_exp
from fairnesseval.metrics import getEO, TPRB
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.metrics import equalized_odds_difference
from functools import partial

import cvxpy as cp
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import traceback

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
        privileged_protected_attributes = [Y.groupby(A).mean().sort_values(ascending=False).index.tolist()[0:1]]
        unprivileged_protected_attributes = [Y.groupby(A).mean().sort_values(ascending=False).index.tolist()[1:]]
        if Y.name is None:
            Y.name = 'label'
        if A.name is None:
            A.name = 'sensitive_attribute'
        return BinaryLabelDataset(df=pd.concat([X, Y, A], axis=1),
                               label_names=[Y.name],   
                               protected_attribute_names=[A.name],
                               privileged_protected_attributes=privileged_protected_attributes,
                               unprivileged_protected_attributes=unprivileged_protected_attributes,
                               metadata=dict(label_maps=[{1: 1, 0: 0}],
                                             protected_attribute_maps=[{x: x for x in A.unique()}]
                                             )
                               )

    def fit(self, X, y, sensitive_features):
        pass

    # predict(self, X, sensitive_features):

    def predict(self, X):
        pass


def replace_values_aif360_dataset(X, y, sensitive_features, aif360_dataset):
    aif360_dataset = aif360_dataset.copy()
    y = y if y is not None else np.zeros_like(sensitive_features)
    aif360_dataset.features = np.array(pd.concat([X, sensitive_features], axis=1))
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
        X,y,A = datasets[:3]
        u = np.unique(A)
        self.pam = [dict(zip(u, u))]
        u = np.unique(y)
        self.lm = [dict(zip(u, u))]
        
        self.label_name=y.name
        
        self.op = OptimPreproc(OptTools, optim_options=self.get_option(self.aif_dataset),
                                privileged_groups=[{self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.privileged_protected_attributes[0]}],
                                unprivileged_groups=[{self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.unprivileged_protected_attributes[0]}],
                                seed=random_state)
        self.base_model = base_model
        self.method_str = method_str

    def fit(self, X, y, sensitive_features):
        X, self.selected_columns = utils_prepare_data.convert_floats_to_categorical(X, y, self.base_model)
        aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name], protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        aif_dataset.metadata['protected_attribute_maps'] = self.pam
        aif_dataset.metadata['label_maps'] = self.lm
        
        self.op = self.op.fit(aif_dataset)
        dataset_transf_train = self.op.transform(aif_dataset, transform_Y=True)
        dataset_transf_train = aif_dataset.align_datasets(dataset_transf_train)
        train = dataset_transf_train.features[:,:-1]
        self.base_model.fit(train, dataset_transf_train.labels.ravel())

    def predict(self, X, sensitive_features):
        X, _ = utils_prepare_data.convert_floats_to_categorical(X, None, self.base_model, selected_columns=self.selected_columns)
        df=pd.concat([X, sensitive_features], axis=1)
        df[self.label_name] = 0
        
        aif_dataset = BinaryLabelDataset(df=df, label_names=[self.label_name], protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        aif_dataset.metadata['protected_attribute_maps'] = self.pam
        aif_dataset.metadata['label_maps'] = self.lm
        
        df_transformed = self.op.transform(aif_dataset, transform_Y=True)
        df_transformed = aif_dataset.align_datasets(df_transformed)
        X = df_transformed.features[:,:-1]
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
        features = np.array(aif_dataset.features).tolist()
        index = aif_dataset.feature_names.index(self.sensitive_attribute)
        self.repairer = self.preprocess_model.Repairer(features, index, self.preprocess_model.repair_level, False)

        repaired_ds = self.transform(aif_dataset)
        train = repaired_ds.features[:,:-1]
        self.base_model.fit(train, repaired_ds.labels.ravel())

    def transform(self, aif_dataset):
        # Code took from original aif360 code and modified to save fitted model
        features = np.array(aif_dataset.features).tolist()
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
        X = df_transformed.features[:,:-1]
        return self.base_model.predict(X)


class LFRWrapper(GeneralAifModel):
    def __init__(self, method_str, base_model, datasets, eps=None, constraint_code=None, random_state=None):
        super().__init__(datasets)
        self.lfr = LFR(
            privileged_groups=[{self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.privileged_protected_attributes[0]}],
            unprivileged_groups=[{self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.unprivileged_protected_attributes[0]}],
            k=10, Ax=0.1, Ay=1.0, Az=2.0,
            seed=random_state)
        self.base_model = base_model
        self.method_str = method_str

    def fit(self, X, y, sensitive_features):
        #aif_dataset = replace_values_aif360_dataset(X, y, sensitive_features, self.aif_dataset)
        aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name], protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        
        self.lfr = self.lfr.fit(aif_dataset)
        dataset_transf_train = self.lfr.transform(aif_dataset)
        
        train = dataset_transf_train.features[:,:-1]
        if len(np.unique(dataset_transf_train.labels)) == 1:
            dataset_transf_train.labels[0] = 1 - dataset_transf_train.labels[0]
        self.base_model.fit(train, dataset_transf_train.labels)

    def predict(self, X, sensitive_features):
        aif_dataset = replace_values_aif360_dataset(X, None, sensitive_features, self.aif_dataset)
        # (self.aif360_dataset.convert_to_dataframe()[0].iloc[:,:-1].values == aif_dataset.convert_to_dataframe()[0].iloc[:,:-1].values).all()
        df_transformed = self.lfr.transform(aif_dataset)
        
        X = df_transformed.features[:,:-1]
        return self.base_model.predict(X)
    
    
class ReweighingWrapper(GeneralAifModel):
    def __init__(self, method_str, base_model, datasets, eps=None, constraint_code=None, random_state=None):
        super().__init__(datasets)
        self.preprocessor = Reweighing(
            privileged_groups=[{self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.privileged_protected_attributes[0]}],
            unprivileged_groups=[{self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.unprivileged_protected_attributes[0]}])
        self.base_model = base_model
        self.method_str = method_str

    def fit(self, X, y, sensitive_features):
        #aif_dataset = replace_values_aif360_dataset(X, y, sensitive_features, self.aif_dataset)
        aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name], protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        
        self.preprocessor = self.preprocessor.fit(aif_dataset)
        dataset_transf_train = self.preprocessor.transform(aif_dataset)
        
        train = dataset_transf_train.features[:,:-1]
        if len(np.unique(dataset_transf_train.labels)) == 1:
            dataset_transf_train.labels[0] = 1 - dataset_transf_train.labels[0]
        self.base_model.fit(train, dataset_transf_train.labels, sample_weight=dataset_transf_train.instance_weights)

    def predict(self, X, sensitive_features):
        #aif_dataset = replace_values_aif360_dataset(X, None, sensitive_features, self.aif_dataset)
        df=pd.concat([X, sensitive_features], axis=1)
        df['label'] = 0
        aif_dataset = BinaryLabelDataset(df=df, label_names=['label'], protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        
        df_transformed = self.preprocessor.transform(aif_dataset)
        
        X = df_transformed.features[:,:-1]
        return self.base_model.predict(X)


class Hardt(GeneralAifModel):
    def __init__(self, random_state, method_str=None, base_model=None, datasets=None, eps=None, constraint_code=None, ):
        super().__init__(datasets)
        self.method_str = method_str
        X, y, A = datasets[:3]
        self.base_model = base_model
        self.postprocess_model = EqOddsPostprocessing(
            privileged_groups=[
                {self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.privileged_protected_attributes[0]}],
            unprivileged_groups=[
                {self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.unprivileged_protected_attributes[0]}],
            seed=random_state)

    def fit(self, X, y, sensitive_features):
        aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name], protected_attribute_names=[sensitive_features.name])
        
        if not isinstance(aif_dataset.features, np.ndarray):
            aif_dataset.features = aif_dataset.features.to_numpy()
        
        self.base_model.fit(X, y)
        y_pred = self.base_model.predict(X).reshape(-1, 1)
        aif_dataset_pred = aif_dataset.copy()
        aif_dataset_pred.labels = y_pred
        self.postprocess_model.fit(dataset_true=aif_dataset, dataset_pred=aif_dataset_pred)

    def predict(self, X, sensitive_features):
        X = X.reset_index(drop=True)
        sensitive_features = sensitive_features.reset_index(drop=True)
        y_pred = self.base_model.predict(X)
        y_pred_df = pd.DataFrame(y_pred, columns=['y_pred'])
        aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y_pred_df], axis=1), label_names=['y_pred'], protected_attribute_names=[sensitive_features.name])
        
        if not isinstance(aif_dataset.features, np.ndarray):
            aif_dataset.features = aif_dataset.features.to_numpy()            
            
        aif_corrected = self.postprocess_model.predict(aif_dataset)
        return aif_corrected.labels
    

class PleissWrapper(GeneralAifModel):
    def __init__(self, method_str, base_model, datasets, eps=None, constraint_code=None, random_state=None):
        super().__init__(datasets)
        self.postprocessor = CalibratedEqOddsPostprocessing(
            privileged_groups=[{self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.privileged_protected_attributes[0]}],
            unprivileged_groups=[{self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.unprivileged_protected_attributes[0]}],
            seed=random_state)
        self.base_model = base_model
        self.method_str = method_str

    def fit(self, X, y, sensitive_features):
        #aif_dataset = replace_values_aif360_dataset(X, y, sensitive_features, self.aif_dataset)
        aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name], protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        
        self.base_model.fit(X, y)
        y_pred = self.base_model.predict(X).reshape(-1, 1)
        aif_dataset_pred = aif_dataset.copy()
        aif_dataset_pred.labels = y_pred
        self.postprocessor.fit(dataset_true=aif_dataset, dataset_pred=aif_dataset_pred)

    def predict(self, X, sensitive_features):
        y_pred = self.base_model.predict(X)
        
        df=pd.concat([X, sensitive_features], axis=1)
        df['label'] = y_pred
        
        aif_dataset = BinaryLabelDataset(df=df, label_names=['label'], protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        
        aif_corrected = self.postprocessor.predict(aif_dataset)
        return aif_corrected.labels
    
    
class PrejudiceRemoverWrapper(GeneralAifModel):
    def __init__(self, method_str, base_model, datasets, eps=None, constraint_code=None, random_state=None):
        super().__init__(datasets)
        self.label_name = datasets[1].name
        self.model = PrejudiceRemover(sensitive_attr=datasets[2].name, class_attr=self.label_name, eta=1.0)
        self.method_str = method_str

    def fit(self, X, y, sensitive_features):
        #aif_dataset = replace_values_aif360_dataset(X, y, sensitive_features, self.aif_dataset)
        aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name], protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)   
        self.model = self.model.fit(aif_dataset)

    def predict(self, X, sensitive_features):
        df=pd.concat([X, sensitive_features], axis=1)
        df[self.label_name] = 0
        aif_dataset = BinaryLabelDataset(df=df, label_names=[self.label_name], protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        
        df_transformed = self.model.predict(aif_dataset)
        return df_transformed.labels
    
        
class AdversarialDebiasingWrapper(GeneralAifModel):
    def __init__(self, method_str, base_model, datasets, eps=None, constraint_code=None, random_state=None):
        super().__init__(datasets)
        tf.compat.v1.reset_default_graph()
        self.sess = tf.compat.v1.Session()
        self.model = AdversarialDebiasing(
            privileged_groups=[{self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.privileged_protected_attributes[0]}],
            unprivileged_groups=[{self.aif_dataset.protected_attribute_names[0]: self.aif_dataset.unprivileged_protected_attributes[0]}],
            scope_name=f'adversarial_debiasing_{random_state}',
            sess=self.sess
        )
        self.method_str = method_str

    def fit(self, X, y, sensitive_features):
        #aif_dataset = replace_values_aif360_dataset(X, y, sensitive_features, self.aif_dataset)
        aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name], protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)   
        self.model = self.model.fit(aif_dataset)

    def predict(self, X, sensitive_features):
        df=pd.concat([X, sensitive_features], axis=1)
        df['label'] = 0
        aif_dataset = BinaryLabelDataset(df=df, label_names=['label'], protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        
        df_transformed = self.model.predict(aif_dataset)
        return df_transformed.labels
    
    def __del__(self):
        """Ensure the TensorFlow session is closed when the object is deleted."""
        if self.sess is not None:
            self.sess.close()
            print("TensorFlow session closed.")


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
        sensitive_attrs_to_cov_thresh = {A.name: {v: {0:0, 1:0} for v in A.unique()}}  # zero covariance threshold, means try to get the fairest solution

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
        sensitive_features = sensitive_features.astype(int)
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
    def __init__(self, method_str, base_model, datasets, eps=None, constraint_code=None, random_state=None):
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
        aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name], protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        self.kearns.fit(aif_dataset, **self.fit_params)
        return self

    def predict(self, X, sensitive_features):
        df=pd.concat([X, sensitive_features], axis=1)
        df['label'] = 0
        aif_dataset = BinaryLabelDataset(df=df, label_names=['label'], protected_attribute_names=[sensitive_features.name])
        aif_dataset.features = np.array(aif_dataset.features)
        
        df_transformed = self.kearns.predict(aif_dataset)
        return df_transformed.labels
        

class ThresholdOptimizerWrapper(ThresholdOptimizer):
    def __init__(self, *args, random_state=0, datasets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = random_state

    def fit(self, X, y, sensitive_features):
        return super().fit(X, y, sensitive_features=sensitive_features)

    def predict(self, X, sensitive_features):
        return super().predict(X, sensitive_features=sensitive_features, random_state=self.random_state)


class ExponentiatedGradientPmf(ExponentiatedGradient):
    def __init__(self, base_model, eps=0.005, random_state=None, run_linprog_step=None, eta0=None,
                 method_str='fairlearn_full', datasets=None, constraint_code='eo', constraint=None, **kwargs):
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

        self.subsample = kwargs.pop('subsample', None)
        super(ExponentiatedGradientPmf, self).__init__(base_model, constraints=copy.deepcopy(constraint), eps=eps, random_state=random_state,
                                                       **kwargs)

    def fit(self, X, y, sensitive_features, **kwargs):
        if self.subsample is not None and self.subsample < 1:
            self.subsample = int(X.shape[0] * self.subsample)
        elif self.subsample == 1:
            self.subsample = None
        if hasattr(X, 'values'):
            X = X.values
        return super().fit(X, y, sensitive_features=sensitive_features, **kwargs)

    def predict(self, X):
        return self._pmf_predict(X)[:, 1]

    #def get_stats_dict(self):
    #    res_dict = {}
    #    for key in ['best_iter_', 'best_gap_',
    #                # 'weights_', '_hs',  'predictors_', 'lambda_vecs_',
    #                'last_iter_', 'n_oracle_calls_',
    #                'n_oracle_calls_dummy_returned_', 'oracle_execution_times_', ]:
    #        res_dict[key] = getattr(self, key)
    #    return res_dict
    

class EnsembleFairness(GeneralAifModel):
    def __init__(self,
                random_state=0,
                method_str=None,
                datasets=None,
                base_model=None,
                postprocessor=None,
                ensemble_train_mode='latest',
                ensemble_mode='exp_grad',
                ensemble_number=5,
                constraint='equalized_odds',
                threshold=0.2):
        """
        Initialize the EnsembleFairnessWrapper with a base model and a post-processing method.

        Parameters:
        - random_state: int, seed for random operations.
        - base_model: A machine learning model (e.g., LogisticRegression).
        - postprocessor: An AIF360 post-processing method (e.g., EqOddsPostprocessing).
        - datasets: tuple, containing (X, y, A) where:
            X: Features (pd.DataFrame)
            y: Target labels (pd.Series or np.ndarray)
            A: Sensitive attribute (pd.Series or np.ndarray)
        - n_iterations: int, number of iterations for the boosting algorithm.
        """
        X, Y, A = datasets[:3]
        priviliged_class = [Y.groupby(A).mean().sort_values(ascending=False).index.tolist()[0]]
        unpriviliged_class = [Y.groupby(A).mean().sort_values(ascending=False).index.tolist()[1]]
        self.random_state = random_state
        self.method_str = method_str
        self.base_model = base_model
        self.postprocessor = postprocessor
        self.ensemble_train_mode = ensemble_train_mode
        self.ensemble_mode = ensemble_mode
        self.ensemble_number = ensemble_number
        self.constraint = constraint
        self.threshold = threshold
                
        self.classifiers = []
        self.fairness_metrics = []
        self.is_fitted = False
        
        np.random.seed(self.random_state)
        
        self.protected_name = A.name
        self.privileged_groups = [{self.protected_name: priviliged_class}]
        self.unprivileged_groups = [{self.protected_name: unpriviliged_class}]

    def fit(self, X, y, sensitive_features):
        """
        Train the ensemble using the boosting algorithm and post-processing.

        Parameters:
        - X: Features (pd.DataFrame)
        - y: Target labels (pd.Series or np.ndarray)
        - sensitive_features: Sensitive attribute(s) used for fairness (pd.Series or np.ndarray).
        """
        self.is_fitted = True
        try:
            if self.protected_name in X.columns:
                X = X.drop(columns=self.protected_name)
                
            if isinstance(self.postprocessor, EqOddsPostprocessing):
                aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name], protected_attribute_names=[self.protected_name])
                if not isinstance(aif_dataset.features, np.ndarray):
                    aif_dataset.features = aif_dataset.features.to_numpy()
                if not isinstance(aif_dataset.labels, np.ndarray):
                    aif_dataset.labels = aif_dataset.labels.to_numpy()
                
            current_target = y.copy()

            base_train_times=[]
            ensemble_prediction_times=[]
            postprocessor_times=[]
            for i in range(self.ensemble_number):
                X_train, X_val, target_train, target_val, sensitive_features_train, sensitive_features_val, y_train, y_val \
                    = train_test_split(X, current_target, sensitive_features, y, test_size=0.75, random_state=self.random_state+i)
                                
                #train base model
                a = datetime.now()
                model = sklearn.clone(self.base_model)
                model.fit(X_train, target_train)
                b = datetime.now()
                base_train_times.append(b-a)

                self.classifiers.append(model)
                self.fairness_metrics.append(getEO(X_val, y_val, sensitive_features_val, model.predict))
                #self.fairness_metrics.append(TPRB(X_val, y_val, sensitive_features_val, model.predict(X)))

                #get ensemble prediction
                a = datetime.now()
                if self.ensemble_train_mode == 'latest':
                    ensemble_predictions = self.classifiers[-1].predict(X)
                elif self.ensemble_train_mode == 'majority':
                    ensemble_predictions = self.ensemble_majority(X)
                elif self.ensemble_train_mode == 'random':
                    ensemble_predictions = self.ensemble_random(X)
                elif self.ensemble_train_mode == 'meta':
                    self.optimize_ensemble_meta(X_train, y_train, sensitive_features_train)
                    ensemble_predictions = self.ensemble_weighted(X)
                    self._last_ensemble_fairness = getEO(X_val, y_val, sensitive_features_val, model.predict)
                b = datetime.now()
                ensemble_prediction_times.append(b-a)

                #apply postprocessor
                a = datetime.now()

                postprocessor = copy.deepcopy(self.postprocessor)
                if isinstance(self.postprocessor, EqOddsPostprocessing):
                    aif_dataset_pred = aif_dataset.copy()
                    aif_dataset_pred.labels = ensemble_predictions
                    postprocessor.privileged_groups = self.privileged_groups
                    postprocessor.unprivileged_groups = self.unprivileged_groups
                
                    postprocessor.fit(dataset_true=aif_dataset, dataset_pred=aif_dataset_pred)
                    mitigated_predictions = postprocessor.predict(aif_dataset_pred)
                    mitigated_predictions = mitigated_predictions.labels.ravel().astype(np.int64)
                elif isinstance(self.postprocessor, ThresholdOptimizer):
                    postprocessor.estimator=self
                    postprocessor.fit(X=X, y=y, sensitive_features=sensitive_features)
                    mitigated_predictions = postprocessor.predict(X=X, sensitive_features=sensitive_features, random_state=self.random_state)
                b = datetime.now()
                postprocessor_times.append(b-a)


                if len(mitigated_predictions) == np.sum(mitigated_predictions == mitigated_predictions[0]):
                    current_target = y.copy()
                else:
                    current_target = mitigated_predictions
                
            self.avg_base_train_time = np.average(base_train_times)
            self.avg_ensemble_pred_time = np.average(ensemble_prediction_times)
            self.avg_postprocessor_time = np.average(postprocessor_times)
            return self
        except Exception as e: print(traceback.format_exc())

    def predict(self, X):
        """
        Make predictions using the trained ensemble and apply post-processing.

        Parameters:
        - X: Features (pd.DataFrame)
        - sensitive_features: Sensitive attribute(s) (pd.Series or np.ndarray).

        Returns:
        - np.ndarray: Post-processed predictions.
        """
        if self.protected_name in X.columns:
            X = X.drop(columns=self.protected_name)
            
        if self.ensemble_mode == 'majority':
            return self.ensemble_majority(X)
        if self.ensemble_mode == 'meta':
            return self.ensemble_weighted(X)
        
    
    def get_stats_dict(self):
        stats = {'avg_base_train_time': self.avg_base_train_time,
            'avg_ensemble_pred_time': self.avg_ensemble_pred_time,
            'avg_postprocessor_time': self.avg_postprocessor_time
            }
        formatted_list=[f"{x:.4f}" for x in self.fairness_metrics]
        stats['model_fairness_metrics'] = '[' + ', '.join(formatted_list) + ']'
        if hasattr(self, '_last_meta_weights'):
            formatted_list=[f"{x:.2f}" for x in self._last_meta_weights]
            stats['meta_model_weights'] = '[' + ', '.join(formatted_list) + ']'
        return stats
    
    def __sklearn_is_fitted__(self):
        return self.is_fitted
    
    def ensemble_random(self, X):
        ensemble_predictions = np.array([clf.predict(X) for clf in self.classifiers])
        n_models, n_samples = ensemble_predictions.shape
        random_model_indices = np.random.randint(0, n_models, size=n_samples)
        final_predictions = ensemble_predictions[random_model_indices, np.arange(n_samples)]
        return final_predictions.reshape(-1, 1)
        
        
    def ensemble_majority(self, X):
        ensemble_predictions = np.array([clf.predict(X) for clf in self.classifiers])
        ensemble_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=ensemble_predictions)
        return ensemble_predictions.reshape(-1, 1)
    
    def ensemble_weighted(self, X):
        meta_features = np.array([clf.predict(X) for clf in self.classifiers])
        meta_X = np.column_stack(meta_features)
        preds = meta_X @ self._last_meta_weights
        return (preds > 0.5).astype(int)
    
    def optimize_ensemble_meta(self, X, y, sensitive_features):
        if len(self.classifiers) == 1:
            self._last_meta_weights = np.array([1])
            return
                
        if self._last_ensemble_fairness > self.threshold:
            self.optimize_fairness(X,y,sensitive_features)
        else:
            self.optimize_accuracy(X,y,sensitive_features)
    
    def optimize_fairness(self, X, y, sensitive_features):
        meta_features = []
        for clf in self.classifiers:
            meta_features.append(clf.predict(X))
        meta_X = np.column_stack(meta_features)
        
        if not isinstance(y, np.ndarray):
            y = y.to_numpy()
        if not isinstance(sensitive_features, np.ndarray):
            sensitive_features = sensitive_features.to_numpy()
            
        def fairness_objective(w):
            y_pred = meta_X @ w

            mean_pred_y0 = np.mean(y_pred[y == 0])
            mean_pred_y1 = np.mean(y_pred[y == 1])

            max_group_diff = 0
            
            sensitive_values = np.unique(sensitive_features)

            for z_value in sensitive_values:
                if len(y_pred[(y == 0) & (sensitive_features == z_value)] > 0):
                    group_mean_y0 = np.mean(y_pred[(y == 0) & (sensitive_features == z_value)])
                    diff_y0 = np.abs(group_mean_y0 - mean_pred_y0)
                else:
                    diff_y0 = 0
                    
                if len(y_pred[(y == 1) & (sensitive_features == z_value)] > 0):
                    group_mean_y1 = np.mean(y_pred[(y == 1) & (sensitive_features == z_value)])
                    diff_y1 = np.abs(group_mean_y1 - mean_pred_y1)
                else:
                    diff_y1 = 0

                max_group_diff = max(max_group_diff, diff_y0, diff_y1)

            return max_group_diff

        initial_w = np.random.randn(len(self.classifiers))

        weight_sum_constraint = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        }

        bounds = [(0, 1) for _ in range(len(self.classifiers))]

        result = minimize(fairness_objective, initial_w, method='SLSQP', constraints=[weight_sum_constraint], bounds=bounds, options={'disp': False})

        self._last_meta_weights = result.x
        return
       
    def optimize_accuracy(self, X, y, sensitive_features): 
        meta_features = []
        for clf in self.classifiers:
            meta_features.append(clf.predict(X))
        meta_X = np.column_stack(meta_features)
        
        if not isinstance(y, np.ndarray):
            y = y.to_numpy()
        if not isinstance(sensitive_features, np.ndarray):
            sensitive_features = sensitive_features.to_numpy()
        
        w = cp.Variable(len(self.classifiers))
        y_pred = meta_X @ w

        objective = cp.Minimize(cp.sum_squares(y_pred - y))

        if self.constraint == "equalized_odds":       
            sensitive_values = np.unique(sensitive_features)

            mean_pred_y0 = cp.sum(cp.multiply(y == 0, y_pred)) / cp.sum(y == 0)
            mean_pred_y1 = cp.sum(cp.multiply(y == 1, y_pred)) / cp.sum(y == 1)

            group_diffs = []

            for z_value in sensitive_values:
                group_mean_y0 = cp.sum(cp.multiply(cp.multiply((y == 0), (sensitive_features == z_value)), y_pred)) / cp.sum(cp.multiply((y == 0), (sensitive_features == z_value)) + 1e-8)
                group_mean_y1 = cp.sum(cp.multiply(cp.multiply((y == 1), (sensitive_features == z_value)), y_pred)) / cp.sum(cp.multiply((y == 1), (sensitive_features == z_value)) + 1e-8)

                diff_y0 = cp.abs(group_mean_y0 - mean_pred_y0)
                diff_y1 = cp.abs(group_mean_y1 - mean_pred_y1)

                group_diffs.append(diff_y0)
                group_diffs.append(diff_y1)

            max_group_diff = cp.max(cp.vstack(group_diffs))

            constraints = [
                max_group_diff <= self.threshold
            ]

        else:
            raise ValueError(f"Unsupported constraint: {self.constraint}")
        

        def evaluate_constraint(y, y_pred, sensitive_features):
            # Unique sensitive attribute values
            sensitive_values = np.unique(sensitive_features)

            # Overall means for y = 0 and y = 1
            mean_pred_y0 = np.sum((y == 0) * y_pred) / (np.sum(y == 0) + 1e-8)
            mean_pred_y1 = np.sum((y == 1) * y_pred) / (np.sum(y == 1) + 1e-8)

            group_diffs = []

            # Iterate over each sensitive value
            for z_value in sensitive_values:
                # Group-specific mean predictions for y = 0 and y = 1
                group_mean_y0 = np.sum((y == 0) * (sensitive_features == z_value) * y_pred) / (np.sum((y == 0) * (sensitive_features == z_value)) + 1e-8)
                group_mean_y1 = np.sum((y == 1) * (sensitive_features == z_value) * y_pred) / (np.sum((y == 1) * (sensitive_features == z_value)) + 1e-8)

                # Calculate absolute differences from the overall mean
                diff_y0 = np.abs(group_mean_y0 - mean_pred_y0)
                diff_y1 = np.abs(group_mean_y1 - mean_pred_y1)

                # Append differences to the list
                group_diffs.append(diff_y0)
                group_diffs.append(diff_y1)

            # Find the maximum group difference
            max_group_diff = np.max(group_diffs)

            return max_group_diff

        
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
        except cp.SolverError:
            self.optimize_fairness(X,y,sensitive_features)
            return
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or getEO(X, y, sensitive_features, lambda _: ((meta_X @ w.value) > 0.5).astype(int)) > self.threshold:
            self.optimize_fairness(X,y,sensitive_features)
            return
        self._last_meta_weights = w.value
        return
        


class Postprocessor(GeneralAifModel):
    def __init__(self, random_state, method_str=None, base_model=None, postprocessor=None, datasets=None):
        X, Y, A = datasets[:3]
        priviliged_class = [Y.groupby(A).mean().sort_values(ascending=False).index.tolist()[0]]
        unpriviliged_class = [Y.groupby(A).mean().sort_values(ascending=False).index.tolist()[1]]
        self.method_str = method_str
        self.base_model = base_model
        self.postprocessor = postprocessor
        self.random_state = random_state
        
        self.protected_name = A.name
        self.privileged_groups = [{self.protected_name: priviliged_class}]
        self.unprivileged_groups = [{self.protected_name: unpriviliged_class}]

    def fit(self, X, y, sensitive_features):
        try:
            if self.protected_name in X.columns:
                X = X.drop(columns=self.protected_name)
            
            self.base_model = self.base_model
            self.base_model.fit(X, y)
            predictions = self.base_model.predict(X).reshape(-1,1)
            
            if isinstance(self.postprocessor, EqOddsPostprocessing):
                aif_dataset = BinaryLabelDataset(df=pd.concat([X, sensitive_features, y], axis=1), label_names=[y.name], protected_attribute_names=[self.protected_name])
                if not isinstance(aif_dataset.features, np.ndarray):
                    aif_dataset.features = aif_dataset.features.to_numpy()
                if not isinstance(aif_dataset.labels, np.ndarray):
                    aif_dataset.labels = aif_dataset.labels.to_numpy()
                aif_dataset_pred = aif_dataset.copy()
                aif_dataset_pred.labels = predictions

                self.postprocessor.privileged_groups = self.privileged_groups
                self.postprocessor.unprivileged_groups = self.unprivileged_groups

                self.postprocessor.fit(dataset_true=aif_dataset, dataset_pred=aif_dataset_pred)
            elif isinstance(self.postprocessor, ThresholdOptimizer):
                self.postprocessor.estimator=self.base_model
                self.postprocessor.fit(X=X, y=y, sensitive_features=sensitive_features)
            return self
        except Exception as e: print(traceback.format_exc())

    def predict(self, X, sensitive_features):
        if self.protected_name in X.columns:
            X = X.drop(columns=self.protected_name)
        predictions = self.base_model.predict(X)
        
        if isinstance(self.postprocessor, EqOddsPostprocessing):
            df = X.copy()
            df[self.protected_name]=sensitive_features
            df['label']=predictions
            aif_dataset_pred = BinaryLabelDataset(df=df, label_names=['label'], protected_attribute_names=[self.protected_name])
            if not isinstance(aif_dataset_pred.features, np.ndarray):
                    aif_dataset_pred.features = aif_dataset_pred.features.to_numpy()
            if not isinstance(aif_dataset_pred.labels, np.ndarray):
                aif_dataset_pred.labels = aif_dataset_pred.labels.to_numpy()
            mitigated_predictions = self.postprocessor.predict(aif_dataset_pred).labels
            
        elif isinstance(self.postprocessor, ThresholdOptimizer):
            mitigated_predictions = self.postprocessor.predict(X=X, sensitive_features=sensitive_features)
        return mitigated_predictions

additional_models_dict = {
    'most_frequent': partial(sklearn.dummy.DummyClassifier, strategy="most_frequent"),
    'LogisticRegression': sklearn.linear_model.LogisticRegression,
}


def create_wrapper(method_str, random_state=42, datasets=None, **kwargs):
    model_class = additional_models_dict.get(method_str)

    class PersonalizedWrapper:
        def __init__(self, method_str, random_state=42, **kwargs):
            self.method_str = method_str
            self.model = model_class(random_state=random_state)

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
