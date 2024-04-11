import os
from copy import deepcopy

import aif360.datasets
import numpy as np
import pandas as pd
import requests
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, \
    load_preproc_data_compas, load_preproc_data_german
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import folktables
import utils_experiment_parameters
from fairlearn.reductions import DemographicParity, EqualizedOdds, UtilityParity
from folktables import ACSDataSource, generate_categories
from utils_general import Singleton

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

baseline_results_file_name = 'results/baseline_results (yeeha).json'
adult_github_data_url = "https://github.com/slundberg/shap/raw/master/data/"
german_credit_kaggle_url = 'https://www.kaggle.com/datasets/uciml/german-credit/download?datasetVersionNumber=1'
compas_credit_github_maliha_url = 'https://github.com/maliha93/Fairness-Analysis-Code/raw/master/dataset/compas.csv'


def raise_dataset_name_error(x):
    raise Exception(f'dataset_name {x} not allowed.')


def cache(url, file_name=None):
    if file_name is None:
        file_name = os.path.basename(url)
    data_dir = os.path.join(os.path.dirname(""), "cached_data")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    file_path = os.path.join(data_dir, file_name)
    if not os.path.isfile(file_path):
        urlretrieve(url, file_path)

    return file_path


def split_label_sensitive_attr(df: pd.DataFrame, label_name, sensitive_name):
    Y = df[label_name].copy()
    A = df[sensitive_name].copy()
    return df, Y, A


def adult(display=False):
    """ Return the Adult census data in a nice package. """
    dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
    raw_data = pd.read_csv(
        cache(adult_github_data_url + "adult.data"),
        names=[d[0] for d in dtypes],
        na_values="?",
        dtype=dict(dtypes)
    )
    data = raw_data.drop(["Education"], axis=1)  # redundant with Education-Num
    filt_dtypes = list(filter(lambda x: not (x[0] in ["Target", "Education"]), dtypes))
    data["Target"] = data["Target"] == " >50K"
    rcode = {
        "Not-in-family": 0,
        "Unmarried": 1,
        "Other-relative": 2,
        "Own-child": 3,
        "Husband": 4,
        "Wife": 5
    }
    for k, dtype in filt_dtypes:
        if dtype == "category":
            if k == "Relationship":
                data[k] = np.array([rcode[v.strip()] for v in data[k]])
            else:
                data[k] = data[k].cat.codes

    if display:
        return raw_data.drop(["Education", "Target", "fnlwgt"], axis=1), data["Target"].values
    else:
        return data.drop(["Target", "fnlwgt"], axis=1), data["Target"].values


def check_download_dataset(dataset_name='compas'):
    if 'compas' in dataset_name:
        compas_raw_data_github_url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
        compas_path = '/home/fairlearn/anaconda3/lib/python3.9/site-packages/aif360/data/raw/compas'
        compas_raw_path = compas_path + '/compas-scores-two-years.csv'
        if not os.path.exists(compas_raw_path):
            df = pd.read_csv(compas_raw_data_github_url)
            os.makedirs(compas_path, exist_ok=True)
            df.to_csv(compas_raw_path, index=False)
        return
    elif 'german' in dataset_name:
        base_path = '/home/fairlearn/anaconda3/lib/python3.9/site-packages/aif360/data/raw/german/'
        base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/'
        file_names = ['german.data', 'german.doc']
    elif 'adult_sigmod' in dataset_name:
        base_path = '/home/fairlearn/anaconda3/lib/python3.9/site-packages/aif360/data/raw/adult'
        base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
        file_names = ['adult.data', 'adult.test', 'adult.names']
    else:
        raise_dataset_name_error(dataset_name)
    for file_name in file_names:
        turn_path = os.path.join(base_path, file_name)
        if not os.path.exists(turn_path):
            os.makedirs(base_path, exist_ok=True)
            response = requests.get(base_url + file_name)
            open(turn_path, "wb").write(response.content)


def convert_from_aif360_to_df(dataset, dataset_name=None):
    # X = pd.DataFrame(dataset.features, columns=dataset.feature_names)
    # data, _ = dataset.convert_to_dataframe()
    # X = data.iloc[:, :-1]
    # y = data.iloc[:, -1] #pd.Series(dataset.labels.flatten(), name=dataset.label_names[0])
    # A = pd.Series(dataset.protected_attributes.flatten(), name=dataset.protected_attribute_names[0])
    X = pd.DataFrame(dataset.features, columns=dataset.feature_names)
    y = pd.Series(dataset.labels.astype(int).ravel(), name=dataset.label_names[0])
    A = pd.Series(dataset.protected_attributes.astype(int).ravel(), name=dataset.protected_attribute_names[0])
    # if dataset.__class__.__name__ == 'GermanDataset':
    #     y[y == 2] = 0
    return X, y, A


def load_convert_dataset_aif360(dataset_name='compas'):
    ret_dict = {}
    check_download_dataset(dataset_name)
    if 'compas' in dataset_name:
        # ret_dict['protected'] = 'race'
        # ret_dict['privileged_groups'] = [{'Race': 1}]
        # ret_dict['unprivileged_groups'] = [{'Race': 0}]
        dataset_orig = load_preproc_data_compas(protected_attributes=['race'])
        # dataset_orig = CompasDataset(protected_attribute_names=['race'])
    elif 'german' in dataset_name:
        # ret_dict['protected'] = 'sex'
        # ret_dict['privileged_groups'] = [{'Sex': 1}]
        # ret_dict['unprivileged_groups'] = [{'Sex': 0}]
        # dataset_orig = GermanDataset(protected_attribute_names=['Sex'])

        dataset_orig = load_preproc_data_german(protected_attributes=['sex'])
        dataset_orig.metadata['label_maps'] = [{1: 'Good Credit', 0: 'Bad Credit'}]
        dataset_orig.unfavorable_label = 0
        dataset_orig.labels[dataset_orig.labels == 2] = 0
    elif 'adult_sigmod' in dataset_name:
        # ret_dict['protected'] = 'sex'
        # ret_dict['privileged_groups'] = [{'Sex': 1}]
        # ret_dict['unprivileged_groups'] = [{'Sex': 0}]

        dataset_orig = load_preproc_data_adult(['sex'])
        # dataset_orig = AdultDataset(protected_attribute_names=['sex'],
        #                             privileged_classes=[['Male']],
        #                             categorical_features=['age', 'education-num'],
        #                             features_to_keep=['age', 'education-num', 'sex']
        #                             )
    else:
        raise_dataset_name_error(dataset_name)
    # ret_dict['aif360_dataset'] = dataset_orig
    X, y, A = convert_from_aif360_to_df(dataset_orig, dataset_name)
    ret_dict['df'] = dict(zip(['X', 'y', 'A'], [X, y, A]))
    return X, y, A, dataset_orig


def split_dataset_aif360(aif360_dataset: aif360.datasets.StandardDataset, train_test_seed):
    ret_dict = {}
    if train_test_seed == 0:
        dataset_orig_train, dataset_orig_test = aif360_dataset.split([0.7], shuffle=False)
    else:
        dataset_orig_train, dataset_orig_test = aif360_dataset.split([0.7], shuffle=True, seed=train_test_seed)
    ret_dict['aif360_train'] = dataset_orig_train
    ret_dict['aif360_test'] = dataset_orig_test
    ret_dict['train_df'] = convert_from_aif360_to_df(dataset_orig_train)
    ret_dict['test_df'] = convert_from_aif360_to_df(dataset_orig_test)
    return ret_dict


def load_transform_Adult(sensitive_attribute='Sex', test_size=0.3, random_state=42):
    # https://archive.ics.uci.edu/ml/datasets/adult
    features = ['Age', 'Workclass', 'Education-Num', 'Marital Status',
                'Occupation', 'Relationship', 'Race', 'Sex',
                'Capital Gain', 'Capital Loss', 'Hours per week', 'Country']
    categorical_cols = ['Workclass',  # 'Education-Num',
                        'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    numerical_cols = np.setdiff1d(features, categorical_cols)

    X, Y = adult()
    Y = pd.Series(LabelEncoder().fit_transform(Y))
    A = X[sensitive_attribute].copy()
    X_transformed = pd.get_dummies(X, dtype=int, columns=categorical_cols)
    X_transformed[numerical_cols] = StandardScaler().fit_transform(X[numerical_cols])
    return X_transformed, Y, A


def load_transform_ACS(dataset_str, states=None, return_acs_data=False):
    loader_method: folktables.BasicProblem = getattr(folktables, dataset_str)
    if loader_method.group in loader_method.features:  # remove sensitive feature from data
        loader_method.features.remove(loader_method.group)
    data_source = ACSDataSource(survey_year=2018, horizon='1-Year', survey='person', root_dir='cached_data')
    definition_df = data_source.get_definitions(download=True)
    categories = generate_categories(features=loader_method.features, definition_df=definition_df)
    acs_data = data_source.get_data(download=True,
                                    states=states)  # TODO # with density 1  random_seed=0 do nothing | join_household=False ???

    df, label, group = loader_method.df_to_pandas(acs_data, categories=categories)
    # df, label, group = fix_nan(df, label, group, mode=fillna_mode)

    categorical_cols = list(categories.keys())
    # See here for data documentation of cols https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/
    # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict00_02.pdf
    df = pd.get_dummies(df, dtype=np.uint8, columns=categorical_cols)
    numerical_cols = np.setdiff1d(loader_method.features, categorical_cols)
    df[numerical_cols] = StandardScaler().fit_transform(df[numerical_cols])
    # choice of the model todo add possibility to chose preprocessing based on the model
    # df[df.columns] = StandardScaler().fit_transform(df)
    print(f'Loaded data memory used by df: {df.memory_usage().sum() / (2 ** (10 * 3)):.3f} GB')
    ret_value = df, label.iloc[:, 0].astype(int), group.iloc[:, 0]
    # df.to_csv('datasets/datasets/ACSPublicCoverage/X.csv')
    # label.iloc[:, 0].astype(int).to_csv('datasets/datasets/ACSPublicCoverage/y.csv')
    # group.iloc[:, 0].to_csv('datasets/datasets/ACSPublicCoverage/groups.csv')
    if return_acs_data:
        ret_value += tuple([acs_data])
    else:
        del acs_data

    return ret_value


def fix_nan(X: pd.DataFrame, y, A, mode='mean'):
    if mode == 'mean':
        X.fillna(X.mean())
    if mode == 'remove':
        notna_mask = X.notna().all(1)
        X, y, A = X[notna_mask], y[notna_mask], A[notna_mask]

    return X, y, A


def load_split_data(sensitive_attribute='Sex', test_size=0.3, random_state=42):
    X, Y, A = load_transform_Adult(sensitive_attribute, test_size, random_state)
    train_index, test_index = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=random_state)
    results = []
    for turn_index in [train_index, test_index]:
        for turn_df in [X, Y, A]:
            results.append(turn_df.iloc[turn_index])
    return results


def get_data_from_expgrad(expgrad):
    res_dict = {}
    for key in ['best_iter_', 'best_gap_',
                # 'weights_', '_hs',  'predictors_', 'lambda_vecs_',
                'last_iter_', 'n_oracle_calls_',
                'n_oracle_calls_dummy_returned_', 'oracle_execution_times_', ]:
        res_dict[key] = getattr(expgrad, key)
    return res_dict


def find_privileged_unprivileged(X, y, sensitive_features):
    sensitive_values = sensitive_features.unique()
    groups_mean = pd.Series(y).groupby(sensitive_features).mean()
    mean_y = groups_mean.mean()
    sensitive_features_name = sensitive_features.name

    privileged_groups = [{'Race': 1}]
    unprivileged_groups = [{'Race': 0}]

    return privileged_groups, unprivileged_groups


def get_dataset(dataset_str, prm=None):
    if dataset_str == "adult":
        return load_transform_Adult()
    elif dataset_str in utils_experiment_parameters.sigmod_datasets + utils_experiment_parameters.sigmod_datasets_aif360:
        return load_convert_dataset_aif360(dataset_str)
    elif dataset_str in utils_experiment_parameters.ACS_dataset_names:
        return load_transform_ACS(dataset_str=dataset_str, states=prm['states'])
    else:
        raise_dataset_name_error(dataset_str)




def white_alone(datasets):
    data_values = DataValuesSingleton()
    data_values.set_original_sensitive_attr(datasets[2])
    datasets = list(datasets)
    datasets[2] = datasets[2].map({1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0})
    return datasets

def binary_split_by_mean_y(datasets):
    data_values = DataValuesSingleton()
    data_values.set_original_sensitive_attr(datasets[2])
    _, y, A = datasets[:3]
    mean_y = y.mean()
    mean_by_group = y.groupby(A).mean()
    priviliged = mean_by_group[mean_by_group > mean_y].index
    group_map = {x:1 if x in priviliged else 0 for x in mean_by_group.index}
    datasets = list(datasets)
    datasets[2] = datasets[2].map(group_map)
    return datasets

preprocessing_function_map = dict(conversion_to_binary_sensitive_attribute=white_alone,
                                  binary_split_by_mean_y=binary_split_by_mean_y,
                                  default=lambda x: x)


def preprocess_dataset(datasets, prm):
    return preprocessing_function_map[prm['preprocessing']](datasets)


def split_dataset_generator(dataset_str, datasets, train_test_seed, split_strategy, test_size):

    # if dataset_str in utils_experiment.sigmod_datasets + utils_experiment.sigmod_datasets_aif360:
    #     convert_to_df_aif360(datasets[3])
    #     logging.warning('split_strategy has no effect when using aif 360 datasets ')
    #     dataset_dict = split_dataset_aif360(datasets[3], train_test_seed=train_test_seed)
    #     # if dataset_str in utils_experiment.sigmod_datasets:
    #     yield dataset_dict['train_df'], dataset_dict['test_df']
    #     # elif dataset_str in utils_experiment.sigmod_datasets_aif360:
    #     #     dataset_dict = split_dataset_aif360(datasets[3], train_test_seed=train_test_seed)
    #     #     dataset_dict['train_df'] += tuple([dataset_dict['aif360_train']])
    #     #     dataset_dict['test_df'] += tuple([dataset_dict['aif360_test']])
    #     #     yield dataset_dict['train_df'], dataset_dict['test_df']
    #     # train_data = dict(zip(['X', 'y', 'A'], dataset_dict['train_df']))
    #     # train_data['X'] = dataset_dict['aif360_train']
    #     # test_data = dict(zip(['X', 'y', 'A'], dataset_dict['train_df']))
    #     # test_data['X'] = dataset_dict['aif360_test']
    #     # yield list(train_data.values()), list(test_data.values())
    # elif dataset_str in utils_experiment.dataset_names:
    data_values = DataValuesSingleton()
    if split_strategy == 'StratifiedKFold':
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=train_test_seed) # todo add to prm the number of splits
        X, y, A = datasets[:3]
        to_stratify = pd.Series(A).astype(str) + '_' + pd.Series(y).astype(str)
        if data_values.original_sensitive_attr is not None:
            A_orig = data_values.original_sensitive_attr
            to_stratify = pd.Series(A_orig).astype(str) + '_' + pd.Series(y).astype(str)
        for train_index, test_index in skf.split(X, to_stratify):
            datasets_divided = []
            for turn_index in [train_index, test_index]:  # train test split of datasets
                datasets_divided.append([df.iloc[turn_index] for df in [X, y, A]])
            data_values.set_train_test_index(train_index=train_index, test_index=test_index)
            yield datasets_divided
    elif split_strategy == 'stratified_train_test_split':
        X, y, A = datasets[:3]
        to_stratify = pd.concat([A, y], axis=1).astype('category').apply(lambda x: '_'.join(x.astype(str)), axis=1)
        sample_mask = np.arange(X.shape[0])
        train_index, test_index = train_test_split(sample_mask, test_size=test_size, stratify=to_stratify,
                                          random_state=train_test_seed, shuffle=True)

        datasets_divided = []
        for turn_index in [train_index, test_index]:
            datasets_divided.append([df.iloc[turn_index] for df in [X, y, A]])
        data_values.train_index = train_index
        data_values.test_index = test_index
        yield datasets_divided



def split_X_y_A(df: pd.DataFrame):
    return [df.iloc[:, :-2], df.iloc[:, -2], df.iloc[:, -1]]


def get_constraint(constraint_code, eps):
    code_to_constraint = {'dp': DemographicParity,
                          'eo': EqualizedOdds,
                          'demographic_parity': DemographicParity,
                          'equalized_odds': EqualizedOdds,
                          }
    if constraint_code not in code_to_constraint.keys():
        assert False, f'available constraint_code are: {list(code_to_constraint.keys())}'
    constraint: UtilityParity = code_to_constraint[constraint_code]
    return constraint(difference_bound=eps)


class DataValuesSingleton(metaclass=Singleton):
    original_sensitive_attr = None
    train_index = None
    test_index = None
    phase = None

    def set_phase(self, phase):
        #if phase not in ['train', 'test']:
        #    raise Exception(f'phase {phase} not allowed.')
        self.phase = phase

    def set_train_test_index(self, train_index, test_index):
        self.train_index = train_index
        self.test_index = test_index

    def get_current_original_sensitive_attr(self):
        if self.phase == 'train':
            return self.original_sensitive_attr[self.train_index]
        elif self.phase == 'test':
            return self.original_sensitive_attr[self.test_index]
        else:
            raise Exception(f'phase {self.phase} not allowed.')

    def set_original_sensitive_attr(self, original_sensitive_attr):
        self.original_sensitive_attr = deepcopy(original_sensitive_attr)
