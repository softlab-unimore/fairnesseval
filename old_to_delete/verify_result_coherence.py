import os

import pandas as pd

from .utils_results_data import filter_results, seed_columns
from utils_experiment_parameters import dataset_names


def check_expgrad_time(df:pd.DataFrame)-> pd.DataFrame:
    sub_mask = df['model_name'].str.contains('sub')
    lp_off_mask = df['model_name'].str.contains('LP_off')
    no_h7_mask =  ~(df['model_name'].str.contains('hybrid_7'))
    true_serie = df['model_name'] != ''
    for lp_mask in [lp_off_mask, ~lp_off_mask]:
        if lp_mask.any():
            turn_mask = ~sub_mask & lp_mask & no_h7_mask  # not sub is static sampling
            static_time = df[turn_mask & (df['model_name'].str.contains('expgrad_frac'))]['time'].values[0]
            true_serie[turn_mask] &= (df.loc[turn_mask, 'time'] == static_time)
            assert (df.loc[turn_mask, 'time'] == static_time).all()

            turn_mask = sub_mask & lp_mask  # sub is adaptive sampling --> take h7 that is expgrad adaptive
            adaptive_time = df[lp_mask & (df['model_name'].str.contains('hybrid_7'))]['time'].values[0]
            true_serie[turn_mask] &= (df.loc[turn_mask, 'time'] == adaptive_time)
            true_serie[lp_mask & (df['model_name'].str.contains('hybrid_7'))] &= adaptive_time > static_time
            assert (df.loc[turn_mask, 'time'] == adaptive_time).all()
            assert adaptive_time > static_time

    return true_serie


if __name__ == '__main__':
    save = True
    show = False
    df_list = []
    for base_model_code in ['lr', 'lgbm']:
        for dataset_name in dataset_names:
            base_dir = os.path.join("../results", "fairlearn-2", dataset_name)
            all_model_df = filter_results(base_dir, conf=dict(exp_grid_ratio='sqrt', states='', exp_subset='True',
                                                              base_model_code=base_model_code, ))

            expgrad_phase_mask = all_model_df['phase'] == "expgrad_fracs"
            expgrad_df = all_model_df[expgrad_phase_mask]
            check_df = expgrad_df.groupby(seed_columns + ['frac']).apply(check_expgrad_time).stack()
            assert check_df.all()
