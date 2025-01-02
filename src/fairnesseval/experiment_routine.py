from fairnesseval import run
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
done_conf = [
    # done on fairlearn-2
    'sigmod_h_exp_1.0',
    's_h_exp_EO_1.0',
    's_h_EO_1.0',
    'acs_eps_EO_1.0',
    'acs_eps_EO_1.1',
    'acs_eps_EO_2.0',
    'acs_eps_EO_2.1',
    'acs_h_gs1_EO_2.0',
    'acs_h_gsSR_1.0',
    'acs_h_gsSR_2.0',
    'acs_h_gsSR_1.1',
    'acsE_h_gsSR_1.1',
    'acsE_h_gs1_1.0',
    's_h_EO_1.0',
    'acs_h_gsSR_2.1',
    'acs_h_gs1_1.1',
    'sigmod_h_exp_2.0',
    'sigmod_h_exp_3.0',
    'acs_eps_EO_2.0',
    'acs_eps_EO_2.1',
    'acs_h_gs1_EO_1.0',
    'acs_h_gs1_EO_2.0',
    'acs_h_gsSR_2.0',
    'acs_h_gsSR_2.1',
    's_tr_1.1',
    's_tr_2.0',
    's_tr_2.1',
    'f_eta0_1.0',
    'f_eta0_2.0',
    's_h_1.0r',
    's_h_EO_1.0r',
    'acs_eps_EO_1.0r',
    'acs_h_eps_1.0r',
    's_h_exp_1.0r',
    's_h_exp_EO_1.0r',
    's_h_exp_2.0r',
    's_h_exp_EO_2.0r',
    'acs_h_gs1_1.0r',
    'acs_h_gs1_EO_1.0r',
    'acs_h_gsSR_1.0r',
    'acs_h_gsSR_EO_1.0r',
    's_c_1.0r',
    's_tr_1.0r',
    's_tr_1.1r',
    's_tr_2.0r',
    's_tr_2.1r',
    'most_frequent_sig.0r',
    'most_frequent_ACS.0r',
    'acs_to_binary1.0r',
    'acs_to_binaryEO1.0r',
    'acsE_eps_EO_1.0r',
    'acsE_h_eps_1.0r',
    'f_eta0_1.1',
    'f_eta0_2.1',
    'acsE_h_gs1_1.0r',
    'acsE_h_gs1_EO_1.0r',
    'acsE_h_gsSR_1.0r',
    'acsE_h_gsSR_1.1r',
    'acsE_h_gs1_EO_1.1r',
    'acs_h_eps_1.2r',
    'f_eta0_1.2',
    'f_eta0_2.2',
    'f_eta0_eps.3P',
    'f_eta0_eps.4P',
    'f_eta0_eps.3E',
    'f_eta0_eps.4E',
    'f_eta0_eps.4.2E',
    'f_eta0_eps.3.2E',
    'e_s.1',
    'e_s.0',
    'e_m.0',
    'e_m.1',
    'e_l.fast',
    'e_l.0',
    'e_l.1',

    'e_l.fast.2',
    'rlp_F_ACS.1',

    # doing in fairlearn-2

    # done on fairlearn-3
    's_zDI_1.2',
    's_zDI_1.22',
    's_zEO_1.1',
    's_f_1.0r',
    's_f_1.1r',
    'acsER_bin2.0r',
    'acsER_bin2.1r',
    'acsER_bin2.2r',
    'acs_to_binary1.1r',
    'acsER_bin3.0r',
    'acsER_bin3.1r',
    'acsE_h_gs1_1.1r',
    'acsE_h_gs1_1.2r',
    'acsER_bin2.0r',
    'acsER_bin2.1r',
    'acsER_bin3.0r',
    'acsER_bin4.0r',
    'f_eta0_eps.1',
    'f_eta0_eps.2',
    'acsER_binB1.0Mr',
    'acsER_binB1.0r',
    'demo.0'
    'demo.2r',
    'demo.2.1r',
    'demo.A.1r',
    # to update on fairlearn-3
    'acsER_binB1.1Mr',
    'acsER_binB1.1r',
    'acsER_binB2.0r',
    'acsER_binB2.0Mr',
    'demo.C.0r',

    'demo.2r',
    'demo.B2r',
    'demo.B1r',
    'demo.D.0r',

    # doing on fairlearn-3

    'exp_h_eps_PUB.1r',
]

new_exp_done = ['s_c_1.0N',  # done
                's_zDI_1.1N',  # done
                's_zEO_1.1N',  # small
                's_zDI_1.2N',  # BIG
                's_e_1.0N',
                's_tr_1.0N',  # small
                's_tr_2.0N',  # medium/BIG
                's_f_1.0N',  # small
                's_f_1.1N',  # BIG
                's_u_1.0N',  # unmitigated
                's_KearnsPleiss_1.0N',  # small
                's_k_1.1N',
                'rlp_F_1.0N',  # small
                'a_e_1.0N',
                's_e_1.0N',
                ]

if __name__ == "__main__":

    conf_todo = [
        # pending
        's_zDI_1.2N',  # BIG # todo repeat

        'rlp_F_1.12N',

        # debugging

        # ###
        # # running now
        # #AB2
        # #AB3
        # 'a_e_1.1N',
        # #AB4
        # 'rlp_F_1.1N',  # big

    ]

    # run.launch_experiment_by_config({"experiment_id": "q12",
    # "dataset_names": [
    # "ACSIncome",
    # "ACSPublicCoverage"],
    # "model_names": ["LogisticRegression"],
    # "random_seed": 1,
    # "train_fractions": [0.016, 0.063, 0.251, 1],
    # "results_path": "demo_results",
    # "params": ["--debug"]})

    # run.launch_experiment_by_config(
    #     {"experiment_id": "demo.00",
    # "dataset_names": ["adult"],
    # "model_names": ["expgrad"],
    # "model_params": {"base_model_code": "lr", 'eps': [0.005],
    # "base_model_grid_params": {"C": [0.1, 1]}},
    # "results_path": "c:\\users\\andrea\\gdrive\\drive condivisi\\softlab\\projects\\fairness\\scalable-expgrad\\streamlit\\demo_results",
    # "params": ["--debug"]}

    # {"experiment_id": "demo.A.1rs",
    # "dataset_names": ["adult"],
    # "model_names": ["Feld",
    # "ThresholdOptimizer",
    # "ZafarDI"],
    # "model_params": {"base_model_code": ["lr"],
    # "constraint_code": "dp",
    # "base_model_grid_params": {"C": [0.1, 1]}},
    # "results_path": "c:\\users\\andrea\\gdrive\\drive condivisi\\softlab\\projects\\fairness\\scalable-expgrad\\streamlit\\demo_results",
    # "params": ["--debug"]}
    # )
    for x in conf_todo:
        run.launch_experiment_by_id(x)
