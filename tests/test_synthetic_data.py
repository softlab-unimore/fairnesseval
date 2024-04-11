from old_to_delete.synthetic_data import *


def test_get_miro_synthetic_data():
    num_data_pts, num_features, random_seed = 1000, 4, 42
    n_groups = 3
    prob = np.array([5, 6, 8])
    prob = prob / prob.sum()
    ratios = {'group': np.arange(n_groups),
              'group_prob': prob,
              'y_prob': [.7, .6, .65],
              'switch_pos': [.1, .2, .15],
              'switch_neg': [.2, .15, .2]}
    X, Y, A = get_miro_synthetic_data(num_data_pts, num_features, random_seed, ratios=ratios, theta=.5)
    # synth_old = get_synthetic_data_old(num_data_pts, num_features, type_ratio, t0_ratio, t1_ratio, random_seed)
    # assert (synth_old.values == synth_new.values).all()
    assert X.shape == (1000, num_features + 1)
    assert len(pd.unique(A)) == n_groups
    assert (A[:20] == [2, 1, 2, 2, 0, 2, 2, 2, 0, 1, 1, 2, 2, 2, 1, 0, 1, 0, 2, 2]).all()
    assert (Y[:20] == [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1]).all()
    assert (X[:20].values == [[2, 0, 0, 1, 0], [1, 0, 1, 1, 0], [2, 0, 1, 0, 1], [2, 0, 0, 1, 0], [0, 1, 0, 0, 0],
                      [2, 1, 0, 1, 0], [2, 0, 0, 0, 1], [2, 0, 1, 0, 1], [0, 0, 0, 1, 1], [1, 0, 1, 1, 0],
                      [1, 0, 1, 1, 0], [2, 0, 0, 1, 0], [2, 1, 0, 1, 0], [2, 0, 0, 0, 1], [1, 1, 0, 0, 0],
                      [0, 1, 0, 0, 1], [1, 1, 1, 1, 1], [0, 1, 0, 0, 1], [2, 1, 1, 1, 1], [2, 0, 1, 1, 0]]).all()
    X, Y, A = get_miro_synthetic_data(num_data_pts, num_features, random_seed + 1, ratios=ratios, theta=.5)
    assert (A[:20] != [2, 1, 2, 2, 0, 2, 2, 2, 0, 1, 1, 2, 2, 2, 1, 0, 1, 0, 2, 2]).any()
    assert (Y[:20] != [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1]).any()
    assert (X[:20].values != [[2, 0, 0, 1, 0], [1, 0, 1, 1, 0], [2, 0, 1, 0, 1], [2, 0, 0, 1, 0], [0, 1, 0, 0, 0],
                              [2, 1, 0, 1, 0], [2, 0, 0, 0, 1], [2, 0, 1, 0, 1], [0, 0, 0, 1, 1], [1, 0, 1, 1, 0],
                              [1, 0, 1, 1, 0], [2, 0, 0, 1, 0], [2, 1, 0, 1, 0], [2, 0, 0, 0, 1], [1, 1, 0, 0, 0],
                              [0, 1, 0, 0, 1], [1, 1, 1, 1, 1], [0, 1, 0, 0, 1], [2, 1, 1, 1, 1],
                              [2, 0, 1, 1, 0]]).any()


if __name__ == '__main__':
    test_get_miro_synthetic_data()
