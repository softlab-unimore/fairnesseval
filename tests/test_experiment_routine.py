from run import to_arg
import numpy as np

def test_to_arg():
    list_p = ['p1', 'parameter-number-two']
    dict_p = {'-n': 1000,
              '--sample_seeds': np.arange(10)}
    original_argv = ['caio']
    res = to_arg(list_p, dict_p,
                             original_argv)
    assert res == ['caio', 'p1', 'parameter-number-two',
                   '-n=1000', '--sample_seeds=0,1,2,3,4,5,6,7,8,9']
