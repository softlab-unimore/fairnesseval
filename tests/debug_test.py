# from test_save_model import TestSaveModels
#
# if __name__ == '__main__':
#     test = TestSaveModels()
#     test.setUp()
#     test.test_save_model()
#     test.tearDown()

from test_models import TestModels

if __name__ == '__main__':
    test = TestModels()
    test.setUp()
    test.test_expgrad_and_base_model_grid_params()
    test.tearDown()