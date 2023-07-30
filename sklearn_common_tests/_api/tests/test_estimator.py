from sklearn.utils._testing import raises

from sklearn_common_tests._minimal_estimator import (
    EstimatorWithGetSetParams,
    EstimatorWithSklearnClone,
)

from sklearn_common_tests._api.estimator import check_estimator_api_clone


class EstimatorNoGetSetParams:
    def __init__(self, param=None):
        self.param = param


class EstimatorWrongSklearnClone:
    def __init__(self, param=None):
        self.param = param

    def __sklearn_clone__(self):
        return "xxx"


def test_check_estimator_api_clone():
    """Check that estimator implementing the scikit-learn cloning API are passing the
    cloning test.
    """
    estimator = EstimatorWithGetSetParams()
    check_estimator_api_clone(estimator.__class__.__name__, estimator)

    estimator = EstimatorWithSklearnClone()
    check_estimator_api_clone(estimator.__class__.__name__, estimator)


def test_check_estimator_api_clone_error():
    """Check error message of estimators"""
    err_msg = "does not implement a 'get_params' method"
    estimator = EstimatorNoGetSetParams()
    with raises(TypeError, match=err_msg):
        check_estimator_api_clone(estimator.__class__.__name__, estimator)

    err_msg = (
        "Cloning an estimator should return an estimator instance of the same class"
    )
    estimator = EstimatorWrongSklearnClone()
    with raises(AssertionError, match=err_msg):
        check_estimator_api_clone(estimator.__class__.__name__, estimator)
