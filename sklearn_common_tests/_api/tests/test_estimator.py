from copy import deepcopy

from sklearn.base import BaseEstimator
from sklearn.utils._testing import raises

from sklearn_common_tests._minimal_estimator import (
    EstimatorWithGetSetParams,
    EstimatorWithSklearnClone,
    EstimatorArgsOptionalArgs,
    EstimatorWithFit,
)

from sklearn_common_tests._api.estimator import (
    check_estimator_api_clone,
    check_estimator_api_parameter_init,
    check_estimator_api_fit,
)


def test_check_estimator_api_clone():
    """Check that estimator implementing the scikit-learn cloning API are passing the
    cloning test.
    """
    estimator = EstimatorWithGetSetParams()
    check_estimator_api_clone(estimator.__class__.__name__, estimator)

    estimator = EstimatorWithSklearnClone()
    check_estimator_api_clone(estimator.__class__.__name__, estimator)


class EstimatorNoGetSetParams:
    """Does not implement any cloning interface."""

    def __init__(self, param=None):
        self.param = param


class EstimatorWrongSklearnClone:
    """Estimator does not return an instance of the same class when cloning."""

    def __init__(self, param=None):
        self.param = param

    def __sklearn_clone__(self):
        return "xxx"


def test_check_estimator_api_clone_error():
    """Check handling of estimators that does not implement or wrongly implement the
    scikit-learn cloning API.
    """
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


class EstimatorWithPrivateAttributes(EstimatorArgsOptionalArgs):
    """Estimator storing private attributes at `__init__`."""

    def __init__(self, arg1, *, arg2=None):
        super().__init__(arg1=arg1, arg2=arg2)
        self._private_attribute = 1


def test_check_estimator_api_parameter_init():
    """Check that estimator implementing the regular parameter init are passing the
    parameter init test.
    """
    estimator = EstimatorArgsOptionalArgs(1, arg2=2)
    check_estimator_api_parameter_init(estimator.__class__.__name__, estimator)

    estimator = EstimatorWithPrivateAttributes(1, arg2=2)
    check_estimator_api_parameter_init(estimator.__class__.__name__, estimator)


class EstimatorNotStoringParams(BaseEstimator):
    """Estimator that does not store attributes at `__init__`."""

    def __init__(self, param=None):
        pass


class EstimatorAdditionalParams(BaseEstimator):
    """Estimator that stores additional attributes at `__init__`."""

    def __init__(self, param=None):
        self.param = param
        self.additional_param = 1


class EstimatorCopyingInInit(BaseEstimator):
    """Estimator that validate attribute in `__init__`."""

    def __init__(self, param):
        self.param = deepcopy(param)


class EstimatorModifyInitAttributes(BaseEstimator):
    """Estimator that validate attribute in `__init__`."""

    def __init__(self, param):
        if not isinstance(param, list):
            raise ValueError(
                "`param` should be a list for the purpose of this estimator"
            )
        self.param = param
        del self.param[0]


def test_check_estimator_api_parameter_init_error():
    """Check handling of estimators that does not implement or wrongly implement the
    regular parameter init.
    """
    err_msg = "should store all parameters as an attribute during init."
    estimator = EstimatorNotStoringParams()
    with raises(AttributeError, match=err_msg):
        check_estimator_api_parameter_init(estimator.__class__.__name__, estimator)

    err_msg = "should not set any attribute apart from parameters during init."
    estimator = EstimatorAdditionalParams()
    with raises(AssertionError, match=err_msg):
        check_estimator_api_parameter_init(estimator.__class__.__name__, estimator)

    err_msg = "should not modify the input attribute in any ways"
    estimator = EstimatorCopyingInInit(param=[1, 2, 3])
    with raises(AssertionError, match=err_msg):
        check_estimator_api_parameter_init(estimator.__class__.__name__, estimator)

    # TODO: not able to detect this case
    # estimator = EstimatorModifyInitAttributes(param=[1, 2, 3])
    # with raises(AssertionError, match=err_msg):
    #     check_estimator_api_parameter_init(estimator.__class__.__name__, estimator)


def test_check_estimator_api_fit():
    """Check that estimator implementing the regular fit API specs are passing the
    fit test.
    """
    estimator = EstimatorWithFit(param=None)
    check_estimator_api_fit(estimator.__class__.__name__, estimator)


class EstimatorNotImplementingFit(EstimatorWithGetSetParams):
    """Estimator that does not implement the fit method."""


def test_check_estimator_api_fit_error():
    """Check that estimator not implementing or wrongly implement the fit API specs are
    failing the fit test.
    """
    err_msg = "does not implement a `fit` method"
    estimator = EstimatorNotImplementingFit()
    with raises(AssertionError, match=err_msg):
        check_estimator_api_fit(estimator.__class__.__name__, estimator)
