from copy import deepcopy

import numpy as np

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
    check_estimator_api_get_params,
    check_estimator_api_parameter_init,
    check_estimator_api_fit,
)


def test_check_estimator_api_clone():
    """Check that estimator implementing the scikit-learn cloning API are passing the
    cloning test.
    """
    for Estimator in (EstimatorWithGetSetParams, EstimatorWithSklearnClone):
        estimator = Estimator()
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
    # parametrization with a tuple of (Estimator, type_error, error_message)
    parametrize = [
        (
            EstimatorNoGetSetParams,
            TypeError,
            "does not implement a 'get_params' method",
        ),
        (
            EstimatorWrongSklearnClone,
            AssertionError,
            "Cloning an estimator should return an estimator instance of the same",
        ),
    ]

    for Estimator, type_err, err_msg in parametrize:
        estimator = Estimator()
        with raises(type_err, match=err_msg):
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
    for Estimator in (EstimatorArgsOptionalArgs, EstimatorWithPrivateAttributes):
        estimator = Estimator(1)
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


class EstimatorMutableInitAttributes(BaseEstimator):
    """Estimator that as a default mutable attribute in `__init__`."""

    def __init__(self, param=[]):
        self.param = param


class EstimatorModifyDefaultAttribute(BaseEstimator):
    """Estimator that create modify a default parameter in `__init__`."""

    def __init__(self, *, param=None, replace_by_nan=True):
        if replace_by_nan:
            self.param = np.nan
        else:
            self.param = "random"
        self.replace_by_nan = replace_by_nan


def test_check_estimator_api_parameter_init_error():
    """Check handling of estimators that does not implement or wrongly implement the
    regular parameter init.
    """
    # parametrization with a tuple (estimator, type_error, error_message)
    parametrize = [
        (
            EstimatorNotStoringParams(),
            AttributeError,
            "should store all parameters as an attribute during init.",
        ),
        (
            EstimatorAdditionalParams(),
            AssertionError,
            "should not set any attribute apart from parameters during init.",
        ),
        (
            EstimatorCopyingInInit(param=[1, 2, 3]),
            AssertionError,
            "should not modify the input attribute in any ways",
        ),
        (
            EstimatorMutableInitAttributes(),
            AssertionError,
            "is of type list which is not allowed",
        ),
        (
            EstimatorModifyDefaultAttribute(replace_by_nan=True),
            AssertionError,
            "param was mutated on init",
        ),
        (
            EstimatorModifyDefaultAttribute(replace_by_nan=False),
            AssertionError,
            "param was mutated on init",
        ),
    ]

    for estimator, type_err, err_msg in parametrize:
        with raises(type_err, match=err_msg):
            check_estimator_api_parameter_init(estimator.__class__.__name__, estimator)


def test_checK_estimator_api_get_params():
    """Check that an estimator implementing `get_params` specs does not fail."""
    for Estimator in (BaseEstimator, EstimatorWithGetSetParams):
        estimator = Estimator()
        check_estimator_api_get_params(estimator.__class__.__name__, estimator)


class EstimatorGetParamsWithoutDeep:
    """Check that an estimator with `get_params` but without the optional
    parameter `deep` fails.
    """

    def get_params(self):
        return {}


class EstimatorGetParamsDeepWrongDefault:
    """Check that an estimator with `get_params` but the wrong default for the `deep`
    parameter fails.
    """

    def get_params(self, deep=False):
        return {}


class EstimatorGetParamsNotEquivalentInit:
    """Check that an estimator that does not return the same parameters as the init
    fails.
    """

    def __init__(self, *, param=1):
        self.param = param

    def get_params(self, deep=True):
        return {"additional_param": 2}


class EstimatorGetParamsNotSubsetDeep:
    """Check that we raise if `estimator.get_params(deep=True)` is not a subset of
    `estimator.get_params(deep=False)`.
    """

    def __init__(self, *, param=1):
        self.param = param

    def get_params(self, deep=True):
        return {} if deep else {"param": 1}


class EstimatorGetParamsWrongDeepMode:
    """Check that we raise an error if `estimator.get_params(deep=True)` is not
    returning properly nested information.
    """

    def __init__(self, *, param=None, estimator=None):
        self.param = param
        self.estimator = estimator

    def get_params(self, deep=True):
        return {"param": self.param, "estimator": self.estimator}


def test_check_estimator_api_get_params_error():
    """Check that an estimator that doesn't implement the `get_params` specs fails."""
    # parametrization with a tuple (estimator, type_error, error_message)
    parametrize = [
        (
            EstimatorNoGetSetParams(),
            AssertionError,
            "should have a `get_params` method",
        ),
        (
            EstimatorGetParamsWithoutDeep(),
            AssertionError,
            "method does not have a `deep` optional parameter",
        ),
        (
            EstimatorGetParamsDeepWrongDefault(),
            AssertionError,
            "this parameter is not set to True by default",
        ),
        (
            EstimatorGetParamsNotEquivalentInit(),
            AssertionError,
            "the parameters between the `__init__` method and the `get_params` method",
        ),
        (
            EstimatorGetParamsNotSubsetDeep(),
            AssertionError,
            "is not subset of the ones returned by `get_params` with `deep=False`",
        ),
        (
            EstimatorGetParamsWrongDeepMode(estimator=EstimatorWithGetSetParams()),
            AssertionError,
            r"the parameters returned by `get_params\(deep=True\)` are incorrect",
        ),
    ]
    for estimator, type_err, err_msg in parametrize:
        with raises(type_err, match=err_msg):
            check_estimator_api_get_params(estimator.__class__.__name__, estimator)


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
