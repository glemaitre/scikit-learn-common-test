from sklearn.base import clone
from sklearn.utils import IS_PYPY
from sklearn.utils._testing import _get_args


def check_estimator_api_clone(name, estimator):
    """Check that an estimator passes the cloning API.

    API specs defined here:
    https://scikit-learn.org/dev/developers/develop.html#cloning
    """
    try:
        cloned_estimator = clone(estimator)
    except TypeError as exc:
        if "does not implement a 'get_params' method" in str(exc):
            # TODO: improve the error message in `_clone_parametrized` to include a link
            # to the API documentation. We re-raise this error with this information
            raise TypeError(
                str(exc)
                + " Refer to the following development guide to implement the expected "
                "API: https://scikit-learn.org/dev/developers/develop.html#cloning"
            ) from exc
        raise exc

    assert isinstance(cloned_estimator, estimator.__class__), (
        "Cloning an estimator should return an estimator instance of the same class. "
        f"Got {cloned_estimator.__class__.__name__} instead of "
        f"{name}. Refer to the following development guide to implement the expected "
        "API: https://scikit-learn.org/dev/developers/develop.html#cloning"
    )


def check_estimator_api_parameter_init(name, estimator):
    """Check that an estimator passes the API regarding the `__init__` parameters.

    API specs defined here:
    https://scikit-learn.org/dev/developers/develop.html#parameters_init
    """
    try:
        cloned_estimator = clone(estimator)
    except AttributeError as exc:
        raise AttributeError(
            f"Estimator {name} should store all parameters as an attribute during init."
            " Refer to the following development guide to implement the expected API: "
            "https://scikit-learn.org/dev/developers/develop.html#parameters_init"
        ) from exc

    init_params = _get_args(type(cloned_estimator).__init__)
    if IS_PYPY:
        # __init__ signature has additional objects in PyPy
        for key in ["obj"]:
            if key in init_params:
                init_params.remove(key)
    parents_init_params = [
        param
        for params_parent in (
            _get_args(parent) for parent in type(cloned_estimator).__mro__
        )
        for param in params_parent
    ]

    # Test for no setting apart from parameters during init
    invalid_attr = (
        set(vars(cloned_estimator)) - set(init_params) - set(parents_init_params)
    )
    # Ignore private attributes
    invalid_attr = set([attr for attr in invalid_attr if not attr.startswith("_")])
    assert not invalid_attr, (
        f"Estimator {name} should not set any attribute apart from parameters during "
        f"init. Found attributes {sorted(invalid_attr)}."
    )
