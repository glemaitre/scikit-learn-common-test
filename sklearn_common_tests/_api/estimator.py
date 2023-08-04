from inspect import signature
from itertools import chain

import numpy as np

from sklearn.base import clone
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import assert_array_equal


def yield_estimator_api_checks(estimator):
    yield check_estimator_api_clone
    yield check_estimator_api_parameter_init
    yield check_estimator_api_fit


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


def _get_init_params_from_signature(estimator):
    """Function to get the parameters of the `__init__` method."""
    # For deprecated classes, `getattr` on `__init__` will get `"deprecated_original"`.
    # We need to workaround.
    init_method = getattr(estimator.__init__, "deprecated_original", estimator.__init__)
    try:
        init_params = signature(init_method).parameters
    except (TypeError, ValueError):
        # Error on builtin C function and Mixin classes
        return []

    # filters parameters such that it is:
    # - not `self`;
    # - not `*args`;
    # - not `**kwargs`.
    init_params = [
        param
        for param in init_params.values()
        if param.name not in ("self", "obj")  # "obj" is an additional parameter in PyPy
        and param.kind != param.VAR_POSITIONAL
        and param.kind != param.VAR_KEYWORD
    ]
    return init_params


def check_estimator_api_parameter_init(name, estimator):
    """Check that an estimator passes the API regarding the `__init__` parameters.

    API specs defined here:
    https://scikit-learn.org/dev/developers/develop.html#parameters_init
    """
    # Check that we generally can clone the estimator. `clone` catch two errors:
    # - if an attribute from the `__init__` is not stored in `self`;
    # - if the `id` of an object is change in `__init__`.
    try:
        clone(estimator)
    except RuntimeError as exc:
        raise AssertionError(
            f"Estimator {name} should not modify the input attribute in any ways. "
            "Refer to the following development guide to implement the expected API: "
            "https://scikit-learn.org/dev/developers/develop.html#parameters_init"
        ) from exc
    except AttributeError as exc:
        raise AttributeError(
            f"Estimator {name} should store all parameters as an attribute during init."
            " Refer to the following development guide to implement the expected API: "
            "https://scikit-learn.org/dev/developers/develop.html#parameters_init"
        ) from exc

    estimator_init_params = _get_init_params_from_signature(estimator)
    estimator_parent_init_params = chain.from_iterable(
        _get_init_params_from_signature(parent)
        for parent in estimator.__class__.__mro__
    )

    # Check that we only set attributes with the same name as in the parameters of the
    # `__init__` method. We tolerate to have additional private attributes starting
    # with `_` as they are not considered as part of the public API.
    estimator_init_params_names = set(param.name for param in estimator_init_params)
    estimator_parent_init_params_names = set(
        param.name for param in estimator_parent_init_params
    )
    invalid_attributes = (
        set(vars(estimator))
        - set(estimator_init_params_names)
        - set(estimator_parent_init_params_names)
    )
    invalid_attributes = set(
        [attr for attr in invalid_attributes if not attr.startswith("_")]
    )
    assert not invalid_attributes, (
        f"Estimator {name} should not set any attribute apart from parameters during "
        f"init. Found attributes {sorted(invalid_attributes)}."
    )

    # Check the constraint apply for the init parameters. They should have:
    # - a default value;
    # - belong to a certain type;
    # - not be mutated during the `__init__` call.
    allowed_types = {
        str,
        int,
        float,
        bool,
        tuple,
        type(None),
        type,
    }
    # Any numpy numeric such as np.int32.
    allowed_types.update(np.core.numerictypes.allTypes.values())
    # filter the non-default arguments
    estimator_init_params = estimator_init_params[
        len(getattr(estimator, "_required_parameters", [])) :
    ]
    estimator_get_params = estimator.get_params()
    for param in estimator_init_params:
        # check that init parameters have a default value
        assert (
            param.default != param.empty
        ), f"Parameter {param.name} for {name} has no default value"
        # check that the init parameter type is allowed
        allowed_value = (
            type(param.default) in allowed_types
            or
            # Although callables are mutable, we accept them as argument
            # default value and trust that neither the implementation of
            # the callable nor of the estimator changes the state of the
            # callable.
            callable(param.default)
        )

        assert allowed_value, (
            f"Parameter '{param.name}' of estimator "
            f"'{name}' is of type "
            f"{type(param.default).__name__} which is not allowed. "
            f"'{param.name}' must be a callable or must be of type "
            f"{set(type.__name__ for type in allowed_types)}."
        )

        param_value_from_get_params = estimator_get_params[param.name]
        if isinstance(param_value_from_get_params, np.ndarray):
            assert_array_equal(param_value_from_get_params, param.default)
        else:
            failure_text = (
                f"Parameter {param.name} was mutated on init. All parameters must be "
                "stored unchanged."
            )
            if is_scalar_nan(param_value_from_get_params):
                # Allows to set default parameters to np.nan
                assert param_value_from_get_params is param.default, failure_text
            else:
                assert param_value_from_get_params == param.default, failure_text


# def check_parameters_default_constructible(name, estimator):
#     # test default-constructibility
#     # get rid of deprecation warnings

#     Estimator = estimator.__class__

#     # # test cloning
#     # clone(estimator)
#     # # test __repr__
#     # repr(estimator)
#     # # test that set_params returns self
#     # assert estimator.set_params() is estimator

#     # test if init does nothing but set parameters
#     # this is important for grid_search etc.
#     # We get the default parameters from init and then
#     # compare these against the actual values of the attributes.

#     # this comes from getattr. Gets rid of deprecation decorator.
#     init = getattr(estimator.__init__, "deprecated_original", estimator.__init__)

#     try:

#         def param_filter(p):
#             """Identify hyper parameters of an estimator."""
#             return (
#                 p.name != "self"
#                 and p.kind != p.VAR_KEYWORD
#                 and p.kind != p.VAR_POSITIONAL
#             )

#         init_params = [
#             p for p in signature(init).parameters.values() if param_filter(p)
#         ]

#     except (TypeError, ValueError):
#         # init is not a python function.
#         # true for mixins
#         return
#     params = estimator.get_params()
#     # they can need a non-default argument
#     init_params = init_params[len(getattr(estimator, "_required_parameters", [])) :]

#     for init_param in init_params:
#         assert (
#             init_param.default != init_param.empty
#         ), "parameter %s for %s has no default value" % (
#             init_param.name,
#             type(estimator).__name__,
#         )
#         allowed_types = {
#             str,
#             int,
#             float,
#             bool,
#             tuple,
#             type(None),
#             type,
#         }
#         # Any numpy numeric such as np.int32.
#         allowed_types.update(np.core.numerictypes.allTypes.values())

#         allowed_value = (
#             type(init_param.default) in allowed_types
#             or
#             # Although callables are mutable, we accept them as argument
#             # default value and trust that neither the implementation of
#             # the callable nor of the estimator changes the state of the
#             # callable.
#             callable(init_param.default)
#         )

#         assert allowed_value, (
#             f"Parameter '{init_param.name}' of estimator "
#             f"'{Estimator.__name__}' is of type "
#             f"{type(init_param.default).__name__} which is not allowed. "
#             f"'{init_param.name}' must be a callable or must be of type "
#             f"{set(type.__name__ for type in allowed_types)}."
#         )
#         if init_param.name not in params.keys():
#             # deprecated parameter, not in get_params
#             assert init_param.default is None, (
#                 f"Estimator parameter '{init_param.name}' of estimator "
#                 f"'{Estimator.__name__}' is not returned by get_params. "
#                 "If it is deprecated, set its default value to None."
#             )
#             continue

#         param_value = params[init_param.name]
#         if isinstance(param_value, np.ndarray):
#             assert_array_equal(param_value, init_param.default)
#         else:
#             failure_text = (
#                 f"Parameter {init_param.name} was mutated on init. All "
#                 "parameters must be stored unchanged."
#             )
#             if is_scalar_nan(param_value):
#                 # Allows to set default parameters to np.nan
#                 assert param_value is init_param.default, failure_text
#             else:
#                 assert param_value == init_param.default, failure_text


def check_estimator_api_fit(name, estimator):
    """Check that an estimator passes the API specification for the `fit` method.

    API specs defined here:
    https://scikit-learn.org/dev/developers/develop.html#fit_api
    """
    if not hasattr(estimator, "fit"):
        raise AssertionError(
            f"Estimator {name} does not implement a `fit` method. "
            "Refer to the following development guide to implement the expected API: "
            "https://scikit-learn.org/dev/developers/develop.html#fit_api"
        )


# @ignore_warnings(category=FutureWarning)
# def check_dont_overwrite_parameters(name, estimator_orig):
#     # check that fit method only changes or sets private attributes
#     if hasattr(estimator_orig.__init__, "deprecated_original"):
#         # to not check deprecated classes
#         return
#     estimator = clone(estimator_orig)
#     rnd = np.random.RandomState(0)
#     X = 3 * rnd.uniform(size=(20, 3))
#     X = _enforce_estimator_tags_X(estimator_orig, X)
#     y = X[:, 0].astype(int)
#     y = _enforce_estimator_tags_y(estimator, y)

#     if hasattr(estimator, "n_components"):
#         estimator.n_components = 1
#     if hasattr(estimator, "n_clusters"):
#         estimator.n_clusters = 1

#     set_random_state(estimator, 1)
#     dict_before_fit = estimator.__dict__.copy()
#     estimator.fit(X, y)

#     dict_after_fit = estimator.__dict__

#     public_keys_after_fit = [
#         key for key in dict_after_fit.keys() if _is_public_parameter(key)
#     ]

#     attrs_added_by_fit = [
#         key for key in public_keys_after_fit if key not in dict_before_fit.keys()
#     ]

#     # check that fit doesn't add any public attribute
#     assert not attrs_added_by_fit, (
#         "Estimator adds public attribute(s) during"
#         " the fit method."
#         " Estimators are only allowed to add private attributes"
#         " either started with _ or ended"
#         " with _ but %s added"
#         % ", ".join(attrs_added_by_fit)
#     )

#     # check that fit doesn't change any public attribute
#     attrs_changed_by_fit = [
#         key
#         for key in public_keys_after_fit
#         if (dict_before_fit[key] is not dict_after_fit[key])
#     ]

#     assert not attrs_changed_by_fit, (
#         "Estimator changes public attribute(s) during"
#         " the fit method. Estimators are only allowed"
#         " to change attributes started"
#         " or ended with _, but"
#         " %s changed"
#         % ", ".join(attrs_changed_by_fit)
#     )
