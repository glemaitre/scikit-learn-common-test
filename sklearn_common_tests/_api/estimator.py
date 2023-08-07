from copy import deepcopy
from inspect import signature
from itertools import chain, product
from pprint import pformat
from queue import LifoQueue

import numpy as np

from sklearn.base import clone
from sklearn.utils import is_scalar_nan


def yield_estimator_api_checks(estimator):
    yield check_estimator_api_clone
    yield check_estimator_api_parameter_init
    yield check_estimator_api_get_params
    yield check_estimator_api_set_params
    yield check_estimator_api_round_trip_get_set_params
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
            " Refer to the following development guide to implement the expected API: "
            "https://scikit-learn.org/dev/developers/develop.html#parameters_init"
        )

        param_value_from_get_params = estimator_get_params[param.name]
        failure_text = (
            f"Parameter {param.name} was mutated on init. All parameters must be "
            "stored unchanged. Refer to the following development guide to implement "
            "the expected API: "
            "https://scikit-learn.org/dev/developers/develop.html#parameters_init"
        )
        if is_scalar_nan(param_value_from_get_params):
            # Allows to set default parameters to np.nan
            assert param_value_from_get_params is param.default, failure_text
        else:
            assert param_value_from_get_params == param.default, failure_text


def check_estimator_api_get_params(name, estimator):
    """Check that the estimator passes the API specification for `get_params` method.

    API specs defined here:
    https://scikit-learn.org/dev/developers/develop.html#get_set_params
    """
    # check the general API of `get_params`:
    # - it is implemented
    # - as a `deep` optional parameter
    # - with a default value of `True`
    if not hasattr(estimator, "get_params"):
        raise AssertionError(
            f"Estimator {name} should have a `get_params` method. "
            "Refer to the following development guide to implement the expected API: "
            "https://scikit-learn.org/dev/developers/develop.html#get_set_params"
        )

    get_params_signature = signature(estimator.get_params)
    assert "deep" in get_params_signature.parameters, (
        f"Estimator {name} implements a `get_params` method. However this method does "
        "not have a `deep` optional parameter. This parameter should be set to True by "
        "by default."
        "Refer to the following development guide to implement the expected API: "
        "https://scikit-learn.org/dev/developers/develop.html#get_set_params"
    )
    assert get_params_signature.parameters["deep"].default is True, (
        f"Estimator {name} implements a `get_params` method with the `deep` optional "
        "parameter. However this parameter is not set to True by default and it is "
        f"instead set to {get_params_signature.parameters['deep'].default}. "
        "Refer to the following development guide to implement the expected API: "
        "https://scikit-learn.org/dev/developers/develop.html#get_set_params"
    )

    # check the consistency between `__init__` and the output of
    # `get_params(deep=False)`
    estimator_init_params = _get_init_params_from_signature(estimator)
    estimator_get_params_shallow = estimator.get_params(deep=False)

    estimator_init_params_names = set(param.name for param in estimator_init_params)
    estimator_get_params_names = set(estimator_get_params_shallow.keys())
    if estimator_init_params_names != estimator_get_params_names:
        missing_get_params = estimator_init_params_names - estimator_get_params_names
        additional_get_params = estimator_get_params_names - estimator_init_params_names
        msg = (
            "The not an exact matching of the parameters between the "
            "`__init__` method and the `get_params` method."
        )
        if missing_get_params:
            msg += (
                f" The following parameters are defined in the `__init__` method "
                "but are missing from the `get_params` method: "
                f"{sorted(missing_get_params)}."
            )
        if additional_get_params:
            msg += (
                f" The following parameters are returned by the `get_params` "
                "method but are missing from the `__init__` method: "
                f"{sorted(additional_get_params)}."
            )
        msg += (
            " Refer to the following development guide to implement the expected "
            "API:"
            "https://scikit-learn.org/dev/developers/develop.html#get_set_params"
        )
        raise AssertionError(msg)

    # check the consistency between `get_params(deep=False)` and
    # `get_params(deep=True)`
    estimator_get_params_deep = estimator.get_params(deep=True)
    assert all(
        item in estimator_get_params_deep.items()
        for item in estimator_get_params_shallow.items()
    ), (
        f"For estimator {name}, the parameters returned by `get_params` with "
        "`deep=True` is not subset of the ones returned by `get_params` with "
        "`deep=False`."
        " Refer to the following development guide to implement the expected API: "
        "https://scikit-learn.org/dev/developers/develop.html#get_set_params"
    )

    # When using `get_params(deep=True)`, we need to recurse estimators to make sure
    # that we show all parameters. This test does not handle list of estimators as
    # parameters. This case is usually linked with the estimator having a private
    # attribute `_required_params`.
    # TODO: do we want to handle such composition case? This is somehow tested in
    # round-trip get_params -> set_params -> get_params.
    if not hasattr(estimator, "_required_parameters"):
        # this is an alternative implementation of `get_params` that uses a
        # LIFO queue and store any estimator to get parameters from in the
        # queue.
        nested_estimator = LifoQueue()
        # initialize the queue with the top-level estimator
        nested_estimator.put((name, estimator))
        expected_params = estimator.get_params(deep=False)
        nesting_level = 0  # later on used to preprend the name of the parameter
        while not nested_estimator.empty():
            est_name, est = nested_estimator.get()
            est_params = est.get_params(deep=False)
            for param_name, param_value in est_params.items():
                pname = f"{est_name}__{param_name}" if nesting_level > 0 else param_name
                expected_params[pname] = param_value
                if hasattr(param_value, "get_params"):
                    # to be recurse in a later iteration
                    nested_estimator.put((pname, param_value))
            nesting_level += 1

        assert estimator.get_params(deep=True) == expected_params, (
            f"For estimator {name}, the parameters returned by `get_params(deep=True)` "
            "are incorrect. We would expect the following parameters:\n"
            f"{pformat(expected_params)}\n"
            " Refer to the following development guide to implement the expected API: "
            "https://scikit-learn.org/dev/developers/develop.html#get_set_params"
        )


def check_estimator_api_set_params(name, estimator):
    """Check that the estimator passes the API specification for `set_params` method.

    API specs defined here:
    https://scikit-learn.org/dev/developers/develop.html#get_set_params
    """
    if not hasattr(estimator, "set_params"):
        raise AssertionError(
            f"Estimator {name} should have a `set_params` method. "
            "Refer to the following development guide to implement the expected API: "
            "https://scikit-learn.org/dev/developers/develop.html#get_set_params"
        )

    # check that set_params returns self
    # make a deep clone since `set_params` could modify the state of the estimator
    cloned_estimator = deepcopy(estimator)
    assert cloned_estimator.set_params() is cloned_estimator, (
        f"Estimator {name} does not return `self` from `set_params`. "
        "Refer to the following development guide to implement the expected API: "
        "https://scikit-learn.org/dev/developers/develop.html#get_set_params"
    )

    all_param_names = estimator.get_params(deep=False).keys()
    # Try different type of values that we can easily detect. Since there is no
    # validation in `set_params` we can pass anything.
    test_values = [-np.inf, np.inf, None]

    for param_name, test_value in product(all_param_names, test_values):
        # make a deepcopy of the original estimator since we are going to change nested
        # parameter that will not be copied using `clone`
        cloned_estimator = deepcopy(estimator)
        original_params = cloned_estimator.get_params(deep=False).copy()
        # make a round-trip to make sure that the estimator is not modified by
        cloned_estimator.set_params(**original_params)
        # set specifically the current parameter
        cloned_estimator.set_params(**{param_name: test_value})

        current_params = cloned_estimator.get_params(deep=False)
        original_params_name = set(original_params.keys())
        current_params_name = set(current_params.keys())
        assert original_params_name == current_params_name, (
            f"Estimator {name} does not implement properly `get_params` and"
            "`set_params`. After setting parameters, `get_params` does not return the "
            "same set of parameters. The problematic parameter(s) is(are): "
            f"{original_params_name.symmetric_difference(current_params_name)}. "
            "Refer to the following development guide to implement the expected API: "
            "https://scikit-learn.org/dev/developers/develop.html#get_set_params"
        )
        msg = (
            "Estimator {0} does not implement properly `get_params` and"
            "`set_params`. The parameter `{1}` was not set properly set. The memory "
            "address of the parameter changed. "
            "Refer to the following development guide to implement the expected API: "
            "https://scikit-learn.org/dev/developers/develop.html#get_set_params"
        )
        for current_param_name, current_param_value in current_params.items():
            if current_param_name is param_name:
                assert current_param_value is test_value, msg.format(
                    name, current_param_name
                )
            else:
                assert (
                    current_param_value is original_params[current_param_name]
                ), msg.format(name, current_param_name)


def check_estimator_api_round_trip_get_set_params(name, estimator):
    """Check the API for `get_params` and `set_params` by a round-trip.

    Complementary to `set_params` test since it would test the case `deep=True`.

    API specs defined here:
    https://scikit-learn.org/dev/developers/develop.html#get_set_params
    """
    # Make an original deepcopy of the parameters that are used in `set_params`. In this
    # way, they will not be impacted by a wrong `set_params` implementation that changes
    # the internal state of the estimator.
    original_params = deepcopy(estimator.get_params(deep=True))
    estimator.set_params(**original_params)
    updated_params = estimator.get_params(deep=True)

    original_params_name = set(original_params.keys())
    updated_params_name = set(updated_params.keys())
    assert original_params_name == updated_params_name, (
        f"Estimator {name} does not implement properly `get_params` and `set_params`. "
        "The names of the parameters does not correspond after a round-trip "
        "get_params/set_params. The problematic parameter(s) is(are): "
        f"{original_params_name.symmetric_difference(updated_params_name)}. "
        "Refer to the following development guide to implement the expected API: "
        "https://scikit-learn.org/dev/developers/develop.html#get_set_params"
    )

    for updated_param_name, updated_param_value in updated_params.items():
        if (
            (required_parameters := getattr(estimator, "_required_parameters", None))
            is not None
            and updated_param_name in required_parameters
            # We want to treat differently estimator that accepts a list of tuples
            # containing estimators. Setting those will change the memory address of
            # the parameter because tuple are immutable.
            and isinstance(original_params[updated_param_name], list)
            and isinstance(original_params[updated_param_name][0], tuple)
        ):
            for tuple_updated, tuple_original in zip(
                updated_param_value, original_params[updated_param_name]
            ):
                for elt_updated, elt_original in zip(tuple_updated, tuple_original):
                    assert elt_updated is elt_original, (
                        f"Estimator {name} does not implement properly `get_params` "
                        "and `set_params`. The memory address of inner parameters has "
                        f"changed for the parameter `{updated_param_name}`. "
                        "Refer to the following development guide to implement the "
                        "expected API: https://scikit-learn.org/dev/developers/"
                        "develop.html#get_set_params"
                    )
        else:
            assert original_params[updated_param_name] is updated_param_value, (
                f"Estimator {name} does not implement properly `get_params` and "
                "`set_params`. Implementing a round-trip get_params/set_params does "
                "not return the exact same object. The memory address of the parameter "
                f"`{updated_param_name}` has changed. "
                "Refer to the following development guide to implement the expected "
                "API: https://scikit-learn.org/dev/developers/"
                "develop.html#get_set_params"
            )


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
