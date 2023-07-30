from sklearn.base import clone


def check_estimator_api_clone(name, estimator):
    """Check that an estimator passes the cloning API."""
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
        f"{estimator.__class__.__name__}. Refer to the following development guide to "
        "implement the expected API: "
        "https://scikit-learn.org/dev/developers/develop.html#cloning"
    )
