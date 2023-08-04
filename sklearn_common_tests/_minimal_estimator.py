class EstimatorWithGetSetParams:
    """Estimator implemented the `get_params` and `set_params` interface."""

    def __init__(self, param=None):
        self.param = param

    def get_params(self, deep=True):
        return {"param": self.param}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class EstimatorWithSklearnClone:
    """Estimator implemented the `__sklearn_clone__` interface."""

    def __init__(self, param=None):
        self.param = param

    def __sklearn_clone__(self):
        return self


class EstimatorArgsOptionalArgs(EstimatorWithGetSetParams):
    """Estimator implemented the `__init__` interface with args and optional args."""

    # Since `arg1` is a required argument, it needs to be listed in
    # `_required_parameters`.
    _required_parameters = ["arg1"]

    def __init__(self, arg1, *, arg2=None):
        self.arg1 = arg1
        self.arg2 = arg2

    def get_params(self, deep=True):
        return {"arg1": self.arg1, "arg2": self.arg2}


class EstimatorWithFit(EstimatorWithGetSetParams):
    """Estimator implemented the `fit` interface."""

    def fit(self, X, y=None):
        self._is_fitted_ = True
        return self
