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
