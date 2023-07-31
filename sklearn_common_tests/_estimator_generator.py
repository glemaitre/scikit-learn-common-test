import warnings

from sklearn.base import RegressorMixin
from sklearn.exceptions import SkipTestWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import (
    RANSACRegressor,
    LinearRegression,
    Ridge,
    SGDRegressor,
    LogisticRegression,
)
from sklearn.utils.estimator_checks import _construct_instance
from sklearn.utils.discovery import all_estimators
from sklearn.utils._testing import SkipTest


# These estimators are special cases and should be tested separately
SKIPPED_ESTIMATORS = [
    "ColumnTransformer",
    "FeatureUnion",
    "GridSearchCV",
    "Pipeline",
    "RandomizedSearchCV",
]


def construct_instance(Estimator):
    """Construct Estimator instance if possible."""
    required_parameters = getattr(Estimator, "_required_parameters", [])
    if len(required_parameters):
        if required_parameters in (["estimator"], ["base_estimator"]):
            # `RANSACRegressor` will raise an error with any model other
            # than `LinearRegression` if we don't fix `min_samples` parameter.
            # For common test, we can enforce using `LinearRegression` that
            # is the default estimator in `RANSACRegressor` instead of `Ridge`.
            if issubclass(Estimator, RANSACRegressor):
                estimator = Estimator(LinearRegression())
            elif issubclass(Estimator, RegressorMixin):
                estimator = Estimator(Ridge())
            elif issubclass(Estimator, SelectFromModel):
                # Increases coverage because SGDRegressor has partial_fit
                estimator = Estimator(SGDRegressor(random_state=0))
            else:
                estimator = Estimator(LogisticRegression(C=1))
        elif required_parameters in (["estimators"],):
            # Heterogeneous ensemble classes (i.e. stacking, voting)
            if issubclass(Estimator, RegressorMixin):
                estimator = Estimator(
                    estimators=[("est1", Ridge(alpha=0.1)), ("est2", Ridge(alpha=1))]
                )
            else:
                estimator = Estimator(
                    estimators=[
                        ("est1", LogisticRegression(C=0.1)),
                        ("est2", LogisticRegression(C=1)),
                    ]
                )
        else:
            # TODO: we should be able to test the SparseCoder transformer
            msg = (
                f"Can't instantiate estimator {Estimator.__name__} parameters "
                f"{required_parameters}"
            )
            warnings.warn(msg, SkipTestWarning)
            raise SkipTest(msg)
    else:
        estimator = Estimator()
    return estimator


def _tested_estimators(type_filter=None):
    for _, Estimator in all_estimators(type_filter=type_filter):
        if Estimator.__name__ in SKIPPED_ESTIMATORS:
            continue
        try:
            estimator = _construct_instance(Estimator)
        except SkipTest:
            continue

        yield estimator
