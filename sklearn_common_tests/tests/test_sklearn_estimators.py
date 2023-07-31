from .._api.estimator import yield_estimator_api_checks
from .._estimator_generator import _tested_estimators
from .._test_generator import parametrize_with_checks


@parametrize_with_checks(_tested_estimators(), yield_estimator_api_checks)
def test_estimator_api(estimator, check, request):
    check(estimator)
