from functools import partial

from sklearn.utils.estimator_checks import (
    _get_check_estimator_ids,
    _maybe_mark_xfail,
)


def parametrize_with_checks(estimators, checks):
    import pytest

    def checks_generator():
        for estimator in estimators:
            name = type(estimator).__name__
            for check in checks(estimator):
                check = partial(check, name)
                yield _maybe_mark_xfail(estimator, check, pytest)

    return pytest.mark.parametrize(
        "estimator, check", checks_generator(), ids=_get_check_estimator_ids
    )
