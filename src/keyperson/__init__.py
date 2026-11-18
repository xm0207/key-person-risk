"""Key-person and CEO-event studies: a general event-study engine with single-firm (SQ) and
cross-sectional tests, built on the market model of mega7-decomposition.

    returns (any source) -> EventStudyEngine -> windows table -> measures (priced-in share, drift)
                                                               -> cross_section (t, BMP, Kolari-Pynnönen, sign)
"""

from .dates import day_zero
from .engine import DEFAULT_WINDOWS, EstimationRule, EventResult, EventStudyEngine, ModelFit, Window, fit_model
from .sq_test import SQResult, reject, sq_test, window_sums

__all__ = [
    "DEFAULT_WINDOWS",
    "EstimationRule",
    "EventResult",
    "EventStudyEngine",
    "ModelFit",
    "SQResult",
    "Window",
    "day_zero",
    "fit_model",
    "reject",
    "sq_test",
    "window_sums",
]
