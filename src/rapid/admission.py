"""Response-window admission for proactive expansion.

This module implements the rule described in the manuscript section
"Response-Window Admission for Proactive Expansion".  The rule is an optional
overlay on top of IPDG.  While a cloud response is still pending, the runtime
admits another proactive expansion step only when the predicted step cost fits
the remaining predicted response window.  The CAAC length plan is unchanged.

Both predictors use an exponential moving average.  The rule is a soft rule:
it does not guarantee a deadline, and it can stop a step that would have
produced useful tokens.  Once the response has arrived the rule no longer
applies, so a root match can still expand to the planned length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


@dataclass(frozen=True, slots=True)
class WindowAdmissionConfig:
    """Predictor weights and safety fraction for the admission rule.

    ``window_weight`` and ``step_weight`` are the EMA observation weights for
    the response window and the proactive step cost.  ``safety_fraction`` is
    the fraction of the predicted window that a new step may occupy.
    """

    window_weight: float = 0.3
    step_weight: float = 0.3
    safety_fraction: float = 0.9

    def __post_init__(self) -> None:
        for name, value in (
            ("window_weight", self.window_weight),
            ("step_weight", self.step_weight),
        ):
            if not isinstance(value, Real) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if not isinstance(self.safety_fraction, Real) or isinstance(self.safety_fraction, bool):
            raise TypeError("safety_fraction must be a real number")
        safety = float(self.safety_fraction)
        if not math.isfinite(safety) or not 0.0 < safety <= 1.0:
            raise ValueError("safety_fraction must be in (0, 1]")


class WindowAdmissionController:
    """Online predictors and the per-boundary admission decision."""

    def __init__(self, config: WindowAdmissionConfig | None = None) -> None:
        self._config = config if config is not None else WindowAdmissionConfig()
        self._window_estimate: float | None = None
        self._step_estimate: float | None = None
        self._window_observations = 0
        self._step_observations = 0
        self._admitted = 0
        self._blocked = 0

    @property
    def config(self) -> WindowAdmissionConfig:
        return self._config

    @property
    def ready(self) -> bool:
        """Whether both predictors have at least one observation."""

        return self._window_estimate is not None and self._step_estimate is not None

    @property
    def window_estimate(self) -> float | None:
        return self._window_estimate

    @property
    def step_estimate(self) -> float | None:
        return self._step_estimate

    @property
    def counters(self) -> tuple[int, int, int, int]:
        """Return (window observations, step observations, admitted, blocked)."""

        return (
            self._window_observations,
            self._step_observations,
            self._admitted,
            self._blocked,
        )

    def observe_window(self, seconds: float) -> None:
        """Record one completed response window."""

        value = _positive_seconds(seconds, "response window")
        self._window_estimate = _ema(self._window_estimate, value, self._config.window_weight)
        self._window_observations += 1

    def observe_step_cost(self, seconds: float) -> None:
        """Record the cost of one completed proactive expansion step."""

        value = _positive_seconds(seconds, "step cost")
        self._step_estimate = _ema(self._step_estimate, value, self._config.step_weight)
        self._step_observations += 1

    def should_admit(
        self,
        *,
        now: float,
        request_start: float,
        steps_completed: int,
        planned_depth: int,
    ) -> bool:
        """Return whether one more proactive step may start.

        The decision follows the manuscript rule.  Before both predictors are
        initialized the runtime follows the original policy, so the rule admits
        the step.  A caller that has already reached the planned length must not
        call this method, and the method returns ``False`` defensively.
        """

        for name, value in (("now", now), ("request_start", request_start)):
            if not isinstance(value, Real) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name, value in (
            ("steps_completed", steps_completed),
            ("planned_depth", planned_depth),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if planned_depth < 1:
            raise ValueError("planned_depth must be at least one")
        if steps_completed < 0:
            raise ValueError("steps_completed must be nonnegative")
        if steps_completed >= planned_depth:
            return False
        window_estimate = self._window_estimate
        step_estimate = self._step_estimate
        if window_estimate is None or step_estimate is None:
            return True

        elapsed = max(0.0, float(now) - float(request_start))
        remaining = max(0.0, self._config.safety_fraction * window_estimate - elapsed)
        admitted = step_estimate <= remaining
        if admitted:
            self._admitted += 1
        else:
            self._blocked += 1
        return admitted


def _positive_seconds(value: float, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{label} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return value


def _ema(previous: float | None, sample: float, weight: float) -> float:
    if previous is None:
        return sample
    return weight * sample + (1.0 - weight) * previous
