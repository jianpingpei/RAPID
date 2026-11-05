# SPDX-License-Identifier: MIT
# RAPID reference implementation. See LICENSE and NOTICE.

"""Congestion-Aware Adaptive Controller (CAAC).

The controller keeps its draft depth as a continuous value.  Only the depth
returned to the inference engine is quantized, so rounding does not feed back
into the next PID update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class CAACConfig:
    """Parameters reported for CAAC in the RAPID paper."""

    target_acceptance: float = 0.75
    initial_acceptance: float = 0.8
    gamma: float = 0.5
    kappa: float = 0.01
    alpha: float = 0.5
    beta: float = 0.6
    kp: float = 5.0
    ki: float = 0.1
    kd: float = 1.0
    min_depth: int = 1
    max_depth: int = 8
    integral_min: float = -1.0
    integral_max: float = 1.0
    esr_cap: float = 0.99
    low_acceptance_reset: float = 0.6

    def __post_init__(self) -> None:
        scalar_values = {
            "target_acceptance": self.target_acceptance,
            "initial_acceptance": self.initial_acceptance,
            "gamma": self.gamma,
            "kappa": self.kappa,
            "alpha": self.alpha,
            "beta": self.beta,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "integral_min": self.integral_min,
            "integral_max": self.integral_max,
            "esr_cap": self.esr_cap,
            "low_acceptance_reset": self.low_acceptance_reset,
        }
        for name, value in scalar_values.items():
            _require_finite(name, value)

        if not 0.0 <= self.target_acceptance <= 1.0:
            raise ValueError("target_acceptance must be in [0, 1]")
        if not 0.0 <= self.initial_acceptance <= 1.0:
            raise ValueError("initial_acceptance must be in [0, 1]")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 < self.kappa <= 1.0:
            raise ValueError("kappa must be in (0, 1]")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must be in (0, 1]")
        if not isinstance(self.min_depth, int) or isinstance(self.min_depth, bool):
            raise TypeError("min_depth must be an integer")
        if not isinstance(self.max_depth, int) or isinstance(self.max_depth, bool):
            raise TypeError("max_depth must be an integer")
        if self.min_depth < 1 or self.max_depth < self.min_depth:
            raise ValueError("depth range must satisfy 1 <= min_depth <= max_depth")
        if self.integral_min >= self.integral_max:
            raise ValueError("integral_min must be smaller than integral_max")
        if not self.integral_min <= 0.0 <= self.integral_max:
            raise ValueError("integral bounds must contain the initial state 0.0")
        if not 0.0 <= self.esr_cap < 1.0:
            raise ValueError("esr_cap must be in [0, 1)")
        if not 0.0 <= self.low_acceptance_reset <= 1.0:
            raise ValueError("low_acceptance_reset must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CAACDecision:
    """Controller output and the intermediate values that produced it."""

    depth: int
    continuous_depth: float
    proposed_depth: float
    acceptance_ema: float
    error: float
    integral_before_reset: float
    integral_state: float
    fast_latency: float
    baseline_latency: float
    slowdown_ratio: float
    slowdown_factor: float


class CAACController:
    """Stateful CAAC controller operating in draft-depth units.

    ``initial_depth`` defaults to the lower depth bound.  The acceptance EMA
    starts at ``CAACConfig.initial_acceptance``.  Continuous depths are
    quantized with ``ceil`` because a partial depth level requires the next
    complete expansion level.  The continuous value remains unchanged by that
    quantization.
    """

    def __init__(
        self,
        config: CAACConfig | None = None,
        *,
        initial_depth: float | None = None,
        initial_acceptance: float | None = None,
    ) -> None:
        self.config = config or CAACConfig()
        selected_depth = self.config.min_depth if initial_depth is None else initial_depth
        selected_acceptance = (
            self.config.initial_acceptance if initial_acceptance is None else initial_acceptance
        )
        _require_finite("initial_depth", selected_depth)
        _require_finite("initial_acceptance", selected_acceptance)
        if not self.config.min_depth <= selected_depth <= self.config.max_depth:
            raise ValueError("initial_depth must be inside the configured depth range")
        if not 0.0 <= selected_acceptance <= 1.0:
            raise ValueError("initial_acceptance override must be in [0, 1]")

        self._continuous_depth = float(selected_depth)
        self._acceptance_ema = float(selected_acceptance)
        self._integral = 0.0
        self._previous_error = 0.0
        self._fast_latency: float | None = None
        self._baseline_latency: float | None = None

    @property
    def continuous_depth(self) -> float:
        return self._continuous_depth

    @property
    def acceptance_ema(self) -> float:
        return self._acceptance_ema

    @property
    def integral(self) -> float:
        return self._integral

    @property
    def fast_latency(self) -> float | None:
        return self._fast_latency

    @property
    def baseline_latency(self) -> float | None:
        return self._baseline_latency

    def update(self, current_acceptance: float, per_token_latency: float) -> CAACDecision:
        """Consume one round's measurements and choose the next draft depth.

        Args:
            current_acceptance: Acceptance measured for the completed round.
            per_token_latency: Local generation time divided by drafted tokens.

        Returns:
            The integer depth for the next round plus formula-level diagnostics.
        """

        _require_finite("current_acceptance", current_acceptance)
        _require_finite("per_token_latency", per_token_latency)
        if not 0.0 <= current_acceptance <= 1.0:
            raise ValueError("current_acceptance must be in [0, 1]")
        if per_token_latency <= 0.0:
            raise ValueError("per_token_latency must be positive")

        config = self.config

        acceptance_ema = (
            config.beta * current_acceptance + (1.0 - config.beta) * self._acceptance_ema
        )
        error = acceptance_ema - config.target_acceptance
        integral_before_reset = _clip(
            self._integral + error,
            config.integral_min,
            config.integral_max,
        )
        derivative = error - self._previous_error
        proposed_depth = (
            self._continuous_depth
            + config.kp * error
            + config.ki * integral_before_reset
            + config.kd * derivative
        )

        if self._fast_latency is None:
            fast_latency = per_token_latency
            baseline_latency = per_token_latency
        else:
            fast_latency = (
                config.gamma * per_token_latency + (1.0 - config.gamma) * self._fast_latency
            )
            previous_baseline = self._baseline_latency
            if previous_baseline is None:
                raise RuntimeError("baseline latency must be initialized with fast latency")
            if fast_latency < previous_baseline:
                baseline_latency = fast_latency
            else:
                baseline_latency = previous_baseline + config.kappa * (
                    fast_latency - previous_baseline
                )

        slowdown_ratio = max(0.0, 1.0 - baseline_latency / fast_latency)
        slowdown_ratio = min(config.esr_cap, slowdown_ratio)
        slowdown_factor = 1.0 / (1.0 + config.alpha * slowdown_ratio / (1.0 - slowdown_ratio))
        continuous_depth = _clip(
            proposed_depth * slowdown_factor,
            float(config.min_depth),
            float(config.max_depth),
        )

        # This reset occurs after the current proposal, as described in the paper.
        integral_state = (
            0.0 if acceptance_ema < config.low_acceptance_reset else integral_before_reset
        )

        self._continuous_depth = continuous_depth
        self._acceptance_ema = acceptance_ema
        self._integral = integral_state
        self._previous_error = error
        self._fast_latency = fast_latency
        self._baseline_latency = baseline_latency

        depth = math.ceil(continuous_depth)
        return CAACDecision(
            depth=depth,
            continuous_depth=continuous_depth,
            proposed_depth=proposed_depth,
            acceptance_ema=acceptance_ema,
            error=error,
            integral_before_reset=integral_before_reset,
            integral_state=integral_state,
            fast_latency=fast_latency,
            baseline_latency=baseline_latency,
            slowdown_ratio=slowdown_ratio,
            slowdown_factor=slowdown_factor,
        )
