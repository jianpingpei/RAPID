"""Expansion-boundary stopping for Interruptible Proactive Draft Generation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from .admission import WindowAdmissionController

StateT = TypeVar("StateT")
RootT = TypeVar("RootT")
ResponseT = TypeVar("ResponseT")


@dataclass(frozen=True, slots=True)
class IPDGExpansionResult(Generic[StateT, ResponseT]):
    """State reached by IPDG's expansion loop."""

    state: StateT
    response: ResponseT | None
    expansions_completed: int
    stopped_after_mismatch: bool
    paused_for_window: bool = False


def run_ipdg_expansion(
    *,
    state: StateT,
    branch_root: RootT,
    planned_depth: int,
    expand_one_step: Callable[[StateT], StateT],
    response_ready: Callable[[], bool],
    receive_response: Callable[[], ResponseT],
    root_matches: Callable[[RootT, ResponseT], bool],
    admission: WindowAdmissionController | None = None,
    clock: Callable[[], float] | None = None,
    request_start: float | None = None,
) -> IPDGExpansionResult[StateT, ResponseT]:
    """Expand a proactive branch and inspect feedback at complete boundaries.

    A response is first inspected after an expansion finishes.  A root match is
    remembered and expansion continues to ``planned_depth``.  A root mismatch
    stops the loop before the next expansion begins.  If no response is ready by
    the last boundary, reconciliation is left to the caller after it waits.
    Response callbacks must not use ``None`` as a response value because it is
    reserved here to mean that no response has been received.

    When ``admission`` is provided, the optional response-window rule is applied
    before each new step while the cloud response is still pending.  The rule
    uses ``clock`` and ``request_start`` to measure the pending time, records
    every completed response window and step cost, and can finish the loop
    early with ``paused_for_window=True``.  The caller then waits for the
    response before reconciliation.  Passing ``None`` keeps the original
    behaviour.
    """

    if not isinstance(planned_depth, int) or isinstance(planned_depth, bool):
        raise TypeError("planned_depth must be an integer")
    if planned_depth < 1:
        raise ValueError("planned_depth must be at least one")
    if admission is not None and (clock is None or request_start is None):
        raise ValueError("window admission requires clock and request_start")

    response: ResponseT | None = None
    expansions_completed = 0
    stopped_after_mismatch = False
    paused_for_window = False

    for _ in range(planned_depth):
        if admission is not None and response is None:
            assert clock is not None and request_start is not None
            if not admission.should_admit(
                now=clock(),
                request_start=request_start,
                steps_completed=expansions_completed,
                planned_depth=planned_depth,
            ):
                paused_for_window = True
                break

        step_start = clock() if clock is not None else None
        state = expand_one_step(state)
        if admission is not None and step_start is not None:
            assert clock is not None
            admission.observe_step_cost(clock() - step_start)
        expansions_completed += 1

        if response is None and response_ready():
            response = receive_response()
            if response is None:
                raise ValueError("receive_response must not return None")
            if admission is not None:
                assert clock is not None and request_start is not None
                admission.observe_window(clock() - request_start)
            if not root_matches(branch_root, response):
                stopped_after_mismatch = True
                break

    return IPDGExpansionResult(
        state=state,
        response=response,
        expansions_completed=expansions_completed,
        stopped_after_mismatch=stopped_after_mismatch,
        paused_for_window=paused_for_window,
    )
