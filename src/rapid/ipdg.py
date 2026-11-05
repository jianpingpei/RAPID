"""Expansion-boundary stopping for Interruptible Proactive Draft Generation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

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


def run_ipdg_expansion(
    *,
    state: StateT,
    branch_root: RootT,
    planned_depth: int,
    expand_one_step: Callable[[StateT], StateT],
    response_ready: Callable[[], bool],
    receive_response: Callable[[], ResponseT],
    root_matches: Callable[[RootT, ResponseT], bool],
) -> IPDGExpansionResult[StateT, ResponseT]:
    """Expand a proactive branch and inspect feedback at complete boundaries.

    A response is first inspected after an expansion finishes.  A root match is
    remembered and expansion continues to ``planned_depth``.  A root mismatch
    stops the loop before the next expansion begins.  If no response is ready by
    the last boundary, reconciliation is left to the caller after it waits.
    Response callbacks must not use ``None`` as a response value because it is
    reserved here to mean that no response has been received.
    """

    if not isinstance(planned_depth, int) or isinstance(planned_depth, bool):
        raise TypeError("planned_depth must be an integer")
    if planned_depth < 1:
        raise ValueError("planned_depth must be at least one")

    response: ResponseT | None = None
    expansions_completed = 0
    stopped_after_mismatch = False

    for _ in range(planned_depth):
        state = expand_one_step(state)
        expansions_completed += 1

        if response is None and response_ready():
            response = receive_response()
            if response is None:
                raise ValueError("receive_response must not return None")
            if not root_matches(branch_root, response):
                stopped_after_mismatch = True
                break

    return IPDGExpansionResult(
        state=state,
        response=response,
        expansions_completed=expansions_completed,
        stopped_after_mismatch=stopped_after_mismatch,
    )
