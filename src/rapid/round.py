"""Round-level composition of CAAC and IPDG."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Generic, Protocol, TypeVar

from .caac import CAACController, CAACDecision
from .ipdg import run_ipdg_expansion

DraftT = TypeVar("DraftT")
DraftTContra = TypeVar("DraftTContra", contravariant=True)
StateT = TypeVar("StateT")
BoundaryT = TypeVar("BoundaryT")
HandleT = TypeVar("HandleT")
RootT = TypeVar("RootT")
ResponseT = TypeVar("ResponseT")


class RapidRoundOps(Protocol[DraftTContra, StateT, BoundaryT, HandleT, RootT, ResponseT]):
    """Callbacks that connect one RAPID round to a serving runtime."""

    def save_boundary(self, state: StateT) -> BoundaryT:
        """Save aligned draft-tree and KV-cache state."""
        ...

    def send_async(self, draft: DraftTContra) -> HandleT:
        """Start cloud verification without blocking local generation."""
        ...

    def start_proactive_branch(self, state: StateT) -> tuple[StateT, RootT]:
        """Create the predicted root without consuming an expansion step."""
        ...

    def expand_one_step(self, state: StateT) -> StateT:
        """Complete one recoverable tree-expansion step."""
        ...

    def cloud_response_ready(self, handle: HandleT) -> bool:
        """Return whether verification feedback can be received now."""
        ...

    def receive(self, handle: HandleT) -> ResponseT:
        """Receive feedback known to be ready."""
        ...

    def wait_for_feedback(self, handle: HandleT) -> ResponseT:
        """Wait for feedback after planned expansion has completed."""
        ...

    def root_matches(self, branch_root: RootT, response: ResponseT) -> bool:
        """Return whether the proactive branch starts at the verified root."""
        ...

    def retain_and_remap(
        self,
        state: StateT,
        response: ResponseT,
        boundary: BoundaryT,
    ) -> StateT:
        """Attach a matching branch to the verified sequence."""
        ...

    def restore_and_discard(
        self,
        state: StateT,
        response: ResponseT,
        boundary: BoundaryT,
    ) -> StateT:
        """Restore the boundary and discard an invalid branch."""
        ...

    def read_local_counters(self) -> tuple[float, int]:
        """Return local generation time and drafted-token count."""
        ...

    def observe_acceptance(self, response: ResponseT, state: StateT) -> float:
        """Return the completed round's acceptance rate."""
        ...


@dataclass(frozen=True, slots=True)
class RapidRoundResult(Generic[StateT, ResponseT]):
    """Reconciled state and control output from one RAPID round."""

    state: StateT
    response: ResponseT
    next_depth: int
    caac_decision: CAACDecision
    expansions_completed: int
    stopped_after_mismatch: bool
    root_matched: bool


def run_round(
    *,
    draft: DraftT,
    state: StateT,
    planned_depth: int,
    controller: CAACController,
    operations: RapidRoundOps[DraftT, StateT, BoundaryT, HandleT, RootT, ResponseT],
) -> RapidRoundResult[StateT, ResponseT]:
    """Execute the paper's round-level RAPID workflow.

    The runtime-specific operations are callbacks so this reference algorithm
    does not depend on a model framework, GPU library, or networking package.
    The runtime owns request cancellation and boundary recovery if one of its
    callbacks raises.
    """

    if not isinstance(planned_depth, int) or isinstance(planned_depth, bool):
        raise TypeError("planned_depth must be an integer")
    if planned_depth < 1:
        raise ValueError("planned_depth must be at least one")

    boundary = operations.save_boundary(state)
    handle = operations.send_async(draft)
    proactive_state, branch_root = operations.start_proactive_branch(state)

    expansion = run_ipdg_expansion(
        state=proactive_state,
        branch_root=branch_root,
        planned_depth=planned_depth,
        expand_one_step=operations.expand_one_step,
        response_ready=lambda: operations.cloud_response_ready(handle),
        receive_response=lambda: operations.receive(handle),
        root_matches=operations.root_matches,
    )

    response = expansion.response
    if response is None:
        response = operations.wait_for_feedback(handle)
        if response is None:
            raise ValueError("wait_for_feedback must not return None")
        root_matched = operations.root_matches(branch_root, response)
    else:
        # The response received at an expansion boundary was already checked.
        root_matched = not expansion.stopped_after_mismatch

    local_time, local_tokens = operations.read_local_counters()
    if not isinstance(local_time, Real) or isinstance(local_time, bool):
        raise TypeError("local generation time must be a real number")
    if not math.isfinite(local_time) or local_time <= 0.0:
        raise ValueError("local generation time must be a positive finite value")
    if not isinstance(local_tokens, int) or isinstance(local_tokens, bool):
        raise TypeError("local token count must be an integer")
    if local_tokens <= 0:
        raise ValueError("local token count must be positive")
    per_token_latency = local_time / local_tokens

    if root_matched:
        reconciled_state = operations.retain_and_remap(
            expansion.state,
            response,
            boundary,
        )
    else:
        reconciled_state = operations.restore_and_discard(
            expansion.state,
            response,
            boundary,
        )

    current_acceptance = operations.observe_acceptance(response, reconciled_state)
    decision = controller.update(current_acceptance, per_token_latency)

    return RapidRoundResult(
        state=reconciled_state,
        response=response,
        next_depth=decision.depth,
        caac_decision=decision,
        expansions_completed=expansion.expansions_completed,
        stopped_after_mismatch=expansion.stopped_after_mismatch,
        root_matched=root_matched,
    )
