# =============================================================================
# prevention.py — Deadlock Prevention Policies & Recovery
# =============================================================================
# This module implements:
#   1. terminate_process()      — recover from deadlock by killing a process
#   2. suggest_termination()    — auto-pick the best victim
#   3. check_resource_ordering() — prevention: fixed acquisition order
#   4. check_all_upfront()      — prevention: request-all-at-once
#   5. compare_policies()       — side-by-side comparison of original vs policy
# =============================================================================

from __future__ import annotations
import copy
from typing import Any, Dict, List, Tuple

from detection import detect_deadlock_multi


# =====================================================================
# 1.  RECOVERY — terminate a process and release its resources
# =====================================================================
# Steps:
#   1. Copy the current Allocation, Request, Available.
#   2. Release all resources held by the chosen process:
#      Available[j] += Allocation[pid][j] for all j.
#   3. Set Allocation[pid] and Request[pid] to all zeros.
#   4. Re-run the detection algorithm on the new state.
#   5. Return the new state and the detection result.
# =====================================================================
def terminate_process(
    process_id: int,
    allocation: List[List[int]],
    request: List[List[int]],
    available: List[int],
) -> Dict[str, Any]:
    """
    Terminate *process_id*, release its resources, and re-run detection.

    Returns:
        terminated      — the process id that was killed
        released        — the resources freed (old Allocation[pid])
        new_allocation  — updated Allocation matrix
        new_request     — updated Request matrix
        new_available   — updated Available vector
        detection       — result of re-running detect_deadlock_multi
    """
    num_resources = len(available)

    new_alloc   = copy.deepcopy(allocation)
    new_request = copy.deepcopy(request)
    new_avail   = available[:]

    released = new_alloc[process_id][:]

    for j in range(num_resources):
        new_avail[j] += new_alloc[process_id][j]
        new_alloc[process_id][j] = 0
        new_request[process_id][j] = 0

    detection = detect_deadlock_multi(new_alloc, new_request, new_avail)

    return {
        "terminated":     process_id,
        "released":       released,
        "new_allocation": new_alloc,
        "new_request":    new_request,
        "new_available":  new_avail,
        "detection":      detection,
    }


# =====================================================================
# 2.  SUGGEST TERMINATION — pick the deadlocked process holding the most
# =====================================================================
# Strategy: among the deadlocked set, choose the process whose total
# allocation (sum of Allocation[i]) is largest.  This releases the most
# resources and gives the best chance of breaking the deadlock.
# =====================================================================
def suggest_termination(
    deadlocked: List[int],
    allocation: List[List[int]],
) -> Dict[str, Any]:
    """
    Suggest which deadlocked process to terminate.

    Returns:
        suggested_process — the process id to kill
        total_held        — how many total resource units it holds
        reasoning         — human-readable explanation
    """
    best_pid = deadlocked[0]
    best_total = sum(allocation[best_pid])

    for pid in deadlocked[1:]:
        total = sum(allocation[pid])
        if total > best_total:
            best_total = total
            best_pid = pid

    return {
        "suggested_process": best_pid,
        "total_held":        best_total,
        "reasoning": (
            f"P{best_pid} holds the most total resources ({best_total} units) "
            f"among the deadlocked set {{{', '.join(f'P{p}' for p in deadlocked)}}}. "
            f"Terminating it releases the most resources, giving other processes "
            f"the best chance of completing."
        ),
    }


# =====================================================================
# 3.  PREVENTION — Resource Ordering (eliminates circular wait)
# =====================================================================
# Rule: A process may only request a resource whose index is STRICTLY
# GREATER than the highest-indexed resource it currently holds.
# If it holds nothing, any request is allowed.
#
# Steps:
#   1. Find the highest resource index currently held by the process.
#   2. For each requested resource, check if its index > highest held.
#   3. If any requested resource has a lower or equal index, REJECT.
# =====================================================================
def check_resource_ordering(
    held_resources: List[int],
    requested_indices: List[int],
) -> Dict[str, Any]:
    """
    Check whether a request respects the resource-ordering policy.

    Parameters:
        held_resources    — list of resource indices currently held (may be empty).
        requested_indices — list of resource indices being requested.

    Returns:
        allowed    — True if the request respects the ordering
        violations — list of (requested_idx, max_held) tuples for failures
        reason     — human-readable explanation
    """
    if not held_resources:
        return {
            "allowed":    True,
            "violations": [],
            "reason":     "Process holds no resources — any request is allowed.",
        }

    max_held = max(held_resources)
    violations: List[Tuple[int, int]] = []

    for req_idx in requested_indices:
        if req_idx <= max_held:
            violations.append((req_idx, max_held))

    if violations:
        parts = []
        for req_idx, mh in violations:
            parts.append(
                f"resource {chr(65+req_idx)} (index {req_idx}) ≤ "
                f"max held index {mh} ({chr(65+mh)})"
            )
        return {
            "allowed":    False,
            "violations": violations,
            "reason": (
                f"Ordering violation(s): {'; '.join(parts)}.  "
                f"Under resource ordering, a process must request only "
                f"resources with index > highest held ({max_held})."
            ),
        }

    return {
        "allowed":    True,
        "violations": [],
        "reason": (
            f"All requested indices are > max held index {max_held}.  "
            f"Ordering policy satisfied."
        ),
    }


# =====================================================================
# 4.  PREVENTION — Request All Up-front (eliminates hold-and-wait)
# =====================================================================
# Rule: A process must request ALL resources it will ever need at once,
# before it starts executing.  If all are available, grant them all;
# otherwise the process waits (holding nothing).
#
# Metric: since every process either gets everything or nothing, we can
# count how many processes can run concurrently (those whose full need
# fits in Available).
# =====================================================================
def check_all_upfront(
    max_needs: List[List[int]],
    available: List[int],
) -> Dict[str, Any]:
    """
    Simulate the request-all-upfront policy.

    Parameters:
        max_needs — each process's total resource need (Max matrix).
        available — current Available vector.

    Returns:
        can_run          — list of process indices that can be granted immediately
        must_wait         — list of process indices that must wait
        concurrent_count — number of processes that can run concurrently
    """
    num_processes = len(max_needs)
    num_resources = len(available)

    can_run: List[int] = []
    must_wait: List[int] = []

    remaining = available[:]

    # Greedy: try to fit each process in order
    for i in range(num_processes):
        fits = all(max_needs[i][j] <= remaining[j] for j in range(num_resources))
        if fits:
            for j in range(num_resources):
                remaining[j] -= max_needs[i][j]
            can_run.append(i)
        else:
            must_wait.append(i)

    return {
        "can_run":          can_run,
        "must_wait":        must_wait,
        "concurrent_count": len(can_run),
    }


# =====================================================================
# 5.  COMPARE POLICIES — run original vs. prevention side by side
# =====================================================================
def apply_resource_ordering_to_scenario(
    allocation: List[List[int]],
    request: List[List[int]],
    available: List[int],
) -> Dict[str, Any]:
    """
    Apply resource ordering to a scenario and report results.

    For each process, check whether its outstanding request respects
    the ordering policy.  Rejected requests are set to zero (the process
    gives up that request).  Then re-run detection on the modified state.

    Returns:
        original_detection — detection result on original state
        ordering_results   — per-process ordering check results
        modified_request   — request matrix after rejecting violations
        modified_detection — detection result on modified state
        coffman_eliminated — which Coffman condition is eliminated
        tradeoffs          — list of tradeoff descriptions
    """
    num_processes = len(allocation)
    num_resources = len(available)

    # Detection on original state
    original_detection = detect_deadlock_multi(allocation, request, available)

    # Check each process's request against the ordering policy
    ordering_results: List[Dict[str, Any]] = []
    modified_request = copy.deepcopy(request)

    for i in range(num_processes):
        # Which resources does process i hold?
        held_indices = [j for j in range(num_resources) if allocation[i][j] > 0]
        # Which resources does process i request?
        requested_indices = [j for j in range(num_resources) if request[i][j] > 0]

        result = check_resource_ordering(held_indices, requested_indices)
        result["process"] = i
        ordering_results.append(result)

        # If violated, zero out the request (process must release and re-acquire)
        if not result["allowed"]:
            for j in range(num_resources):
                modified_request[i][j] = 0

    # Detection on modified state
    modified_detection = detect_deadlock_multi(allocation, modified_request, available)

    return {
        "original_detection": original_detection,
        "ordering_results":   ordering_results,
        "modified_request":   modified_request,
        "modified_detection": modified_detection,
        "coffman_eliminated": "Circular Wait",
        "tradeoffs": [
            "Lower resource utilization — processes may hold resources idle "
            "while waiting for higher-numbered ones.",
            "Reduced concurrency — processes that need low-numbered resources "
            "late in their execution must release and re-acquire.",
            "Possible starvation — a process needing many low-numbered "
            "resources may be repeatedly preempted.",
        ],
    }


def apply_all_upfront_to_scenario(
    allocation: List[List[int]],
    request: List[List[int]],
    max_matrix: List[List[int]],
    available: List[int],
) -> Dict[str, Any]:
    """
    Compare original policy with request-all-upfront.

    Returns:
        original_detection — detection on original state
        upfront_result     — result of check_all_upfront()
        coffman_eliminated — which Coffman condition is eliminated
        tradeoffs          — list of tradeoff descriptions
    """
    original_detection = detect_deadlock_multi(allocation, request, available)

    # Total available = Available + sum of all Allocation (everything released first)
    num_resources = len(available)
    total_available = available[:]
    for row in allocation:
        for j in range(num_resources):
            total_available[j] += row[j]

    upfront_result = check_all_upfront(max_matrix, total_available)

    return {
        "original_detection": original_detection,
        "upfront_result":     upfront_result,
        "coffman_eliminated": "Hold and Wait",
        "tradeoffs": [
            "Lower resource utilization — resources allocated but unused until "
            "the process actually needs them.",
            "Reduced concurrency — fewer processes can run simultaneously "
            f"(only {upfront_result['concurrent_count']} can start at once).",
            "Possible starvation — processes needing many resources may "
            "never get all of them at once.",
        ],
    }
