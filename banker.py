# =============================================================================
# banker.py — Banker's Algorithm (Deadlock Avoidance)
# =============================================================================
# This module implements:
#   1. compute_need()       — calculate Need = Max − Allocation
#   2. safety_algorithm()   — determine if a state is safe, return step trace
#   3. request_resources()  — validate and process a resource request
#
# All functions use plain Python lists / integers so that every group member
# can read and explain them in a viva.  Each function returns a structured
# result dict (never just a boolean) so the UI can display every step.
# =============================================================================

from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional

from validation import (
    SAFE, UNSAFE, WAITING, DENIED_UNSAFE, INVALID_INPUT, GRANTED,
    validate_system, validate_request,
)


# ---------------------------------------------------------------------------
# 1.  compute_need  —  Need[i][j] = Max[i][j] − Allocation[i][j]
# ---------------------------------------------------------------------------
def compute_need(
    allocation: List[List[int]],
    max_matrix: List[List[int]],
) -> List[List[int]]:
    """
    Compute the Need matrix.

    Steps:
        For every process i and resource j:
            Need[i][j] = Max[i][j] − Allocation[i][j]
    """
    num_processes = len(allocation)
    num_resources = len(allocation[0])
    need: List[List[int]] = []
    for i in range(num_processes):
        row: List[int] = []
        for j in range(num_resources):
            row.append(max_matrix[i][j] - allocation[i][j])
        need.append(row)
    return need


# ---------------------------------------------------------------------------
# 2.  safety_algorithm  —  find a safe sequence (or prove none exists)
# ---------------------------------------------------------------------------
# Algorithm (textbook Banker's safety check):
#   1. Let Work = copy of Available;  Finish = [False] * n.
#   2. Repeat:
#       a. Find an i where Finish[i] == False AND Need[i] <= Work.
#       b. If found:  Work = Work + Allocation[i];  Finish[i] = True;
#          record this step.
#       c. If not found: stop.
#   3. If all Finish[i] are True → state is SAFE (return the sequence).
#      Otherwise → UNSAFE (return which processes are stuck and why).
# ---------------------------------------------------------------------------
def safety_algorithm(
    allocation: List[List[int]],
    need: List[List[int]],
    available: List[int],
) -> Dict[str, Any]:
    """
    Run the Banker's safety algorithm.

    Returns a dict with keys:
        status        — SAFE or UNSAFE
        sequence      — list of process indices in safe order (or partial)
        steps         — list of step-dicts for the UI table
        stuck         — list of process indices that could not finish
        stuck_reasons — dict mapping stuck process to reason string
    """
    num_processes = len(allocation)
    num_resources = len(available)

    # Work and Finish vectors
    work = available[:]                          # copy
    finish = [False] * num_processes

    sequence: List[int] = []
    steps: List[Dict[str, Any]] = []

    # Repeat until no more processes can be picked
    changed = True
    while changed:
        changed = False
        for i in range(num_processes):
            if finish[i]:
                continue

            # Check Need[i] <= Work
            can_finish = True
            for j in range(num_resources):
                if need[i][j] > work[j]:
                    can_finish = False
                    break

            if can_finish:
                # Record the step BEFORE updating Work
                work_before = work[:]
                for j in range(num_resources):
                    work[j] += allocation[i][j]
                work_after = work[:]

                finish[i] = True
                sequence.append(i)
                changed = True

                steps.append({
                    "step":       len(steps) + 1,
                    "process":    f"P{i}",
                    "need":       need[i][:],
                    "work_before": work_before,
                    "work_after":  work_after,
                    "finish":     finish[:],
                })
                break  # restart scan from P0

    # Build result
    if all(finish):
        return {
            "status":   SAFE,
            "sequence": sequence,
            "steps":    steps,
            "stuck":    [],
            "stuck_reasons": {},
        }

    # Determine which processes are stuck and why
    stuck: List[int] = []
    stuck_reasons: Dict[int, str] = {}
    for i in range(num_processes):
        if not finish[i]:
            stuck.append(i)
            # Identify which resources are short
            shortfalls: List[str] = []
            for j in range(num_resources):
                if need[i][j] > work[j]:
                    shortfalls.append(
                        f"{chr(65+j)}: need {need[i][j]}, work has {work[j]}"
                    )
            stuck_reasons[i] = "; ".join(shortfalls)

    return {
        "status":        UNSAFE,
        "sequence":      sequence,      # partial
        "steps":         steps,
        "stuck":         stuck,
        "stuck_reasons": stuck_reasons,
    }


# ---------------------------------------------------------------------------
# 3.  request_resources  —  handle a resource request from process *pid*
# ---------------------------------------------------------------------------
# Order of checks (per the spec):
#   1. Validate input (valid pid, non-negative ints, correct length).
#      Failure → INVALID_INPUT.
#   2. request > Need[pid] → INVALID_INPUT (exceeded declared max claim).
#   3. request > Available → WAITING (cannot be granted now; state unchanged).
#   4. Tentatively allocate, run safety algorithm on the copy.
#      Safe → GRANTED (commit).  Unsafe → DENIED_UNSAFE (rollback).
# ---------------------------------------------------------------------------
def request_resources(
    process_id: int,
    request_vector: List[int],
    allocation: List[List[int]],
    max_matrix: List[List[int]],
    available: List[int],
) -> Dict[str, Any]:
    """
    Process a resource request using the Banker's algorithm.

    Parameters are NOT modified in-place when the request is denied or waiting.
    When GRANTED, the caller should use the returned 'new_allocation',
    'new_available', and 'new_need' to update its state.

    Returns a dict with:
        status           — one of the status constants
        message          — human-readable explanation
        safety_trace     — output of safety_algorithm (if run)
        new_allocation   — updated matrix (only when GRANTED)
        new_available    — updated vector (only when GRANTED)
        new_need         — updated matrix (only when GRANTED)
        before_matrices  — snapshot before tentative alloc (DENIED_UNSAFE)
        after_matrices   — snapshot after tentative alloc (DENIED_UNSAFE)
    """
    num_processes = len(allocation)
    num_resources = len(available)

    # ------------------------------------------------------------------
    # Step 1: basic input validation
    # ------------------------------------------------------------------
    valid, msg = validate_request(process_id, request_vector, num_processes, num_resources)
    if not valid:
        return {"status": INVALID_INPUT, "message": msg}

    # Also check non-negative integer (validate_request already does this,
    # but we keep it explicit for clarity)

    need = compute_need(allocation, max_matrix)

    # ------------------------------------------------------------------
    # Step 2: request > Need → INVALID_INPUT
    # ------------------------------------------------------------------
    for j in range(num_resources):
        if request_vector[j] > need[process_id][j]:
            return {
                "status": INVALID_INPUT,
                "message": (
                    f"P{process_id} requested {chr(65+j)}={request_vector[j]} "
                    f"but Need[P{process_id}][{chr(65+j)}]={need[process_id][j]}.  "
                    f"Process exceeded its declared maximum claim."
                ),
            }

    # ------------------------------------------------------------------
    # Step 3: request > Available → WAITING
    # ------------------------------------------------------------------
    for j in range(num_resources):
        if request_vector[j] > available[j]:
            return {
                "status": WAITING,
                "message": (
                    f"P{process_id}'s request is within Need but "
                    f"{chr(65+j)}={request_vector[j]} > Available[{chr(65+j)}]"
                    f"={available[j]}.  Request deferred; state unchanged."
                ),
            }

    # ------------------------------------------------------------------
    # Step 4: tentatively allocate on a COPY, run safety check
    # ------------------------------------------------------------------
    # Deep-copy the state so we can roll back if unsafe
    new_alloc = copy.deepcopy(allocation)
    new_avail = available[:]

    for j in range(num_resources):
        new_avail[j]                -= request_vector[j]
        new_alloc[process_id][j]    += request_vector[j]

    new_need = compute_need(new_alloc, max_matrix)

    # Snapshot for the UI (shown on DENIED_UNSAFE)
    before_matrices = {
        "allocation": copy.deepcopy(allocation),
        "available":  available[:],
        "need":       copy.deepcopy(need),
    }
    after_matrices = {
        "allocation": copy.deepcopy(new_alloc),
        "available":  new_avail[:],
        "need":       copy.deepcopy(new_need),
    }

    trace = safety_algorithm(new_alloc, new_need, new_avail)

    if trace["status"] == SAFE:
        return {
            "status":         GRANTED,
            "message":        (
                f"Request granted.  Safe sequence: "
                f"<{', '.join(f'P{p}' for p in trace['sequence'])}>."
            ),
            "safety_trace":   trace,
            "new_allocation": new_alloc,
            "new_available":  new_avail,
            "new_need":       new_need,
        }
    else:
        # DENIED_UNSAFE — the copy is discarded; original state unchanged
        return {
            "status":          DENIED_UNSAFE,
            "message":         (
                f"Granting P{process_id}'s request would leave the system "
                f"in an unsafe state.  Request denied; original state restored."
            ),
            "safety_trace":    trace,
            "before_matrices": before_matrices,
            "after_matrices":  after_matrices,
        }
