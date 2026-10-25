# =============================================================================
# validation.py — Shared input-validation helpers for the Deadlock Dashboard
# =============================================================================
# Every public function returns a tuple (is_valid: bool, message: str).
# If is_valid is False, message explains what went wrong so the UI can show it.
# =============================================================================

from __future__ import annotations
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Status constants (single source of truth for the entire project)
# ---------------------------------------------------------------------------
SAFE          = "SAFE"
UNSAFE        = "UNSAFE"
WAITING       = "WAITING"          # request within Need but > Available
DENIED_UNSAFE = "DENIED_UNSAFE"    # would make state unsafe; request denied
DEADLOCKED    = "DEADLOCKED"
INVALID_INPUT = "INVALID_INPUT"
GRANTED       = "GRANTED"

# Mapping each status to a colour name used by the Streamlit UI
STATUS_COLORS = {
    SAFE:          "green",
    UNSAFE:        "orange",
    WAITING:       "blue",
    DENIED_UNSAFE: "red",
    DEADLOCKED:    "red",
    INVALID_INPUT: "gray",
    GRANTED:       "green",
}

# Human-readable one-line descriptions
STATUS_DESCRIPTIONS = {
    SAFE:          "All processes can finish — the system is in a safe state.",
    UNSAFE:        "No safe sequence exists — deadlock is *possible* (not certain).",
    WAITING:       "Request is within Need but exceeds Available — deferred, state unchanged.",
    DENIED_UNSAFE: "Granting would make the state unsafe — request denied, original state restored.",
    DEADLOCKED:    "Processes are blocked in a cycle — deadlock has occurred.",
    INVALID_INPUT: "Input failed validation — see the error message for details.",
    GRANTED:       "Request granted — new state is safe.",
}


# ---------------------------------------------------------------------------
# Helper: check that a value is a non-negative integer
# ---------------------------------------------------------------------------
def _is_non_negative_int(value) -> bool:
    """Return True if *value* is an int (or numpy int) and >= 0."""
    # Accept plain int and numpy integer types
    try:
        # numpy integers have .item()
        if hasattr(value, "item"):
            value = value.item()
        return isinstance(value, int) and value >= 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Validate the system configuration (Allocation, Max, Available)
# ---------------------------------------------------------------------------
# Steps:
#   1. Check that Allocation and Max have the same shape (P x R).
#   2. Check that Available has length R.
#   3. Check every element is a non-negative integer.
#   4. Check Allocation[i][j] <= Max[i][j] for all i, j.
# ---------------------------------------------------------------------------
def validate_system(
    allocation: List[List[int]],
    max_matrix: List[List[int]],
    available: List[int],
) -> Tuple[bool, str]:
    """Validate the full system configuration matrices."""

    # --- shape checks ---
    num_processes = len(allocation)
    if num_processes == 0:
        return False, "Allocation matrix is empty — need at least one process."

    num_resources = len(allocation[0])
    if num_resources == 0:
        return False, "Allocation matrix has zero resource columns."

    if len(max_matrix) != num_processes:
        return False, (
            f"Max matrix has {len(max_matrix)} rows but Allocation has "
            f"{num_processes} — they must match."
        )

    for i, row in enumerate(allocation):
        if len(row) != num_resources:
            return False, (
                f"Allocation row P{i} has {len(row)} columns, expected {num_resources}."
            )

    for i, row in enumerate(max_matrix):
        if len(row) != num_resources:
            return False, (
                f"Max row P{i} has {len(row)} columns, expected {num_resources}."
            )

    if len(available) != num_resources:
        return False, (
            f"Available vector length is {len(available)}, expected {num_resources}."
        )

    # --- element checks ---
    for i in range(num_processes):
        for j in range(num_resources):
            if not _is_non_negative_int(allocation[i][j]):
                return False, (
                    f"Allocation[P{i}][{chr(65+j)}] = {allocation[i][j]!r} "
                    f"is not a non-negative integer."
                )
            if not _is_non_negative_int(max_matrix[i][j]):
                return False, (
                    f"Max[P{i}][{chr(65+j)}] = {max_matrix[i][j]!r} "
                    f"is not a non-negative integer."
                )

    for j in range(num_resources):
        if not _is_non_negative_int(available[j]):
            return False, (
                f"Available[{chr(65+j)}] = {available[j]!r} "
                f"is not a non-negative integer."
            )

    # --- Allocation <= Max ---
    for i in range(num_processes):
        for j in range(num_resources):
            if allocation[i][j] > max_matrix[i][j]:
                return False, (
                    f"Allocation[P{i}][{chr(65+j)}] = {allocation[i][j]} exceeds "
                    f"Max[P{i}][{chr(65+j)}] = {max_matrix[i][j]}."
                )

    return True, "System configuration is valid."


# ---------------------------------------------------------------------------
# Validate a single resource-request vector
# ---------------------------------------------------------------------------
# Steps:
#   1. process_id must be a valid index (0 .. num_processes-1).
#   2. request_vector must have length == num_resources.
#   3. Every element must be a non-negative integer.
# ---------------------------------------------------------------------------
def validate_request(
    process_id: int,
    request_vector: List[int],
    num_processes: int,
    num_resources: int,
) -> Tuple[bool, str]:
    """Validate the raw request parameters (before any Need/Available check)."""

    # --- process id ---
    if not isinstance(process_id, int) or process_id < 0 or process_id >= num_processes:
        return False, (
            f"Process id {process_id!r} is invalid — must be 0..{num_processes - 1}."
        )

    # --- vector length ---
    if len(request_vector) != num_resources:
        return False, (
            f"Request vector length is {len(request_vector)}, "
            f"expected {num_resources}."
        )

    # --- element checks ---
    for j, val in enumerate(request_vector):
        if not _is_non_negative_int(val):
            return False, (
                f"Request[{chr(65+j)}] = {val!r} is not a non-negative integer."
            )

    return True, "Request vector is valid."


# ---------------------------------------------------------------------------
# Validate detection inputs (Allocation, Request, Available)
# ---------------------------------------------------------------------------
def validate_detection_inputs(
    allocation: List[List[int]],
    request: List[List[int]],
    available: List[int],
) -> Tuple[bool, str]:
    """Validate matrices used by the deadlock-detection algorithm."""

    num_processes = len(allocation)
    if num_processes == 0:
        return False, "Allocation matrix is empty."

    num_resources = len(allocation[0])
    if num_resources == 0:
        return False, "Allocation matrix has zero resource columns."

    if len(request) != num_processes:
        return False, (
            f"Request matrix has {len(request)} rows but Allocation has "
            f"{num_processes}."
        )

    for i, row in enumerate(allocation):
        if len(row) != num_resources:
            return False, f"Allocation row P{i} has wrong width."

    for i, row in enumerate(request):
        if len(row) != num_resources:
            return False, f"Request row P{i} has wrong width."

    if len(available) != num_resources:
        return False, (
            f"Available vector length is {len(available)}, expected {num_resources}."
        )

    for i in range(num_processes):
        for j in range(num_resources):
            if not _is_non_negative_int(allocation[i][j]):
                return False, (
                    f"Allocation[P{i}][{chr(65+j)}] is not a non-negative integer."
                )
            if not _is_non_negative_int(request[i][j]):
                return False, (
                    f"Request[P{i}][{chr(65+j)}] is not a non-negative integer."
                )

    for j in range(num_resources):
        if not _is_non_negative_int(available[j]):
            return False, (
                f"Available[{chr(65+j)}] is not a non-negative integer."
            )

    return True, "Detection inputs are valid."
