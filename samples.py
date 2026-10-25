# =============================================================================
# samples.py — Preloaded test-data scenarios A–F
# =============================================================================
# Each scenario is a plain dictionary so every team member can inspect and
# modify it easily.  Resource names are A, B, C …; processes are P0, P1 …
# =============================================================================

from __future__ import annotations
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Helper to build resource / process labels
# ---------------------------------------------------------------------------

def resource_labels(n: int) -> List[str]:
    """Return ['A', 'B', ..] for *n* resource types."""
    return [chr(65 + j) for j in range(n)]


def process_labels(n: int) -> List[str]:
    """Return ['P0', 'P1', ..] for *n* processes."""
    return [f"P{i}" for i in range(n)]


# ===================================================================
# BASE SAMPLE  (5 processes, 3 resource types with totals 10, 5, 7)
# ===================================================================
BASE_ALLOCATION = [
    [0, 1, 0],   # P0
    [2, 0, 0],   # P1
    [3, 0, 2],   # P2
    [2, 1, 1],   # P3
    [0, 0, 2],   # P4
]

BASE_MAX = [
    [7, 5, 3],   # P0
    [3, 2, 2],   # P1
    [9, 0, 2],   # P2
    [2, 2, 2],   # P3
    [4, 3, 3],   # P4
]

BASE_AVAILABLE = [3, 3, 2]

# Expected Need = Max - Allocation
BASE_EXPECTED_NEED = [
    [7, 4, 3],   # P0
    [1, 2, 2],   # P1
    [6, 0, 0],   # P2
    [0, 1, 1],   # P3
    [4, 3, 1],   # P4
]


# ===================================================================
# SCENARIO A / TC1  — Safe state
# ===================================================================
SCENARIO_A: Dict[str, Any] = {
    "name": "Scenario A – Safe State",
    "description": (
        "Base sample with 5 processes and resources A, B, C.  "
        "A valid safe sequence exists, e.g. <P1, P3, P4, P0, P2>."
    ),
    "allocation": [row[:] for row in BASE_ALLOCATION],
    "max_matrix": [row[:] for row in BASE_MAX],
    "available": BASE_AVAILABLE[:],
    "expected_need": [row[:] for row in BASE_EXPECTED_NEED],
    "expected_status": "SAFE",
}


# ===================================================================
# SCENARIO B / TC2  — WAITING (request within Need, > Available)
# ===================================================================
SCENARIO_B: Dict[str, Any] = {
    "name": "Scenario B – Waiting Request",
    "description": (
        "P2 requests (6,0,0).  Need_P2 = (6,0,0) so request <= Need, "
        "but Available = (3,3,2) so request > Available.  "
        "Status = WAITING; state unchanged."
    ),
    "allocation": [row[:] for row in BASE_ALLOCATION],
    "max_matrix": [row[:] for row in BASE_MAX],
    "available": BASE_AVAILABLE[:],
    "request_process": 2,
    "request_vector": [6, 0, 0],
    "expected_status": "WAITING",
}


# ===================================================================
# SCENARIO C / TC3  — DENIED_UNSAFE + a valid contrast request
# ===================================================================
SCENARIO_C_UNSAFE: Dict[str, Any] = {
    "name": "Scenario C – Unsafe Request (P4 asks 3,3,0)",
    "description": (
        "P4 requests (3,3,0).  Within Need (4,3,1) and Available (3,3,2).  "
        "Tentative allocation leaves Available = (0,0,2); no process can "
        "finish → unsafe.  Status = DENIED_UNSAFE; state restored."
    ),
    "allocation": [row[:] for row in BASE_ALLOCATION],
    "max_matrix": [row[:] for row in BASE_MAX],
    "available": BASE_AVAILABLE[:],
    "request_process": 4,
    "request_vector": [3, 3, 0],
    "expected_status": "DENIED_UNSAFE",
}

SCENARIO_C_GRANTED: Dict[str, Any] = {
    "name": "Scenario C – Valid Request (P1 asks 1,0,2)",
    "description": (
        "P1 requests (1,0,2).  Within Need (1,2,2) and Available (3,3,2).  "
        "After tentative allocation a safe sequence still exists.  "
        "Status = GRANTED."
    ),
    "allocation": [row[:] for row in BASE_ALLOCATION],
    "max_matrix": [row[:] for row in BASE_MAX],
    "available": BASE_AVAILABLE[:],
    "request_process": 1,
    "request_vector": [1, 0, 2],
    "expected_status": "GRANTED",
}


# ===================================================================
# SCENARIO D / TC4  — Deadlock (multi-instance + single-instance)
# ===================================================================

# --- Multi-instance deadlock ---
# 4 processes, 3 resources.  Available is all zeros.
# Every process is waiting, none can finish.
SCENARIO_D_MULTI: Dict[str, Any] = {
    "name": "Scenario D – Multi-Instance Deadlock",
    "description": (
        "4 processes, 3 resource types.  Available = (0,0,0).  "
        "Every process has outstanding requests that cannot be satisfied."
    ),
    "allocation": [
        [1, 0, 1],   # P0 holds 1 of A, 1 of C
        [0, 1, 0],   # P1 holds 1 of B
        [0, 0, 1],   # P2 holds 1 of C
        [1, 0, 0],   # P3 holds 1 of A
    ],
    "request": [
        [0, 1, 0],   # P0 wants 1 of B
        [1, 0, 1],   # P1 wants 1 of A, 1 of C
        [1, 0, 0],   # P2 wants 1 of A
        [0, 1, 0],   # P3 wants 1 of B
    ],
    "available": [0, 0, 0],
    "expected_status": "DEADLOCKED",
    "expected_deadlocked": [0, 1, 2, 3],
}

# --- Single-instance RAG with cycle ---
# Resources: R0, R1  (1 instance each)
# P1 holds R1, requests R2   (edge R1->P1, P1->R2)
# P2 holds R2, requests R1   (edge R2->P2, P2->R1)
# Cycle: P1 -> R2 -> P2 -> R1 -> P1
SCENARIO_D_SINGLE: Dict[str, Any] = {
    "name": "Scenario D – Single-Instance RAG Cycle",
    "description": (
        "2 processes, 2 resources (1 instance each).  "
        "P1 holds R1 and requests R2; P2 holds R2 and requests R1.  "
        "Cycle: P1→R2→P2→R1→P1  ⇒ deadlock."
    ),
    "assignments": {
        # resource -> process that holds it
        "R1": "P1",
        "R2": "P2",
    },
    "requests": {
        # process -> resource it is requesting
        "P1": "R2",
        "P2": "R1",
    },
    "expected_status": "DEADLOCKED",
    "expected_cycle": ["P1", "R2", "P2", "R1", "P1"],
}

# --- Multi-instance RAG with cycle but NO deadlock ---
# This demonstrates why the matrix algorithm is needed for multi-instance.
# 3 processes, 2 resources with 2 instances each.
# The RAG has a cycle, but the matrix algorithm shows no deadlock
# because P2 can finish and release resources.
SCENARIO_D_CYCLE_NO_DEADLOCK: Dict[str, Any] = {
    "name": "Scenario D – Multi-Instance Cycle ≠ Deadlock",
    "description": (
        "3 processes, 2 resources (2 instances each).  "
        "The RAG has a cycle (P0→R1→P1→R0→P0), but P2 can finish and "
        "release resources, so there is NO deadlock.  This shows why "
        "a cycle in a multi-instance RAG is necessary but NOT sufficient."
    ),
    "allocation": [
        [1, 0],   # P0 holds 1 of R0
        [0, 1],   # P1 holds 1 of R1
        [1, 1],   # P2 holds 1 of R0, 1 of R1
    ],
    "request": [
        [0, 1],   # P0 requests 1 of R1
        [1, 0],   # P1 requests 1 of R0
        [0, 0],   # P2 requests nothing
    ],
    "available": [0, 0],
    "expected_status": "SAFE",  # NOT deadlocked despite cycle
}


# ===================================================================
# SCENARIO E / TC5  — Prevention (resource ordering)
# ===================================================================
SCENARIO_E: Dict[str, Any] = {
    "name": "Scenario E – Prevention via Resource Ordering",
    "description": (
        "Apply resource ordering to the Scenario D multi-instance deadlock.  "
        "Under the ordering policy, processes must acquire resources in "
        "increasing order (A < B < C).  Any request that violates this "
        "is rejected, breaking circular wait."
    ),
    "allocation": [
        [1, 0, 1],   # P0
        [0, 1, 0],   # P1
        [0, 0, 1],   # P2
        [1, 0, 0],   # P3
    ],
    "request": [
        [0, 1, 0],   # P0 wants B (index 1) — holds A(0) and C(2), B < C ⇒ VIOLATION
        [1, 0, 1],   # P1 wants A(0) and C(2) — holds B(1), A < B ⇒ VIOLATION for A
        [1, 0, 0],   # P2 wants A(0) — holds C(2), A < C ⇒ VIOLATION
        [0, 1, 0],   # P3 wants B(1) — holds A(0), A < B ⇒ OK
    ],
    "available": [0, 0, 0],
    "expected_status": "DEADLOCKED",
}


# ===================================================================
# SCENARIO F / TC6  — Boundary and invalid inputs
# ===================================================================
SCENARIO_F_ZERO_AVAILABLE: Dict[str, Any] = {
    "name": "F1 – Available all zeros (valid boundary)",
    "description": "All resources are allocated.  Valid config if Alloc <= Max.",
    "allocation": [
        [2, 1, 1],
        [1, 1, 1],
    ],
    "max_matrix": [
        [3, 2, 2],
        [2, 2, 2],
    ],
    "available": [0, 0, 0],
}

SCENARIO_F_ZERO_NEED: Dict[str, Any] = {
    "name": "F2 – Process with zero remaining Need (finishes immediately)",
    "description": (
        "P0 has Allocation == Max, so Need = 0.  It can finish immediately "
        "and release its resources."
    ),
    "allocation": [
        [3, 2, 2],   # P0: Need = 0
        [1, 0, 0],   # P1
    ],
    "max_matrix": [
        [3, 2, 2],   # P0
        [2, 1, 1],   # P1
    ],
    "available": [0, 0, 0],
}

SCENARIO_F_INVALID_CASES: List[Dict[str, Any]] = [
    {
        "name": "F3 – Invalid process id",
        "process_id": 99,
        "request_vector": [1, 0, 0],
        "num_processes": 5,
        "num_resources": 3,
    },
    {
        "name": "F4 – Negative request",
        "process_id": 0,
        "request_vector": [-1, 0, 0],
        "num_processes": 5,
        "num_resources": 3,
    },
    {
        "name": "F5 – Wrong vector length",
        "process_id": 0,
        "request_vector": [1, 0],
        "num_processes": 5,
        "num_resources": 3,
    },
    {
        "name": "F6 – Allocation exceeding Max",
        "allocation": [
            [8, 1, 0],   # 8 > 7 in Max
            [2, 0, 0],
        ],
        "max_matrix": [
            [7, 5, 3],
            [3, 2, 2],
        ],
        "available": [3, 3, 2],
    },
    {
        "name": "F7 – Request exceeding Need",
        "description": "P1 requests (2,0,0) but Need_P1 = (1,2,2).  Request[A]=2 > Need[A]=1.",
        "allocation": [row[:] for row in BASE_ALLOCATION],
        "max_matrix": [row[:] for row in BASE_MAX],
        "available": BASE_AVAILABLE[:],
        "process_id": 1,
        "request_vector": [2, 0, 0],  # Need_P1[A] = 1, so 2 > 1
    },
]


# ===================================================================
# Convenience: collect all scenarios in one dict for the UI dropdown
# ===================================================================
ALL_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "A – Safe State":                      SCENARIO_A,
    "B – Waiting Request":                 SCENARIO_B,
    "C – Unsafe (P4 asks 3,3,0)":          SCENARIO_C_UNSAFE,
    "C – Granted (P1 asks 1,0,2)":         SCENARIO_C_GRANTED,
    "D – Multi-Instance Deadlock":         SCENARIO_D_MULTI,
    "D – Single-Instance RAG Cycle":       SCENARIO_D_SINGLE,
    "D – Cycle ≠ Deadlock Demo":           SCENARIO_D_CYCLE_NO_DEADLOCK,
    "E – Prevention (Resource Ordering)":  SCENARIO_E,
}
