# =============================================================================
# tests/test_banker.py — Tests for the Banker's algorithm module
# =============================================================================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import pytest
from banker import compute_need, safety_algorithm, request_resources
from validation import SAFE, UNSAFE, WAITING, DENIED_UNSAFE, INVALID_INPUT, GRANTED, DEADLOCKED
from samples import (
    BASE_ALLOCATION, BASE_MAX, BASE_AVAILABLE, BASE_EXPECTED_NEED,
    SCENARIO_A, SCENARIO_B, SCENARIO_C_UNSAFE, SCENARIO_C_GRANTED,
)


# ---------------------------------------------------------------------------
# Test compute_need
# ---------------------------------------------------------------------------
class TestComputeNeed:
    def test_base_sample(self):
        """TC1: Need = Max - Allocation matches expected values."""
        need = compute_need(BASE_ALLOCATION, BASE_MAX)
        assert need == BASE_EXPECTED_NEED

    def test_zero_need(self):
        """Boundary: a process with Allocation == Max has Need = 0."""
        alloc = [[3, 2, 2]]
        max_m = [[3, 2, 2]]
        need = compute_need(alloc, max_m)
        assert need == [[0, 0, 0]]


# ---------------------------------------------------------------------------
# Test safety_algorithm
# ---------------------------------------------------------------------------
class TestSafetyAlgorithm:
    def test_scenario_a_safe(self):
        """TC1: The base sample must produce a SAFE result."""
        need = compute_need(SCENARIO_A["allocation"], SCENARIO_A["max_matrix"])
        result = safety_algorithm(
            SCENARIO_A["allocation"], need, SCENARIO_A["available"],
        )
        assert result["status"] == SAFE
        assert len(result["sequence"]) == 5       # all 5 processes finish
        assert result["stuck"] == []

    def test_scenario_a_sequence_is_valid(self):
        """TC1: Replay the returned safe sequence and confirm every process
        can actually finish in that order."""
        alloc = copy.deepcopy(SCENARIO_A["allocation"])
        need = compute_need(alloc, SCENARIO_A["max_matrix"])
        avail = SCENARIO_A["available"][:]

        result = safety_algorithm(alloc, need, avail)
        seq = result["sequence"]

        # Replay
        work = SCENARIO_A["available"][:]
        for pid in seq:
            for j in range(len(work)):
                assert need[pid][j] <= work[j], (
                    f"P{pid} cannot finish: Need[{j}]={need[pid][j]}, Work[{j}]={work[j]}"
                )
            for j in range(len(work)):
                work[j] += alloc[pid][j]

    def test_steps_trace_has_correct_fields(self):
        """Each step in the trace must have the expected keys."""
        need = compute_need(BASE_ALLOCATION, BASE_MAX)
        result = safety_algorithm(BASE_ALLOCATION, need, BASE_AVAILABLE)
        for step in result["steps"]:
            assert "step" in step
            assert "process" in step
            assert "need" in step
            assert "work_before" in step
            assert "work_after" in step


# ---------------------------------------------------------------------------
# Test request_resources
# ---------------------------------------------------------------------------
class TestRequestResources:
    def test_scenario_b_waiting(self):
        """TC2: P2 requests (6,0,0) — within Need but > Available → WAITING."""
        result = request_resources(
            process_id=SCENARIO_B["request_process"],
            request_vector=SCENARIO_B["request_vector"],
            allocation=copy.deepcopy(SCENARIO_B["allocation"]),
            max_matrix=copy.deepcopy(SCENARIO_B["max_matrix"]),
            available=SCENARIO_B["available"][:],
        )
        assert result["status"] == WAITING

    def test_scenario_b_state_unchanged(self):
        """TC2: After WAITING, the original state must be unchanged."""
        alloc = copy.deepcopy(SCENARIO_B["allocation"])
        avail = SCENARIO_B["available"][:]
        alloc_before = copy.deepcopy(alloc)
        avail_before = avail[:]

        request_resources(
            process_id=SCENARIO_B["request_process"],
            request_vector=SCENARIO_B["request_vector"],
            allocation=alloc,
            max_matrix=copy.deepcopy(SCENARIO_B["max_matrix"]),
            available=avail,
        )
        assert alloc == alloc_before
        assert avail == avail_before

    def test_scenario_c_denied_unsafe(self):
        """TC3: P4 requests (3,3,0) — safe check fails → DENIED_UNSAFE."""
        result = request_resources(
            process_id=SCENARIO_C_UNSAFE["request_process"],
            request_vector=SCENARIO_C_UNSAFE["request_vector"],
            allocation=copy.deepcopy(SCENARIO_C_UNSAFE["allocation"]),
            max_matrix=copy.deepcopy(SCENARIO_C_UNSAFE["max_matrix"]),
            available=SCENARIO_C_UNSAFE["available"][:],
        )
        assert result["status"] == DENIED_UNSAFE
        assert "safety_trace" in result

    def test_scenario_c_granted(self):
        """TC3 contrast: P1 requests (1,0,2) → GRANTED."""
        result = request_resources(
            process_id=SCENARIO_C_GRANTED["request_process"],
            request_vector=SCENARIO_C_GRANTED["request_vector"],
            allocation=copy.deepcopy(SCENARIO_C_GRANTED["allocation"]),
            max_matrix=copy.deepcopy(SCENARIO_C_GRANTED["max_matrix"]),
            available=SCENARIO_C_GRANTED["available"][:],
        )
        assert result["status"] == GRANTED
        assert "new_allocation" in result
        assert "new_available" in result

    def test_invalid_process_id(self):
        """TC6: invalid process id → INVALID_INPUT."""
        result = request_resources(
            process_id=99,
            request_vector=[1, 0, 0],
            allocation=copy.deepcopy(BASE_ALLOCATION),
            max_matrix=copy.deepcopy(BASE_MAX),
            available=BASE_AVAILABLE[:],
        )
        assert result["status"] == INVALID_INPUT

    def test_negative_request(self):
        """TC6: negative value in request → INVALID_INPUT."""
        result = request_resources(
            process_id=0,
            request_vector=[-1, 0, 0],
            allocation=copy.deepcopy(BASE_ALLOCATION),
            max_matrix=copy.deepcopy(BASE_MAX),
            available=BASE_AVAILABLE[:],
        )
        assert result["status"] == INVALID_INPUT

    def test_wrong_vector_length(self):
        """TC6: request vector with wrong length → INVALID_INPUT."""
        result = request_resources(
            process_id=0,
            request_vector=[1, 0],
            allocation=copy.deepcopy(BASE_ALLOCATION),
            max_matrix=copy.deepcopy(BASE_MAX),
            available=BASE_AVAILABLE[:],
        )
        assert result["status"] == INVALID_INPUT

    def test_request_exceeds_need(self):
        """TC6: request > Need → INVALID_INPUT (not WAITING)."""
        # Need_P1 = [1,2,2];  requesting [2,0,0] means A=2 > NeedA=1
        result = request_resources(
            process_id=1,
            request_vector=[2, 0, 0],
            allocation=copy.deepcopy(BASE_ALLOCATION),
            max_matrix=copy.deepcopy(BASE_MAX),
            available=BASE_AVAILABLE[:],
        )
        assert result["status"] == INVALID_INPUT

    # ------------------------------------------------------------------
    # Critical invariant: WAITING must NEVER be reported as INVALID_INPUT
    # ------------------------------------------------------------------
    def test_waiting_is_never_invalid(self):
        """WAITING requests (within Need, > Available) must return WAITING,
        NOT INVALID_INPUT."""
        # P2 Need = [6,0,0], Available = [3,3,2].  Request [6,0,0] is within
        # Need but 6 > 3 (Available A).  Must be WAITING.
        result = request_resources(
            process_id=2,
            request_vector=[6, 0, 0],
            allocation=copy.deepcopy(BASE_ALLOCATION),
            max_matrix=copy.deepcopy(BASE_MAX),
            available=BASE_AVAILABLE[:],
        )
        assert result["status"] == WAITING
        assert result["status"] != INVALID_INPUT

    # ------------------------------------------------------------------
    # Critical invariant: UNSAFE must NEVER be reported as DEADLOCKED
    # ------------------------------------------------------------------
    def test_unsafe_never_labelled_deadlock(self):
        """An unsafe state (from Banker's) must be DENIED_UNSAFE, not DEADLOCKED."""
        result = request_resources(
            process_id=SCENARIO_C_UNSAFE["request_process"],
            request_vector=SCENARIO_C_UNSAFE["request_vector"],
            allocation=copy.deepcopy(SCENARIO_C_UNSAFE["allocation"]),
            max_matrix=copy.deepcopy(SCENARIO_C_UNSAFE["max_matrix"]),
            available=SCENARIO_C_UNSAFE["available"][:],
        )
        assert result["status"] != DEADLOCKED
        assert result["status"] == DENIED_UNSAFE
