# =============================================================================
# tests/test_detection.py — Tests for the detection module
# =============================================================================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from detection import detect_deadlock_multi, build_rag, detect_cycle_single
from validation import SAFE, DEADLOCKED
from samples import (
    SCENARIO_D_MULTI, SCENARIO_D_SINGLE, SCENARIO_D_CYCLE_NO_DEADLOCK,
)


class TestMultiInstanceDetection:
    def test_scenario_d_deadlock(self):
        """TC4: all processes deadlocked when Available is all zeros and
        every process has outstanding requests."""
        result = detect_deadlock_multi(
            SCENARIO_D_MULTI["allocation"],
            SCENARIO_D_MULTI["request"],
            SCENARIO_D_MULTI["available"],
        )
        assert result["status"] == DEADLOCKED
        assert set(result["deadlocked"]) == set(SCENARIO_D_MULTI["expected_deadlocked"])

    def test_no_deadlock_when_all_can_finish(self):
        """A state where processes have zero requests should not deadlock."""
        alloc = [[1, 0], [0, 1]]
        req   = [[0, 0], [0, 0]]
        avail = [1, 1]
        result = detect_deadlock_multi(alloc, req, avail)
        assert result["status"] == SAFE
        assert result["deadlocked"] == []

    def test_cycle_no_deadlock_demo(self):
        """TC4 demo: multi-instance RAG has cycle but no deadlock.
        P2 holds resources and requests nothing, so it can finish first."""
        data = SCENARIO_D_CYCLE_NO_DEADLOCK
        result = detect_deadlock_multi(
            data["allocation"], data["request"], data["available"],
        )
        assert result["status"] == SAFE
        assert result["deadlocked"] == []

    def test_steps_recorded(self):
        """Detection trace must include step details."""
        result = detect_deadlock_multi(
            SCENARIO_D_MULTI["allocation"],
            SCENARIO_D_MULTI["request"],
            SCENARIO_D_MULTI["available"],
        )
        # In the deadlock case no process can finish so steps may be empty
        # but the result must still have the key
        assert "steps" in result


class TestSingleInstanceDetection:
    def test_scenario_d_single_cycle(self):
        """TC4: single-instance RAG has a cycle → deadlock."""
        graph = build_rag(
            SCENARIO_D_SINGLE["assignments"],
            SCENARIO_D_SINGLE["requests"],
        )
        result = detect_cycle_single(graph)
        assert result["has_cycle"] is True
        assert result["status"] == DEADLOCKED

    def test_no_cycle(self):
        """RAG with no cycle → no deadlock."""
        graph = build_rag(
            assignments={"R1": "P1"},
            requests={"P2": "R1"},
        )
        # P2 waits for R1, R1 held by P1, but P1 doesn't wait for anything
        result = detect_cycle_single(graph)
        assert result["has_cycle"] is False
        assert result["status"] == SAFE

    def test_graph_has_correct_node_types(self):
        """Nodes must have the correct ntype attribute."""
        graph = build_rag(
            SCENARIO_D_SINGLE["assignments"],
            SCENARIO_D_SINGLE["requests"],
        )
        for node, data in graph.nodes(data=True):
            assert data["ntype"] in ("process", "resource")


class TestDetectionNeverConfusedWithUnsafe:
    """Ensure detection returns DEADLOCKED or SAFE — never UNSAFE or DENIED_UNSAFE."""
    def test_multi_returns_correct_statuses(self):
        result = detect_deadlock_multi(
            SCENARIO_D_MULTI["allocation"],
            SCENARIO_D_MULTI["request"],
            SCENARIO_D_MULTI["available"],
        )
        assert result["status"] in (SAFE, DEADLOCKED)

    def test_single_returns_correct_statuses(self):
        graph = build_rag(
            SCENARIO_D_SINGLE["assignments"],
            SCENARIO_D_SINGLE["requests"],
        )
        result = detect_cycle_single(graph)
        assert result["status"] in (SAFE, DEADLOCKED)
