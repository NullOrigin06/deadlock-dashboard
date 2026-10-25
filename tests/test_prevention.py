# =============================================================================
# tests/test_prevention.py — Tests for the prevention & recovery module
# =============================================================================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import pytest
from prevention import (
    terminate_process, suggest_termination,
    check_resource_ordering, check_all_upfront,
    apply_resource_ordering_to_scenario,
)
from detection import detect_deadlock_multi
from validation import SAFE, DEADLOCKED
from samples import SCENARIO_D_MULTI, SCENARIO_E


class TestTerminateProcess:
    def test_recovery_breaks_deadlock(self):
        """TC4 recovery: terminating a process should break (or reduce) deadlock."""
        data = SCENARIO_D_MULTI
        # First confirm it's deadlocked
        det = detect_deadlock_multi(data["allocation"], data["request"], data["available"])
        assert det["status"] == DEADLOCKED

        # Suggest a victim
        suggestion = suggest_termination(det["deadlocked"], data["allocation"])
        victim = suggestion["suggested_process"]

        # Terminate
        result = terminate_process(
            victim,
            copy.deepcopy(data["allocation"]),
            copy.deepcopy(data["request"]),
            data["available"][:],
        )

        assert result["terminated"] == victim
        assert result["released"] == data["allocation"][victim]
        # After termination, at least one fewer deadlocked process
        new_deadlocked = result["detection"].get("deadlocked", [])
        assert victim not in new_deadlocked

    def test_released_resources_added_to_available(self):
        """Available must increase by the terminated process's allocation."""
        data = SCENARIO_D_MULTI
        old_avail = data["available"][:]
        result = terminate_process(
            0,
            copy.deepcopy(data["allocation"]),
            copy.deepcopy(data["request"]),
            old_avail[:],
        )
        released = data["allocation"][0]
        for j in range(len(old_avail)):
            assert result["new_available"][j] == old_avail[j] + released[j]


class TestSuggestTermination:
    def test_picks_largest_holder(self):
        """Should pick the process holding the most total resources."""
        alloc = [
            [1, 0, 1],   # P0 total=2
            [0, 1, 0],   # P1 total=1
            [0, 0, 1],   # P2 total=1
            [1, 0, 0],   # P3 total=1
        ]
        suggestion = suggest_termination([0, 1, 2, 3], alloc)
        assert suggestion["suggested_process"] == 0
        assert suggestion["total_held"] == 2


class TestResourceOrdering:
    def test_valid_ordering(self):
        """Request for higher-indexed resource when holding lower → allowed."""
        result = check_resource_ordering(
            held_resources=[0],          # holds A (index 0)
            requested_indices=[1, 2],    # wants B, C
        )
        assert result["allowed"] is True

    def test_ordering_violation(self):
        """Request for lower-indexed resource when holding higher → rejected."""
        result = check_resource_ordering(
            held_resources=[2],          # holds C (index 2)
            requested_indices=[0],       # wants A (index 0) — violation!
        )
        assert result["allowed"] is False
        assert len(result["violations"]) == 1

    def test_no_resources_held(self):
        """If process holds nothing, any request is allowed."""
        result = check_resource_ordering(
            held_resources=[],
            requested_indices=[0, 1, 2],
        )
        assert result["allowed"] is True


class TestResourceOrderingScenario:
    def test_scenario_e_prevention(self):
        """TC5: resource ordering applied to Scenario E should break the deadlock."""
        data = SCENARIO_E
        result = apply_resource_ordering_to_scenario(
            data["allocation"], data["request"], data["available"],
        )
        # Original should be deadlocked
        assert result["original_detection"]["status"] == DEADLOCKED
        # After applying ordering, the modified state should not be deadlocked
        # (violating requests are zeroed out)
        assert result["modified_detection"]["status"] == SAFE
        assert result["coffman_eliminated"] == "Circular Wait"


class TestAllUpfront:
    def test_basic_upfront(self):
        """With enough resources, all processes can run."""
        max_needs = [[2, 1], [1, 2]]
        available = [3, 3]
        result = check_all_upfront(max_needs, available)
        assert result["concurrent_count"] == 2
        assert result["must_wait"] == []

    def test_not_enough_for_all(self):
        """When resources are scarce, some processes must wait."""
        max_needs = [[3, 3], [3, 3], [3, 3]]
        available = [4, 4]
        result = check_all_upfront(max_needs, available)
        assert result["concurrent_count"] < 3
        assert len(result["must_wait"]) > 0
