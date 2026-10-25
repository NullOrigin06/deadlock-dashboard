# =============================================================================
# tests/test_validation.py — Tests for the validation module
# =============================================================================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from validation import (
    validate_system, validate_request, validate_detection_inputs,
    SAFE, UNSAFE, WAITING, DENIED_UNSAFE, DEADLOCKED, INVALID_INPUT, GRANTED,
)
from samples import (
    BASE_ALLOCATION, BASE_MAX, BASE_AVAILABLE,
    SCENARIO_F_ZERO_AVAILABLE, SCENARIO_F_ZERO_NEED, SCENARIO_F_INVALID_CASES,
)


class TestValidateSystem:
    def test_base_sample_valid(self):
        """The base sample must pass validation."""
        valid, msg = validate_system(BASE_ALLOCATION, BASE_MAX, BASE_AVAILABLE)
        assert valid is True

    def test_zero_available_valid(self):
        """TC6 F1: Available all zeros is a valid boundary case."""
        data = SCENARIO_F_ZERO_AVAILABLE
        valid, msg = validate_system(
            data["allocation"], data["max_matrix"], data["available"],
        )
        assert valid is True

    def test_zero_need_valid(self):
        """TC6 F2: Process with Allocation == Max (zero Need) is valid."""
        data = SCENARIO_F_ZERO_NEED
        valid, msg = validate_system(
            data["allocation"], data["max_matrix"], data["available"],
        )
        assert valid is True

    def test_allocation_exceeds_max_invalid(self):
        """TC6 F6: Allocation > Max must be rejected."""
        data = SCENARIO_F_INVALID_CASES[3]  # F6: allocation exceeding Max
        valid, msg = validate_system(
            data["allocation"], data["max_matrix"], data["available"],
        )
        assert valid is False
        assert "exceeds" in msg.lower() or "Exceeds" in msg

    def test_empty_allocation_invalid(self):
        """Empty Allocation matrix is invalid."""
        valid, msg = validate_system([], [[1]], [1])
        assert valid is False

    def test_mismatched_rows_invalid(self):
        """Max with different row count than Allocation is invalid."""
        valid, msg = validate_system(
            [[1, 0], [0, 1]],
            [[1, 0]],
            [1, 1],
        )
        assert valid is False


class TestValidateRequest:
    def test_valid_request(self):
        """A well-formed request should pass."""
        valid, msg = validate_request(0, [1, 0, 0], 5, 3)
        assert valid is True

    def test_invalid_process_id(self):
        """TC6 F3: Process id out of range."""
        valid, msg = validate_request(99, [1, 0, 0], 5, 3)
        assert valid is False

    def test_negative_value(self):
        """TC6 F4: Negative request value."""
        valid, msg = validate_request(0, [-1, 0, 0], 5, 3)
        assert valid is False

    def test_wrong_length(self):
        """TC6 F5: Wrong vector length."""
        valid, msg = validate_request(0, [1, 0], 5, 3)
        assert valid is False

    def test_negative_process_id(self):
        """Negative process id is invalid."""
        valid, msg = validate_request(-1, [1, 0, 0], 5, 3)
        assert valid is False


class TestStatusConstants:
    """Verify the status constants are all distinct."""
    def test_all_distinct(self):
        statuses = [SAFE, UNSAFE, WAITING, DENIED_UNSAFE, DEADLOCKED, INVALID_INPUT, GRANTED]
        assert len(set(statuses)) == 7

    def test_unsafe_is_not_deadlocked(self):
        """An unsafe state must never be labelled as deadlocked."""
        assert UNSAFE != DEADLOCKED

    def test_waiting_is_not_invalid(self):
        """A WAITING request must never be labelled as INVALID_INPUT."""
        assert WAITING != INVALID_INPUT
