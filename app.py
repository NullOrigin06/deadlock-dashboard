# =============================================================================
# app.py — Streamlit UI for the Deadlock Analysis & Resource Allocation Dashboard
# =============================================================================
# This file contains ONLY the UI logic.  All algorithms live in:
#   banker.py     — avoidance (Banker's algorithm)
#   detection.py  — deadlock detection (multi-instance + single-instance)
#   prevention.py — prevention policies & recovery
#   validation.py — shared input validation & status constants
#   samples.py    — preloaded test scenarios A–F
# =============================================================================

from __future__ import annotations
import copy, csv, io
from typing import Any, Dict, List

import streamlit as st
import pandas as pd
import numpy as np

from validation import (
    SAFE, UNSAFE, WAITING, DENIED_UNSAFE, DEADLOCKED, INVALID_INPUT, GRANTED,
    STATUS_COLORS, STATUS_DESCRIPTIONS,
    validate_system,
)
from banker import compute_need, safety_algorithm, request_resources
from detection import (
    detect_deadlock_multi, build_rag, detect_cycle_single, draw_rag,
)
from prevention import (
    terminate_process, suggest_termination,
    apply_resource_ordering_to_scenario, apply_all_upfront_to_scenario,
    check_resource_ordering, check_all_upfront,
)
from samples import (
    ALL_SCENARIOS, resource_labels, process_labels,
    SCENARIO_A, SCENARIO_B, SCENARIO_C_UNSAFE, SCENARIO_C_GRANTED,
    SCENARIO_D_MULTI, SCENARIO_D_SINGLE, SCENARIO_D_CYCLE_NO_DEADLOCK,
    SCENARIO_E,
    SCENARIO_F_ZERO_AVAILABLE, SCENARIO_F_ZERO_NEED, SCENARIO_F_INVALID_CASES,
    BASE_ALLOCATION, BASE_MAX, BASE_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deadlock Analysis Dashboard",
    page_icon="🔒",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS for status colours
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.status-safe       { background:#064e3b; color:#6ee7b7; border:1px solid #059669; padding:12px; border-radius:8px; font-weight:600; }
.status-unsafe     { background:#7c2d12; color:#fdba74; border:1px solid #ea580c; padding:12px; border-radius:8px; font-weight:600; }
.status-waiting    { background:#1e3a8a; color:#93c5fd; border:1px solid #2563eb; padding:12px; border-radius:8px; font-weight:600; }
.status-denied     { background:#7f1d1d; color:#fca5a5; border:1px solid #dc2626; padding:12px; border-radius:8px; font-weight:600; }
.status-deadlocked { background:#881337; color:#fda4af; border:1px solid #e11d48; padding:12px; border-radius:8px; font-weight:600; }
.status-invalid    { background:#334155; color:#cbd5e1; border:1px solid #64748b; padding:12px; border-radius:8px; font-weight:600; }
.status-granted    { background:#065f46; color:#a7f3d0; border:1px solid #10b981; padding:12px; border-radius:8px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def show_status_banner(status: str, message: str = ""):
    """Show a colour-coded status banner."""
    css_map = {
        SAFE:          "status-safe",
        UNSAFE:        "status-unsafe",
        WAITING:       "status-waiting",
        DENIED_UNSAFE: "status-denied",
        DEADLOCKED:    "status-deadlocked",
        INVALID_INPUT: "status-invalid",
        GRANTED:       "status-granted",
    }
    css = css_map.get(status, "status-invalid")
    desc = STATUS_DESCRIPTIONS.get(status, "")
    text = f"**Status: {status}** — {desc}"
    if message:
        text += f"<br>{message}"
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)


def matrix_to_df(
    matrix: List[List[int]],
    num_processes: int,
    num_resources: int,
    label: str = "",
) -> pd.DataFrame:
    """Convert a matrix to a DataFrame with Px / A,B,C labels."""
    cols = resource_labels(num_resources)
    idx  = process_labels(num_processes)
    df = pd.DataFrame(matrix, columns=cols, index=idx)
    if label:
        df.index.name = label
    return df


def vector_to_df(
    vector: List[int], num_resources: int, label: str = "",
) -> pd.DataFrame:
    cols = resource_labels(num_resources)
    df = pd.DataFrame([vector], columns=cols, index=[label] if label else None)
    return df


def show_safety_trace(trace: Dict[str, Any], num_resources: int):
    """Display the step-by-step safety algorithm trace."""
    if not trace["steps"]:
        st.info("No processes could be picked — the algorithm terminated immediately.")
        return

    rows = []
    for step in trace["steps"]:
        row = {
            "Step":        step["step"],
            "Process":     step["process"],
        }
        # Add Need columns
        key = "need" if "need" in step else "request"
        for j in range(num_resources):
            row[f"Need/{chr(65+j)}"] = step[key][j]
        for j in range(num_resources):
            row[f"Work_before/{chr(65+j)}"] = step["work_before"][j]
        for j in range(num_resources):
            row[f"Work_after/{chr(65+j)}"] = step["work_after"][j]
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔒 Deadlock Dashboard")
    st.divider()

    # Sample loader
    st.subheader("📂 Load Sample")
    sample_choice = st.selectbox(
        "Choose a scenario",
        ["(none)"] + list(ALL_SCENARIOS.keys()),
        key="sample_choice",
    )

    if st.button("Load Selected Sample", use_container_width=True):
        if sample_choice != "(none)":
            data = ALL_SCENARIOS[sample_choice]
            if "allocation" in data:
                st.session_state["alloc"]     = copy.deepcopy(data["allocation"])
                st.session_state["max_mat"]   = copy.deepcopy(data["max_matrix"]) if "max_matrix" in data else None
                st.session_state["available"] = data["available"][:]
                st.session_state["num_proc"]  = len(data["allocation"])
                st.session_state["num_res"]   = len(data["allocation"][0])
            st.rerun()

    if st.button("🔄 Reset State", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()

    # Glossary
    st.subheader("📖 Status Glossary")
    for status, desc in STATUS_DESCRIPTIONS.items():
        colour = STATUS_COLORS[status]
        st.markdown(f"**:{colour}[{status}]** — {desc}")

    st.divider()
    st.caption("Built for OS IA-2 Assignment")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_avoid, tab_detect, tab_prevent, tab_tests = st.tabs([
    "🛡️ Avoidance", "🔍 Detection", "🛑 Prevention & Recovery", "📋 Test Cases",
])


# ========================== TAB 1: AVOIDANCE ==============================
with tab_avoid:
    st.header("Deadlock Avoidance — Banker's Algorithm")

    st.info(
        "💡 **An unsafe state is not necessarily a deadlocked state:** unsafe means "
        "a deadlock is *possible*, not that it has occurred.  Banker's checks whether "
        "granting a request preserves a safe state; it does NOT detect an existing deadlock."
    )

    col1, col2 = st.columns(2)
    with col1:
        num_proc = st.number_input("Number of processes", 1, 20, st.session_state.get("num_proc", 5), key="np_avoid")
    with col2:
        num_res  = st.number_input("Number of resource types", 1, 10, st.session_state.get("num_res", 3), key="nr_avoid")

    # --- editable matrices ---
    st.subheader("Allocation Matrix")
    default_alloc = st.session_state.get("alloc", [[0]*num_res for _ in range(num_proc)])
    # Ensure dimensions match
    if len(default_alloc) != num_proc or (default_alloc and len(default_alloc[0]) != num_res):
        default_alloc = [[0]*num_res for _ in range(num_proc)]

    alloc_df = pd.DataFrame(
        default_alloc, columns=resource_labels(num_res), index=process_labels(num_proc),
    )
    edited_alloc = st.data_editor(alloc_df, key="alloc_editor", use_container_width=True)

    st.subheader("Maximum Matrix")
    default_max = st.session_state.get("max_mat", [[0]*num_res for _ in range(num_proc)])
    if len(default_max) != num_proc or (default_max and len(default_max[0]) != num_res):
        default_max = [[0]*num_res for _ in range(num_proc)]

    max_df = pd.DataFrame(
        default_max, columns=resource_labels(num_res), index=process_labels(num_proc),
    )
    edited_max = st.data_editor(max_df, key="max_editor", use_container_width=True)

    st.subheader("Available Vector")
    default_avail = st.session_state.get("available", [0]*num_res)
    if len(default_avail) != num_res:
        default_avail = [0]*num_res

    avail_df = pd.DataFrame([default_avail], columns=resource_labels(num_res), index=["Available"])
    edited_avail = st.data_editor(avail_df, key="avail_editor", use_container_width=True)

    # Convert edited data back to lists
    alloc_list = edited_alloc.values.tolist()
    max_list   = edited_max.values.tolist()
    avail_list = edited_avail.values.tolist()[0]

    # Convert to int (data_editor may return float)
    try:
        alloc_list = [[int(v) for v in row] for row in alloc_list]
        max_list   = [[int(v) for v in row] for row in max_list]
        avail_list = [int(v) for v in avail_list]
    except (ValueError, TypeError):
        st.error("All values must be integers.")
        st.stop()

    # Validate
    valid, vmsg = validate_system(alloc_list, max_list, avail_list)
    if not valid:
        show_status_banner(INVALID_INPUT, vmsg)
    else:
        # Compute and display Need
        need = compute_need(alloc_list, max_list)
        st.subheader("Need Matrix (computed: Max − Allocation)")
        st.dataframe(matrix_to_df(need, num_proc, num_res), use_container_width=True)

        # --- Run Safety Algorithm ---
        if st.button("▶️ Run Safety Algorithm", key="run_safety"):
            result = safety_algorithm(alloc_list, need, avail_list)
            show_status_banner(result["status"])

            if result["status"] == SAFE:
                seq_str = ", ".join(f"P{p}" for p in result["sequence"])
                st.success(f"Safe sequence: **<{seq_str}>**")
            else:
                stuck_str = ", ".join(f"P{p}" for p in result["stuck"])
                st.warning(f"Stuck processes: {stuck_str}")
                for pid, reason in result["stuck_reasons"].items():
                    st.caption(f"  P{pid}: {reason}")

            st.subheader("Work / Finish Trace")
            show_safety_trace(result, num_res)

        # --- Resource Request ---
        st.divider()
        st.subheader("🔧 Resource Request")
        rcol1, rcol2 = st.columns([1, 3])
        with rcol1:
            req_pid = st.number_input("Process ID", 0, num_proc - 1, 0, key="req_pid")
        with rcol2:
            req_vec_str = st.text_input(
                "Request vector (comma-separated)",
                ",".join(["0"] * num_res),
                key="req_vec",
            )

        if st.button("📤 Submit Request", key="submit_req"):
            try:
                req_vec = [int(x.strip()) for x in req_vec_str.split(",")]
            except ValueError:
                show_status_banner(INVALID_INPUT, "Could not parse request — enter comma-separated integers.")
                st.stop()

            result = request_resources(
                req_pid, req_vec,
                copy.deepcopy(alloc_list), copy.deepcopy(max_list), avail_list[:],
            )
            show_status_banner(result["status"], result.get("message", ""))

            if result["status"] == GRANTED:
                st.session_state["alloc"]     = result["new_allocation"]
                st.session_state["available"] = result["new_available"]
                st.subheader("Updated State")
                st.dataframe(
                    matrix_to_df(result["new_allocation"], num_proc, num_res, "Allocation"),
                    use_container_width=True,
                )
                st.dataframe(
                    vector_to_df(result["new_available"], num_res, "Available"),
                    use_container_width=True,
                )
                if "safety_trace" in result:
                    show_safety_trace(result["safety_trace"], num_res)

            elif result["status"] == DENIED_UNSAFE:
                if "before_matrices" in result and "after_matrices" in result:
                    bcol, acol = st.columns(2)
                    with bcol:
                        st.caption("Before (original — preserved)")
                        st.dataframe(matrix_to_df(
                            result["before_matrices"]["allocation"], num_proc, num_res,
                        ))
                    with acol:
                        st.caption("After (tentative — discarded)")
                        st.dataframe(matrix_to_df(
                            result["after_matrices"]["allocation"], num_proc, num_res,
                        ))
                if "safety_trace" in result:
                    st.subheader("Failed Safety Trace")
                    show_safety_trace(result["safety_trace"], num_res)


# ========================== TAB 2: DETECTION ==============================
with tab_detect:
    st.header("Deadlock Detection")

    mode = st.radio(
        "Select detection mode",
        ["Multi-Instance (Matrix Algorithm)", "Single-Instance (RAG / Wait-For Graph)"],
        key="det_mode",
    )

    # ---- MULTI-INSTANCE ----
    if mode.startswith("Multi"):
        st.subheader("🔢 Multi-Instance Resource Detection")
        st.caption(
            "Uses Allocation, Request (current outstanding requests), and Available.  "
            "Processes left unfinished at the end are **deadlocked**."
        )

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            d_nproc = st.number_input("Processes", 1, 20, 4, key="d_nproc")
        with dcol2:
            d_nres  = st.number_input("Resource types", 1, 10, 3, key="d_nres")

        st.subheader("Allocation")
        d_alloc_default = st.session_state.get("det_alloc", [[0]*d_nres for _ in range(d_nproc)])
        if len(d_alloc_default) != d_nproc or (d_alloc_default and len(d_alloc_default[0]) != d_nres):
            d_alloc_default = [[0]*d_nres for _ in range(d_nproc)]
        d_alloc_df = pd.DataFrame(d_alloc_default, columns=resource_labels(d_nres), index=process_labels(d_nproc))
        d_alloc_ed = st.data_editor(d_alloc_df, key="d_alloc_ed", use_container_width=True)

        st.subheader("Request (outstanding)")
        d_req_default = st.session_state.get("det_req", [[0]*d_nres for _ in range(d_nproc)])
        if len(d_req_default) != d_nproc or (d_req_default and len(d_req_default[0]) != d_nres):
            d_req_default = [[0]*d_nres for _ in range(d_nproc)]
        d_req_df = pd.DataFrame(d_req_default, columns=resource_labels(d_nres), index=process_labels(d_nproc))
        d_req_ed = st.data_editor(d_req_df, key="d_req_ed", use_container_width=True)

        st.subheader("Available")
        d_avail_default = st.session_state.get("det_avail", [0]*d_nres)
        if len(d_avail_default) != d_nres:
            d_avail_default = [0]*d_nres
        d_avail_df = pd.DataFrame([d_avail_default], columns=resource_labels(d_nres), index=["Available"])
        d_avail_ed = st.data_editor(d_avail_df, key="d_avail_ed", use_container_width=True)

        try:
            d_alloc_list = [[int(v) for v in row] for row in d_alloc_ed.values.tolist()]
            d_req_list   = [[int(v) for v in row] for row in d_req_ed.values.tolist()]
            d_avail_list = [int(v) for v in d_avail_ed.values.tolist()[0]]
        except (ValueError, TypeError):
            st.error("All values must be integers.")
            st.stop()

        if st.button("▶️ Run Detection", key="run_detect_multi"):
            det = detect_deadlock_multi(d_alloc_list, d_req_list, d_avail_list)
            show_status_banner(det["status"])

            if det["status"] == DEADLOCKED:
                dl_str = ", ".join(f"P{p}" for p in det["deadlocked"])
                st.error(f"Deadlocked processes: **{{{dl_str}}}**")
            else:
                st.success("No deadlock detected — all processes can complete.")

            st.subheader("Work / Finish Trace")
            show_safety_trace(det, d_nres)

        # Load Scenario D button
        if st.button("📥 Load Scenario D (Deadlock)", key="load_d_multi"):
            st.session_state["det_alloc"] = copy.deepcopy(SCENARIO_D_MULTI["allocation"])
            st.session_state["det_req"]   = copy.deepcopy(SCENARIO_D_MULTI["request"])
            st.session_state["det_avail"] = SCENARIO_D_MULTI["available"][:]
            st.rerun()

        if st.button("📥 Load Cycle ≠ Deadlock Demo", key="load_cycle_demo"):
            st.session_state["det_alloc"] = copy.deepcopy(SCENARIO_D_CYCLE_NO_DEADLOCK["allocation"])
            st.session_state["det_req"]   = copy.deepcopy(SCENARIO_D_CYCLE_NO_DEADLOCK["request"])
            st.session_state["det_avail"] = SCENARIO_D_CYCLE_NO_DEADLOCK["available"][:]
            st.rerun()

    # ---- SINGLE-INSTANCE ----
    else:
        st.subheader("🔗 Single-Instance Resource Detection (RAG)")
        st.caption(
            "For **single-instance** resources, a cycle in the Resource Allocation Graph "
            "(RAG) means deadlock.  For multi-instance resources, a cycle is necessary "
            "but **NOT sufficient** — the matrix algorithm must be used."
        )

        st.markdown("#### Define Assignments (Resource → Process)")
        assign_str = st.text_area(
            "One per line: ResourceName,ProcessName  (e.g. R1,P1)",
            "R1,P1\nR2,P2",
            key="rag_assign",
        )
        st.markdown("#### Define Requests (Process → Resource)")
        req_str = st.text_area(
            "One per line: ProcessName,ResourceName  (e.g. P1,R2)",
            "P1,R2\nP2,R1",
            key="rag_req",
        )

        if st.button("▶️ Build Graph & Detect Cycle", key="run_detect_single"):
            try:
                assignments = {}
                for line in assign_str.strip().split("\n"):
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        assignments[parts[0]] = parts[1]

                requests = {}
                for line in req_str.strip().split("\n"):
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        requests[parts[0]] = parts[1]
            except (IndexError, ValueError):
                st.error("Could not parse input — use the format ResourceName,ProcessName.")
                st.stop()

            graph = build_rag(assignments, requests)
            result = detect_cycle_single(graph)
            show_status_banner(result["status"])

            if result["has_cycle"]:
                cycle_str = " → ".join(result["cycle"])
                st.error(f"Cycle found: **{cycle_str}**")
                fig = draw_rag(graph, result["cycle"], "RAG — Cycle Detected")
            else:
                st.success("No cycle — no deadlock.")
                fig = draw_rag(graph, title="RAG — No Cycle")

            st.pyplot(fig)

        # Load Scenario D single-instance
        if st.button("📥 Load Scenario D (Single-Instance)", key="load_d_single"):
            lines_a = "\n".join(f"{r},{p}" for r, p in SCENARIO_D_SINGLE["assignments"].items())
            lines_r = "\n".join(f"{p},{r}" for p, r in SCENARIO_D_SINGLE["requests"].items())
            st.session_state["rag_assign"] = lines_a
            st.session_state["rag_req"]    = lines_r
            st.rerun()

        st.divider()
        with st.expander("📚 Why cycle ≠ deadlock for multi-instance resources"):
            st.markdown(
                "In a multi-instance RAG, a cycle is a **necessary but not sufficient** "
                "condition for deadlock.  Consider 3 processes and 2 resources with 2 "
                "instances each:\n\n"
                "- P0 holds 1×R0, requests 1×R1\n"
                "- P1 holds 1×R1, requests 1×R0\n"
                "- P2 holds 1×R0 + 1×R1, requests nothing\n\n"
                "The RAG has a cycle (P0→R1→P1→R0→P0), but P2 can finish first, "
                "releasing both R0 and R1, breaking the apparent cycle.  "
                "The matrix algorithm correctly identifies this as **not deadlocked**."
            )


# ========================== TAB 3: PREVENTION & RECOVERY ==================
with tab_prevent:
    st.header("Deadlock Prevention & Recovery")

    prev_mode = st.radio(
        "Choose section",
        ["Recovery (Terminate Process)", "Prevention (Resource Ordering)", "Prevention (Request All Up-front)"],
        key="prev_mode",
    )

    # ---- RECOVERY ----
    if prev_mode.startswith("Recovery"):
        st.subheader("🔧 Recovery by Process Termination")
        st.caption(
            "After deadlock is detected, terminate a process to release its resources "
            "and re-run detection."
        )

        # Use detection matrices from session state or defaults
        r_alloc = st.session_state.get("rec_alloc", SCENARIO_D_MULTI["allocation"])
        r_req   = st.session_state.get("rec_req",   SCENARIO_D_MULTI["request"])
        r_avail = st.session_state.get("rec_avail",  SCENARIO_D_MULTI["available"])

        r_nproc = len(r_alloc)
        r_nres  = len(r_avail)

        st.markdown("**Current State:**")
        col_a, col_r, col_v = st.columns(3)
        with col_a:
            st.caption("Allocation")
            st.dataframe(matrix_to_df(r_alloc, r_nproc, r_nres))
        with col_r:
            st.caption("Request")
            st.dataframe(matrix_to_df(r_req, r_nproc, r_nres))
        with col_v:
            st.caption("Available")
            st.dataframe(vector_to_df(r_avail, r_nres, "Available"))

        # Run detection first
        det = detect_deadlock_multi(r_alloc, r_req, r_avail)
        show_status_banner(det["status"])

        if det["status"] == DEADLOCKED:
            dl = det["deadlocked"]
            dl_str = ", ".join(f"P{p}" for p in dl)
            st.warning(f"Deadlocked: {{{dl_str}}}")

            # Suggest
            suggestion = suggest_termination(dl, r_alloc)
            st.info(f"💡 **Suggestion:** {suggestion['reasoning']}")

            victim = st.selectbox(
                "Process to terminate",
                dl,
                format_func=lambda x: f"P{x}",
                key="rec_victim",
            )

            if st.button("⚡ Terminate & Re-run Detection", key="run_recovery"):
                rec = terminate_process(victim, copy.deepcopy(r_alloc), copy.deepcopy(r_req), r_avail[:])
                st.success(
                    f"Terminated P{rec['terminated']}.  "
                    f"Released: {rec['released']}"
                )

                st.session_state["rec_alloc"] = rec["new_allocation"]
                st.session_state["rec_req"]   = rec["new_request"]
                st.session_state["rec_avail"] = rec["new_available"]

                st.markdown("**New Available:**")
                st.dataframe(vector_to_df(rec["new_available"], r_nres, "Available"))

                show_status_banner(rec["detection"]["status"])
                if rec["detection"]["status"] == DEADLOCKED:
                    st.warning(f"Still deadlocked: {rec['detection']['deadlocked']}")
                else:
                    st.success("System can now progress!")

                show_safety_trace(rec["detection"], r_nres)
        else:
            st.success("No deadlock — no recovery needed.")

        if st.button("📥 Load Scenario D for Recovery", key="load_d_rec"):
            st.session_state["rec_alloc"] = copy.deepcopy(SCENARIO_D_MULTI["allocation"])
            st.session_state["rec_req"]   = copy.deepcopy(SCENARIO_D_MULTI["request"])
            st.session_state["rec_avail"] = SCENARIO_D_MULTI["available"][:]
            st.rerun()

    # ---- PREVENTION: RESOURCE ORDERING ----
    elif "Ordering" in prev_mode:
        st.subheader("📏 Prevention — Resource Ordering (Eliminates Circular Wait)")
        st.caption(
            "Rule: processes must request resources in strictly increasing index order.  "
            "Any request for a resource with index ≤ highest held is rejected."
        )

        if st.button("▶️ Run Scenario E (Resource Ordering)", key="run_ordering"):
            result = apply_resource_ordering_to_scenario(
                SCENARIO_E["allocation"], SCENARIO_E["request"], SCENARIO_E["available"],
            )
            n_p = len(SCENARIO_E["allocation"])
            n_r = len(SCENARIO_E["available"])

            col_orig, col_mod = st.columns(2)
            with col_orig:
                st.markdown("**Original Policy**")
                show_status_banner(result["original_detection"]["status"])
                st.caption("Request Matrix")
                st.dataframe(matrix_to_df(SCENARIO_E["request"], n_p, n_r))

            with col_mod:
                st.markdown("**After Resource Ordering**")
                show_status_banner(result["modified_detection"]["status"])
                st.caption("Modified Request Matrix")
                st.dataframe(matrix_to_df(result["modified_request"], n_p, n_r))

            st.subheader("Per-Process Ordering Check")
            for r in result["ordering_results"]:
                icon = "✅" if r["allowed"] else "❌"
                st.markdown(f"{icon} **P{r['process']}**: {r['reason']}")

            st.subheader("Summary")
            st.success(f"**Coffman condition eliminated:** {result['coffman_eliminated']}")
            st.markdown("**Trade-offs:**")
            for t in result["tradeoffs"]:
                st.markdown(f"- {t}")

    # ---- PREVENTION: ALL UP-FRONT ----
    else:
        st.subheader("🔒 Prevention — Request All Up-front (Eliminates Hold-and-Wait)")
        st.caption(
            "Rule: each process must request ALL resources it needs at once before starting.  "
            "If not all are available, the process waits (holding nothing)."
        )

        if st.button("▶️ Run Scenario E (All Up-front)", key="run_upfront"):
            # We need a Max matrix — for the detection scenario we use allocation + request as max
            e_alloc = SCENARIO_E["allocation"]
            e_req   = SCENARIO_E["request"]
            e_avail = SCENARIO_E["available"]
            n_r = len(e_avail)

            # Max = Allocation + Request (the total a process will ever need)
            e_max = []
            for i in range(len(e_alloc)):
                row = [e_alloc[i][j] + e_req[i][j] for j in range(n_r)]
                e_max.append(row)

            result = apply_all_upfront_to_scenario(e_alloc, e_req, e_max, e_avail)
            n_p = len(e_alloc)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Policy**")
                show_status_banner(result["original_detection"]["status"])
            with col2:
                st.markdown("**All Up-front Policy**")
                up = result["upfront_result"]
                can_str = ", ".join(f"P{p}" for p in up["can_run"]) or "(none)"
                wait_str = ", ".join(f"P{p}" for p in up["must_wait"]) or "(none)"
                st.metric("Concurrent Processes", up["concurrent_count"])
                st.markdown(f"Can run: {can_str}")
                st.markdown(f"Must wait: {wait_str}")

            st.success(f"**Coffman condition eliminated:** {result['coffman_eliminated']}")
            st.markdown("**Trade-offs:**")
            for t in result["tradeoffs"]:
                st.markdown(f"- {t}")

    st.divider()
    with st.expander("📊 Concept Summary — Prevention vs Avoidance vs Detection vs Recovery"):
        concept_data = {
            "Approach":       ["Prevention",       "Avoidance",             "Detection",                    "Recovery"],
            "When Applied":   ["Before requests",  "At request time",       "Periodically or on timeout",   "After deadlock found"],
            "Mechanism":      ["Eliminate a Coffman condition", "Banker's safe-state check", "Matrix algo / RAG cycle check", "Terminate process / preempt resources"],
            "Overhead":       ["Low runtime, high restriction", "Moderate (safety check)", "Low to moderate", "High (process killed)"],
            "Drawback":       ["Reduced utilization & concurrency", "Must know max claims in advance", "Deadlock already occurred before detection", "Work lost; possible starvation"],
        }
        st.table(pd.DataFrame(concept_data))


# ========================== TAB 4: TEST CASES =============================
with tab_tests:
    st.header("📋 Test Cases — TC1 to TC6")

    test_results: List[Dict[str, str]] = []

    # ---- TC1: Safe State ----
    with st.expander("TC1 — Safe State (Scenario A)", expanded=False):
        if st.button("▶️ Run TC1", key="run_tc1"):
            need = compute_need(SCENARIO_A["allocation"], SCENARIO_A["max_matrix"])
            result = safety_algorithm(SCENARIO_A["allocation"], need, SCENARIO_A["available"])

            st.markdown("**Need Matrix:**")
            st.dataframe(matrix_to_df(need, 5, 3))

            show_status_banner(result["status"])
            seq_str = ", ".join(f"P{p}" for p in result["sequence"])
            st.success(f"Safe sequence: <{seq_str}>")

            show_safety_trace(result, 3)

            actual = result["status"]
            expected = "SAFE"
            passed = actual == expected
            test_results.append({
                "Test Case": "TC1", "System Condition": "Base sample — safe state",
                "Actual Result": actual, "Expected Result": expected,
                "Analysis/Conclusion": "PASS — valid safe sequence found" if passed else "FAIL",
            })
            st.session_state["tc1"] = test_results[-1]

    # ---- TC2: Waiting ----
    with st.expander("TC2 — Waiting Request (Scenario B)", expanded=False):
        if st.button("▶️ Run TC2", key="run_tc2"):
            result = request_resources(
                SCENARIO_B["request_process"], SCENARIO_B["request_vector"],
                copy.deepcopy(SCENARIO_B["allocation"]),
                copy.deepcopy(SCENARIO_B["max_matrix"]),
                SCENARIO_B["available"][:],
            )
            show_status_banner(result["status"], result.get("message", ""))

            actual = result["status"]
            expected = "WAITING"
            passed = actual == expected
            test_results.append({
                "Test Case": "TC2", "System Condition": "P2 requests (6,0,0) — within Need, > Available",
                "Actual Result": actual, "Expected Result": expected,
                "Analysis/Conclusion": "PASS — correctly deferred" if passed else "FAIL",
            })
            st.session_state["tc2"] = test_results[-1]

    # ---- TC3: Unsafe + Granted ----
    with st.expander("TC3 — Unsafe & Granted Requests (Scenario C)", expanded=False):
        if st.button("▶️ Run TC3", key="run_tc3"):
            # Unsafe
            r1 = request_resources(
                SCENARIO_C_UNSAFE["request_process"], SCENARIO_C_UNSAFE["request_vector"],
                copy.deepcopy(SCENARIO_C_UNSAFE["allocation"]),
                copy.deepcopy(SCENARIO_C_UNSAFE["max_matrix"]),
                SCENARIO_C_UNSAFE["available"][:],
            )
            st.markdown("**P4 requests (3,3,0):**")
            show_status_banner(r1["status"], r1.get("message", ""))
            if "safety_trace" in r1:
                show_safety_trace(r1["safety_trace"], 3)

            st.divider()

            # Granted
            r2 = request_resources(
                SCENARIO_C_GRANTED["request_process"], SCENARIO_C_GRANTED["request_vector"],
                copy.deepcopy(SCENARIO_C_GRANTED["allocation"]),
                copy.deepcopy(SCENARIO_C_GRANTED["max_matrix"]),
                SCENARIO_C_GRANTED["available"][:],
            )
            st.markdown("**P1 requests (1,0,2) [contrast]:**")
            show_status_banner(r2["status"], r2.get("message", ""))
            if "safety_trace" in r2:
                show_safety_trace(r2["safety_trace"], 3)

            actual_u = r1["status"]
            actual_g = r2["status"]
            passed = actual_u == "DENIED_UNSAFE" and actual_g == "GRANTED"
            test_results.append({
                "Test Case": "TC3",
                "System Condition": "P4→(3,3,0) unsafe; P1→(1,0,2) granted",
                "Actual Result": f"{actual_u} / {actual_g}",
                "Expected Result": "DENIED_UNSAFE / GRANTED",
                "Analysis/Conclusion": "PASS" if passed else "FAIL",
            })
            st.session_state["tc3"] = test_results[-1]

    # ---- TC4: Deadlock ----
    with st.expander("TC4 — Deadlock Detection (Scenario D)", expanded=False):
        if st.button("▶️ Run TC4", key="run_tc4"):
            # Multi-instance
            st.markdown("**Multi-Instance Deadlock:**")
            det_m = detect_deadlock_multi(
                SCENARIO_D_MULTI["allocation"],
                SCENARIO_D_MULTI["request"],
                SCENARIO_D_MULTI["available"],
            )
            show_status_banner(det_m["status"])
            if det_m["status"] == DEADLOCKED:
                st.error(f"Deadlocked: {{{', '.join(f'P{p}' for p in det_m['deadlocked'])}}}")
            show_safety_trace(det_m, 3)

            st.divider()

            # Single-instance
            st.markdown("**Single-Instance RAG Cycle:**")
            graph = build_rag(SCENARIO_D_SINGLE["assignments"], SCENARIO_D_SINGLE["requests"])
            det_s = detect_cycle_single(graph)
            show_status_banner(det_s["status"])
            if det_s["has_cycle"]:
                st.error(f"Cycle: {' → '.join(det_s['cycle'])}")
            fig = draw_rag(graph, det_s.get("cycle"), "RAG — Scenario D")
            st.pyplot(fig)

            st.divider()

            # Recovery demo
            st.markdown("**Recovery — terminate suggested process:**")
            suggestion = suggest_termination(det_m["deadlocked"], SCENARIO_D_MULTI["allocation"])
            st.info(suggestion["reasoning"])
            rec = terminate_process(
                suggestion["suggested_process"],
                copy.deepcopy(SCENARIO_D_MULTI["allocation"]),
                copy.deepcopy(SCENARIO_D_MULTI["request"]),
                SCENARIO_D_MULTI["available"][:],
            )
            st.success(f"Terminated P{rec['terminated']}, released {rec['released']}")
            show_status_banner(rec["detection"]["status"])

            passed = det_m["status"] == DEADLOCKED and det_s["status"] == DEADLOCKED
            test_results.append({
                "Test Case": "TC4",
                "System Condition": "Multi-instance deadlock + single-instance RAG cycle",
                "Actual Result": f"Multi:{det_m['status']} / Single:{det_s['status']}",
                "Expected Result": "DEADLOCKED / DEADLOCKED",
                "Analysis/Conclusion": "PASS — deadlock correctly detected & recovery shown" if passed else "FAIL",
            })
            st.session_state["tc4"] = test_results[-1]

    # ---- TC5: Prevention ----
    with st.expander("TC5 — Prevention via Resource Ordering (Scenario E)", expanded=False):
        if st.button("▶️ Run TC5", key="run_tc5"):
            result = apply_resource_ordering_to_scenario(
                SCENARIO_E["allocation"], SCENARIO_E["request"], SCENARIO_E["available"],
            )
            n_p = len(SCENARIO_E["allocation"])
            n_r = len(SCENARIO_E["available"])

            st.markdown("**Original:**")
            show_status_banner(result["original_detection"]["status"])

            st.markdown("**After Resource Ordering:**")
            show_status_banner(result["modified_detection"]["status"])

            for r in result["ordering_results"]:
                icon = "✅" if r["allowed"] else "❌"
                st.markdown(f"{icon} P{r['process']}: {r['reason']}")

            st.success(f"Coffman condition eliminated: **{result['coffman_eliminated']}**")

            orig_dl = result["original_detection"]["status"] == DEADLOCKED
            mod_safe = result["modified_detection"]["status"] == SAFE
            passed = orig_dl and mod_safe
            test_results.append({
                "Test Case": "TC5",
                "System Condition": "Resource ordering breaks circular wait",
                "Actual Result": f"Original:{result['original_detection']['status']} → Modified:{result['modified_detection']['status']}",
                "Expected Result": "DEADLOCKED → SAFE",
                "Analysis/Conclusion": "PASS — ordering prevents deadlock" if passed else "FAIL",
            })
            st.session_state["tc5"] = test_results[-1]

    # ---- TC6: Boundary / Invalid ----
    with st.expander("TC6 — Boundary & Invalid Inputs (Scenario F)", expanded=False):
        if st.button("▶️ Run TC6", key="run_tc6"):
            all_pass = True

            # F1: Available all zeros — valid boundary
            st.markdown("**F1: Available all zeros (valid boundary)**")
            v1, m1 = validate_system(
                SCENARIO_F_ZERO_AVAILABLE["allocation"],
                SCENARIO_F_ZERO_AVAILABLE["max_matrix"],
                SCENARIO_F_ZERO_AVAILABLE["available"],
            )
            st.markdown(f"Valid: {v1} — {m1}")
            if not v1:
                all_pass = False

            # F2: Zero need — finishes immediately
            st.markdown("**F2: Process with zero Need (finishes immediately)**")
            need_f2 = compute_need(
                SCENARIO_F_ZERO_NEED["allocation"],
                SCENARIO_F_ZERO_NEED["max_matrix"],
            )
            st.dataframe(matrix_to_df(need_f2, 2, 3))
            res_f2 = safety_algorithm(
                SCENARIO_F_ZERO_NEED["allocation"], need_f2,
                SCENARIO_F_ZERO_NEED["available"],
            )
            show_status_banner(res_f2["status"])
            # P0 should finish first (need=0)
            if res_f2["status"] == SAFE and res_f2["sequence"][0] == 0:
                st.success("P0 finishes immediately (Need = 0)")
            else:
                all_pass = False

            st.divider()
            st.markdown("**Invalid Input Tests:**")

            # F3: Invalid process id
            st.markdown("*F3: Invalid process id (99)*")
            r3 = request_resources(99, [1,0,0], copy.deepcopy(BASE_ALLOCATION), copy.deepcopy(BASE_MAX), BASE_AVAILABLE[:])
            st.markdown(f"Status: {r3['status']} — {r3.get('message','')}")
            if r3["status"] != INVALID_INPUT:
                all_pass = False

            # F4: Negative request
            st.markdown("*F4: Negative request (-1,0,0)*")
            r4 = request_resources(0, [-1,0,0], copy.deepcopy(BASE_ALLOCATION), copy.deepcopy(BASE_MAX), BASE_AVAILABLE[:])
            st.markdown(f"Status: {r4['status']} — {r4.get('message','')}")
            if r4["status"] != INVALID_INPUT:
                all_pass = False

            # F5: Wrong vector length
            st.markdown("*F5: Wrong vector length (1,0)*")
            r5 = request_resources(0, [1,0], copy.deepcopy(BASE_ALLOCATION), copy.deepcopy(BASE_MAX), BASE_AVAILABLE[:])
            st.markdown(f"Status: {r5['status']} — {r5.get('message','')}")
            if r5["status"] != INVALID_INPUT:
                all_pass = False

            # F6: Allocation exceeding Max
            st.markdown("*F6: Allocation exceeding Max*")
            f6 = SCENARIO_F_INVALID_CASES[3]
            v6, m6 = validate_system(f6["allocation"], f6["max_matrix"], f6["available"])
            st.markdown(f"Valid: {v6} — {m6}")
            if v6:
                all_pass = False

            # F7: Request exceeding Need
            st.markdown("*F7: Request exceeding Need (P1 asks 2,0,0 but Need A=1)*")
            r7 = request_resources(1, [2,0,0], copy.deepcopy(BASE_ALLOCATION), copy.deepcopy(BASE_MAX), BASE_AVAILABLE[:])
            st.markdown(f"Status: {r7['status']} — {r7.get('message','')}")
            if r7["status"] != INVALID_INPUT:
                all_pass = False

            test_results.append({
                "Test Case": "TC6",
                "System Condition": "Boundary valid + invalid inputs",
                "Actual Result": "All handled correctly" if all_pass else "Some failures",
                "Expected Result": "Valid boundaries accepted; invalid rejected",
                "Analysis/Conclusion": "PASS" if all_pass else "FAIL — see details above",
            })
            st.session_state["tc6"] = test_results[-1]

    # ---- Summary Table ----
    st.divider()
    st.subheader("📊 Test Results Summary")

    # Collect from session state
    summary_rows = []
    for key in ["tc1", "tc2", "tc3", "tc4", "tc5", "tc6"]:
        if key in st.session_state:
            summary_rows.append(st.session_state[key])

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # CSV download
        csv_buf = io.StringIO()
        summary_df.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Download Results CSV",
            csv_buf.getvalue(),
            "test_results.csv",
            "text/csv",
        )
    else:
        st.info("Run the test cases above to populate the summary table.")
