# =============================================================================
# detection.py — Deadlock Detection Algorithms
# =============================================================================
# Two separate sub-modes:
#
#   1. detect_deadlock_multi()  — matrix-based detection for MULTIPLE-instance
#      resources.  Uses Allocation, Request (outstanding), and Available.
#
#   2. build_rag() / detect_cycle_single()  — resource-allocation graph and
#      cycle detection for SINGLE-instance resources, using networkx.
#
# IMPORTANT: This module does NOT call anything from banker.py.
# The detection algorithm is structurally similar to the safety algorithm but
# operates on a *Request* matrix (current outstanding requests), not a Need
# matrix (maximum future needs).
# =============================================================================

from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx            # type: ignore
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for Streamlit
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as mpatches  # type: ignore

from validation import SAFE, DEADLOCKED, validate_detection_inputs


# =====================================================================
# 1.  MULTI-INSTANCE DETECTION  (matrix algorithm)
# =====================================================================
# Algorithm (textbook deadlock detection):
#   1. Let Work = copy of Available;  Finish[i] = True if Allocation[i] == 0,
#      else Finish[i] = False.
#   2. Repeat:
#       a. Find an i where Finish[i] == False AND Request[i] <= Work.
#       b. If found: Work += Allocation[i];  Finish[i] = True; record step.
#       c. If not found: stop.
#   3. Any process with Finish[i] == False is DEADLOCKED.
# =====================================================================
def detect_deadlock_multi(
    allocation: List[List[int]],
    request: List[List[int]],
    available: List[int],
) -> Dict[str, Any]:
    """
    Matrix-based deadlock detection for multi-instance resources.

    Returns:
        status     — SAFE or DEADLOCKED
        steps      — list of step dicts (like safety_algorithm)
        deadlocked — list of deadlocked process indices
        sequence   — order in which processes were marked finished
    """
    # Validate inputs first
    valid, msg = validate_detection_inputs(allocation, request, available)
    if not valid:
        return {"status": "INVALID_INPUT", "message": msg}

    num_processes = len(allocation)
    num_resources = len(available)

    work = available[:]

    # Finish[i] starts True only if the process holds nothing
    finish: List[bool] = []
    for i in range(num_processes):
        all_zero = all(allocation[i][j] == 0 for j in range(num_resources))
        finish.append(all_zero)

    sequence: List[int] = []
    steps: List[Dict[str, Any]] = []

    changed = True
    while changed:
        changed = False
        for i in range(num_processes):
            if finish[i]:
                continue

            # Check Request[i] <= Work
            can_proceed = True
            for j in range(num_resources):
                if request[i][j] > work[j]:
                    can_proceed = False
                    break

            if can_proceed:
                work_before = work[:]
                for j in range(num_resources):
                    work[j] += allocation[i][j]
                work_after = work[:]

                finish[i] = True
                sequence.append(i)
                changed = True

                steps.append({
                    "step":        len(steps) + 1,
                    "process":     f"P{i}",
                    "request":     request[i][:],
                    "work_before": work_before,
                    "work_after":  work_after,
                    "finish":      finish[:],
                })
                break  # restart scan from P0

    deadlocked = [i for i in range(num_processes) if not finish[i]]

    if deadlocked:
        return {
            "status":     DEADLOCKED,
            "steps":      steps,
            "deadlocked": deadlocked,
            "sequence":   sequence,
        }
    else:
        return {
            "status":     SAFE,
            "steps":      steps,
            "deadlocked": [],
            "sequence":   sequence,
        }


# =====================================================================
# 2.  SINGLE-INSTANCE DETECTION  (RAG + cycle detection)
# =====================================================================
# Steps:
#   1. Build a directed graph (resource-allocation graph / wait-for graph).
#      - Nodes: processes (type="process"), resources (type="resource").
#      - Edges:
#          assignment edge: resource → process  (resource is held by process)
#          request edge:    process → resource  (process is waiting for resource)
#   2. Use networkx to find a directed cycle.
#   3. For single-instance resources a cycle ⇔ deadlock.
# =====================================================================
def build_rag(
    assignments: Dict[str, str],
    requests: Dict[str, str],
) -> nx.DiGraph:
    """
    Build a Resource Allocation Graph.

    Parameters:
        assignments — {resource_name: process_name} for each held resource.
        requests    — {process_name: resource_name} for each waiting process.

    Returns a networkx DiGraph with node attribute 'ntype' ∈ {"process","resource"}.
    """
    graph = nx.DiGraph()

    # Add assignment edges: resource → process (the resource is assigned to that process)
    for resource, process in assignments.items():
        graph.add_node(resource, ntype="resource")
        graph.add_node(process, ntype="process")
        graph.add_edge(resource, process, etype="assignment")

    # Add request edges: process → resource (the process is requesting that resource)
    for process, resource in requests.items():
        graph.add_node(resource, ntype="resource")
        graph.add_node(process, ntype="process")
        graph.add_edge(process, resource, etype="request")

    return graph


def detect_cycle_single(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Detect a cycle in a single-instance RAG.

    Returns:
        has_cycle  — bool
        cycle      — list of node names forming the cycle (empty if none)
        status     — DEADLOCKED or SAFE
    """
    try:
        cycle = nx.find_cycle(graph, orientation="original")
        # cycle is a list of (u, v, direction) tuples
        cycle_nodes = [edge[0] for edge in cycle]
        cycle_nodes.append(cycle[-1][1])  # close the cycle

        return {
            "has_cycle": True,
            "cycle":     cycle_nodes,
            "status":    DEADLOCKED,
        }
    except nx.NetworkXNoCycle:
        return {
            "has_cycle": False,
            "cycle":     [],
            "status":    SAFE,
        }


# =====================================================================
# 3.  draw_rag  — visualise the RAG with matplotlib
# =====================================================================
def draw_rag(
    graph: nx.DiGraph,
    cycle_nodes: Optional[List[str]] = None,
    title: str = "Resource Allocation Graph",
) -> plt.Figure:
    """
    Draw the RAG.  Processes = circles, resources = squares.
    Assignment edges = solid blue, request edges = dashed red.
    Cycle edges (if any) are highlighted in thick orange.

    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title, fontsize=14, fontweight="bold")

    pos = nx.spring_layout(graph, seed=42, k=2.0)

    # Separate node lists
    process_nodes = [n for n, d in graph.nodes(data=True) if d.get("ntype") == "process"]
    resource_nodes = [n for n, d in graph.nodes(data=True) if d.get("ntype") == "resource"]

    # Determine cycle edges for highlighting
    cycle_edge_set = set()
    if cycle_nodes and len(cycle_nodes) > 1:
        for k in range(len(cycle_nodes) - 1):
            cycle_edge_set.add((cycle_nodes[k], cycle_nodes[k + 1]))

    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos, nodelist=process_nodes,
        node_shape="o", node_color="#4FC3F7", node_size=800,
        edgecolors="black", linewidths=1.5, ax=ax,
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=resource_nodes,
        node_shape="s", node_color="#AED581", node_size=800,
        edgecolors="black", linewidths=1.5, ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=11, font_weight="bold", ax=ax)

    # Draw edges by type
    for u, v, data in graph.edges(data=True):
        is_cycle_edge = (u, v) in cycle_edge_set
        etype = data.get("etype", "request")

        if is_cycle_edge:
            color = "#FF6F00"
            width = 3.5
            style = "solid"
        elif etype == "assignment":
            color = "#1565C0"
            width = 2.0
            style = "solid"
        else:
            color = "#C62828"
            width = 2.0
            style = "dashed"

        nx.draw_networkx_edges(
            graph, pos, edgelist=[(u, v)],
            edge_color=color, width=width, style=style,
            arrows=True, arrowsize=20,
            connectionstyle="arc3,rad=0.1", ax=ax,
        )

    # Legend
    legend_handles = [
        mpatches.Patch(color="#4FC3F7", label="Process (circle)"),
        mpatches.Patch(color="#AED581", label="Resource (square)"),
        mpatches.Patch(color="#1565C0", label="Assignment edge (solid)"),
        mpatches.Patch(color="#C62828", label="Request edge (dashed)"),
    ]
    if cycle_edge_set:
        legend_handles.append(mpatches.Patch(color="#FF6F00", label="Cycle edge"))
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    ax.axis("off")
    fig.tight_layout()
    return fig
