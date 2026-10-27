# Deadlock Analysis & Resource Allocation Dashboard

A comprehensive, interactive dashboard for understanding and demonstrating deadlock concepts in Operating Systems. Built with Python and Streamlit.

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
cd deadlock-dashboard
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Run Tests

```bash
pytest tests/ -v
```

---

## 📁 Module Overview

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI only (4 tabs). No algorithm logic. |
| `banker.py` | Banker's algorithm: `compute_need()`, `safety_algorithm()`, `request_resources()` |
| `detection.py` | Multi-instance matrix detection + single-instance RAG cycle detection |
| `prevention.py` | Prevention policies (resource ordering, request-all-upfront) + recovery |
| `validation.py` | Shared input validation + status constants (`SAFE`, `UNSAFE`, `WAITING`, etc.) |
| `samples.py` | Preloaded test data for Scenarios A–F |
| `tests/` | pytest test suite for all modules |

---

## 🏗️ Architecture

```
┌─────────┐     ┌────────────┐     ┌──────────────┐     ┌───────────────┐
│ app.py  │────▶│ banker.py  │     │ detection.py │     │ prevention.py │
│ (UI)    │────▶│ (avoidance)│     │ (detection)  │     │ (prevention)  │
│         │────▶│            │     │              │     │ (recovery)    │
└─────────┘     └─────┬──────┘     └──────┬───────┘     └───────┬───────┘
                      │                   │                     │
                      └───────────┬───────┘─────────────────────┘
                                  ▼
                          ┌──────────────┐     ┌────────────┐
                          │ validation.py│     │ samples.py │
                          │ (shared)     │     │ (test data)│
                          └──────────────┘     └────────────┘
```

---

## 🔑 Status Model

| Status | Colour | Meaning |
|---|---|---|
| **SAFE** | 🟢 Green | All processes can finish |
| **UNSAFE** | 🟠 Orange | No safe sequence — deadlock is *possible* |
| **WAITING** | 🔵 Blue | Request within Need but > Available — deferred |
| **DENIED_UNSAFE** | 🔴 Red | Would make state unsafe — denied, state restored |
| **DEADLOCKED** | 🔴 Red | Processes blocked in cycle — deadlock has occurred |
| **INVALID_INPUT** | ⚪ Gray | Input failed validation |
| **GRANTED** | 🟢 Green | Request granted, new state is safe |

> ⚠️ An unsafe state is NEVER labelled as deadlocked. A WAITING request is NEVER labelled as invalid.

---

## 📋 Test Case Screenshot Checklist

| TC | Scenario | What to screenshot |
|---|---|---|
| TC1 | Safe state | Safe sequence + Work/Finish trace table |
| TC2 | Waiting | WAITING banner + state-unchanged confirmation |
| TC3 | Unsafe + Granted | DENIED_UNSAFE banner + before/after matrices + GRANTED contrast |
| TC4 | Deadlock | Multi-instance detection + RAG graph with cycle + recovery |
| TC5 | Prevention | Original vs modified side-by-side + ordering violations |
| TC6 | Boundary/Invalid | Valid boundaries accepted + all invalid inputs rejected |

---

## 📝 AI Usage Record

| AI Tool Used | Purpose | Output Generated | Student Verification |
|---|---|---|---|
| **Google Antigravity IDE / Gemini 3.8 Flash** | Architecture & Interface Design | Decoupled modular design separating UI (`app.py`) from algorithms (`banker.py`, `detection.py`, `prevention.py`, `validation.py`). | Verified strict separation: zero algorithm calls in `app.py`. Standardized return dictionaries across all modules. |
| **Google Antigravity IDE / Gemini 3.8 Flash** | Algorithmic Implementation | Implementations of Banker's safety algorithm, rollback on `DENIED_UNSAFE`, NetworkX RAG cycle detection, and Havender's resource ordering. | Hand-calculated safety algorithm traces against textbook benchmark; verified that multi-instance cycles do not necessarily deadlock. |
| **Google Antigravity IDE / Gemini 3.8 Flash** | Automated Unit Testing | Scaffolded 47 `pytest` unit test cases across `test_banker.py`, `test_detection.py`, `test_prevention.py`, and `test_validation.py`. | Executed test suite locally (`python -m pytest -v`); achieved 100% pass rate (47/47) across all edge, boundary, and illegal input tests. |
| **Google Antigravity IDE / Gemini 3.8 Flash** | Frontend UI & Interactive Visualization | Streamlit 4-tab interface, dynamic data editors, status banners, NetworkX RAG plots, and CSV export. | Verified interactive editing, state resets, and strict status invariant display (SAFE, UNSAFE, WAITING, DENIED_UNSAFE, DEADLOCKED). |
| **Google Antigravity IDE / Gemini 3.8 Flash** | Analytical Evaluation & Report Generation | Completed Analytical Test-Case Table synthesis, Work vector traces, and publication-ready `.docx` report generation. | Audited all technical explanations against Operating System Concepts (Silberschatz) 10th Ed. and Coffman condition principles. |

---

## 🏷️ Assumptions

1. Resource names are single uppercase letters (A, B, C, …) and process names are P0, P1, ….
2. The safety algorithm scans processes in order P0, P1, … and picks the first eligible process (any valid safe sequence is acceptable, not just a specific one).
3. For detection, the `Request` matrix represents **current outstanding requests**, not the Need (maximum future claim).
4. In the resource-ordering prevention policy, violating requests are zeroed out (the process gives up the request) rather than killing the process.
5. The request-all-upfront policy uses a greedy allocation in process-id order to determine how many can run concurrently.

## ⚠️ Possible Differences from Textbook

- Some textbooks initialize `Finish[i] = False` for all processes in the detection algorithm; this implementation sets `Finish[i] = True` when `Allocation[i] == 0` (following Silberschatz, Galvin & Gagne).
- The safe sequence returned may differ from a textbook example because any valid ordering is correct.
- The request-all-upfront concurrency metric is a simplified estimate (greedy, not optimal).
