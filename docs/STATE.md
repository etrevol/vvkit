# STATE.md — Current Project State

## Status
- **Current Milestone:** Post-M6 — Audit & Report Enrichment
- **Date:** 2026-07-28

## Completed Milestones
- [x] **M0 — Repository skeleton** (tag: `m0`)
- [x] **M1 — Norms and convergence core** (tag: `m1`)
- [x] **M2 — MMS engine** (tag: `m2`)
- [x] **M3 — Runner and adapters** (tag: `m3`)
- [x] **M4 — Checks and regression** (tag: `m4`)
- [x] **M5 — Reporting and CLI** (tag: `m5`)
- [x] **M6 — pytest plugin, docs, release** (tag: `v1.0.0-rc1`)
- [x] **Post-M6 — Full PROJECT_SPEC audit** (enriched HTML reports, multi-plot support, bug fixes, test expansion)

## Recent Changes (Post-M6 Audit)
- **Report enrichment**: 10-section HTML template with metrics dashboard, pairwise orders, error table, GCI table, convergence plot, asymptotic diagnostics, MMS diagnostics, conservation checks, and environment provenance.
- **Multi-graph support**: `plots.py` renders multi-norm convergence plots with fitted slopes, reference slope lines, and round-off floor markers; conservation time-series plots with tolerance bands and departure markers.
- **Conservation fix**: Corrected off-by-one bug in `checks/conservation.py` `imbalance_series` computation.
- **CLI integration**: `main.py` populates all enriched data models (GCITableRow, EnvironmentInfo, MMSDiagnostics, ConservationResultSummary) with system provenance.
- **Test expansion**: Added 10 new test cases covering order recovery, convergence classifications, and conservation departures.

## Test Coverage
- Overall coverage: **95%** (meets quality bar >= 90%)
- `convergence/` coverage: **95%**
- `mms/` coverage: **96%**
- All 30 unit and integration tests passing.
