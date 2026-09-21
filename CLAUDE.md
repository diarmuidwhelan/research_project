# research_project — computer vision research

Research-stage computer vision: detection, tracking, calibration, colour and number
classification, action recognition.

**Status: dormant** — last commit January 2025, no specs, not previously wired to the brain.
**Open decision:** revive under the sport-agnostic ontology, or archive. Until that is decided, treat
anything here as a research corpus, not a product input.

## Plugins

`cortex` `cortex-sport` from the `diarmuid-brain` marketplace. Setup: `C:/Users/Diarmuid/brain/docs/INSTALL.md`.

## The brain

**Read `C:/Users/Diarmuid/brain/CORTEX.md` first.** It is the single operating contract — mission, locked
decisions, hard rules, memory tiers and agent policies. It is not duplicated here, because four
copies of the same rules is how they drift.

| Need | Where |
|---|---|
| Operating contract | `C:/Users/Diarmuid/brain/CORTEX.md` |
| Search past failures and skills | `recall` MCP tool, or `C:/Users/Diarmuid/brain/index/knowledge-index.md` |
| What depends on what | `graph_query` MCP tool |
| Decision gates | `C:/Users/Diarmuid/brain/cortex/gates.md`, or the `gates` MCP tool |
| Data rights basis | `C:/Users/Diarmuid/brain/data/contracts.md`, or `rights_check` |
| Approval before writing to the brain | `C:/Users/Diarmuid/brain/meta/approval-process.md` |

## Enforcement

Five invariants block at the hook layer, not as advice: random splits on dated data, fitting against
a holdout, trading code inside a product repo, secrets, and dataset writes with no rights basis
(advisory). A block names the rule, the line and the remedy.

Genuine exceptions are recorded with a reason — `# cortex:allow <rule> <reason>`. A bare pragma with
no reason suppresses nothing.

Run it by hand or in CI:

```
python C:/Users/Diarmuid/brain/plugins/cortex/scripts/cortex_check.py --all
```

## Method

Spec-driven for anything non-trivial: `sdd-specify` → approve → `sdd-plan-tasks` → implement against
tests derived from the spec's acceptance criteria, written in a separate context from the
implementation.

At session end run `/evolve` — it mines the session for learnings and ADRs, records them on approval,
and regenerates the index and graph.

## If revived

Any event output must conform to the sport-agnostic ontology in `quantiphi-sport-data` — core
`possession -> sequence -> action -> outcome`, per-sport extension, regime indicator, provenance and
rights basis on every row. Research output that does not carry provenance cannot enter the product or
the benchmark population.

Load `gaa-cv-pipeline` and `extract-events-from-video` before building anything here; most of what
this repo was reaching for now exists in `cv_pipeline`.
