# CircularLoop Lab: Implementation Specification (v2, source of truth)

> **Status:** revised specification. Version 1 was critically reviewed before implementation;
> Section 0 records what changed and why. Everything below Section 0 is the binding
> description of what is implemented.

---

## 0. Review of v1 and summary of changes

v1 had the right central idea (*lifecycle record → RUL prediction → transparent decision layer*), but
several modelling and design choices would have produced a demo that looks plausible while being
methodologically weak. The main findings and the resulting changes:

| # | Weakness in v1 | Why it matters | Change in v2 |
|---|---|---|---|
| 1 | Strategy cost and CO₂ compared as **absolute** values | Recycling "costs" almost nothing and reuse emits almost nothing, but they deliver very different amounts of service. Comparing absolute numbers rewards strategies that deliver little or no function. | All strategies are compared on a **functional basis**: cost and CO₂ **per year of service delivered** (equivalent annual cost/carbon). "Recycle" is modelled as *recycle + replace with a new component*, because the function must still be provided. |
| 2 | **Min–max normalisation** across the five strategies | Exaggerates trivial differences (0.91 vs 0.93 becomes 0 vs 1), causes rank reversal when an option is added or blocked, and makes scores incomparable between components. | **Reference-point normalisation** against a fixed "buy new" baseline and fixed scales. A score of 0.6 means the same thing for every component. |
| 3 | Feasibility used both as a weighted criterion **and** as a penalty, unspecified | Double counting, and with a low feasibility weight unsafe options can still win. | Two distinct mechanisms: (a) **hard gates** that block a strategy regardless of weights; (b) a graded **technical-feasibility criterion** inside the weighted sum. Gates are tested with the feasibility weight set to zero. |
| 4 | Point RUL used for safety-relevant gates | Gating on the point estimate ignores prediction error. | An approximate **80 % prediction interval** from cross-validated residuals; reuse gates use the **conservative (P10) RUL**. |
| 5 | `manufacture_year` and `age_years` both used as features | Perfectly collinear (one is derived from the other) → unstable linear coefficients. | `manufacture_year` is kept in the lifecycle record but **excluded from model features**. Economic fields (cost, CO₂, recyclability, spare parts) are also excluded: they are not physical degradation drivers. |
| 6 | Hidden RUL function fully determined by observed features | A model could trivially reconstruct the target; metrics would be meaninglessly high. | Two **latent, unobserved** factors (duty severity, build quality) and heteroscedastic noise create an irreducible error floor. |
| 7 | Condition variables sampled independently | Unrealistic: corrosion, thickness loss, vibration and failures should co-evolve with age and duty. | **Causal generation order**: type → age → duty → usage → corrosion → thickness → vibration → failures → maintenance. |
| 8 | Model selection "best by test MAE" | Selecting on the test set leaks test information into the decision pipeline. | Model used for decisions is selected by **5-fold CV MAE on the training set**; the test set is only reported. User can override. |
| 9 | Impurity-based GB feature importance | Biased towards high-cardinality features; not comparable with LR coefficients. | **Permutation importance on the held-out set for both models** + standardised LR coefficients. |
| 10 | "Insufficient data" only for missing fields | Invalid or contradictory values are equally dangerous. | Validation layer with severities (*critical / warning / info*); any critical issue → **no recommendation**. Non-critical missing values are imputed and lower confidence. |
| 11 | Strategy semantics undefined (reuse vs. repair vs. refurbish) | Ambiguous definitions make any scoring arbitrary. | Explicit operational definitions (Section 7.1). |
| 12 | Single score as the answer | Hides how fragile the recommendation is. | **Decision margin** and **weight-sensitivity analysis** (Monte-Carlo perturbation of weights) → decision confidence. |
| 13 | Package named `src/`, repo and spec named after an existing research project | `import src.x` is poor practice; reusing a project name implies affiliation. | Package `circularloop/`, UI package `ui/`, neutral project identity "CircularLoop Lab". |
| 14 | Demo story hard-coded the answer ("→ Refurbish") | Tuning a model to reach a predetermined answer is bad research practice. | Showcase cases are defined by **input condition only**; the outcome is whatever the engine produces and is reported honestly. |
| 15 | Six component types incl. brake housing, three materials | Gearbox and motor housings are typically cast. | Five types (rail bogie frame, industrial gearbox, drive shaft, steel structural profile, electric motor housing); four materials incl. cast iron, with type-specific material mixes. |
| 16 | Training on unvalidated rows | Contradicts the claim that validation guards the pipeline. | The fleet file contains a few deliberately defective records; **only records passing validation are used for training**. |

**Changes found necessary during implementation** (each was discovered by testing the built system):

| # | Observation | Change |
|---|---|---|
| 17 | Unconstrained boosting learned "longer since maintenance → longer life" (never-maintained parts are young). "Improved maintenance" then *lowered* RUL for 209 of 500 parts. | Gradient Boosting implemented as `HistGradientBoostingRegressor` with **monotonic constraints** encoding engineering priors; the unconstrained model is reported as an ablation. |
| 18 | Fixed life uplifts ("RUL + 0.35 D") recommended refurbishing nearly new parts. | Repair/refurbish recover a **share of consumed life** (max(D − RUL, 0)); a healthy part gains nothing from intervention. |
| 19 | One absolute thickness/corrosion scale made remanufacturing look infeasible for recoverable cores, so recycling won too often. | Thickness/corrosion feasibility measured as the **margin to each strategy's own gate**. |
| 20 | Imputed fields were only reflected in completeness; a record missing four fields still showed "Sufficient". | −5 quality points per imputed field. |
| 21 | Constant-width RUL interval (±45 months) made the conservative RUL zero for most short-lived parts. | Residuals scaled by prediction size → interval width proportional to predicted life. |

---

## 1. Purpose and positioning

Build a polished, end-to-end **synthetic research demonstrator** in Python exploring:

> How can structured product lifecycle data support data-driven decisions between repair,
> reuse, refurbishment, remanufacturing and recycling for durable industrial components?

The demonstrator is **inspired by publicly described circular lifecycle decision-support research**
in production and maintenance management. It is:

- an **independent** synthetic research demonstrator,
- **not** an official implementation of any research project, university or company,
- **not** compliant with any regulatory product-data standard,
- **not** validated for real industrial decisions.

Project identity: **CircularLoop Lab**: *Circular Strategy Decision Support from Lifecycle Data*.
Note shown in UI: *Independent synthetic research demonstrator inspired by publicly described circular
lifecycle decision-support challenges.*

No logos, institutional branding, partner names as endorsements, or copied text/figures anywhere.

---

## 2. Architecture

```text
lifecycle record
      ↓
Data validation            (critical / warning / info; quality score)
      ↓
RUL prediction             (LR baseline vs. Gradient Boosting; P10–P90 interval)
      ↓
Technical feasibility      (hard gates + graded feasibility per strategy)
      ↓
Cost / carbon / life-extension evaluation   (functional basis: per service-year)
      ↓
MCDA                       (reference-point normalisation, adjustable weights)
      ↓
Circular strategy recommendation  (+ margin, sensitivity, deterministic explanation)
```

Prediction answers *"how long can it last?"*; the decision layer answers *"what should we do with it?"*.
The ML model never outputs a strategy.

---

## 3. Technology stack

Python ≥ 3.11 · pandas · numpy · scikit-learn · Streamlit · Plotly · pytest. Optional dev: ruff.
No deep learning, no LLM APIs, no paid services, no secrets, no databases. Runs locally and on
Streamlit Community Cloud (relative paths only).

---

## 4. Repository structure

```text
.
├── app.py                       # thin Streamlit entry point (layout + routing only)
├── CircularLoop_Lab_Spec.md     # this specification
├── README.md
├── requirements.txt
├── pyproject.toml               # ruff + pytest configuration
├── .gitignore
├── .streamlit/config.toml       # neutral theme
├── circularloop/                # domain logic, no Streamlit imports
│   ├── __init__.py
│   ├── config.py                # all assumptions / thresholds / weights / scenario deltas
│   ├── record_schema.py            # field catalogue, sections, JSON export
│   ├── synthetic_data.py        # fleet generator + hidden degradation function
│   ├── data_io.py               # load-or-create dataset
│   ├── validation.py            # record validation + quality score
│   ├── rul_model.py             # training, evaluation, explainability, prediction interval
│   ├── decision_engine.py       # strategy estimates, gates, MCDA, sensitivity
│   ├── scenarios.py             # scenario transformations
│   ├── explanation.py           # deterministic natural-language explanations
│   ├── showcase.py              # hand-defined demonstration cases (inputs only)
│   ├── pipeline.py              # single orchestration: record → Assessment
│   └── __main__.py              # `python -m circularloop` regenerates data files
├── ui/                          # Streamlit-only helpers
│   ├── components.py  charts.py  state.py
│   └── views/ (decision, why, scenarios, sandbox, benchmark, record, research)
├── data/ sample_components.csv  sample_lifecycle_record.json
├── tests/
└── docs/methodology.md
```

---

## 5. Synthetic data

### 5.1 Fleet
- 500 components + 6 deliberately defective records (for validation demonstration), seed `42`,
  reference year 2026.
- Component types and their illustrative design parameters (all synthetic):

| Type | Design life (y) | Nominal h/year | Major load cycles / h | Materials (share) |
|---|---:|---:|---:|---|
| Rail bogie frame | 30 | 4 000 | 30 | HS steel .85, stainless .15 |
| Industrial gearbox | 20 | 6 000 | 15 | cast iron .55, HS steel .45 |
| Drive shaft | 15 | 5 000 | 25 | HS steel .80, stainless .20 |
| Steel structural profile | 40 | 8 000 | 2 | HS steel .75, stainless .25 |
| Electric motor housing | 25 | 5 000 | 8 | aluminium .50, cast iron .40, stainless .10 |

`load_cycles` = counted **major** load events (start/stop, load reversal), not gear-mesh cycles.

### 5.2 Fields

Identity: `component_id, component_type, material, manufacture_year`
Usage: `age_years, operating_hours, load_cycles`
Condition: `vibration_rms` (mm/s), `temperature_mean_c`, `corrosion_score` (0–10), `residual_thickness_pct` (0–100)
Maintenance: `maintenance_count, failure_events, last_maintenance_months_ago, spare_parts_available`
Sustainability/economics: `recyclability_pct, embodied_co2_kg, replacement_cost_eur`
Target: `rul_months` (synthetic).

### 5.3 Causal generation order
type → material → age (Beta-distributed fraction of design life, up to 1.05×) → latent duty severity
→ utilisation → hours → cycles → maintenance policy/count → environment aggressiveness → corrosion
(material-specific rate, reduced by maintenance) → residual thickness (corrosion loss + cycle wear)
→ vibration (type base × duty × ageing) → failure events (Poisson, driven by duty, vibration, age)
→ temperature → spare-part availability (obsolescence with age) → mass → embodied CO₂ (mass ×
material factor) → replacement cost.

### 5.4 Hidden degradation function (documented in `docs/methodology.md`, not prominent in UI)
```text
consumption L = 0.4·age/D + 0.3·hours/(D·h_nom) + 0.3·cycles/(D·h_nom·c_ph)
base          = 12·D·max(0, 1.15 − L)
RUL           = base · f_corr · f_thk · f_vib · f_fail · f_temp · f_latent + maint_credit + ε
f_corr = exp(−0.012·corr²)            f_thk = logistic((thk − 68)/5)
f_vib  = 1/(1 + 2·(max(0, vib − v_alarm)/v_alarm)²)
f_fail = 0.88^failures                 f_temp = exp(−0.03·max(0, T − T_limit))
f_latent = build_quality · duty^−0.3   (both unobserved)
maint_credit = 1.5·min(maint_count, 8)·(1 − min(last_maint/36, 1))
ε ~ N(0, 3 + 0.07·signal);  RUL clipped at 0
```
Linear (consumption), multiplicative, hinge, logistic and interaction effects are all present, so the
comparison between a linear and a nonlinear model is meaningful but not trivial.

### 5.5 Defective records
Six records with: missing residual thickness, negative operating hours, corrosion 14, recyclability
130 %, negative replacement cost, operating hours exceeding calendar hours. Excluded from training,
selectable in the UI to show the "insufficient data" path.

---

## 6. Validation layer (`validation.py`)

Each record yields a `ValidationReport` with a list of `Issue(field, severity, message)`.

- **critical** → no recommendation ("Insufficient data for a reliable recommendation"):
  missing critical field (type, material, age, operating hours, corrosion, residual thickness,
  replacement cost, embodied CO₂); value outside physically possible range (negative hours/cost/CO₂,
  percentages outside 0–100, corrosion outside 0–10, temperature outside −60…250 °C); operating hours
  > calendar hours since commissioning.
- **warning** → recommendation shown with reduced confidence: missing non-critical field (imputed
  from training-fleet median), implausible-but-possible values (temperature > 120 °C, vibration >
  20 mm/s, > 3 600 cycles/h), last maintenance older than the component, manufacture year
  inconsistent with age, more failures than maintenance interventions.
- **info**: e.g. component beyond design life.

`quality_score` (0–100) = weighted completeness (critical fields weight 3, others 1) − 8 per warning
− 5 per imputed field, clipped to [0, 100]. Status: *Sufficient* (no critical, score ≥ 85), *Reduced confidence*,
*Insufficient* (any critical issue).

---

## 7. Decision layer (`decision_engine.py`)

### 7.1 Strategy definitions (operational, for this demonstrator)
- **Reuse**: return to service as-is after inspection/requalification; no material restoration.
- **Repair**: targeted correction of a specific defect (replace bearing/seal, local weld); does not
  restore wall thickness.
- **Refurbish**: restore to good working condition (surface treatment, wear parts, re-coating);
  not to as-new specification.
- **Remanufacture**: disassemble, restore core to as-new specification with renewed warranty.
- **Recycle**: end-of-life: material recovery; the function is provided by a **new replacement
  component** (cost and CO₂ of that replacement are attributed to this option).

### 7.2 Per-strategy estimates (fractions of replacement cost R, embodied CO₂ E, design life D months)

| Strategy | Cost | CO₂ | Service life after intervention |
|---|---|---|---|
| Reuse | 0.04 R | 0.01 E | RUL (point) |
| Repair | 0.12 R × (1.6 if no spare parts) × (1 + 0.05·failures) | 0.05 E | RUL + 0.25 · max(D − RUL, 0) |
| Refurbish | 0.28 R × (1 + 0.04·corrosion) | 0.12 E | RUL + 0.55 · max(D − RUL, 0) |
| Remanufacture | 0.55 R | 0.28 E | 0.85 D |
| Recycle + replace | R − scrap credit (0.05·R·recyclability) | E·(1 − 0.30·recyclability) | D |

Service life capped at D. Annualised: `cost/yr = cost / (life/12)`, `CO₂/yr = CO₂ / (life/12)`.

### 7.3 Technical feasibility (0–1, graded)
Built from condition sub-scores (thickness, corrosion, vibration vs. alarm, failures, conservative
RUL vs. D, spare parts) with strategy-specific weights; Recycle = 1.0. Thickness and corrosion are
scored as the margin to the strategy's own gate (0 at the gate, 1 comfortably inside it). Defined in
`config.py`.

### 7.4 Hard gates (block regardless of weights; editable in `config.py`)
| Strategy | Blocked if |
|---|---|
| Reuse | thickness < 80 % · corrosion > 5 · P10 RUL < 12 months · failures ≥ 3 · vibration > 1.5 × alarm |
| Repair | thickness < 65 % · corrosion > 7 · failures ≥ 5 |
| Refurbish | thickness < 60 % · corrosion > 8.5 |
| Remanufacture | thickness < 45 % · corrosion > 9.5 |
| any (except Recycle) | graded feasibility < 0.20 |

Recycle is never blocked → a feasible recommendation always exists.

### 7.5 MCDA
Criteria scores in [0, 1], **reference-point normalisation** against the "new component" baseline:
- feasibility: as computed
- cost: `clip(1 − (cost/yr ÷ new cost/yr) / 2, 0, 1)` (equal to new → 0.5)
- CO₂: `clip(1 − (CO₂/yr ÷ new CO₂/yr) / 2, 0, 1)`
- life: `life / D`
- circularity: fixed retention hierarchy Reuse 1.0, Repair 0.9, Refurbish 0.8, Remanufacture 0.65,
  Recycle 0.4 × recyclability.

Default weights: feasibility 0.30, CO₂ 0.20, cost 0.20, life 0.20, circularity 0.10; adjustable,
auto-normalised to sum 1 (all-zero → defaults). Score = 100 × Σ wᵢ·sᵢ; blocked strategies score 0
and rank last with the blocking reason.

### 7.6 Robustness
- **Decision margin** = score(1st) − score(2nd).
- **Weight sensitivity**: 400 Dirichlet draws centred on current weights (concentration 30, seed 7);
  stability = share of draws in which the recommended strategy still wins.
- **Decision confidence**: High (quality sufficient, stability ≥ 0.80), Low (stability < 0.55 or
  quality < 60), otherwise Medium.

---

## 8. RUL modelling (`rul_model.py`)

- Features: age, hours, cycles, vibration, temperature, corrosion, thickness, maintenance count,
  failures, months since maintenance, component type, material.
- Pipelines: median imputer + StandardScaler + OneHotEncoder → `LinearRegression`;
  median imputer + OneHotEncoder → `HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
  max_depth=4, min_samples_leaf=10, l2_regularization=1.0, random_state=42)` with monotonic
  constraints (RUL non-increasing in age, hours, cycles, vibration, temperature, corrosion, failures,
  months since maintenance; non-decreasing in thickness and maintenance count). The unconstrained
  variant is reported as an ablation only.
- Split 75/25, `random_state=42`, training only on validated records.
- Reported: test MAE, RMSE, R²; 5-fold CV MAE on training set (used for default model choice).
- Interval: P10/P90 of out-of-fold residuals divided by (prediction + 12 months), re-applied to the
  point prediction (approximate 80 %, width proportional to predicted life).
- Predictions clipped at 0 for display and decisions.
- Explainability: permutation importance (held-out, both models), standardised LR coefficients,
  single-feature response curves for the selected component.

---

## 9. Scenarios (`scenarios.py`)
- **Current state**: record as is.
- **Improved maintenance**: vibration × 0.7, failures − 1, corrosion − 1.0 (protective treatment),
  last maintenance → ≤ 2 months, temperature − 5 °C. (Maintenance count is left unchanged: it is a
  history counter, not a condition.) **Thickness unchanged**
  (material loss is irreversible).
- **Severe degradation**: corrosion + 3, thickness − 15 pp, failures + 2, vibration × 1.6,
  temperature + 10 °C. All values clipped to valid ranges.

---

## 10. Explanations (`explanation.py`)
Deterministic, template-based, built from actual values: why the winner wins (its strongest
criteria vs. the runner-up), what blocks or penalises other strategies (named gate conditions with
the observed value and threshold), and a robustness sentence (margin, stability). No LLM.

---

## 11. Frontend (Streamlit)

Visual direction: technical research prototype + industrial decision support. Neutral slate/teal
palette, restrained typography, bordered cards, compact explanatory captions, Plotly charts in a
consistent style, status pills for data quality and confidence.

Sidebar: component selector (showcase cases first, then fleet), RUL model (auto/LR/GB), MCDA
weight sliders with normalised display, reset, compact disclaimer.

Tabs:
0. **About**: the problem, why it is hard, the four-step approach, design choices, how to explore.
1. **Decision**: identity strip; condition metrics; RUL with interval; data-quality card;
   recommendation card (strategy, score, margin, confidence); strategy ranking chart.
2. **Why this recommendation?**: explanation; per-strategy comparison (cost, CO₂, life, feasibility,
   circularity, score) with absolute and annualised values; stacked weighted-contribution chart;
   normalised criteria and weights; gates triggered; weight-sensitivity chart.
3. **Scenarios**: three scenarios side by side: RUL, recommendation, score chart, key
   changed inputs.
4. **Sandbox**: sliders for condition/economic values and "treat as missing" options;
   live RUL and recommendation vs. the original record.
5. **Benchmark**: metrics table, predicted-vs-actual, residual distributions, permutation
   importance, LR coefficients, response curves.
6. **Record**: sectioned JSON, download, field catalogue with units and criticality.
7. **Method**: pipeline, research questions, limitations, interoperability note.

---

## 12. Lifecycle-record JSON
Sections `identity, usage, condition, maintenance, sustainability, decision_support, metadata`.
`metadata` states: synthetic, schema "circularloop-lifecycle-record/0.3", not standards-compliant.
`data/sample_lifecycle_record.json` is generated for one showcase component.

---

## 13. Interoperability (documentation only)
A real implementation could map this simplified model towards the Asset Administration Shell
(submodels), GS1 EPCIS (lifecycle events), OPC UA (live condition data) and emerging EU product-data requirements (ESPR). Phrase as *potential future interoperability layer*; no fake mappings.

---

## 14. Tests (≥ 15 meaningful, pytest)
Data constraints and determinism; validation rules (percentages, negatives, missing critical,
consistency); training, finite & non-negative predictions, interval ordering, no excluded features;
weight normalisation; full ranking returned and sorted; hard gates hold **even when feasibility
weight is 0 and CO₂ weight is 1**; healthy component gives a high-retention strategy a real chance;
determinism; scenarios monotonic (severe ≤ current RUL); record export round-trip; explanation mentions
winner and blocked strategies; insufficient data → no recommendation.

---

## 15. Research questions surfaced in the app
1. Which lifecycle-record attributes are actually required for circular lifecycle decisions?
2. How should missing or uncertain lifecycle data affect recommendations?
3. How accurately can RUL be estimated from real (censored, sparse) industrial data?
4. How should lifecycle cost and lifecycle carbon be estimated and allocated between life cycles?
5. Who sets MCDA weights (engineers, LCC models, policy), and how?
6. How should recommendations be validated against expert decisions?
7. How can lifecycle records be exchanged across organisations without losing semantics?
8. How should prediction uncertainty enter strategy selection?

---

## 16. Not built (deliberately)
Blockchain, chatbots/LLMs, neural networks, computer vision, IoT streaming, digital-twin claims,
cloud dependencies, PDF export, fake standards compliance.

---

## 17. Acceptance criteria
`streamlit run app.py` works from a fresh clone; dataset auto-created if missing; both models train
and report genuine held-out metrics; five strategies ranked with visible gates; recommendation
reacts to condition and weight changes; insufficient data blocks recommendation; record JSON visible
and downloadable; ≥ 15 tests pass; README and methodology explain method and limitations; no
implied affiliation or copied branding anywhere.
