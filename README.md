# CircularLoop Lab

**Circular strategy decision support from lifecycle data: an independent synthetic research demonstrator.**

CircularLoop Lab explores one applied research question:

> How can structured product lifecycle data support data-driven decisions between
> repair, reuse, refurbishment, remanufacturing and recycling for durable industrial components?

It takes a structured lifecycle record of an industrial component (rail bogie frame, gearbox, drive shaft,
structural profile, motor housing), checks the data, predicts remaining useful life (RUL), and then
ranks five circular strategies with a transparent, rule-guarded multi-criteria analysis, showing
*why* a strategy wins, how close the call is, and what would change it.

> **Disclaimer.** This is an independent demonstrator inspired by publicly described circular
> lifecycle decision-support challenges. It is **not** an official implementation of any research
> project, university or company, **not** compliant with any regulatory product-data standard, and
> **not** validated for real industrial or safety-relevant decisions. All data are synthetic and
> all thresholds, costs and carbon factors are illustrative assumptions.

## Architecture

```
lifecycle record
      ↓
Data validation
      ↓
RUL prediction
      ↓
Technical feasibility
      ↓
Cost / carbon / life-extension evaluation
      ↓
MCDA
      ↓
Circular strategy recommendation
```

Prediction (*how long can it last?*) and decision support (*what should we do with it?*) are
deliberately separate. The ML model never outputs a strategy; strategies are ranked by an
explainable decision layer in which hard technical gates cannot be outweighed.

```
app.py                  thin Streamlit entry point (layout + routing)
circularloop/           domain logic, no Streamlit imports
  config.py             every assumption, threshold, weight and scenario delta
  synthetic_data.py     fleet generator + hidden degradation function
  validation.py         critical / warning / info checks, quality score
  rul_model.py          LR vs. Gradient Boosting, metrics, intervals, explainability
  decision_engine.py    estimates, gates, feasibility, MCDA, weight sensitivity
  explanation.py        deterministic plain-English explanations
  scenarios.py          current / improved maintenance / severe degradation
  record_schema.py         field catalogue and lifecycle-record JSON export
  pipeline.py           single orchestration: record → Assessment
  showcase.py           hand-defined demonstration inputs
ui/                     Streamlit views, charts and components
tests/                  pytest suite (51 tests)
docs/methodology.md     full method, formulas and limitations
data/                   generated sample dataset and sample lifecycle record
```

## Installation and running

Requires Python 3.11+.

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

The first start trains both models on the synthetic fleet (≈ 15 s); afterwards they are cached.
`data/sample_components.csv` is regenerated automatically if it is missing. To regenerate both data
files explicitly:

```bash
python -m circularloop
```

Run the tests and the linter:

```bash
pytest
```

```bash
ruff check .
```

No secrets, paid services or external APIs are required, so the app can be deployed to Streamlit
Community Cloud directly from a GitHub repository (entry point `app.py`).

## What the app shows

| Tab | Purpose |
|---|---|
| About | the problem, why it is hard, the approach, design choices and a guided tour |
| Decision | lifecycle and condition data, data quality, RUL with interval, recommendation, ranking |
| Why this recommendation? | plain-English explanation, per-strategy cost / CO₂ / life / feasibility / circularity, weighted contributions, normalised criteria, hard-gate checks, weight sensitivity |
| Scenarios | current vs. improved maintenance vs. severe degradation |
| Sandbox | edit condition and economic values, hide fields, and watch RUL and ranking respond |
| Benchmark | Linear Regression vs. Gradient Boosting on held-out data, errors, importance, coefficients, response curves |
| Record | sectioned lifecycle-record JSON (downloadable) and field catalogue |
| Method | pipeline, assumptions, limitations, research questions |

Five showcase cases cover a healthy part, a mid-life gearbox, a heavily corroded frame, a partially
documented record (imputation) and a record without thickness data (no recommendation).

## Methodology in brief

**Synthetic data.** 500 components generated in causal order (age → duty → usage → corrosion →
thickness → vibration → failures → economics). The RUL target comes from a hidden function with
additive consumption, multiplicative condition penalties, a logistic thickness threshold and
hinge effects, plus **unobserved** duty-severity and build-quality factors and heteroscedastic
noise, so it cannot be reconstructed exactly. Six deliberately defective records exercise the
validation layer and are excluded from training.

**Regression.** Linear Regression (scaled, one-hot) is the interpretable baseline. Gradient Boosting
(`HistGradientBoostingRegressor`) is the nonlinear comparison, with monotonic constraints that
encode engineering priors, added after the unconstrained model learned a confounded "longer since
maintenance → longer life" pattern. The decision model is selected by cross-validated MAE on the
training split; the test split is only reported.

**Decision layer.** For each strategy the engine estimates cost, CO₂ and service life on a
**functional basis** (per service-year; recycling includes the new replacement), computes graded
technical feasibility as the margin to that strategy's own limits, applies **hard gates** (e.g. no
reuse below 80 % thickness, above corrosion 5, or with a conservative RUL under 12 months), and
scores criteria on fixed scales against a "buy new" reference. A Monte-Carlo weight perturbation
reports how stable the recommendation is.

Full details, formulas and parameter values: [docs/methodology.md](docs/methodology.md).

## Model comparison (synthetic data, seed 42, 125-component test split)

| Model | MAE (months) | RMSE (months) | R² |
|---|---:|---:|---:|
| Linear Regression | 34.4 | 46.8 | 0.840 |
| Gradient Boosting (monotonic) | 28.1 | 39.7 | 0.885 |
| Gradient Boosting (unconstrained, ablation) | 29.2 | 40.5 | 0.880 |

Gradient Boosting is more accurate here because the synthetic degradation is multiplicative and
thresholded. These numbers describe recovery of an assumed function, not real-world accuracy.

## Decision logic

| Strategy | Hard gates (blocked if…) | Service life delivered |
|---|---|---|
| Reuse | thickness < 80 %, corrosion > 5, P10 RUL < 12 months, failures > 2, vibration > 1.5 × alarm | predicted RUL |
| Repair | thickness < 65 %, corrosion > 7, failures > 4 | RUL + 25 % of consumed life |
| Refurbish | thickness < 60 %, corrosion > 8.5 | RUL + 55 % of consumed life |
| Remanufacture | thickness < 45 %, corrosion > 9.5 | 85 % of design life |
| Recycle (+ replace) | never blocked | design life of the new part |

Default weights: technical feasibility 0.30, CO₂ 0.20, cost 0.20, service life 0.20,
circularity 0.10 (adjustable in the sidebar, automatically normalised).

## Limitations

- Synthetic dataset and an assumed degradation function.
- Illustrative lifecycle, cost and carbon assumptions, not LCC or LCA results.
- Gate thresholds are placeholders, not safety-validated engineering limits.
- Not an official implementation of any research project, university or company.
- Not compliant with any regulatory product-data, AAS or EPCIS specification.
- Not intended for real industrial decision-making.

## Questions this prototype raises

1. Which lifecycle-record attributes are actually required for circular lifecycle decisions?
2. How should missing or uncertain lifecycle data affect recommendations?
3. How accurately can RUL be estimated from real, sparse and censored industrial data?
4. How should lifecycle cost and lifecycle carbon be estimated and allocated between life cycles?
5. Who should set MCDA weights, and how should disagreement be handled?
6. How should recommendations be validated against expert decisions?
7. How can lifecycle records be exchanged across organisations without losing semantics?
8. How should prediction uncertainty enter strategy selection?

A real implementation could later map the simplified record towards the Asset Administration Shell,
GS1 EPCIS, OPC UA and emerging EU product-data requirements under the Ecodesign for Sustainable Products Regulation (ESPR); no such mapping is
implemented or claimed here.

## Licence, attribution and disclaimer

- **Licence.** The code and documentation are released under the MIT License (see `LICENSE`).
- **Third-party components.** The bundled Inter typeface is used under the SIL Open Font License 1.1;
  Python dependencies are installed from PyPI under their own permissive licences. Details are in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **No borrowed branding or content.** The repository contains no logos, institutional branding,
  screenshots, proprietary diagrams, images or copied text. Icons are the Material Symbols set
  shipped with Streamlit; all charts are generated from the synthetic data.
- **No affiliation.** This is an independent project. It is not affiliated with, endorsed by or
  derived from any university, research project, company or standards body. Names of standards and
  products appear only to identify them and may be trademarks of their respective owners.
- **Data and privacy.** All data are synthetic and contain no personal information. Run locally with
  the included configuration, the app collects no usage statistics, sets no tracking or analytics
  cookies, and serves its fonts locally rather than from a third-party font service. A hosting
  platform used for deployment may apply its own terms and privacy policy.
- **No warranty, not for operational use.** The software is provided "as is", without warranty of
  any kind (see `LICENSE`). Results are illustrative and must not be used for engineering, safety,
  financial or regulatory decisions.
