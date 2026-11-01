# Methodology: CircularLoop Lab

> Independent synthetic research demonstrator. Every number in this document is an
> illustrative assumption chosen to be plausible in order of magnitude. Nothing here is
> validated engineering data, LCA data, or an official methodology of any research project,
> university or company.

## 1. Research question and framing

*How can structured product lifecycle data support data-driven decisions between
repair, reuse, refurbishment, remanufacturing and recycling for durable industrial components?*

The demonstrator splits this into two questions with different natures:

| Question | Nature | Method |
|---|---|---|
| A. How much useful life may remain? | Prediction (regression) | Linear Regression vs. Gradient Boosting |
| B. What should we do with the component? | Decision under multiple objectives | Hard technical gates + transparent MCDA |

Question B is deliberately **not** treated as classification. There is no ground truth "correct
strategy" to learn from, and a classifier would hide the trade-offs that decision makers need to
see. The ML model never outputs a strategy.

```text
lifecycle record → validation → RUL prediction → technical feasibility → cost / CO₂ / life → MCDA → recommendation
```

## 2. Synthetic data

### 2.1 Fleet and component families

500 components (seed 42, reference year 2026) across five families, plus six deliberately
defective records used only to exercise validation.

| Family | Design life D (y) | Nominal h/y | Major load cycles/h | Vibration alarm (mm/s) | Temp. limit (°C) | Material mix |
|---|---:|---:|---:|---:|---:|---|
| Rail bogie frame | 30 | 4 000 | 30 | 3.5 | 60 | HS steel 85 %, stainless 15 % |
| Industrial gearbox | 20 | 6 000 | 15 | 4.5 | 80 | cast iron 55 %, HS steel 45 % |
| Drive shaft | 15 | 5 000 | 25 | 4.5 | 70 | HS steel 80 %, stainless 20 % |
| Steel structural profile | 40 | 8 000 | 2 | 1.8 | 55 | HS steel 75 %, stainless 25 % |
| Electric motor housing | 25 | 5 000 | 8 | 3.5 | 75 | aluminium 50 %, cast iron 40 %, stainless 10 % |

Materials carry an illustrative corrosion rate, CO₂ factor (kg CO₂e per kg, incl. processing),
cost multiplier and recyclability range (`circularloop/config.py → MATERIALS`).

### 2.2 Feature definitions

| Field | Unit | Definition |
|---|---|---|
| `age_years` | years | time in service since commissioning |
| `operating_hours` | h | cumulative hours under load |
| `load_cycles` | count | counted **major** load events (start/stop, load reversal), not gear-mesh cycles |
| `vibration_rms` | mm/s | latest RMS vibration velocity |
| `temperature_mean_c` | °C | mean operating temperature |
| `corrosion_score` | 0–10 | inspection-based severity |
| `residual_thickness_pct` | % of nominal | measured remaining wall/section thickness |
| `maintenance_count` | count | recorded maintenance interventions |
| `failure_events` | count | recorded functional failures |
| `last_maintenance_months_ago` | months | since last intervention, or since commissioning if never maintained |
| `spare_parts_available` | bool | repair parts can currently be sourced |
| `recyclability_pct` | % | share of mass recoverable as secondary material |
| `embodied_co2_kg` | kg CO₂e | footprint of a **new** equivalent component |
| `replacement_cost_eur` | EUR | price of a new equivalent component |

### 2.3 Causal generation order

Variables are generated so that condition indicators co-evolve, instead of being sampled
independently:

1. family → material (family-specific mix)
2. age = D · 1.05 · Beta(2, 2.2)
3. latent **duty severity** s ~ LogNormal(0, 0.25), *never exposed*
4. utilisation ∝ s^0.5 → operating hours (capped below calendar hours) → load cycles ∝ hours · s
5. maintenance rate ~ U(0.12, 0.55)/year → maintenance count ~ Poisson(age · rate)
6. latent environment aggressiveness e ~ Beta(2, 3), *never exposed*
7. corrosion = material rate · age · (0.6 + 2e) · (1 − 0.35 · maintenance ratio) + noise, clipped to 0–10
8. thickness loss = 1.8 · corrosion^1.4 + 8 · (cycles / design cycles) + noise
9. vibration = family base · s^0.8 · (1 + 0.6 (age/D)²) · maintenance effect · noise
10. failures ~ Poisson(0.06 · age · s^1.5 · (1 + max(0, vib/alarm − 0.7)) · maintenance effect)
11. temperature = family nominal + 12 (s − 1) + 1.5 · vibration + noise
12. spare parts available with probability 0.97 − 0.55 · age/D (obsolescence)
13. mass → embodied CO₂ (mass × material factor) and replacement cost (mass × €/kg × material multiplier)

### 2.4 Hidden degradation function (synthetic target)

This is the "truth" the models try to recover. It is documented here for transparency but not
shown prominently in the app.

```text
consumption L = 0.4·age/D + 0.3·hours/(D·h_nom) + 0.3·cycles/(D·h_nom·c_ph)
base          = 12·D·max(0, 1.15 − L)                         (months; 15 % design margin)

f_corr   = exp(−0.012 · corrosion²)                            convex corrosion penalty
f_thk    = logistic((thickness − 68) / 5)                      structural threshold near 68 %
f_vib    = 1 / (1 + 2 · (max(0, vib − alarm) / alarm)²)        hinge above the alarm level
f_fail   = 0.88^failures                                       multiplicative failure history
f_temp   = exp(−0.03 · max(0, T − T_limit))                    hinge above the temperature limit
f_latent = build_quality · s^−0.3                              both unobserved

maint_credit = 1.5 · min(maintenance_count, 8) · (1 − min(last_maint / 36, 1))

RUL = base · f_corr · f_thk · f_vib · f_fail · f_temp · f_latent + maint_credit + ε
ε   ~ N(0, 3 + 0.07 · signal)                                  heteroscedastic noise
RUL clipped at 0;  build_quality ~ LogNormal(0, 0.12)
```

Design choices that matter methodologically:

- **Latent factors** (duty severity, build quality, environment) are never features, so the target
  cannot be reconstructed exactly. On held-out data the best model reaches R² ≈ 0.89, not ≈ 1.
- **Mixed functional forms** (additive consumption, multiplicative condition factors, a logistic
  threshold, hinges) make the linear-vs-nonlinear comparison meaningful without being rigged.
- **No leakage:** the target is computed *after* all observable features, from them plus latent
  variables and noise; no feature is derived from the target.

## 3. Data validation

`circularloop/validation.py` checks each record and assigns severities.

| Severity | Examples | Effect |
|---|---|---|
| critical | missing type, material, age, hours, corrosion, thickness, cost or CO₂; percentage outside 0–100; corrosion outside 0–10; negative hours/cost/CO₂; temperature outside −60…250 °C; hours exceeding calendar hours; unknown category | **no RUL, no recommendation**: "Insufficient data for a reliable recommendation" |
| warning | missing non-critical field (imputed from training-fleet median; spare parts assumed *unavailable*); temperature > 120 °C; vibration > 20 mm/s; > 3 600 cycles/h; maintenance dated before commissioning; manufacture year inconsistent with age; more failures than interventions | recommendation shown with reduced quality score / confidence |
| info | component beyond design life | none |

**Quality score** = weighted completeness (critical fields × 3) − 8 per warning − 5 per imputed
field, clipped to 0–100 (capped at 40 if any critical issue). Status: *Sufficient* (no critical
issue and score ≥ 85), *Reduced confidence*, or *Insufficient*.

Only records without critical issues are used for model training.

## 4. RUL modelling

### 4.1 Features

Age, operating hours, load cycles, vibration, temperature, corrosion, residual thickness,
maintenance count, failure events, months since maintenance, component type, material.

Excluded on purpose: `manufacture_year` (perfectly collinear with age), and the economic /
sustainability fields (cost, CO₂, recyclability, spare parts), which are not physical degradation
drivers.

### 4.2 Models

| Model | Preprocessing | Notes |
|---|---|---|
| Linear Regression | median imputation, standard scaling, one-hot (reference level dropped) | interpretable baseline; coefficients per 1 SD |
| Gradient Boosting (`HistGradientBoostingRegressor`) | median imputation, one-hot | 300 iterations, learning rate 0.05, depth 4, min 10 samples/leaf, L2 = 1, **monotonic constraints** |

**Why monotonic constraints.** During development, the unconstrained boosting model learned that a
*longer* time since maintenance means a *longer* remaining life, because never-maintained parts in
the fleet are mostly young. In what-if analyses this made "improved maintenance" *reduce* predicted
life for most components. The constraints encode engineering priors (RUL cannot rise with corrosion,
vibration, temperature, failures, age, hours, cycles or months since maintenance; cannot fall with
thickness or maintenance count). On this dataset the constrained model is also slightly more
accurate on held-out data than the unconstrained ablation, which is reported in the app. The
constraints are a prior, not a causal identification strategy.

### 4.3 Train/test protocol

- 75 / 25 random split, `random_state = 42`, validated records only (375 train / 125 test).
- Reported on the test split: MAE, RMSE, R².
- The model used for decisions is chosen by **5-fold cross-validated MAE on the training split**, so
  the test split is never used for selection. Users may override the choice.

### 4.4 Results (synthetic, seed 42)

| Model | Test MAE (months) | Test RMSE (months) | Test R² | CV MAE (train) |
|---|---:|---:|---:|---:|
| Linear Regression | 34.4 | 46.8 | 0.840 | 34.8 |
| Gradient Boosting (constrained) | 28.1 | 39.7 | 0.885 | 27.5 |
| Gradient Boosting (unconstrained, ablation) | 29.2 | 40.5 | 0.880 | 27.5 |

The nonlinear model is better here because the hidden function is multiplicative and contains
thresholds. This is a property of the synthetic generator, not evidence about real components.
Typical errors of ~2–3 years on components with 15–40-year design lives are also a reminder that
RUL predictions should enter decisions with their uncertainty.

### 4.5 Prediction interval

An approximate 80 % interval is derived from out-of-fold residuals on the training split, scaled by
the prediction size (residual ÷ (prediction + 12 months)), because synthetic degradation noise
grows with remaining life. Interval = point + q₁₀/q₉₀ · (point + 12). This is a simple normalised
split-conformal-style approximation, not a calibrated probabilistic model. The **lower bound
(P10)** is used as the *conservative RUL* in reuse gates.

### 4.6 Explainability

- Permutation importance on the held-out set for both models (comparable across model types;
  impurity importance is avoided because of its bias).
- Standardised linear coefficients (numeric per 1 SD; categories vs. reference level).
- Single-feature response curves for the selected component. Corrosion and thickness are strongly
  correlated in the fleet, so moving one while holding the other fixed leaves the training
  distribution; the app says so explicitly.

## 5. Decision layer

### 5.1 Strategy definitions

| Strategy | Operational meaning in this demonstrator |
|---|---|
| Reuse | return to service as-is after inspection and requalification |
| Repair | targeted correction of a specific defect; wall thickness is not restored |
| Refurbish | restore to good working condition (surface treatment, wear parts); not as-new |
| Remanufacture | disassemble and restore the core to as-new specification |
| Recycle | material recovery; **the function is provided by a new replacement component** |

### 5.2 Functional basis

Strategies deliver very different amounts of service, so absolute cost and CO₂ are not comparable.
Each strategy is evaluated as "deliver the function from today", and cost and CO₂ are expressed
**per year of service delivered**. Recycling therefore carries the cost and footprint of the new
replacement (minus scrap and recycled-content credits); otherwise "recycle" would look free.

| Strategy | Cost | CO₂ | Service life after intervention |
|---|---|---|---|
| Reuse | 0.04 R | 0.01 E | RUL |
| Repair | 0.12 R × (1.6 if no spares) × (1 + 0.05 · failures) | 0.05 E | RUL + 25 % of consumed life |
| Refurbish | 0.28 R × (1 + 0.04 · corrosion) | 0.12 E | RUL + 55 % of consumed life |
| Remanufacture | 0.55 R | 0.28 E | 0.85 D |
| Recycle + replace | R − 0.05 R · recyclability | E · (1 − 0.30 · recyclability) | D |

R = replacement cost, E = embodied CO₂ of a new part, D = design life, consumed life = max(D − RUL, 0).
Service life is capped at D (and floored at one month to keep annualised values finite).

*Consumed-life restoration* is a deliberate choice: an intervention can only recover life that has
actually been lost. A healthy part therefore gains nothing from refurbishment, which prevents the
absurd recommendation to refurbish a nearly new component.

### 5.3 Hard technical gates

Gates block a strategy **regardless of weights** (score 0, ranked last, reason shown).

| Strategy | Blocked if |
|---|---|
| Reuse | thickness < 80 % · corrosion > 5 · conservative (P10) RUL < 12 months · failures > 2 · vibration > 1.5 × alarm |
| Repair | thickness < 65 % · corrosion > 7 · failures > 4 |
| Refurbish | thickness < 60 % · corrosion > 8.5 |
| Remanufacture | thickness < 45 % (core not recoverable) · corrosion > 9.5 |
| any except Recycle | graded feasibility < 0.20 |

Recycle is never gated, so a feasible option always exists.

### 5.4 Graded technical feasibility

A weighted mean of condition sub-scores in [0, 1]. Thickness and corrosion are scored as the
**margin to each strategy's own gate** (0 at the gate, 1 comfortably inside it), e.g. thickness for
reuse 80 → 0 and 97 → 1, for remanufacture 45 → 0 and 78 → 1. Failures (6 → 0, 0 → 1), vibration
relative to alarm (2.0 → 0, 0.5 → 1), conservative RUL / D (0 → 0, 0.3 → 1) and spare parts (1 or
0.3) complete the picture with strategy-specific weights (`FEASIBILITY_WEIGHTS`). Recycle = 1.

### 5.5 MCDA normalisation and scoring

Min–max normalisation across the five options was rejected: it exaggerates trivial differences,
causes rank reversal when an option is blocked, and makes scores incomparable between components.
Instead, every criterion is scored on a fixed scale against a "new component" reference:

```text
feasibility = graded feasibility                     (0–1)
cost        = clip(1 − (cost/yr ÷ (R / D_years)) / 2, 0, 1)    0.5 = same as buying new
co2         = clip(1 − (CO₂/yr ÷ (E / D_years)) / 2, 0, 1)
life        = service life / D
circularity = Reuse 1.0 · Repair 0.9 · Refurbish 0.8 · Remanufacture 0.65 · Recycle 0.4 × recyclability

score = 100 · Σ wᵢ · sᵢ        (0 if any gate is violated)
```

Default weights: feasibility 0.30, CO₂ 0.20, cost 0.20, life 0.20, circularity 0.10; adjustable;
negative values are clipped and the vector is renormalised to sum to 1 (all-zero → defaults).
Ties are broken by the resource-retention order Reuse > Repair > Refurbish > Remanufacture > Recycle.

### 5.6 Robustness and confidence

- **Decision margin**: score of the winner minus the best feasible runner-up.
- **Weight sensitivity**: 400 weight vectors from a Dirichlet distribution centred on the current
  weights (concentration 30, seed 7); *stability* = share in which the recommended strategy still
  wins. Blocked strategies cannot win any draw.
- **Decision confidence**: *Low* if stability < 0.55 or quality < 60; *High* if data quality is
  sufficient and stability ≥ 0.80; otherwise *Medium*.

With default weights the 500 fleet components split into Repair 114, Recycle 110, Remanufacture 98,
Reuse 97 and Refurbish 81 recommendations (synthetic; reported only to show that no strategy wins
by construction).

### 5.7 Deterministic explanations

`circularloop/explanation.py` assembles plain-English text from actual values: the condition
summary (RUL, interval, corrosion, thickness), the criteria on which the winner gains the most
weighted points over the runner-up, every blocked strategy with its first violated gate, the
runner-up's main advantage, and a robustness sentence (margin, stability, plausible alternatives).
No language model is used.

## 6. Scenarios

| Scenario | Changes |
|---|---|
| Current state | none |
| Improved maintenance | vibration × 0.7, failures − 1, corrosion − 1, last maintenance → ≤ 2 months, temperature − 5 °C; **thickness unchanged** (material loss is irreversible) |
| Severe degradation | corrosion + 3, thickness − 15 pp, failures + 2, vibration × 1.6, temperature + 10 °C |

All values are clipped to valid ranges and the same pipeline is re-run. Tests check across the fleet
that improved maintenance never lowers and severe degradation never raises predicted RUL, and that
severe degradation never yields a direct-reuse recommendation.

## 7. Showcase cases

Five hand-defined cases specify *inputs only*; their outcomes are whatever the pipeline produces:
a healthy young drive shaft, a mid-life gearbox with moderate wear, a heavily corroded bogie frame,
a partially documented motor housing (imputation), and a record without thickness measurement
(insufficient data). For the mid-life gearbox the engine reports a near-tie between remanufacturing
and recycling with *Low* confidence, an honest result that shows why margins and sensitivity are
displayed rather than a single score.

## 8. Known limitations

- Synthetic data and an assumed degradation function; metrics only show how well models recover it.
- Real RUL labels are rare and right-censored; survival methods would be required.
- Cost, CO₂ and life factors are illustrative, not lifecycle costing or LCA.
- Gate thresholds are placeholders, not safety standards or expert-elicited values.
- Prediction uncertainty enters only through the conservative bound on reuse gates and the
  interval display; imputation does not widen the interval.
- Weighted-sum MCDA is compensatory; outranking methods (e.g. PROMETHEE, ELECTRE) behave differently.
- Observational confounding (maintenance vs. age) is mitigated by priors, not resolved causally.
- The record structure is conceptual and not aligned with any formal product-data model.

## 9. Interoperability (potential future layer, not implemented)

A real implementation could map this simplified record towards:

- **Asset Administration Shell**: identity, technical data and condition as AAS submodels;
- **GS1 EPCIS**: maintenance, repair and ownership changes as lifecycle events;
- **OPC UA**: live condition-monitoring values (vibration, temperature) from equipment;
- **emerging EU product-data requirements under the Ecodesign for Sustainable Products Regulation (ESPR)**: product-group-specific data sets and access rights.

No mapping to these is implemented, and the demonstrator makes no compliance claim.

## 10. Future research extensions

- Real industrial lifecycle records and real maintenance histories.
- RUL uncertainty propagated into the decision (e.g. expected scores over the predictive distribution).
- Survival analysis for censored lifetime data.
- Expert-elicited weights (AHP, swing weighting) and group decision making.
- Lifecycle costing and process-based LCA in place of fixed fractions.
- Semantic interoperability (shared vocabularies, AAS submodel templates).
- Explicit treatment of missing-data uncertainty instead of median imputation.
- Benchmarking recommendations against expert decisions.
- Optimisation under uncertainty at fleet level (capacity, spare-part and budget constraints).
