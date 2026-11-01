"""Synthetic fleet generator with a hidden degradation function.

Variables are generated in causal order (type → age → duty → usage → condition →
failures → economics) so that condition indicators co-evolve plausibly. The RUL
target depends on two *latent* factors that are never exposed as features
(duty severity and build quality), which creates an irreducible error floor and
prevents a model from trivially reconstructing the target.

The exact formula is documented in ``docs/methodology.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from circularloop.config import (
    COMPONENT_TYPES,
    MATERIALS,
    N_COMPONENTS,
    RANDOM_SEED,
    REFERENCE_YEAR,
)

COLUMN_ORDER: tuple[str, ...] = (
    "component_id",
    "component_type",
    "material",
    "manufacture_year",
    "age_years",
    "operating_hours",
    "load_cycles",
    "vibration_rms",
    "temperature_mean_c",
    "corrosion_score",
    "residual_thickness_pct",
    "maintenance_count",
    "failure_events",
    "last_maintenance_months_ago",
    "spare_parts_available",
    "recyclability_pct",
    "embodied_co2_kg",
    "replacement_cost_eur",
    "rul_months",
)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _type_param(types: np.ndarray, attr: str) -> np.ndarray:
    """Vectorised lookup of a ComponentTypeSpec attribute."""
    return np.array([getattr(COMPONENT_TYPES[t], attr) for t in types], dtype=float)


def _sample_materials(types: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(len(types), dtype=object)
    for i, t in enumerate(types):
        mix = COMPONENT_TYPES[t].materials
        out[i] = rng.choice(list(mix.keys()), p=list(mix.values()))
    return out


def hidden_rul_signal(df: pd.DataFrame, duty_severity: np.ndarray, build_quality: np.ndarray) -> np.ndarray:
    """Noise-free part of the hidden degradation function (months).

    Not used by any model. Exposed separately so the methodology can be tested.
    """
    types = df["component_type"].to_numpy()
    design_years = _type_param(types, "design_life_years")
    h_nom = _type_param(types, "nominal_hours_per_year")
    c_ph = _type_param(types, "load_cycles_per_hour")
    v_alarm = _type_param(types, "vibration_alarm_mm_s")
    t_limit = _type_param(types, "temperature_limit_c")

    consumption = (
        0.4 * df["age_years"].to_numpy() / design_years
        + 0.3 * df["operating_hours"].to_numpy() / (design_years * h_nom)
        + 0.3 * df["load_cycles"].to_numpy() / (design_years * h_nom * c_ph)
    )
    base = 12.0 * design_years * np.maximum(0.0, 1.15 - consumption)

    corr = df["corrosion_score"].to_numpy()
    thk = df["residual_thickness_pct"].to_numpy()
    vib = df["vibration_rms"].to_numpy()
    f_corr = np.exp(-0.012 * corr**2)
    f_thk = _logistic((thk - 68.0) / 5.0)
    f_vib = 1.0 / (1.0 + 2.0 * (np.maximum(0.0, vib - v_alarm) / v_alarm) ** 2)
    f_fail = 0.88 ** df["failure_events"].to_numpy()
    f_temp = np.exp(-0.03 * np.maximum(0.0, df["temperature_mean_c"].to_numpy() - t_limit))
    f_latent = build_quality * duty_severity**-0.3

    maint_credit = (
        1.5
        * np.minimum(df["maintenance_count"].to_numpy(), 8)
        * (1.0 - np.minimum(df["last_maintenance_months_ago"].to_numpy() / 36.0, 1.0))
    )
    return base * f_corr * f_thk * f_vib * f_fail * f_temp * f_latent + maint_credit


def generate_fleet(n: int = N_COMPONENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate ``n`` synthetic, internally consistent component records with RUL target."""
    rng = np.random.default_rng(seed)
    type_names = np.array(list(COMPONENT_TYPES.keys()))
    types = rng.choice(type_names, size=n)
    materials = _sample_materials(types, rng)

    design_years = _type_param(types, "design_life_years")
    h_nom = _type_param(types, "nominal_hours_per_year")
    c_ph = _type_param(types, "load_cycles_per_hour")

    # --- usage --------------------------------------------------------------
    age = np.round(np.clip(design_years * 1.05 * rng.beta(2.0, 2.2, n), 0.3, None), 1)
    duty = rng.lognormal(0.0, 0.25, n)  # latent, unobserved
    utilisation = np.clip(rng.normal(1.0, 0.18, n) * duty**0.5, 0.3, 1.7)
    hours = np.round(age * h_nom * utilisation * rng.normal(1.0, 0.04, n), -1)
    hours = np.round(np.minimum(hours, age * 8_760 * 0.98), -1)
    cycles = np.round(hours * c_ph * duty * rng.lognormal(0.0, 0.15, n), -1)

    # --- maintenance policy ---------------------------------------------------
    maint_rate = rng.uniform(0.12, 0.55, n)  # interventions per year
    maint_count = rng.poisson(age * maint_rate)
    maint_ratio = np.clip(maint_rate / 0.55, 0, 1)

    # --- corrosion and thickness -------------------------------------------
    environment = rng.beta(2.0, 3.0, n)  # latent aggressiveness
    corr_rate = np.array([MATERIALS[m].corrosion_rate_per_year for m in materials])
    corrosion = corr_rate * age * (0.6 + 2.0 * environment) * (1.0 - 0.35 * maint_ratio)
    corrosion = np.round(np.clip(corrosion + rng.normal(0, 0.4, n), 0, 10), 1)

    cycles_rel = cycles / (design_years * h_nom * c_ph)
    thickness_loss = 1.8 * corrosion**1.4 + 8.0 * cycles_rel + rng.normal(0, 2.5, n)
    thickness = np.round(np.clip(100.0 - np.maximum(thickness_loss, 0), 30, 100), 1)

    # --- vibration, failures, temperature -----------------------------------
    vib_base = _type_param(types, "vibration_base_mm_s")
    vibration = vib_base * duty**0.8 * (1.0 + 0.6 * (age / design_years) ** 2)
    vibration *= (1.0 - 0.15 * maint_ratio) * rng.lognormal(0.0, 0.2, n)
    vibration = np.round(np.clip(vibration, 0.1, 30), 2)

    v_alarm = _type_param(types, "vibration_alarm_mm_s")
    fail_lambda = 0.06 * age * duty**1.5 * (1.0 + np.maximum(0, vibration / v_alarm - 0.7))
    fail_lambda *= 1.0 - 0.35 * maint_ratio
    failures = rng.poisson(fail_lambda)

    t_nom = _type_param(types, "temperature_nominal_c")
    temperature = t_nom + 12.0 * (duty - 1.0) + 1.5 * vibration + rng.normal(0, 4.0, n)
    temperature = np.round(temperature, 1)

    months_in_service = age * 12.0
    last_maint = np.where(
        maint_count > 0,
        rng.uniform(0.5, 1.0, n) * np.minimum(36.0, months_in_service) * rng.uniform(0.1, 1.0, n),
        months_in_service,
    )
    last_maint = np.round(np.minimum(last_maint, months_in_service), 1)

    spares = rng.random(n) < np.clip(0.97 - 0.55 * age / design_years, 0.25, 0.97)

    # --- sustainability / economics -----------------------------------------
    mass_lo = np.array([COMPONENT_TYPES[t].mass_range_kg[0] for t in types])
    mass_hi = np.array([COMPONENT_TYPES[t].mass_range_kg[1] for t in types])
    mass = rng.uniform(mass_lo, mass_hi)
    co2_factor = np.array([MATERIALS[m].co2_kg_per_kg for m in materials])
    embodied_co2 = np.round(mass * co2_factor * rng.lognormal(0, 0.08, n), 0)
    cost_kg = _type_param(types, "cost_per_kg_eur")
    cost_mult = np.array([MATERIALS[m].cost_multiplier for m in materials])
    replacement_cost = np.round(mass * cost_kg * cost_mult * rng.lognormal(0, 0.12, n), -1)
    rec_lo = np.array([MATERIALS[m].recyclability_range_pct[0] for m in materials])
    rec_hi = np.array([MATERIALS[m].recyclability_range_pct[1] for m in materials])
    recyclability = np.round(rng.uniform(rec_lo, rec_hi), 1)

    df = pd.DataFrame(
        {
            "component_id": [f"CMP-{i + 1:04d}" for i in range(n)],
            "component_type": types,
            "material": materials,
            "manufacture_year": (REFERENCE_YEAR - np.floor(age)).astype(int),
            "age_years": age,
            "operating_hours": hours,
            "load_cycles": cycles,
            "vibration_rms": vibration,
            "temperature_mean_c": temperature,
            "corrosion_score": corrosion,
            "residual_thickness_pct": thickness,
            "maintenance_count": maint_count.astype(int),
            "failure_events": failures.astype(int),
            "last_maintenance_months_ago": last_maint,
            "spare_parts_available": spares,
            "recyclability_pct": recyclability,
            "embodied_co2_kg": embodied_co2,
            "replacement_cost_eur": replacement_cost,
        }
    )

    # --- hidden target ------------------------------------------------------
    build_quality = rng.lognormal(0.0, 0.12, n)  # latent, unobserved
    signal = hidden_rul_signal(df, duty, build_quality)
    noise = rng.normal(0.0, 3.0 + 0.07 * signal)
    df["rul_months"] = np.round(np.clip(signal + noise, 0.0, None), 1)
    return df.loc[:, list(COLUMN_ORDER)]


def defective_records() -> pd.DataFrame:
    """A handful of deliberately invalid records to exercise the validation layer.

    They carry no RUL target and are never used for training.
    """
    base = {
        "component_type": "Industrial gearbox",
        "material": "Cast iron",
        "manufacture_year": 2014,
        "age_years": 12.0,
        "operating_hours": 70_000.0,
        "load_cycles": 1_100_000.0,
        "vibration_rms": 3.1,
        "temperature_mean_c": 64.0,
        "corrosion_score": 3.5,
        "residual_thickness_pct": 86.0,
        "maintenance_count": 4,
        "failure_events": 1,
        "last_maintenance_months_ago": 10.0,
        "spare_parts_available": True,
        "recyclability_pct": 91.0,
        "embodied_co2_kg": 2_900.0,
        "replacement_cost_eur": 26_000.0,
    }
    defects = [
        {"residual_thickness_pct": None},
        {"operating_hours": -1_200.0},
        {"corrosion_score": 14.0},
        {"recyclability_pct": 130.0},
        {"replacement_cost_eur": -500.0},
        {"operating_hours": 160_000.0},  # exceeds calendar hours for 12 years
    ]
    rows = []
    for i, defect in enumerate(defects, start=1):
        row = {"component_id": f"DQ-{i:02d}", **base, **defect, "rul_months": np.nan}
        rows.append(row)
    return pd.DataFrame(rows).loc[:, list(COLUMN_ORDER)]


def build_dataset(n: int = N_COMPONENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Fleet plus defective records, as stored in ``data/sample_components.csv``."""
    return pd.concat([generate_fleet(n, seed), defective_records()], ignore_index=True)
