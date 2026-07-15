# =============================================================================
# main.py — Housing / Construction Output Forecasting Pipeline
# =============================================================================
#
# HOW TO USE
# ----------
#   python main.py
#
# FLOW
# ----
# 1. Country selection  — pick from France, Germany, Italy, United Kingdom
# 2. KPI selection      — exclude optional KPIs if desired
# 3. Preprocessing      — lag analysis, ElasticNetCV, sign checking
# 4. Forecast           — walk-forward validation + recursive 2026-2027
#
# TARGET DATA
# -----------
#   Euroconstruct_data.xlsx — residential construction output (EUR mn)
#   Years: 2006-2027  (train: 2006-2025, forecast: 2026-2027)
#
# =============================================================================

import sys
import traceback

import numpy as np
import pandas as pd

DEBUG = False

REQUIRED_RETURN_KEYS = [
    "df", "hist_clean", "train_lagged", "lag_col_map",
    "FEATURE_LAG_MAP", "CLEAN_FEATURES", "LONG_KPIS", "SHORT_KPIS",
    "COUNTRY", "TARGET_PCT", "TARGET_LVL", "TRAIN_END", "TEST_END",
]

# ---------------------------------------------------------------------------
# KPI CATALOGUE
# ---------------------------------------------------------------------------
KPI_CATALOGUE = [
    # ── Mandatory ────────────────────────────────────────────────────────────
    {
        "name": "interest_rate_chg",
        "tier": "MANDATORY",
        "description": "Annual change in long-term interest rate — mortgage cost signal",
    },
    {
        "name": "gdp_yoy",
        "tier": "MANDATORY",
        "description": "GDP YoY% — broad economic cycle (from quarterly data, annual mean)",
    },
    {
        "name": "output_yoy_lag1",
        "tier": "MANDATORY",
        "description": "Prior year own construction output YoY% (autoregressive signal)",
    },
    # ── Optional ─────────────────────────────────────────────────────────────
    {
        "name": "disposable_income_yoy",
        "tier": "OPTIONAL",
        "description": "Median disposable income per household YoY% — household purchasing power",
    },
    {
        "name": "gross_income_yoy",
        "tier": "OPTIONAL",
        "description": "Gross income YoY% — pre-tax income growth signal",
    },
    {
        "name": "population_yoy",
        "tier": "OPTIONAL",
        "description": "Total population YoY% — demographic demand driver",
    },
    {
        "name": "housing_permits_yoy",
        "tier": "OPTIONAL",
        "description": "Housing permits YoY% — leading indicator of construction starts",
    },
    {
        "name": "hpi_yoy",
        "tier": "OPTIONAL",
        "description": "House Price Index YoY% — price incentive for new development",
    },
    {
        "name": "house_to_rent_ratio_yoy",
        "tier": "OPTIONAL",
        "description": "House price-to-rent ratio YoY% — buy vs rent relative cost signal",
    },
    {
        "name": "labor_cost_yoy",
        "tier": "OPTIONAL",
        "description": "Unit Labour Cost Index YoY% — construction cost pressure (negative sign)",
    },
    {
        "name": "household_size_chg",
        "tier": "OPTIONAL",
        "description": "Avg household size annual change — shrinking households → more units needed (negative sign)",
    },
]

SELECTABLE_TIERS = {"OPTIONAL"}


# =============================================================================
# COUNTRY LIST
# =============================================================================

AVAILABLE_COUNTRIES = [
    "France",
    "Germany",
    "Italy",
    "United Kingdom",
]


# =============================================================================
# COUNTRY PICKER
# =============================================================================

def _pick_countries():
    print("\n" + "=" * 72)
    print("  STEP 1A — COUNTRY SELECTION")
    print("=" * 72)
    print()
    for i, name in enumerate(AVAILABLE_COUNTRIES, start=1):
        print(f"  [{i:>2}]  {name}")
    print()
    print("  Instructions:")
    print("    1          → single country")
    print("    1,3        → multiple countries")
    print("    all        → all 4 countries")
    print()

    while True:
        raw = input("  Enter number(s) or 'all': ").strip().lower()
        if raw == "all":
            chosen = list(AVAILABLE_COUNTRIES)
            break
        parts = [p.strip() for p in raw.split(",")]
        if not all(p.isdigit() for p in parts):
            print("  Invalid input.")
            continue
        idxs    = [int(p) for p in parts]
        invalid = [i for i in idxs if not (1 <= i <= len(AVAILABLE_COUNTRIES))]
        if invalid:
            print(f"  Invalid numbers: {invalid}.")
            continue
        seen, chosen = set(), []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                chosen.append(AVAILABLE_COUNTRIES[i - 1])
        break

    print(f"\n  ✓ Selected ({len(chosen)} {'country' if len(chosen)==1 else 'countries'}):")
    for c in chosen:
        print(f"      • {c}")
    return chosen


# =============================================================================
# KPI PICKER
# =============================================================================

def _pick_kpis():
    print("\n" + "=" * 72)
    print("  STEP 1B — KPI SELECTION")
    print("  (Applied to ALL selected countries)")
    print("=" * 72)

    mandatory  = [k for k in KPI_CATALOGUE if k["tier"] == "MANDATORY"]
    selectable = [k for k in KPI_CATALOGUE if k["tier"] in SELECTABLE_TIERS]

    print("\n  MANDATORY KPIs — always included:")
    print(f"  {'KPI':<32} Description")
    print(f"  {'-' * 74}")
    for kpi in mandatory:
        print(f"  [LOCKED] {kpi['name']:<23} {kpi['description']}")

    print(f"\n  SELECTABLE KPIs — enter numbers to EXCLUDE (default: keep all):")
    print(f"  {'#':<5} {'Tier':<10} {'KPI':<32} Description")
    print(f"  {'-' * 84}")
    for i, kpi in enumerate(selectable, start=1):
        tier_tag = f"[{kpi['tier']}]"
        print(f"  {i:<5} {tier_tag:<10} {kpi['name']:<32} {kpi['description']}")

    print(f"\n  Instructions:")
    print(f"    ENTER        → include ALL {len(selectable)} selectable KPIs (recommended)")
    print(f"    2,5          → exclude KPIs 2 and 5, keep the rest")
    print(f"    none         → exclude ALL (mandatory only)")

    while True:
        raw = input("\n  Numbers to EXCLUDE (or ENTER for all): ").strip().lower()

        if raw == "":
            chosen, excluded = [k["name"] for k in selectable], []
        elif raw == "none":
            chosen, excluded = [], [k["name"] for k in selectable]
        else:
            parts = [p.strip() for p in raw.split(",")]
            if not all(p.isdigit() for p in parts):
                print("  Invalid input — enter comma-separated numbers, 'none', or ENTER.")
                continue
            exclude_idxs = {int(p) for p in parts}
            invalid = {idx for idx in exclude_idxs if not (1 <= idx <= len(selectable))}
            if invalid:
                print(f"  Invalid numbers: {sorted(invalid)}.")
                continue
            chosen   = [k["name"] for i, k in enumerate(selectable, 1)
                        if i not in exclude_idxs]
            excluded = [k["name"] for i, k in enumerate(selectable, 1)
                        if i in exclude_idxs]

        print("\n  ── KPI Summary ──────────────────────────────────────────────────")
        print(f"  {'KPI':<36} {'Status':<12} Tier")
        print(f"  {'-' * 62}")
        for kpi in mandatory:
            print(f"  {kpi['name']:<36} {'MANDATORY':<12} [locked]")
        for kpi in selectable:
            status = "INCLUDED" if kpi["name"] in chosen else "EXCLUDED"
            print(f"  {kpi['name']:<36} {status:<12} [{kpi['tier']}]")
        print(f"  {'─' * 62}")
        total = len(mandatory) + len(chosen)
        print(f"  KPIs entering preprocessing: {total}  "
              f"({len(mandatory)} mandatory + {len(chosen)} optional)")

        confirm = input("\n  Confirm? (ENTER to proceed, 'r' to redo): ").strip().lower()
        if confirm in ("", "y", "yes"):
            return chosen
        print()


# =============================================================================
# RESULT VALIDATION & SUMMARY
# =============================================================================

def _validate_result(result):
    if result is None:
        raise ValueError("run_preprocessing() returned None.")
    if not isinstance(result, dict):
        raise ValueError(f"run_preprocessing() returned {type(result).__name__}, expected dict.")
    missing = [k for k in REQUIRED_RETURN_KEYS if k not in result]
    if missing:
        raise ValueError(f"Preprocessing output missing required keys: {missing}")
    if not result["CLEAN_FEATURES"]:
        raise ValueError("CLEAN_FEATURES is empty — no usable KPIs survived.")
    if result["TRAIN_END"] >= result["TEST_END"]:
        raise ValueError(
            f"Invalid window: TRAIN_END={result['TRAIN_END']} "
            f"must be < TEST_END={result['TEST_END']}."
        )
    if result["df"].empty:
        raise ValueError("Returned dataframe 'df' is empty.")
    if result["hist_clean"].empty:
        raise ValueError("Returned 'hist_clean' is empty.")
    if result["TARGET_PCT"] not in result["hist_clean"].columns:
        raise ValueError(f"TARGET_PCT '{result['TARGET_PCT']}' not in hist_clean.")
    if result["TARGET_LVL"] not in result["hist_clean"].columns:
        raise ValueError(f"TARGET_LVL '{result['TARGET_LVL']}' not in hist_clean.")


def _print_result_summary(result):
    print("\n" + "=" * 72)
    print("   PREPROCESSING COMPLETE — Summary")
    print("=" * 72)
    print(f"\nCOUNTRY         = '{result['COUNTRY']}'")
    print(f"CLEAN_FEATURES  = {result['CLEAN_FEATURES']}")
    print(f"FEATURE_LAG_MAP = {result['FEATURE_LAG_MAP']}")
    print(f"LONG_KPIS       = {result['LONG_KPIS']}")
    print(f"SHORT_KPIS      = {result['SHORT_KPIS']}")
    print(f"TARGET_PCT      = '{result['TARGET_PCT']}'")
    print(f"TARGET_LVL      = '{result['TARGET_LVL']}'")
    print(f"TRAIN_END       = {result['TRAIN_END']}")
    print(f"TEST_END        = {result['TEST_END']}")
    print(f"\ndf shape         : {result['df'].shape}")
    print(f"hist_clean rows  : {len(result['hist_clean'])}")
    print(f"train_lagged rows: {len(result['train_lagged'])}")
    print(f"features selected: {len(result['CLEAN_FEATURES'])}")
    print("\nSelected features:")
    for i, feat in enumerate(result["CLEAN_FEATURES"], start=1):
        print(f"  {i:>2}. {feat}")
    print("\nLag map:")
    for k, v in result["FEATURE_LAG_MAP"].items():
        print(f"  {k:<36} -> {v}")


# =============================================================================
# PER-COUNTRY RUNNER
# =============================================================================

def _run_one_country(country, user_optional):
    result = {"country": country, "status": "ok", "error": None, "prep": None}

    print(f"\n{'=' * 72}")
    print(f"  [{country}] STEP 2 — Preprocessing")
    print(f"{'=' * 72}")

    from preprocessing import run_preprocessing

    try:
        prep = run_preprocessing(country, user_optional=user_optional)
        _validate_result(prep)
        _print_result_summary(prep)
        result["prep"] = prep
    except FileNotFoundError as e:
        result["status"] = "error"
        result["error"]  = f"File not found: {e}"
        print(f"\n  ERROR — {result['error']}")
        return result
    except (ValueError, RuntimeError) as e:
        result["status"] = "error"
        result["error"]  = str(e)
        print(f"\n  ERROR — {result['error']}")
        return result
    except Exception as e:
        result["status"] = "error"
        result["error"]  = f"Unexpected error: {e}"
        print(f"\n  ERROR — {result['error']}")
        if DEBUG:
            traceback.print_exc()
        return result

    print(f"\n{'=' * 72}")
    print(f"  [{country}] STEP 3 — Forecast")
    print(f"{'=' * 72}\n")

    try:
        from forecastmodel import run_forecast
        run_forecast(prep)
    except Exception as e:
        result["status"] = "error"
        result["error"]  = f"Forecast error: {e}"
        print(f"\n  ERROR — {result['error']}")
        if DEBUG:
            traceback.print_exc()
        return result

    return result


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def _print_run_summary(results):
    print("\n" + "=" * 72)
    print("  PIPELINE RUN SUMMARY")
    print("=" * 72)
    print(f"  {'Country':<22} {'Status':<10} Details")
    print(f"  {'-' * 68}")
    ok_count = err_count = 0
    for r in results:
        if r["status"] == "ok":
            ok_count += 1
            features = len(r["prep"]["CLEAN_FEATURES"]) if r["prep"] else "?"
            print(f"  {r['country']:<22} {'✓ OK':<10} {features} features selected")
        else:
            err_count += 1
            short_err = (r["error"] or "unknown error")[:46]
            print(f"  {r['country']:<22} {'✗ ERROR':<10} {short_err}")
    print(f"  {'─' * 68}")
    print(f"  {ok_count} succeeded, {err_count} failed  (total: {len(results)})")
    print("=" * 72)


# =============================================================================
# MAIN
# =============================================================================

def _print_header():
    print("=" * 72)
    print("   Housing / Construction Output Forecasting Pipeline")
    print("   Target: Euroconstruct residential construction (EUR mn)")
    print("   Period: 2006-2027  |  Countries: France, Germany, Italy, UK")
    print("=" * 72)


def main():
    _print_header()

    try:
        countries = _pick_countries()
    except KeyboardInterrupt:
        print("\n\n  Run cancelled.")
        return None

    multi = len(countries) > 1

    try:
        user_optional = _pick_kpis()
    except KeyboardInterrupt:
        print("\n\n  Run cancelled.")
        return None

    all_results = []
    for idx, country in enumerate(countries, start=1):
        if multi:
            print(f"\n\n{'#' * 72}")
            print(f"  COUNTRY {idx} of {len(countries)}: {country.upper()}")
            print(f"{'#' * 72}")
        try:
            res = _run_one_country(country, user_optional)
        except KeyboardInterrupt:
            print(f"\n\n  Run cancelled at: {country}")
            for remaining in countries[idx:]:
                all_results.append({
                    "country": remaining, "status": "error",
                    "error": "cancelled", "prep": None,
                })
            break
        all_results.append(res)

    if multi:
        _print_run_summary(all_results)

    print("\n" + "=" * 72)
    print("  Pipeline complete.")
    print("=" * 72)

    if len(all_results) == 1:
        return all_results[0]["prep"]
    return all_results


if __name__ == "__main__":
    main()