# =============================================================================
# main.py — Residential Market Forecasting Pipeline
# =============================================================================
#
# HOW TO USE
# ----------
#   python main.py
#
# FLOW
# ----
# 1. Country selection  — Germany, France, Italy, United Kingdom (or all)
# 2. KPI selection      — choose which optional KPIs to include
# 3. Preprocessing      — lag analysis, ElasticNetCV KPI selection
# 4. Forecasting        — walk-forward validation + Stage 1 recursive forecast
# 5. Comparison         — cross-country Excel + chart (if >1 country selected)
#
# MANDATORY KPIs (always on):
#   interest_rate_chg, housing_permits_yoy
#
# OPTIONAL KPIs (user can include/exclude):
#   gdp_yoy, house_price_index_yoy, house_to_rent_ratio_yoy,
#   labor_cost_yoy, gross_income_yoy, disposable_income_yoy,
#   population_yoy, household_size_chg
#
# UNITS
# -----
# All level values are displayed and saved in BILLION (€bn or £bn).
# The Euroconstruct source file stores values in million; division by 1000
# is applied in forecastmodel.py before any output.
# =============================================================================

import sys
import traceback

DEBUG = False

AVAILABLE_COUNTRIES = ["Germany", "France", "Italy", "United Kingdom"]

KPI_CATALOGUE = [
    {"name": "interest_rate_chg",       "tier": "MANDATORY",
     "description": "Annual change in long-term interest rate (key affordability signal)"},
    {"name": "housing_permits_yoy",      "tier": "MANDATORY",
     "description": "Housing permits YoY% — supply pipeline, leading indicator"},
    {"name": "gdp_yoy",                  "tier": "OPTIONAL",
     "description": "GDP YoY% — broad economic activity, income & demand proxy"},
    {"name": "house_price_index_yoy",    "tier": "OPTIONAL",
     "description": "House price index YoY% — market momentum signal"},
    {"name": "house_to_rent_ratio_yoy",  "tier": "OPTIONAL",
     "description": "House-to-rent ratio YoY% — buy-vs-rent balance"},
    {"name": "labor_cost_yoy",           "tier": "OPTIONAL",
     "description": "Labor cost index YoY% — construction cost pressure (negative sign)"},
    {"name": "gross_income_yoy",         "tier": "OPTIONAL",
     "description": "Gross income YoY% — household purchasing power"},
    {"name": "disposable_income_yoy",    "tier": "OPTIONAL",
     "description": "Disposable income per household YoY% — affordability driver"},
    {"name": "population_yoy",           "tier": "OPTIONAL",
     "description": "Population YoY% — structural demand driver"},
    {"name": "household_size_chg",       "tier": "OPTIONAL",
     "description": "Change in avg household size — smaller HH → more units needed"},
]

SELECTABLE_TIERS = {"OPTIONAL"}

REQUIRED_RETURN_KEYS = [
    "df", "hist_clean", "train_lagged", "lag_col_map",
    "FEATURE_LAG_MAP", "CLEAN_FEATURES", "LONG_KPIS", "SHORT_KPIS",
    "COUNTRY", "TARGET_PCT", "TARGET_LVL", "TRAIN_END", "TEST_END",
]


# =============================================================================
# COUNTRY PICKER
# =============================================================================

def _pick_countries():
    print("\n" + "=" * 72)
    print("  STEP 1A — COUNTRY SELECTION")
    print("=" * 72)
    print()
    for i, name in enumerate(AVAILABLE_COUNTRIES, 1):
        print(f"  [{i:>2}]  {name}")
    print()
    print("  Instructions:  1  → single   |  1,3  → multiple   |  all → all 4")
    print()

    while True:
        raw = input("  Enter number(s) or 'all': ").strip().lower()
        if raw == "all":
            chosen = list(AVAILABLE_COUNTRIES)
            break
        parts = [p.strip() for p in raw.split(",")]
        if not all(p.isdigit() for p in parts):
            print("  Invalid input — enter numbers or 'all'.")
            continue
        idxs    = [int(p) for p in parts]
        invalid = [i for i in idxs if not (1 <= i <= len(AVAILABLE_COUNTRIES))]
        if invalid:
            print(f"  Invalid numbers: {invalid}")
            continue
        seen, chosen = set(), []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                chosen.append(AVAILABLE_COUNTRIES[i - 1])
        break

    print(f"\n  ✓ Selected: {', '.join(chosen)}")
    return chosen


# =============================================================================
# KPI PICKER
# =============================================================================

def _pick_kpis():
    print("\n" + "=" * 72)
    print("  STEP 1B — KPI SELECTION  (applied to ALL selected countries)")
    print("=" * 72)

    mandatory  = [k for k in KPI_CATALOGUE if k["tier"] == "MANDATORY"]
    selectable = [k for k in KPI_CATALOGUE if k["tier"] in SELECTABLE_TIERS]

    print("\n  MANDATORY KPIs (always included):")
    for kpi in mandatory:
        print(f"  [LOCKED] {kpi['name']:<35} {kpi['description']}")

    print(f"\n  SELECTABLE KPIs — enter numbers to EXCLUDE (default: keep all):")
    print(f"  {'#':<5} {'KPI':<35} Description")
    print(f"  {'-'*80}")
    for i, kpi in enumerate(selectable, 1):
        print(f"  {i:<5} {kpi['name']:<35} {kpi['description']}")

    print(f"\n  ENTER = include all  |  2,5 = exclude those  |  none = mandatory only")

    while True:
        raw = input("\n  Numbers to EXCLUDE (or ENTER for all): ").strip().lower()
        if raw == "":
            chosen = [k["name"] for k in selectable]
        elif raw == "none":
            chosen = []
        else:
            parts = [p.strip() for p in raw.split(",")]
            if not all(p.isdigit() for p in parts):
                print("  Invalid — enter numbers, 'none', or ENTER.")
                continue
            excl = {int(p) for p in parts}
            inv  = {i for i in excl if not (1 <= i <= len(selectable))}
            if inv:
                print(f"  Invalid numbers: {sorted(inv)}")
                continue
            chosen = [k["name"] for i, k in enumerate(selectable, 1) if i not in excl]

        print(f"\n  KPIs selected (mandatory): {[k['name'] for k in mandatory]}")
        print(f"                + optional : {chosen}")
        confirm = input("  Confirm? (ENTER = proceed, 'r' = redo): ").strip().lower()
        if confirm in ("", "y", "yes"):
            return chosen
        print()


# =============================================================================
# VALIDATION & SUMMARY
# =============================================================================

def _validate_result(result):
    if result is None or not isinstance(result, dict):
        raise ValueError("run_preprocessing() returned None or non-dict.")
    missing = [k for k in REQUIRED_RETURN_KEYS if k not in result]
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    if not result["CLEAN_FEATURES"]:
        raise ValueError("CLEAN_FEATURES is empty.")
    if result["TRAIN_END"] >= result["TEST_END"]:
        raise ValueError("Invalid window.")
    if result["df"].empty:
        raise ValueError("df is empty.")
    if result["hist_clean"].empty:
        raise ValueError("hist_clean is empty.")


def _print_result_summary(result):
    print(f"\n{'='*72}")
    print("   PREPROCESSING COMPLETE")
    print(f"{'='*72}")
    print(f"  COUNTRY         = '{result['COUNTRY']}'")
    print(f"  CLEAN_FEATURES  = {result['CLEAN_FEATURES']}")
    print(f"  FEATURE_LAG_MAP = {result['FEATURE_LAG_MAP']}")
    print(f"  LONG_KPIS       = {result['LONG_KPIS']}")
    print(f"  SHORT_KPIS      = {result['SHORT_KPIS']}")
    print(f"  TRAIN_END       = {result['TRAIN_END']}")
    print(f"  TEST_END        = {result['TEST_END']}")
    print(f"  df shape        : {result['df'].shape}")
    print(f"  hist_clean rows : {len(result['hist_clean'])}")
    print(f"  train rows      : {len(result['train_lagged'])}")


def _print_run_summary(results):
    print(f"\n{'='*72}")
    print("  PIPELINE RUN SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Country':<22} {'Status':<10} Details")
    print(f"  {'-'*68}")
    ok = err = 0
    for r in results:
        if r["status"] == "ok":
            ok += 1
            n = len(r["prep"]["CLEAN_FEATURES"]) if r["prep"] else "?"
            print(f"  {r['country']:<22} {'✓ OK':<10} {n} features selected")
        else:
            err += 1
            print(f"  {r['country']:<22} {'✗ ERROR':<10} {(r['error'] or 'unknown')[:46]}")
    print(f"  {'─'*68}")
    print(f"  {ok} succeeded, {err} failed  (total: {len(results)})")
    print(f"{'='*72}")


# =============================================================================
# PER-COUNTRY RUNNER
# =============================================================================

def _run_one_country(country, user_optional):
    result = {"country": country, "status": "ok", "error": None, "prep": None}

    print(f"\n{'='*72}")
    print(f"  [{country}] STEP 2 — Preprocessing")
    print(f"{'='*72}")

    from preprocessing import run_preprocessing
    try:
        prep = run_preprocessing(country, user_optional=user_optional)
        _validate_result(prep)
        _print_result_summary(prep)
        result["prep"] = prep
    except Exception as e:
        result["status"] = "error"
        result["error"]  = str(e)
        print(f"\n  ERROR — {e}")
        if DEBUG: traceback.print_exc()
        return result

    print(f"\n{'='*72}")
    print(f"  [{country}] STEP 3 — Forecast")
    print(f"{'='*72}\n")

    try:
        from forecastmodel import run_forecast
        run_forecast(prep)
    except Exception as e:
        result["status"] = "error"
        result["error"]  = f"Forecast error: {e}"
        print(f"\n  ERROR — {result['error']}")
        if DEBUG: traceback.print_exc()

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 72)
    print("   Residential Market Forecasting Pipeline")
    print("   Level unit : BILLION  (€bn / £bn)")
    print("   KPIs: Interest Rate + Housing Permits (mandatory)")
    print("         + 8 optional KPIs")
    print("   Countries: Germany, France, Italy, United Kingdom")
    print("=" * 72)

    try:
        countries = _pick_countries()
    except KeyboardInterrupt:
        print("\n  Cancelled."); return None

    try:
        user_optional = _pick_kpis()
    except KeyboardInterrupt:
        print("\n  Cancelled."); return None

    multi       = len(countries) > 1
    all_results = []

    for idx, country in enumerate(countries, 1):
        if multi:
            print(f"\n\n{'#'*72}")
            print(f"  COUNTRY {idx} of {len(countries)}: {country.upper()}")
            print(f"{'#'*72}")

        try:
            res = _run_one_country(country, user_optional)
        except KeyboardInterrupt:
            print(f"\n  Cancelled at country: {country}")
            break

        all_results.append(res)

    if multi:
        _print_run_summary(all_results)

    # ── STEP 4: Cross-country comparison (if >1 country succeeded) ─────────────
    succeeded = [r for r in all_results if r["status"] == "ok"]
    if len(succeeded) > 1:
        print(f"\n{'='*72}")
        print(f"  STEP 4 — Cross-Country Comparison")
        print(f"{'='*72}")
        try:
            from forecastmodel import export_comparison
            export_comparison()
        except Exception as e:
            print(f"  WARNING: Comparison export failed: {e}")
            if DEBUG: traceback.print_exc()

    print(f"\n{'='*72}")
    print("  Pipeline complete.")
    print(f"{'='*72}")

    return (all_results[0]["prep"] if len(all_results) == 1 else all_results)


if __name__ == "__main__":
    main()