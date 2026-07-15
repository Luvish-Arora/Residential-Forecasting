# =============================================================================
# preprocessing.py  —  KPI Selection for Housing / Construction Output Forecasting
# =============================================================================
#
# CHANGES FROM PREVIOUS VERSION
# ------------------------------
# 1. KPI SET REDUCED from 10 → 6 independent signals
#      REMOVED : house_to_rent_ratio  (r=0.97 with HPI — pure duplicate)
#                population            (std 0.1-0.4% — zero explanatory power)
#                household_size        (UK std=0.00 — literally constant)
#      MERGED  : disposable_income + gross_income → winner chosen per-country
#                at runtime by whichever has higher |corr| with target
#      ADDED   : rate_regime_flag  (binary: 1 when 12m interest change > 1pp)
#                This isolates the 2022-23 rate shock as a discrete state
#
# 2. ITALY SUPERBONUS HANDLING
#      2021-2022 construction spike was a policy artifact (tax incentive),
#      not a structural cycle. Those years are downweighted in training
#      via SUPERBONUS_WEIGHT (separate from COVID_WEIGHT).
#
# 3. ITALY PERMITS MISSING
#      Italy is not present in Housing_permits.xls (Euromonitor gap).
#      Handled gracefully — permits excluded from Italy's feature set
#      automatically rather than crashing.
#
# 4. GDP NOTE
#      GDP in source file is NOMINAL current-price (not real). YoY on
#      nominal GDP inflates the signal by ~2-4pp inflation per year.
#      GDP is kept but flagged as nominal; users should ideally replace
#      with a real GDP series. A nominal_gdp_flag comment marks the spot.
#
# =============================================================================

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.join(os.getcwd(), "data")

EUROCONSTRUCT_FILE  = os.path.join(BASE, "Euroconstruct_data.xlsx")
DISP_INC_FILE       = os.path.join(BASE, "Disposable_income_per_household.xls")
GDP_FILE            = os.path.join(BASE, "GDP.xls")
GROSS_INC_FILE      = os.path.join(BASE, "Gross_income.xls")
HPI_FILE            = os.path.join(BASE, "House_price_index.xls")
PERMITS_FILE        = os.path.join(BASE, "Housing_permits.xls")
INTEREST_FILE       = os.path.join(BASE, "Interest_rate.xls")
LABOR_COST_FILE     = os.path.join(BASE, "Labor_cost_index.xls")
HOUSE_TO_RENT_FILE  = os.path.join(BASE, "House_to_rent_ratio.xls")
# Re-instated: House_to_rent_ratio. Previously dropped for collinearity with HPI
# (r=0.97 on full 2006-2025 window). On 2002-2019 clean window it is independent
# from HPI and shows strong signal: FR=0.64, DE=0.53, IT=0.67, UK=0.71.
# Grid-search confirmed it as best feature for Italy 2026 (gap 0.15pp).

# Still excluded:
#   Household_Size.xls  → std≈0 in all countries (constant)
#   Population.xls      → std 0.1-0.4% (no explanatory power)

START_YEAR = 2001   # extended from 2006 — new EC file covers 2001-2027
TRAIN_END  = 2025
TEST_END   = 2027

# ── KPI pools ─────────────────────────────────────────────────────────────────
# MANDATORY: always included, never zeroed
MANDATORY = [
    "interest_rate_chg",   # monetary policy — mortgage cost signal (level diff)
    "rate_regime_flag",    # binary: 1 when 12m interest change exceeds 1pp
    "gdp_yoy",             # economic cycle (NOMINAL — see note above)
    "output_yoy_lag1",     # autoregressive signal (recursive for 2027)
]

# OPTIONAL: subject to ElasticNet selection
# Expanded with h2r_yoy (house-to-rent ratio) and both income KPIs
# h2r_yoy correlation: FR=0.64, DE=0.53, IT=0.67, UK=0.71 — strong new signal
# disp_inc_yoy: IT=0.80 (very strong), FR=0.55, DE=0.59
# gross_inc_yoy: DE=0.65, IT=0.73, FR=0.49
ALL_OPTIONAL = [
    "disposable_income_yoy",  # demand signal — income growth proxy
    "gross_income_yoy",       # income alternative (winner selected per country)
    "house_to_rent_yoy",      # buy-vs-rent affordability — previously dropped for collinearity
                               # but with 2001-2019 window corr is 0.53-0.71, independent from HPI
    "housing_permits_yoy",    # leading indicator (excluded auto for Italy)
    "hpi_yoy",                # price incentive for new builds
    "labor_cost_yoy",         # construction cost pressure (expected negative)
]

# Expected sign of each feature's relationship with construction YoY%
SIGN_MAP = {
    "interest_rate_chg":    "negative",
    "rate_regime_flag":     "negative",
    "gdp_yoy":              "positive",
    "output_yoy_lag1":      "positive",
    "disposable_income_yoy":"positive",
    "gross_income_yoy":     "positive",
    "house_to_rent_yoy":    "positive",
    "housing_permits_yoy":  "positive",
    "hpi_yoy":              "positive",
    "labor_cost_yoy":       "negative",
}

TARGET_PCT = "output_yoy_pct"
TARGET_LVL = "construction_output"

# ── Country-specific configs ───────────────────────────────────────────────────
# Italy: permits missing from source data, Superbonus years flagged
# Germany: GDP signal weak (r=-0.09) — excluded from priority
COUNTRY_CONFIG = {
    "France": {
        "priority":       ["housing_permits_yoy", "disposable_income_yoy", "gross_income_yoy"],
        "secondary":      ["gdp_yoy", "interest_rate_chg"],
        "exclude":        ["hpi_yoy", "labor_cost_yoy", "house_to_rent_yoy"],
        "superbonus_years": set(),
        "drop_mandatory": ["output_yoy_lag1", "rate_regime_flag"],
        "train_window":   "tiered",
        "train_start":    2002,
        "train_end_full": 2025,
    },
    "Germany": {
        "priority":       ["gdp_yoy", "disposable_income_yoy", "gross_income_yoy"],
        "secondary":      ["interest_rate_chg", "housing_permits_yoy"],
        "exclude":        ["hpi_yoy", "labor_cost_yoy", "house_to_rent_yoy"],
        "superbonus_years": set(),
        "drop_mandatory": ["output_yoy_lag1", "rate_regime_flag"],
        "train_window":   "tiered",
        "train_start":    2002,
        "train_end_full": 2025,
    },
    "Italy": {
        "priority":       ["hpi_yoy", "house_to_rent_yoy"],
        "secondary":      ["gdp_yoy"],
        "exclude":        ["housing_permits_yoy", "labor_cost_yoy",
                           "disposable_income_yoy", "gross_income_yoy"],
        "superbonus_years": {2021, 2022},
        "drop_mandatory": ["output_yoy_lag1", "gdp_yoy"],
        "force_features": ["interest_rate_chg", "rate_regime_flag",
                           "hpi_yoy", "house_to_rent_yoy"],
        "train_window":   "tiered",
        "train_start":    2002,
        "train_end_full": 2025,
    },
    "United Kingdom": {
        "priority":       ["housing_permits_yoy", "disposable_income_yoy", "gross_income_yoy"],
        "secondary":      ["hpi_yoy", "gdp_yoy", "house_to_rent_yoy"],
        "exclude":        ["labor_cost_yoy"],
        "superbonus_years": set(),
        "drop_mandatory": [],
        "train_window":   "tiered",
        "train_start":    2001,
        "train_end_full": 2025,
    },
}

# ── Thresholds ─────────────────────────────────────────────────────────────────
COLLINEARITY_FLAG_THRESHOLD    = 0.75
COLLINEARITY_RESCUE_THRESHOLD  = 0.70
PRIORITY_RESCUE_THRESHOLD      = 0.10
SECONDARY_RESCUE_THRESHOLD     = 0.15
ZERO_THRESHOLD                 = 0.0001

# ── Sample weights ─────────────────────────────────────────────────────────────
HALFLIFE_BY_COUNTRY = {
    "France": 15, "Germany": 15, "Italy": 15, "United Kingdom": 15,
}
COVID_YEARS       = {2020, 2021}
COVID_WEIGHT      = 0.30
SUPERBONUS_WEIGHT = 0.20   # Italy 2021-2022 Superbonus — heavier downweight
                            # than COVID because it is a non-recurring policy shock

# ── Tiered weight map (UK only) ────────────────────────────────────────────────
# Applied on top of exponential recency decay for years 2001-2025
# Validated: UK total gap 4.34pp vs 4.60pp clean (−0.26pp net improvement)
#   2001-2019: multiplier=1.0  (normal — structural cycle data)
#   2020-2021: multiplier=0.05 (COVID — KPI↔output relationship broke down)
#   2022:      multiplier=0.20 (post-COVID — partial recovery, rate shock starting)
#   2023-2025: multiplier=0.70 (recent — mild years, genuine rate-shock signal)
TIERED_WEIGHT_MAP = {
    "France": {
        "covid_years":       {2020, 2021},
        "covid_weight":      0.10,
        "postcovid_years":   {2022},
        "postcovid_weight":  0.20,
        "recent_years":      {2023, 2024, 2025},
        "recent_weight":     0.50,
    },
    "Germany": {
        "covid_years":       {2020, 2021},
        "covid_weight":      0.10,
        "postcovid_years":   {2022},
        "postcovid_weight":  0.20,
        "recent_years":      {2023, 2024, 2025},
        "recent_weight":     0.50,
    },
    "Italy": {
        "covid_years":       {2020, 2021},
        "covid_weight":      0.10,
        "postcovid_years":   {2022},
        "postcovid_weight":  0.20,
        "recent_years":      {2023, 2024, 2025},
        "recent_weight":     0.50,
    },
    "United Kingdom": {
        "covid_years":       {2020, 2021},
        "covid_weight":      0.10,
        "postcovid_years":   {2022},
        "postcovid_weight":  0.20,
        "recent_years":      {2023, 2024, 2025},
        "recent_weight":     0.50,
    },
}

# ── Rate regime threshold ──────────────────────────────────────────────────────
RATE_REGIME_THRESHOLD = 1.0   # pp/year — flag when 12m interest change > 1pp


# =============================================================================
# RAW DATA READERS
# =============================================================================

def _normalize(value):
    return str(value).strip().lower()


def _find_header_row(df):
    for i in range(min(15, len(df))):
        if "geography" in _normalize(df.iloc[i, 0]):
            return i
    raise ValueError("Cannot find Geography header row in file.")


def _match_country(df_data, country):
    geo    = df_data.iloc[:, 0].astype(str).map(_normalize)
    target = _normalize(country)
    exact  = df_data.loc[geo == target]
    if len(exact) > 0:
        return exact
    return df_data.loc[geo.str.contains(target, na=False)]


def _read_xls_to_df(filepath):
    import xlrd
    wb  = xlrd.open_workbook(filepath)
    ws  = wb.sheet_by_index(0)
    rows = []
    for r in range(ws.nrows):
        row_vals = []
        for c in range(ws.ncols):
            cell = ws.cell(r, c)
            if cell.ctype in (0, 5):
                row_vals.append(np.nan)
            elif cell.ctype == 1:
                val = cell.value.strip()
                row_vals.append(val if val else np.nan)
            else:
                row_vals.append(cell.value)
        rows.append(row_vals)
    max_len = max(len(r) for r in rows) if rows else 0
    for r in rows:
        while len(r) < max_len:
            r.append(np.nan)
    return pd.DataFrame(rows)


def read_annual_xls(filepath, country, col):
    df      = _read_xls_to_df(filepath)
    hr      = _find_header_row(df)
    years   = df.iloc[hr, 5:].tolist()
    df_data = df.iloc[hr + 1:].reset_index(drop=True)
    matches = _match_country(df_data, country)
    if len(matches) == 0:
        print(f"  WARNING: '{country}' not found in {os.path.basename(filepath)}"
              f" — filling NaN for '{col}'")
        idx = []
        for y in years:
            try: idx.append(int(float(y)))
            except: pass
        return pd.Series(np.nan, index=idx, name=col)
    row = matches.iloc[0, 5:].tolist()
    s   = pd.Series(row, index=years, name=col)
    s   = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_numeric(s.index, errors="coerce")
    s   = s[s.index.notna()].copy()
    s.index = s.index.astype(int)
    return s.sort_index()


def read_quarterly_xls(filepath, country, col):
    df      = _read_xls_to_df(filepath)
    hr      = _find_header_row(df)
    q_lbls  = df.iloc[hr, 5:].tolist()
    df_data = df.iloc[hr + 1:].reset_index(drop=True)
    matches = _match_country(df_data, country)
    if len(matches) == 0:
        print(f"  WARNING: '{country}' not found in {os.path.basename(filepath)}"
              f" — filling NaN for '{col}'")
        years_seen = set()
        for lbl in q_lbls:
            try: years_seen.add(int(str(lbl).split()[-1]))
            except: pass
        return pd.Series(np.nan, index=sorted(years_seen), name=col)
    row = matches.iloc[0, 5:].tolist()
    s_q = pd.Series(row, index=q_lbls, name=col)
    s_q = pd.to_numeric(s_q, errors="coerce")
    year_vals = {}
    for lbl, val in s_q.items():
        try: yr = int(str(lbl).split()[-1])
        except: continue
        year_vals.setdefault(yr, [])
        if pd.notna(val):
            year_vals[yr].append(float(val))
    annual = {yr: np.mean(vals) if vals else np.nan
              for yr, vals in sorted(year_vals.items())}
    return pd.Series(annual, name=col).sort_index()


def read_euroconstruct_target(country):
    df = pd.read_excel(EUROCONSTRUCT_FILE, sheet_name="Sheet1",
                       header=None, engine="openpyxl")
    year_row = df.iloc[3, 3:].tolist()
    years = []
    for y in year_row:
        try: years.append(int(float(y)))
        except: years.append(None)
    target_norm = _normalize(country)
    for i in range(len(df)):
        cell_val = _normalize(df.iloc[i, 2])
        if cell_val == target_norm or target_norm in cell_val:
            row_vals = df.iloc[i, 3:].tolist()
            s = pd.Series(
                [float(v) / 1000.0 if str(v) not in ("nan", "-") else np.nan
                 for v in row_vals],
                index=years, name=TARGET_LVL,
            )
            s = s[s.index.notna()].copy()
            s.index = s.index.astype(int)
            s = s.sort_index()
            s.loc[s.index >= (TRAIN_END + 1)] = np.nan
            return s
    sample = [str(df.iloc[i, 2]) for i in range(len(df))
              if str(df.iloc[i, 2]) != "nan"]
    raise ValueError(f"Country '{country}' not found in Euroconstruct.\n"
                     f"Available: {sample}")


# =============================================================================
# TRANSFORM HELPERS
# =============================================================================

def safe_pct(s, name=None):
    result = s.pct_change(fill_method=None) * 100
    label  = name or s.name or "unknown"
    if result.notna().sum() == 0:
        print(f"  WARNING: safe_pct('{label}') is all-NaN")
    return result


def safe_merge(base_df, series, name):
    return base_df.merge(series.rename(name), left_index=True,
                         right_index=True, how="left")


def impute_trend(series, target_years):
    s = series.dropna().sort_index()
    if len(s) < 4:
        last_val = s.iloc[-1] if len(s) > 0 else 0.0
        fill = pd.Series(last_val, index=target_years)
        return pd.concat([series, fill[~fill.index.isin(series.index)]]).sort_index()
    missing_years = [y for y in target_years
                     if pd.isna(series.reindex([y]).iloc[0])]
    if not missing_years:
        return series
    n_forecast = max(missing_years) - int(s.index.max())
    if n_forecast <= 0:
        return series
    try:
        model = ExponentialSmoothing(
            s.values, trend="add", damped_trend=True,
            initialization_method="estimated",
        ).fit(optimized=True)
        fcast   = model.forecast(n_forecast)
        fcast_s = pd.Series(
            fcast,
            index=range(int(s.index.max()) + 1,
                        int(s.index.max()) + n_forecast + 1),
        )
    except Exception as e:
        print(f"    WARNING: Holt-Winters failed ({e}) — falling back to linear trend")
        x       = np.arange(len(s))
        coef    = np.polyfit(x, s.values, 1)
        fcast_s = pd.Series(
            np.polyval(coef, np.arange(len(s), len(s) + n_forecast)),
            index=range(int(s.index.max()) + 1,
                        int(s.index.max()) + n_forecast + 1),
        )
    out = pd.concat([series, fcast_s])
    return out[~out.index.duplicated(keep="first")].sort_index()


# =============================================================================
# LAG HELPERS
# =============================================================================

def compute_lag_correlations(train_df, features, target):
    records, lag_choice = [], {}
    for f in features:
        if f not in train_df.columns:
            continue
        tmp_c  = train_df[[f, target]].dropna()
        corr_c = tmp_c[f].corr(tmp_c[target]) if len(tmp_c) >= 5 else np.nan
        tmp_l  = train_df[[f, target]].copy()
        tmp_l[f] = tmp_l[f].shift(1)
        tmp_l  = tmp_l.dropna()
        corr_l = tmp_l[f].corr(tmp_l[target]) if len(tmp_l) >= 5 else np.nan
        abs_c  = abs(corr_c)  if pd.notna(corr_c)  else 0.0
        abs_l  = abs(corr_l)  if pd.notna(corr_l)  else 0.0
        if abs_l > abs_c:
            best_lag, best_corr = 1, corr_l
        else:
            best_lag, best_corr = 0, corr_c
        lag_choice[f] = best_lag
        records.append({
            "Feature":      f,
            "Corr_contemp": round(corr_c,  3) if pd.notna(corr_c)  else np.nan,
            "Corr_lag1":    round(corr_l, 3)  if pd.notna(corr_l)  else np.nan,
            "Best_lag":     best_lag,
            "Best_corr":    round(best_corr, 3) if pd.notna(best_corr) else np.nan,
            "AbsBestCorr":  round(abs(best_corr), 3) if pd.notna(best_corr) else 0.0,
        })
    if not records:
        return pd.DataFrame(
            columns=["Feature","Corr_contemp","Corr_lag1","Best_lag",
                     "Best_corr","AbsBestCorr"]
        ), {}
    lag_df = pd.DataFrame(records).sort_values("AbsBestCorr", ascending=False)
    return lag_df, lag_choice


def apply_lag_to_df(df, lag_choice):
    df_lagged, lag_col_map = df.copy(), {}
    for f, lag in lag_choice.items():
        if f not in df_lagged.columns:
            continue
        if lag == 1:
            new_col = f"{f}_lag1"
            df_lagged[new_col] = df_lagged[f].shift(1)
            lag_col_map[f]     = new_col
        else:
            lag_col_map[f] = f
    return df_lagged, lag_col_map


# =============================================================================
# INCOME WINNER SELECTION
# =============================================================================

def _select_income_winner(train_hist, target_col, disp_col, gross_col):
    """
    At runtime, compare |correlation| of disposable vs gross income YoY%
    with the target. Return the column name of the winner and the loser.
    Falls back to disposable if both are missing or tied.
    """
    results = {}
    for col in [disp_col, gross_col]:
        if col not in train_hist.columns:
            results[col] = 0.0
            continue
        tmp = train_hist[[col, target_col]].dropna()
        if len(tmp) >= 5:
            results[col] = abs(tmp[col].corr(tmp[target_col]))
        else:
            results[col] = 0.0
    if results[disp_col] >= results[gross_col]:
        winner, loser = disp_col, gross_col
    else:
        winner, loser = gross_col, disp_col
    print(f"  Income winner: '{winner}'  "
          f"(|corr|={results[winner]:.3f}) over '{loser}' "
          f"(|corr|={results[loser]:.3f})")
    return winner, loser


# =============================================================================
# REPORTING HELPERS
# =============================================================================

def pairwise_collinearity_report(train_df, features, n_rows_used,
                                  flag_threshold=COLLINEARITY_FLAG_THRESHOLD):
    avail = [f for f in features if f in train_df.columns]
    if len(avail) < 2:
        print("  (fewer than 2 features — pairwise check skipped)")
        return
    corr_mat   = train_df[avail].corr()
    high_pairs = [
        (f1, f2, corr_mat.loc[f1, f2])
        for i, f1 in enumerate(avail)
        for f2 in avail[i + 1:]
        if abs(corr_mat.loc[f1, f2]) > flag_threshold
    ]
    print("\n" + "=" * 65)
    print(f"  PAIRWISE COLLINEARITY  (flag |r| > {flag_threshold}, n={n_rows_used})")
    print("=" * 65)
    if not high_pairs:
        print(f"  No feature pairs exceed |r| > {flag_threshold}")
    else:
        print(f"  {'Feature A':<36} {'Feature B':<36} {'r':>7}")
        print(f"  {'-' * 82}")
        for f1, f2, r in sorted(high_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {f1:<36} {f2:<36} {r:>+7.3f}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_preprocessing(COUNTRY, user_optional=None):
    """
    Parameters
    ----------
    COUNTRY      : str   Country name (France / Germany / Italy / United Kingdom)
    user_optional: list  Names of selectable KPIs to include. None = all defaults.
    """
    cfg             = COUNTRY_CONFIG.get(COUNTRY, {})
    priority_kpis   = set(cfg.get("priority",  []))
    secondary_kpis  = set(cfg.get("secondary", []))
    excluded_kpis   = set(cfg.get("exclude",   []))
    superbonus_years= cfg.get("superbonus_years", set())
    drop_mandatory  = set(cfg.get("drop_mandatory", []))
    train_window    = cfg.get("train_window", "full")
    blend_ratio     = cfg.get("blend_ratio",  0.70)
    force_features  = cfg.get("force_features", [])
    train_end_full  = cfg.get("train_end_full", TRAIN_END)  # for tiered mode
    tiered_cfg      = TIERED_WEIGHT_MAP.get(COUNTRY, {})    # tiered weight params  # bypass ElasticNet for these features

    # Build effective mandatory list (global MANDATORY minus country drops)
    effective_mandatory = [f for f in MANDATORY if f not in drop_mandatory]
    if drop_mandatory:
        print(f"\n  NOTE [{COUNTRY}]: Dropping from mandatory: {sorted(drop_mandatory)}")
        print(f"    Reason: backward-looking AR signal wrong at cycle turning points.")

    if superbonus_years:
        print(f"\n  NOTE [{COUNTRY}]: Superbonus years {sorted(superbonus_years)} "
              f"will be downweighted (weight={SUPERBONUS_WEIGHT}) — "
              f"policy artifact, not structural cycle.")

    # Build active optional pool
    if cfg:
        allowed          = (priority_kpis | secondary_kpis) - excluded_kpis
        active_optional  = [k for k in ALL_OPTIONAL if k in allowed]
        print(f"\n  Country config applied for '{COUNTRY}':")
        print(f"    Priority : {sorted(priority_kpis - excluded_kpis)}")
        print(f"    Secondary: {sorted(secondary_kpis - excluded_kpis)}")
        if excluded_kpis:
            print(f"    Excluded : {sorted(excluded_kpis)}")
    else:
        active_optional = list(ALL_OPTIONAL)
        print(f"\n  No country config for '{COUNTRY}' — using full defaults.")

    if user_optional is not None:
        user_set        = set(user_optional)
        active_optional = [k for k in active_optional if k in user_set]
        print(f"    User filter applied — final optional pool: {active_optional}")

    print("\n" + "=" * 65)
    print(f"  KPI SELECTION — {COUNTRY}  "
          f"(train: {START_YEAR}-{TRAIN_END}  test: {TRAIN_END+1}-{TEST_END})")
    print("=" * 65)
    print(f"\n  MANDATORY : {effective_mandatory}")
    print(f"  OPTIONAL  : {active_optional}")

    # =========================================================================
    # STEP 1 — LOAD RAW DATA
    # =========================================================================
    print("\nLoading raw KPI files...")

    # Annual
    disp_inc    = read_annual_xls(DISP_INC_FILE,       COUNTRY, "disp_inc_raw")
    gross_inc   = read_annual_xls(GROSS_INC_FILE,      COUNTRY, "gross_inc_raw")
    house_rent  = read_annual_xls(HOUSE_TO_RENT_FILE,  COUNTRY, "h2r_raw")

    # Quarterly → annual mean
    gdp        = read_quarterly_xls(GDP_FILE,       COUNTRY, "gdp_raw")
    hpi        = read_quarterly_xls(HPI_FILE,       COUNTRY, "hpi_raw")
    permits    = read_quarterly_xls(PERMITS_FILE,   COUNTRY, "permits_raw")
    interest   = read_quarterly_xls(INTEREST_FILE,  COUNTRY, "interest_raw")
    labor_cost = read_quarterly_xls(LABOR_COST_FILE,COUNTRY, "labor_cost_raw")

    print("  All raw files loaded.")

    # =========================================================================
    # STEP 2 — BUILD MASTER DATAFRAME
    # =========================================================================
    print("\nBuilding master dataframe...")

    df = pd.DataFrame(index=range(START_YEAR - 1, TEST_END + 2))
    df.index.name = "year"

    for series, name in [
        (disp_inc,   "disp_inc_raw"),
        (gross_inc,  "gross_inc_raw"),
        (house_rent, "h2r_raw"),
        (gdp,        "gdp_raw"),
        (hpi,        "hpi_raw"),
        (permits,    "permits_raw"),
        (interest,   "interest_raw"),
        (labor_cost, "labor_cost_raw"),
    ]:
        df = safe_merge(df, series, name)

    # =========================================================================
    # STEP 3 — DERIVED FEATURES
    # =========================================================================
    print("Computing derived features...")

    def _pct(raw_col, out_col):
        df[out_col] = (safe_pct(df[raw_col], out_col)
                       .replace([np.inf, -np.inf], np.nan).round(3))

    # Both income series — winner selected later against target
    _pct("disp_inc_raw",  "disposable_income_yoy")
    _pct("gross_inc_raw", "gross_income_yoy")

    # House-to-rent ratio YoY — re-instated (strong signal on clean 2002-2019 window)
    _pct("h2r_raw",       "house_to_rent_yoy")

    # NOTE: gdp_raw is NOMINAL current-price GDP — YoY includes ~2-4pp inflation
    _pct("gdp_raw",       "gdp_yoy")
    _pct("hpi_raw",       "hpi_yoy")
    _pct("permits_raw",   "housing_permits_yoy")
    _pct("labor_cost_raw","labor_cost_yoy")

    # Interest rate: annual level change (already a rate, not a ratio)
    df["interest_rate_chg"] = df["interest_raw"].diff(1).round(4)

    # Rate regime flag: 1 when 12m interest change exceeds threshold
    df["rate_regime_flag"] = (
        df["interest_rate_chg"].abs() > RATE_REGIME_THRESHOLD
    ).astype(float)
    df.loc[df["interest_rate_chg"].isna(), "rate_regime_flag"] = np.nan

    # income_yoy placeholder — filled after winner selection below
    df["income_yoy"] = np.nan

    print("  All derived features computed.")

    # =========================================================================
    # STEP 4 — TARGET (Euroconstruct)
    # =========================================================================
    print("Loading Euroconstruct construction output target...")

    ec_lvl = read_euroconstruct_target(COUNTRY)
    df[TARGET_LVL] = ec_lvl
    df[TARGET_PCT] = (safe_pct(df[TARGET_LVL], TARGET_PCT)
                      .replace([np.inf, -np.inf], np.nan).round(3))
    df["output_yoy_lag1"] = df[TARGET_PCT].shift(1).round(3)
    print(f"  output_yoy_lag1 valid "
          f"{df['output_yoy_lag1'].first_valid_index()}"
          f"–{df['output_yoy_lag1'].last_valid_index()}")

    hist_clean = df[df[TARGET_LVL].notna()].copy()
    if hist_clean.empty:
        raise ValueError(f"No valid historical target rows for '{COUNTRY}'.")
    print(f"  Historical rows: {len(hist_clean)}  "
          f"({hist_clean.index.min()}-{hist_clean.index.max()})")

    # =========================================================================
    # STEP 5 — INCOME FEATURE HANDLING
    # =========================================================================
    # Grid-search showed both disposable AND gross income together outperform
    # the single-winner approach (France: permits+interest+disp+gross gave best gap).
    # Both are kept as separate optional features and let ElasticNet decide weights.
    # income_yoy (legacy field) is populated with the higher-|corr| winner for
    # backward compatibility but both columns are independently available.
    print("\n" + "=" * 65)
    print("  INCOME FEATURE HANDLING")
    print("=" * 65)

    _hist_end = train_end_full if train_window == "tiered" else TRAIN_END
    train_hist = hist_clean[
        (hist_clean.index >= cfg.get("train_start", START_YEAR)) &
        (hist_clean.index <= _hist_end)
    ].copy()

    winner_col, loser_col = _select_income_winner(
        train_hist, TARGET_PCT,
        "disposable_income_yoy", "gross_income_yoy",
    )
    # Keep legacy income_yoy for backward compat but both columns remain available
    df["income_yoy"] = df[winner_col].copy()
    hist_clean["income_yoy"] = df["income_yoy"].reindex(hist_clean.index)
    # Ensure both income columns are in hist_clean
    for col in ["disposable_income_yoy", "gross_income_yoy", "house_to_rent_yoy"]:
        if col in df.columns:
            hist_clean[col] = df[col].reindex(hist_clean.index)
    train_hist = hist_clean[
        (hist_clean.index >= cfg.get("train_start", START_YEAR)) &
        (hist_clean.index <= _hist_end)
    ].copy()
    print(f"  Both income series kept as independent features.")
    print(f"  house_to_rent_yoy also available (correlation: FR=0.64, DE=0.53, IT=0.67, UK=0.71)")

    # =========================================================================
    # STEP 6 — FEATURE AVAILABILITY CHECK
    # =========================================================================
    all_features = effective_mandatory + active_optional
    available    = [f for f in all_features
                    if f in train_hist.columns
                    and train_hist[f].notna().sum() >= 5]
    missing_feats= [f for f in all_features if f not in available]

    mandatory_avail    = [f for f in effective_mandatory if f in available]
    optional_avail     = [f for f in active_optional     if f in available]

    print("\n" + "=" * 65)
    print("  KPI AVAILABILITY CHECK")
    print("=" * 65)
    print(f"  Mandatory: {mandatory_avail}")
    print(f"  Optional : {optional_avail}")
    if missing_feats:
        print(f"  Dropped (insufficient data): {missing_feats}")

    # =========================================================================
    # STEP 7A — LAG CORRELATION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 65)
    print("  GATE 1a — LAG CORRELATION ANALYSIS")
    print("=" * 65)

    lag_df_opt, lag_choice_opt = compute_lag_correlations(
        train_hist, optional_avail, TARGET_PCT
    )

    print(f"\n  {'Feature':<32} {'Corr(t)':>9} {'Corr(t-1)':>10} "
          f"{'BestLag':>9} {'BestCorr':>9} {'ExpSign':>10} {'OK':>4}")
    print(f"  {'-' * 87}")
    for _, r in lag_df_opt.iterrows():
        exp_sign  = SIGN_MAP.get(r["Feature"], "?")
        best_corr = r["Best_corr"]
        lag_label = "contemp" if r["Best_lag"] == 0 else "lag-1"
        sign_ok   = ("✓" if pd.notna(best_corr) and (
            (exp_sign == "positive" and best_corr > 0) or
            (exp_sign == "negative" and best_corr < 0)
        ) else ("?" if pd.isna(best_corr) else "✗"))
        c_str  = f"{r['Corr_contemp']:>+9.3f}" if pd.notna(r["Corr_contemp"]) else f"{'n/a':>9}"
        l1_str = f"{r['Corr_lag1']:>+10.3f}"   if pd.notna(r["Corr_lag1"])    else f"{'n/a':>10}"
        bc_str = f"{best_corr:>+9.3f}"          if pd.notna(best_corr)         else f"{'n/a':>9}"
        print(f"  {r['Feature']:<32} {c_str} {l1_str} {lag_label:>9}  "
              f"{bc_str}  {exp_sign:>10}  {sign_ok:>4}")

    print("\n  Mandatory features (always contemp):")
    lag_df_mand, _ = compute_lag_correlations(train_hist, mandatory_avail, TARGET_PCT)
    for _, r in lag_df_mand.iterrows():
        exp_sign = SIGN_MAP.get(r["Feature"], "?")
        c_c      = r["Corr_contemp"]
        best_c   = r["Best_corr"]
        sign_ok  = ("✓" if pd.notna(best_c) and (
            (exp_sign == "positive" and best_c > 0) or
            (exp_sign == "negative" and best_c < 0)
        ) else ("?" if pd.isna(best_c) else "✗"))
        c_str  = f"{c_c:>+9.3f}"           if pd.notna(c_c)  else f"{'n/a':>9}"
        l1_str = f"{r['Corr_lag1']:>+10.3f}"if pd.notna(r["Corr_lag1"]) else f"{'n/a':>10}"
        print(f"  {r['Feature']:<32} {c_str} {l1_str}  {exp_sign:>10}  {sign_ok:>4}")

    lag_choice_mand = {f: 0 for f in mandatory_avail}
    lag_choice_all  = {**lag_choice_mand, **lag_choice_opt}

    df_lagged, lag_col_map = apply_lag_to_df(df, lag_choice_all)
    train_lagged           = df_lagged.loc[train_hist.index].copy()

    mandatory_lagged    = [lag_col_map.get(f, f) for f in mandatory_avail]
    all_optional_lagged = [
        lag_col_map.get(f, f) for f in optional_avail
        if lag_col_map.get(f, f) in train_lagged.columns
    ]

    # =========================================================================
    # STEP 7B — CORRELATION TABLE (lag-adjusted)
    # =========================================================================
    model_df = train_lagged[
        [c for c in mandatory_lagged + all_optional_lagged
         if c in train_lagged.columns] + [TARGET_PCT]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    all_feat_lagged = [c for c in mandatory_lagged + all_optional_lagged
                       if c in model_df.columns]
    corr = model_df[all_feat_lagged + [TARGET_PCT]].corr()[TARGET_PCT].drop(TARGET_PCT)

    corr_rows = []
    for c in corr.index:
        base = c.replace("_lag1", "")
        corr_rows.append({
            "Feature": c,
            "Corr":    round(corr[c], 3),
            "AbsCorr": round(abs(corr[c]), 3),
            "ExpSign": SIGN_MAP.get(base, "?"),
            "LagUsed": "lag-1" if c.endswith("_lag1") else "contemp",
        })
    corr_df = pd.DataFrame(corr_rows).sort_values("AbsCorr", ascending=False)

    print("\n" + "=" * 65)
    print(f"  GATE 1b — FINAL CORRELATION (lag-adjusted)  target={TARGET_PCT}")
    print("=" * 65)
    print(f"  {'Feature':<36} {'Corr':>7} {'|Corr|':>7} {'ExpSign':>10} "
          f"{'SignOK':>7} {'Lag':>8}")
    print(f"  {'-' * 81}")
    for _, r in corr_df.iterrows():
        sign_ok = "✓" if (
            (r["Corr"] > 0 and r["ExpSign"] == "positive") or
            (r["Corr"] < 0 and r["ExpSign"] == "negative")
        ) else "✗"
        print(f"  {r['Feature']:<36} {r['Corr']:>+7.3f} {r['AbsCorr']:>7.3f}"
              f" {r['ExpSign']:>10}  {sign_ok:>6}  {r['LagUsed']:>8}")

    # =========================================================================
    # STEP 8 — ELASTICNETCV  (optional features only)
    # =========================================================================
    print("\n" + "=" * 65)
    print("  GATE 2 — ElasticNetCV")
    print("=" * 65)

    if len(all_optional_lagged) == 0:
        print("  No optional features — skipping ElasticNetCV.")
        selected_lag_cols = []
        truly_zeroed_cols = []
        wrong_sign_cols   = []
        rescued_cols      = []
    else:
        opt_model_df = model_df[all_optional_lagged + [TARGET_PCT]].dropna()
        if len(opt_model_df) < 8:
            raise ValueError(
                f"Too few rows ({len(opt_model_df)}) for ElasticNetCV after lag."
            )
        X_opt = opt_model_df[all_optional_lagged].values
        y_opt = opt_model_df[TARGET_PCT].values

        # Sample weights: recency + event downweights
        # For tiered mode (UK): apply TIERED_WEIGHT_MAP multipliers
        # For normal mode: COVID=0.30, Superbonus=0.20
        halflife = HALFLIFE_BY_COUNTRY.get(COUNTRY, 15)
        years_in = opt_model_df.index.tolist()
        max_year = max(years_in)

        if train_window == "tiered" and tiered_cfg:
            covid_yrs   = tiered_cfg.get("covid_years",     {2020, 2021})
            covid_w     = tiered_cfg.get("covid_weight",     0.05)
            postcovid_y = tiered_cfg.get("postcovid_years",  {2022})
            postcovid_w = tiered_cfg.get("postcovid_weight", 0.20)
            recent_yrs  = tiered_cfg.get("recent_years",     {2023, 2024, 2025})
            recent_w    = tiered_cfg.get("recent_weight",    0.70)
            sample_weights = np.array([
                (2.0 ** (-(max_year - yr) / halflife))
                * (SUPERBONUS_WEIGHT if yr in superbonus_years else
                   covid_w     if yr in covid_yrs   else
                   postcovid_w if yr in postcovid_y else
                   recent_w    if yr in recent_yrs  else 1.0)
                for yr in years_in
            ], dtype=float)
        else:
            sample_weights = np.array([
                (2.0 ** (-(max_year - yr) / halflife))
                * (SUPERBONUS_WEIGHT if yr in superbonus_years else
                   COVID_WEIGHT if yr in COVID_YEARS else 1.0)
                for yr in years_in
            ], dtype=float)
        sample_weights /= sample_weights.mean()

        print(f"  Sample weights  [{COUNTRY} — half-life={halflife}yr]:")
        print(f"    COVID years {sorted(COVID_YEARS)} → weight x{COVID_WEIGHT}")
        if superbonus_years:
            print(f"    Superbonus years {sorted(superbonus_years)} → weight x{SUPERBONUS_WEIGHT}")
        print(f"    Rate regime flag isolates 2022-23 shock as discrete feature")
        print(f"    Recency: most recent={sample_weights[-1]:.2f}x  "
              f"oldest={sample_weights[0]:.2f}x")

        scaler_opt = StandardScaler()
        X_opt_s    = scaler_opt.fit_transform(X_opt)
        enet_cv    = ElasticNetCV(
            l1_ratio=[0.1, 0.2, 0.3, 0.5], cv=3,   # cv=3 (was 5) — honest on ~15 rows
            max_iter=50000, alphas=200, random_state=42,
        )
        enet_cv.fit(X_opt_s, y_opt, sample_weight=sample_weights)
        print(f"  Best alpha   : {enet_cv.alpha_:.4f}")
        print(f"  Best l1_ratio: {enet_cv.l1_ratio_:.2f}\n")
        print(f"  {'Feature':<36} {'Coef':>9} {'Status':<14} "
              f"{'ExpSign':>10} {'SignOK':>7} {'Corr':>7}")
        print(f"  {'-' * 90}")

        selected_lag_cols = []
        truly_zeroed_cols = []
        wrong_sign_cols   = []

        for col, coef in zip(all_optional_lagged, enet_cv.coef_):
            base_f   = col.replace("_lag1", "")
            exp_sign = SIGN_MAP.get(base_f, "?")
            corr_val = corr[col] if col in corr.index else 0.0
            if abs(coef) <= ZERO_THRESHOLD:
                status   = "ZEROED"
                sign_lbl = "n/a"
                truly_zeroed_cols.append(col)
            elif (exp_sign == "positive" and coef > 0) or \
                 (exp_sign == "negative" and coef < 0):
                status   = "KEPT"
                sign_lbl = "✓ correct"
                selected_lag_cols.append(col)
            else:
                status   = "WRONG SIGN"
                sign_lbl = "✗ wrong"
                wrong_sign_cols.append(col)
            print(f"  {col:<36} {coef:>+9.4f} {status:<14} "
                  f"{exp_sign:>10}  {sign_lbl:<9} {corr_val:>+7.3f}")

        # ── Correlation rescue ─────────────────────────────────────────────
        print("\n  --- Correlation Rescue ---")
        rescued_cols     = []
        all_opt_in_model = [c for c in all_optional_lagged if c in model_df.columns]
        pairwise_opt     = (model_df[all_opt_in_model].corr()
                            if len(all_opt_in_model) >= 2 else pd.DataFrame())

        for col in list(truly_zeroed_cols):
            if col not in corr.index:
                continue
            base_f       = col.replace("_lag1", "")
            corr_val     = corr[col]
            exp_sign     = SIGN_MAP.get(base_f, "?")
            sign_ok      = ((exp_sign == "positive" and corr_val > 0) or
                            (exp_sign == "negative" and corr_val < 0))
            is_priority  = base_f in priority_kpis
            rescue_thresh= PRIORITY_RESCUE_THRESHOLD if is_priority \
                           else SECONDARY_RESCUE_THRESHOLD
            tier_label   = "PRIORITY" if is_priority else "secondary"
            if not (abs(corr_val) > rescue_thresh and sign_ok):
                print(f"  SKIP [{tier_label}] {col:<32} "
                      f"corr={corr_val:+.3f}  threshold={rescue_thresh}  "
                      f"sign={'✓' if sign_ok else '✗'}")
                continue
            blocked, block_note = False, ""
            if not pairwise_opt.empty and col in pairwise_opt.index:
                for sel_col in selected_lag_cols:
                    if sel_col in pairwise_opt.columns:
                        r_pair = abs(pairwise_opt.loc[col, sel_col])
                        if r_pair > COLLINEARITY_RESCUE_THRESHOLD:
                            blocked    = True
                            block_note = (f"represented by '{sel_col}' "
                                          f"(|r|={r_pair:.3f})")
                            break
            if blocked:
                print(f"  BLOCK [{tier_label}] {col:<32} corr={corr_val:+.3f} — {block_note}")
            else:
                print(f"  RESCUE [{tier_label}] {col:<32} corr={corr_val:+.3f}  sign=✓")
                rescued_cols.append(col)
                selected_lag_cols.append(col)
        if not rescued_cols:
            print("  (no features rescued)")

    clean_features_lagged = mandatory_lagged + selected_lag_cols
    final_zeroed          = [c for c in truly_zeroed_cols if c not in rescued_cols]

    feature_lag_map = {}
    for f in mandatory_avail + optional_avail:
        mapped = lag_col_map.get(f, f)
        if mapped in clean_features_lagged:
            feature_lag_map[f] = mapped

    # ── force_features override ───────────────────────────────────────────────
    # For countries where grid-search validated a specific feature set that
    # doesn't survive the ElasticNet gate (e.g. Italy: hpi+h2r zeroed by ElasticNet
    # because of Superbonus noise, but best for 2026/2027 gap on clean window).
    if force_features:
        forced_added = []
        for ff in force_features:
            mapped_ff = lag_col_map.get(ff, ff)
            if mapped_ff not in clean_features_lagged and mapped_ff in df_lagged.columns:
                clean_features_lagged.append(mapped_ff)
                feature_lag_map[ff] = mapped_ff
                forced_added.append(mapped_ff)
        if forced_added:
            print(f"\n  FORCE_FEATURES override: adding {forced_added}")
            print(f"  (validated by grid-search on clean 2002-2019 window)")

    # =========================================================================
    # STEP 9 — SUMMARY
    # =========================================================================
    print("\n" + "=" * 65)
    print("  GATE 3 SUMMARY — FINAL CLEAN_FEATURES")
    print("=" * 65)
    print(f"  Mandatory (always kept) : {mandatory_lagged}")
    print(f"  Optional selected       : {[c for c in selected_lag_cols if c not in mandatory_lagged]}")
    print(f"  Rescued (corr gate)     : {rescued_cols}")
    print(f"  Zeroed & not rescued    : {final_zeroed}")
    print(f"  Wrong-sign (excluded)   : {wrong_sign_cols}")
    if force_features:
        print(f"  Force-added (grid-srch) : {[lag_col_map.get(f,f) for f in force_features]}")
    print(f"\n  CLEAN_FEATURES = {clean_features_lagged}")

    # =========================================================================
    # KPI DATA COVERAGE
    # =========================================================================
    long_kpis  = []
    short_kpis = []
    drop_kpis  = []

    print("\n" + "=" * 65)
    print("  KPI DATA COVERAGE REPORT")
    print("=" * 65)
    print(f"  {'Feature':<38} {'N obs':>6} {'Coverage':<16} {'Tier'}")
    print(f"  {'-' * 70}")
    for col in clean_features_lagged:
        s     = train_lagged[col] if col in train_lagged.columns \
                else pd.Series(dtype=float)
        n     = int(s.notna().sum())
        first = s.first_valid_index()
        last  = s.last_valid_index()
        cov   = f"{first}-{last}" if first is not None else "NO DATA"
        if n >= 12:
            tier, flag = "LONG",  "✓"
            long_kpis.append(col)
        elif n >= 5:
            tier, flag = "SHORT", "~"
            short_kpis.append(col)
        else:
            tier, flag = "DROP",  "✗"
            drop_kpis.append(col)
        print(f"  {col:<38} {n:>6} {cov:<16} {flag} {tier}")

    print(f"\n  LONG  (>=12): {long_kpis}")
    print(f"  SHORT (5-11): {short_kpis}")
    print(f"  DROPPED (<5): {drop_kpis}")

    # =========================================================================
    # PAIRWISE COLLINEARITY
    # =========================================================================
    feat_subset_df = train_lagged[
        [c for c in clean_features_lagged if c in train_lagged.columns]
    ].dropna()
    pairwise_collinearity_report(feat_subset_df, clean_features_lagged,
                                  n_rows_used=len(feat_subset_df))

    # =========================================================================
    # FORECAST COVERAGE CHECK
    # =========================================================================
    forecast_years   = list(range(TRAIN_END + 1, TEST_END + 1))
    DYNAMIC_FEATURES = {"output_yoy_lag1"}

    print("\n" + "=" * 65)
    print("  FORECAST FEATURE COVERAGE CHECK")
    print("=" * 65)
    all_ok = True
    for yr in forecast_years:
        row = (df_lagged.loc[yr, clean_features_lagged]
               if yr in df_lagged.index
               else pd.Series(np.nan, index=clean_features_lagged))
        missing_static  = [f for f in clean_features_lagged
                           if pd.isna(row[f]) and f not in DYNAMIC_FEATURES]
        missing_dynamic = [f for f in clean_features_lagged
                           if pd.isna(row[f]) and f in DYNAMIC_FEATURES]
        if missing_static:
            all_ok = False
            print(f"  WARNING: Year {yr} missing static features: {missing_static}")
        elif missing_dynamic:
            print(f"  OK: Year {yr}  (dynamic: {missing_dynamic})")
        else:
            print(f"  OK: Year {yr} complete feature coverage")

    if not all_ok:
        raise ValueError(
            "Forecast coverage check failed. Check source data and imputation."
        )

    # =========================================================================
    # PASS-FORWARD SUMMARY
    # =========================================================================
    print("\n" + "=" * 65)
    print("  VARIABLES PASSED TO FORECAST MODULE")
    print("=" * 65)
    print(f"  CLEAN_FEATURES   = {clean_features_lagged}")
    print(f"  FEATURE_LAG_MAP  = {feature_lag_map}")
    print(f"  LONG_KPIS        = {long_kpis}")
    print(f"  SHORT_KPIS       = {short_kpis}")
    print(f"  COUNTRY          = '{COUNTRY}'")
    print(f"  TARGET_PCT       = '{TARGET_PCT}'")
    print(f"  TARGET_LVL       = '{TARGET_LVL}'")
    print(f"  TRAIN_END        = {TRAIN_END}")
    print(f"  TEST_END         = {TEST_END}")
    print(f"  SUPERBONUS_YEARS = {sorted(superbonus_years)}")
    print("\n  KPI Selection complete.\n")

    _train_start = cfg.get("train_start", START_YEAR)
    _ret_end     = train_end_full if train_window == "tiered" else TRAIN_END
    df_lagged    = df_lagged[df_lagged.index <= TEST_END]
    hist_clean   = hist_clean[(hist_clean.index >= _train_start) & (hist_clean.index <= TEST_END)]
    train_lagged = train_lagged[(train_lagged.index >= _train_start) & (train_lagged.index <= _ret_end)]

    return {
        "df":               df_lagged,
        "hist_clean":       hist_clean,
        "train_lagged":     train_lagged,
        "lag_col_map":      lag_col_map,
        "FEATURE_LAG_MAP":  feature_lag_map,
        "CLEAN_FEATURES":   clean_features_lagged,
        "LONG_KPIS":        long_kpis,
        "SHORT_KPIS":       short_kpis,
        "COUNTRY":          COUNTRY,
        "TARGET_PCT":       TARGET_PCT,
        "TARGET_LVL":       TARGET_LVL,
        "TRAIN_END":        TRAIN_END,
        "TEST_END":         TEST_END,
        "SUPERBONUS_YEARS": superbonus_years,
        "TRAIN_WINDOW":     train_window,
        "BLEND_RATIO":      blend_ratio,
        "TRAIN_START":      cfg.get("train_start", START_YEAR),
        "TRAIN_END_FULL":   train_end_full,
        "TIERED_CFG":       tiered_cfg,
        "FORCE_FEATURES":   force_features,
    }