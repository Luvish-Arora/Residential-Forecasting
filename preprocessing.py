# =============================================================================
# preprocessing.py — KPI Selection for Housing / Construction Output Forecasting
# =============================================================================
#
# TARGET
# ------
#   Euroconstruct_data.xlsx  — residential construction output (EUR mn)
#   Countries: France, Germany, Italy, United Kingdom
#   Time range: 2006-2027  (train: 2006-2025, test/forecast: 2026-2027)
#
# KPI FILES
# ---------
#   Annual  : Disposable_income_per_household.xls  → disposable_income_yoy
#             Gross_income.xls                     → gross_income_yoy
#             House_to_rent_ratio.xls              → house_to_rent_ratio_yoy
#             Household_Size.xls                   → household_size_chg
#             Population.xls                       → population_yoy
#
#   Quarterly (aggregated to annual mean):
#             GDP.xls                              → gdp_yoy
#             House_price_index.xls                → hpi_yoy
#             Housing_permits.xls                  → housing_permits_yoy
#             Interest_rate.xls                    → interest_rate_chg
#             Labor_cost_index.xls                 → labor_cost_yoy
#
# SIGN MAP (expected relationship with housing construction output YoY%)
# -----------------------------------------------------------------------
#   POSITIVE: gdp_yoy, disposable_income_yoy, gross_income_yoy,
#             population_yoy, housing_permits_yoy, hpi_yoy,
#             house_to_rent_ratio_yoy, output_yoy_lag1
#   NEGATIVE: interest_rate_chg, labor_cost_yoy, household_size_chg
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
HOUSE_RENT_FILE     = os.path.join(BASE, "House_to_rent_ratio.xls")
HOUSEHOLD_SIZE_FILE = os.path.join(BASE, "Household_Size.xls")
PERMITS_FILE        = os.path.join(BASE, "Housing_permits.xls")
INTEREST_FILE       = os.path.join(BASE, "Interest_rate.xls")
LABOR_COST_FILE     = os.path.join(BASE, "Labor_cost_index.xls")
POPULATION_FILE     = os.path.join(BASE, "Population.xls")

START_YEAR = 2006   # changed from 2007
TRAIN_END  = 2025   # changed from 2024
TEST_END   = 2027   # changed from 2027

# ── KPI pools ────────────────────────────────────────────────────────────────
MANDATORY = [
    "interest_rate_chg",        # monetary policy — directly affects mortgage demand
    "gdp_yoy",                  # broad economic cycle
    "output_yoy_lag1",          # autoregressive signal
]

ALL_OPTIONAL = [
    "disposable_income_yoy",    # household purchasing power
    "gross_income_yoy",         # pre-tax income growth
    "population_yoy",           # demographic demand driver
    "housing_permits_yoy",      # leading indicator of construction
    "hpi_yoy",                  # price incentive for new builds
    "house_to_rent_ratio_yoy",  # relative cost signal
    "labor_cost_yoy",           # construction cost pressure (negative)
    "household_size_chg",       # structural demand (smaller households → more units)
]
ALL_TREND_IMPUTED = []   # none needed — all files already extend to 2027
ALL_PARTIAL_DATA  = []

PRE_LAGGED = []

# Expected sign of each KPI's relationship with housing construction YoY%
sign_map = {
    "interest_rate_chg":       "negative",   # higher rates suppress housing
    "gdp_yoy":                 "positive",   # growth supports construction
    "output_yoy_lag1":         "positive",   # momentum / AR signal
    "disposable_income_yoy":   "positive",   # more income → more housing demand
    "gross_income_yoy":        "positive",   # income growth supports demand
    "population_yoy":          "positive",   # more people → more homes needed
    "housing_permits_yoy":     "positive",   # permits precede completions
    "hpi_yoy":                 "positive",   # higher prices incentivise building
    "house_to_rent_ratio_yoy": "positive",   # buying favoured over renting → builds
    "labor_cost_yoy":          "negative",   # higher costs squeeze margins
    "household_size_chg":      "negative",   # shrinking households → more units needed
}

TARGET_PCT = "output_yoy_pct"
TARGET_LVL = "construction_output"

# ── Rescue thresholds ─────────────────────────────────────────────────────────
COLLINEARITY_FLAG_THRESHOLD    = 0.75
COLLINEARITY_RESCUE_THRESHOLD  = 0.70
CORR_RESCUE_THRESHOLD          = 0.15
ZERO_THRESHOLD                 = 0.0001

# Country-specific KPI configs
# Each country: priority KPIs (structural importance), secondary, excluded
COUNTRY_KPI_CONFIG = {
    "France": {
        "priority":  ["housing_permits_yoy", "gdp_yoy", "disposable_income_yoy", "hpi_yoy"],
        "secondary": ["gross_income_yoy", "population_yoy", "house_to_rent_ratio_yoy",
                      "labor_cost_yoy"],
        "exclude":   ["household_size_chg"],
    },
    "Germany": {
        "priority":  ["housing_permits_yoy", "gdp_yoy", "hpi_yoy", "labor_cost_yoy"],
        "secondary": ["disposable_income_yoy", "gross_income_yoy",
                      "house_to_rent_ratio_yoy", "population_yoy"],
        "exclude":   ["household_size_chg"],
    },
    "Italy": {
        "priority":  ["gdp_yoy", "disposable_income_yoy", "hpi_yoy",
                      "house_to_rent_ratio_yoy"],
        "secondary": ["gross_income_yoy", "labor_cost_yoy", "population_yoy"],
        "exclude":   ["housing_permits_yoy", "household_size_chg"],
    },
    "United Kingdom": {
        "priority":  ["housing_permits_yoy", "hpi_yoy", "gdp_yoy",
                      "disposable_income_yoy"],
        "secondary": ["gross_income_yoy", "house_to_rent_ratio_yoy",
                      "labor_cost_yoy", "population_yoy"],
        "exclude":   ["household_size_chg"],
    },
}

PRIORITY_RESCUE_THRESHOLD  = 0.10
SECONDARY_RESCUE_THRESHOLD = 0.15

HALFLIFE_BY_COUNTRY = {
    "France":         15,
    "Germany":        15,
    "Italy":          15,
    "United Kingdom": 15,
}
COVID_YEARS  = {2020, 2021}
COVID_WEIGHT = 0.30   # slightly less aggressive — housing had real COVID demand shift


# =============================================================================
# RAW DATA READERS
# =============================================================================

def _normalize(value):
    return str(value).strip().lower()


def _find_header_row(df):
    """Find the row where col-0 contains 'geography'."""
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


# ── Low-level .xls reader (bypasses pandas xlrd version check) ───────────────
def _read_xls_to_df(filepath):
    """
    Read a .xls file directly via xlrd (any version >= 1.x) and return a
    DataFrame. Bypasses pandas requirement of xlrd >= 2.0.1 so the code
    runs on xlrd 1.2.0 (common in corporate environments).
    """
    import xlrd
    wb  = xlrd.open_workbook(filepath)
    ws  = wb.sheet_by_index(0)
    rows = []
    for r in range(ws.nrows):
        row_vals = []
        for c in range(ws.ncols):
            cell = ws.cell(r, c)
            # ctype: 0=empty, 1=text, 2=number, 3=date, 4=bool, 5=error
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


# ── Annual file reader ────────────────────────────────────────────────────────
def read_annual_xls(filepath, country, col):
    """
    Read an annual KPI file (Geography in col 0, years in cols 5+).
    Returns a pd.Series indexed by integer year.
    """
    df      = _read_xls_to_df(filepath)
    hr      = _find_header_row(df)
    years   = df.iloc[hr, 5:].tolist()
    df_data = df.iloc[hr + 1:].reset_index(drop=True)
    matches = _match_country(df_data, country)
    if len(matches) == 0:
        print(f"  WARNING: '{country}' not found in {os.path.basename(filepath)} "
              f"— filling NaN for '{col}'")
        idx = []
        for y in years:
            try:
                idx.append(int(float(y)))
            except Exception:
                pass
        return pd.Series(np.nan, index=idx, name=col)
    row = matches.iloc[0, 5:].tolist()
    s   = pd.Series(row, index=years, name=col)
    s   = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_numeric(s.index, errors="coerce")
    s   = s[s.index.notna()].copy()
    s.index = s.index.astype(int)
    return s.sort_index()


# ── Quarterly file reader → annual mean ──────────────────────────────────────
def read_quarterly_xls(filepath, country, col):
    """
    Read a quarterly KPI file and collapse to annual means.
    Quarterly cols are labelled 'Q1 YYYY' ... 'Q4 YYYY'.
    Returns a pd.Series indexed by integer year.
    """
    df      = _read_xls_to_df(filepath)
    hr      = _find_header_row(df)
    q_lbls  = df.iloc[hr, 5:].tolist()
    df_data = df.iloc[hr + 1:].reset_index(drop=True)
    matches = _match_country(df_data, country)
    if len(matches) == 0:
        print(f"  WARNING: '{country}' not found in {os.path.basename(filepath)} "
              f"— filling NaN for '{col}'")
        years_seen = set()
        for lbl in q_lbls:
            try:
                yr = int(str(lbl).split()[-1])
                years_seen.add(yr)
            except Exception:
                pass
        return pd.Series(np.nan, index=sorted(years_seen), name=col)

    row = matches.iloc[0, 5:].tolist()
    s_q = pd.Series(row, index=q_lbls, name=col)
    s_q = pd.to_numeric(s_q, errors="coerce")

    year_vals = {}
    for lbl, val in s_q.items():
        try:
            yr = int(str(lbl).split()[-1])
        except Exception:
            continue
        year_vals.setdefault(yr, [])
        if pd.notna(val):
            year_vals[yr].append(float(val))

    annual = {yr: np.mean(vals) if vals else np.nan
              for yr, vals in sorted(year_vals.items())}
    return pd.Series(annual, name=col).sort_index()


# ── Euroconstruct target reader ───────────────────────────────────────────────
def read_euroconstruct_target(country):
    """
    Read construction output from Euroconstruct_data.xlsx.
    Structure: rows 0-2 blank, row 3 = years in cols 3+, row 4+ = country data.
    Country name is in col 2.
    """
    df = pd.read_excel(EUROCONSTRUCT_FILE, sheet_name="Sheet1",
                       header=None, engine="openpyxl")
    # Row 3 (0-indexed) contains years in cols 3 onward
    year_row = df.iloc[3, 3:].tolist()
    years    = []
    for y in year_row:
        try:
            years.append(int(float(y)))
        except Exception:
            years.append(None)

    # Find country row (col 2 has country name)
    target_norm = _normalize(country)
    for i in range(len(df)):
        cell_val = _normalize(df.iloc[i, 2])
        if cell_val == target_norm or target_norm in cell_val:
            row_vals = df.iloc[i, 3:].tolist()
            s = pd.Series(
                [float(v) / 1000.0 if str(v) not in ('nan', '-') else np.nan
                 for v in row_vals],
                index=years,
                name=TARGET_LVL,
            )
            s = s[s.index.notna()].copy()
            s.index = s.index.astype(int)
            s = s.sort_index()
            # Mask forecast years to prevent leakage in training
            s.loc[s.index >= (TRAIN_END + 1)] = np.nan
            return s

    sample = [str(df.iloc[i, 2]) for i in range(len(df)) if str(df.iloc[i, 2]) != 'nan']
    raise ValueError(
        f"Country '{country}' not found in Euroconstruct file.\n"
        f"Available: {sample}"
    )


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
    return base_df.merge(series.rename(name), left_index=True, right_index=True,
                         how="left")


def impute_trend(series, target_years):
    """Extend a series to cover target_years via Holt-Winters (damped additive)."""
    s = series.dropna().sort_index()
    if len(s) < 4:
        last_val = s.iloc[-1] if len(s) > 0 else 0.0
        fill = pd.Series(last_val, index=target_years)
        out  = pd.concat([series, fill[~fill.index.isin(series.index)]])
        return out.sort_index()
    missing_years = [y for y in target_years if pd.isna(series.reindex([y]).iloc[0])]
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
            index=range(int(s.index.max()) + 1, int(s.index.max()) + n_forecast + 1),
        )
    except Exception as e:
        print(f"    WARNING: Holt-Winters failed ({e}) — falling back to linear trend")
        x       = np.arange(len(s))
        coef    = np.polyfit(x, s.values, 1)
        fcast_s = pd.Series(
            np.polyval(coef, np.arange(len(s), len(s) + n_forecast)),
            index=range(int(s.index.max()) + 1, int(s.index.max()) + n_forecast + 1),
        )
    out = pd.concat([series, fcast_s])
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


# =============================================================================
# LAG HELPERS
# =============================================================================

def compute_lag_correlations(train_df, features, target):
    records, lag_choice = [], {}
    for f in features:
        if f not in train_df.columns:
            continue
        tmp_c   = train_df[[f, target]].dropna()
        corr_c  = tmp_c[f].corr(tmp_c[target]) if len(tmp_c) >= 5 else np.nan
        tmp_l   = train_df[[f, target]].copy()
        tmp_l[f]= tmp_l[f].shift(1)
        tmp_l   = tmp_l.dropna()
        corr_l1 = tmp_l[f].corr(tmp_l[target]) if len(tmp_l) >= 5 else np.nan
        abs_c   = abs(corr_c)  if pd.notna(corr_c)  else 0.0
        abs_l1  = abs(corr_l1) if pd.notna(corr_l1) else 0.0
        if abs_l1 > abs_c:
            best_lag, best_corr = 1, corr_l1
        else:
            best_lag, best_corr = 0, corr_c
        lag_choice[f] = best_lag
        records.append({
            "Feature":    f,
            "Corr_contemp": round(corr_c,  3) if pd.notna(corr_c)  else np.nan,
            "Corr_lag1":    round(corr_l1, 3) if pd.notna(corr_l1) else np.nan,
            "Best_lag":     best_lag,
            "Best_corr":    round(best_corr, 3) if pd.notna(best_corr) else np.nan,
            "AbsBestCorr":  round(abs(best_corr), 3) if pd.notna(best_corr) else 0.0,
        })
    if not records:
        return pd.DataFrame(
            columns=["Feature","Corr_contemp","Corr_lag1","Best_lag","Best_corr","AbsBestCorr"]
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
# REPORTING HELPERS
# =============================================================================

def pairwise_collinearity_report(train_df, features,
                                  n_rows_used,
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
    short_names = [f.replace("_lag1", "_L1")[:14] for f in avail]
    print(f"\n  Full correlation matrix ({n_rows_used} rows):")
    print(f"  {'':36}" + "".join(f"{s:>16}" for s in short_names))
    for f1 in avail:
        row = f"  {f1:<36}"
        for f2 in avail:
            r    = corr_mat.loc[f1, f2]
            flag = "*" if f1 != f2 and abs(r) > flag_threshold else " "
            row += f"  {r:>+9.3f}{flag}    "
        print(row)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_preprocessing(COUNTRY, user_optional=None):
    """
    Parameters
    ----------
    COUNTRY      : str   Country name (France / Germany / Italy / United Kingdom)
    user_optional: list  Names of selectable KPIs to include.  None = all defaults.
    """

    # ── Apply country config + user filter ────────────────────────────────────
    country_cfg   = COUNTRY_KPI_CONFIG.get(COUNTRY, {})
    priority_kpis = set(country_cfg.get("priority",  []))
    secondary_kpis= set(country_cfg.get("secondary", []))
    excluded_kpis = set(country_cfg.get("exclude",   []))

    if country_cfg:
        allowed = (priority_kpis | secondary_kpis) - excluded_kpis
        active_optional      = [k for k in ALL_OPTIONAL      if k in allowed]
        active_trend_imputed = []
        active_partial       = []
        print(f"\n  Country-specific KPI config applied for '{COUNTRY}':")
        print(f"    Priority KPIs : {sorted(priority_kpis - excluded_kpis)}")
        print(f"    Secondary KPIs: {sorted(secondary_kpis - excluded_kpis)}")
        print(f"    Excluded KPIs : {sorted(excluded_kpis)}")
    else:
        active_optional      = list(ALL_OPTIONAL)
        active_trend_imputed = list(ALL_TREND_IMPUTED)
        active_partial       = list(ALL_PARTIAL_DATA)
        print(f"\n  No country-specific config for '{COUNTRY}' — using full defaults.")

    if user_optional is not None:
        user_set             = set(user_optional)
        active_optional      = [k for k in active_optional      if k in user_set]
        active_trend_imputed = [k for k in active_trend_imputed if k in user_set]
        active_partial       = [k for k in active_partial       if k in user_set]
        print(f"    User filter applied — final optional pool: {active_optional}")

    print("\n" + "=" * 65)
    print(f"  KPI SELECTION — {COUNTRY}  "
          f"(train: {START_YEAR}-{TRAIN_END}  test: {TRAIN_END+1}-{TEST_END})")
    print("=" * 65)
    print(f"\n  KPI pools for this run:")
    print(f"    MANDATORY (locked) : {MANDATORY}")
    print(f"    OPTIONAL  (active) : {active_optional}")

    # =========================================================================
    # STEP 1 — LOAD RAW DATA
    # =========================================================================
    print("\nLoading raw KPI files...")

    # Annual files
    disp_inc     = read_annual_xls(DISP_INC_FILE,       COUNTRY, "disp_inc_raw")
    gross_inc    = read_annual_xls(GROSS_INC_FILE,       COUNTRY, "gross_inc_raw")
    house_rent   = read_annual_xls(HOUSE_RENT_FILE,      COUNTRY, "house_rent_raw")
    hh_size      = read_annual_xls(HOUSEHOLD_SIZE_FILE,  COUNTRY, "hh_size_raw")
    population   = read_annual_xls(POPULATION_FILE,      COUNTRY, "population_raw")

    # Quarterly → annual
    gdp          = read_quarterly_xls(GDP_FILE,          COUNTRY, "gdp_raw")
    hpi          = read_quarterly_xls(HPI_FILE,          COUNTRY, "hpi_raw")
    permits      = read_quarterly_xls(PERMITS_FILE,      COUNTRY, "permits_raw")
    interest     = read_quarterly_xls(INTEREST_FILE,     COUNTRY, "interest_raw")
    labor_cost   = read_quarterly_xls(LABOR_COST_FILE,   COUNTRY, "labor_cost_raw")

    print("  All raw files loaded")

    # =========================================================================
    # STEP 2 — BUILD MASTER DATAFRAME
    # =========================================================================
    print("\nBuilding master dataframe...")

    # Buffer: one year before START_YEAR to allow pct_change/diff on first row
    df = pd.DataFrame(index=range(START_YEAR - 1, TEST_END + 2))
    df.index.name = "year"

    merge_items = [
        (disp_inc,   "disp_inc_raw"),
        (gross_inc,  "gross_inc_raw"),
        (house_rent, "house_rent_raw"),
        (hh_size,    "hh_size_raw"),
        (population, "population_raw"),
        (gdp,        "gdp_raw"),
        (hpi,        "hpi_raw"),
        (permits,    "permits_raw"),
        (interest,   "interest_raw"),
        (labor_cost, "labor_cost_raw"),
    ]
    for series, name in merge_items:
        df = safe_merge(df, series, name)

    # =========================================================================
    # STEP 3 — DERIVED FEATURES
    # =========================================================================
    print("Computing derived features...")

    def _pct(raw_col, out_col):
        df[out_col] = safe_pct(df[raw_col], out_col).replace(
            [np.inf, -np.inf], np.nan).round(3)

    _pct("disp_inc_raw",  "disposable_income_yoy")
    _pct("gross_inc_raw", "gross_income_yoy")
    _pct("house_rent_raw","house_to_rent_ratio_yoy")
    _pct("population_raw","population_yoy")
    _pct("gdp_raw",       "gdp_yoy")
    _pct("hpi_raw",       "hpi_yoy")
    _pct("permits_raw",   "housing_permits_yoy")
    _pct("labor_cost_raw","labor_cost_yoy")

    # Household size: absolute annual change (not %, as it's a small number)
    df["household_size_chg"] = df["hh_size_raw"].diff(1).round(4)

    # Interest rate: annual change in level (not %, as it's already a rate)
    df["interest_rate_chg"] = df["interest_raw"].diff(1).round(4)

    print("  All derived features computed")

    # =========================================================================
    # STEP 4 — TARGET (Euroconstruct)
    # =========================================================================
    print("Loading Euroconstruct construction output target...")

    ec_lvl = read_euroconstruct_target(COUNTRY)
    df[TARGET_LVL] = ec_lvl
    df[TARGET_PCT] = safe_pct(df[TARGET_LVL], TARGET_PCT).replace(
        [np.inf, -np.inf], np.nan).round(3)
    df["output_yoy_lag1"] = df[TARGET_PCT].shift(1).round(3)
    print(f"  output_yoy_lag1 computed  "
          f"(valid {df['output_yoy_lag1'].first_valid_index()}"
          f"–{df['output_yoy_lag1'].last_valid_index()})")

    hist_clean = df[df[TARGET_LVL].notna()].copy()
    if hist_clean.empty:
        raise ValueError(f"No valid historical target rows found for '{COUNTRY}'.")
    print(f"  Historical rows: {len(hist_clean)}  "
          f"({hist_clean.index.min()}-{hist_clean.index.max()})")

    # =========================================================================
    # STEP 5 — TRAINING WINDOW
    # =========================================================================
    train_hist = hist_clean[hist_clean.index <= TRAIN_END].copy()
    print(f"\n  Training rows for KPI selection: {len(train_hist)} (<={TRAIN_END})")

    all_features = MANDATORY + active_optional + active_trend_imputed + active_partial
    available    = [f for f in all_features
                    if f in train_hist.columns and train_hist[f].notna().sum() >= 5]
    missing_feats= [f for f in all_features if f not in available]

    mandatory_avail     = [f for f in MANDATORY        if f in available]
    optional_avail      = [f for f in active_optional  if f in available]
    all_optional_avail  = optional_avail

    print("\n" + "=" * 65)
    print("  KPI AVAILABILITY CHECK")
    print("=" * 65)
    print(f"  Mandatory  : {mandatory_avail}")
    print(f"  Optional   : {optional_avail}")
    if missing_feats:
        print(f"  Dropped (insufficient data): {missing_feats}")

    # =========================================================================
    # STEP 6A — LAG CORRELATION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 65)
    print("  GATE 1a — LAG CORRELATION ANALYSIS")
    print("=" * 65)

    lag_df_optional, lag_choice_optional = compute_lag_correlations(
        train_hist, all_optional_avail, TARGET_PCT
    )

    print(f"\n  {'Feature':<32} {'Corr(t)':>9} {'Corr(t-1)':>10} "
          f"{'BestLag':>9} {'BestCorr':>9} {'ExpSign':>10} {'OK':>4}")
    print(f"  {'-' * 87}")
    for _, r in lag_df_optional.iterrows():
        exp_sign  = sign_map.get(r["Feature"], "?")
        best_corr = r["Best_corr"]
        lag_label = "contemp" if r["Best_lag"] == 0 else "lag-1"
        if pd.isna(best_corr):
            sign_ok = "?"
        elif ((exp_sign == "positive" and best_corr > 0) or
              (exp_sign == "negative" and best_corr < 0)):
            sign_ok = "✓"
        else:
            sign_ok = "✗"
        c_str  = f"{r['Corr_contemp']:>+9.3f}" if pd.notna(r["Corr_contemp"]) else f"{'n/a':>9}"
        l1_str = f"{r['Corr_lag1']:>+10.3f}"   if pd.notna(r["Corr_lag1"])    else f"{'n/a':>10}"
        bc_str = f"{best_corr:>+9.3f}"          if pd.notna(best_corr)         else f"{'n/a':>9}"
        print(f"  {r['Feature']:<32} {c_str} {l1_str} {lag_label:>9}  "
              f"{bc_str}  {exp_sign:>10}  {sign_ok:>4}")

    print("\n  Mandatory features (always contemp):")
    lag_df_mand, _ = compute_lag_correlations(train_hist, mandatory_avail, TARGET_PCT)
    for _, r in lag_df_mand.iterrows():
        exp_sign = sign_map.get(r["Feature"], "?")
        c_c      = r["Corr_contemp"]
        best_c   = r["Best_corr"]
        if pd.isna(best_c):
            sign_ok = "?"
        elif ((exp_sign == "positive" and best_c > 0) or
              (exp_sign == "negative" and best_c < 0)):
            sign_ok = "✓"
        else:
            sign_ok = "✗"
        c_str  = f"{c_c:>+9.3f}" if pd.notna(c_c) else f"{'n/a':>9}"
        l1_str = f"{r['Corr_lag1']:>+10.3f}" if pd.notna(r["Corr_lag1"]) else f"{'n/a':>10}"
        print(f"  {r['Feature']:<32} {c_str} {l1_str}  {exp_sign:>10}  {sign_ok:>4}")

    lag_choice_mandatory = {f: 0 for f in mandatory_avail}
    lag_choice_all       = {**lag_choice_mandatory, **lag_choice_optional}

    df_lagged, lag_col_map = apply_lag_to_df(df, lag_choice_all)
    train_lagged           = df_lagged.loc[train_hist.index].copy()

    mandatory_lagged    = [lag_col_map.get(f, f) for f in mandatory_avail]
    all_optional_lagged = [
        lag_col_map.get(f, f)
        for f in all_optional_avail
        if lag_col_map.get(f, f) in train_lagged.columns
    ]

    # =========================================================================
    # STEP 6B — CORRELATION TABLE (lag-adjusted)
    # =========================================================================
    model_df = train_lagged[
        [c for c in mandatory_lagged + all_optional_lagged if c in train_lagged.columns]
        + [TARGET_PCT]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    all_feat_lagged = [c for c in mandatory_lagged + all_optional_lagged
                       if c in model_df.columns]
    corr = model_df[all_feat_lagged + [TARGET_PCT]].corr()[TARGET_PCT].drop(TARGET_PCT)

    corr_rows = []
    for c in corr.index:
        base = c.replace("_lag1", "")
        corr_rows.append({
            "Feature":  c,
            "Corr":     round(corr[c], 3),
            "AbsCorr":  round(abs(corr[c]), 3),
            "ExpSign":  sign_map.get(base, "?"),
            "LagUsed":  "lag-1" if c.endswith("_lag1") else "contemp",
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
        print(
            f"  {r['Feature']:<36} {r['Corr']:>+7.3f} {r['AbsCorr']:>7.3f}"
            f" {r['ExpSign']:>10}  {sign_ok:>6}  {r['LagUsed']:>8}"
        )

    # =========================================================================
    # STEP 7 — ELASTICNETCV
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

        # ── Sample weights: recency + COVID downweight ────────────────────────
        halflife = HALFLIFE_BY_COUNTRY.get(COUNTRY, 15)
        years_in = opt_model_df.index.tolist()
        max_year = max(years_in)
        sample_weights = np.array([
            (2.0 ** (-(max_year - yr) / halflife))
            * (COVID_WEIGHT if yr in COVID_YEARS else 1.0)
            for yr in years_in
        ], dtype=float)
        sample_weights /= sample_weights.mean()

        print(f"  Sample weights  [{COUNTRY} — half-life={halflife}yr]:")
        print(f"    COVID years {sorted(COVID_YEARS)} → weight x{COVID_WEIGHT}")
        print(f"    Recency half-life = {halflife} yrs  "
              f"(most recent year = {sample_weights[-1]:.2f}x, "
              f"oldest = {sample_weights[0]:.2f}x)")

        scaler_opt  = StandardScaler()
        X_opt_s     = scaler_opt.fit_transform(X_opt)
        enet_cv = ElasticNetCV(
            l1_ratio=[0.1, 0.2, 0.3, 0.5], cv=5,
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
            exp_sign = sign_map.get(base_f, "?")
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

        # ── Correlation rescue (priority-aware) ──────────────────────────────
        print("\n  --- Correlation Rescue ---")
        print(f"  Priority threshold : |corr| > {PRIORITY_RESCUE_THRESHOLD}")
        print(f"  Secondary threshold: |corr| > {SECONDARY_RESCUE_THRESHOLD}")
        print(f"  Collinearity block : |r| > {COLLINEARITY_RESCUE_THRESHOLD}")

        rescued_cols     = []
        all_opt_in_model = [c for c in all_optional_lagged if c in model_df.columns]
        pairwise_opt     = (model_df[all_opt_in_model].corr()
                            if len(all_opt_in_model) >= 2 else pd.DataFrame())

        for col in list(truly_zeroed_cols):
            if col not in corr.index:
                continue
            base_f    = col.replace("_lag1", "")
            corr_val  = corr[col]
            exp_sign  = sign_map.get(base_f, "?")
            sign_ok   = ((exp_sign == "positive" and corr_val > 0) or
                         (exp_sign == "negative" and corr_val < 0))

            is_priority   = base_f in priority_kpis
            rescue_thresh = PRIORITY_RESCUE_THRESHOLD if is_priority \
                            else SECONDARY_RESCUE_THRESHOLD
            tier_label    = "PRIORITY" if is_priority else "secondary"

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
                print(f"  BLOCK [{tier_label}] {col:<32} "
                      f"corr={corr_val:+.3f} — {block_note}")
            else:
                print(f"  RESCUE [{tier_label}] {col:<32} "
                      f"corr={corr_val:+.3f}  sign=✓")
                rescued_cols.append(col)
                selected_lag_cols.append(col)
        if not rescued_cols:
            print("  (no features rescued)")

    clean_features_lagged = mandatory_lagged + selected_lag_cols
    final_zeroed          = [c for c in truly_zeroed_cols if c not in rescued_cols]

    feature_lag_map = {}
    for f in mandatory_avail + all_optional_avail:
        mapped = lag_col_map.get(f, f)
        if mapped in clean_features_lagged:
            feature_lag_map[f] = mapped

    # =========================================================================
    # STEP 8 — SUMMARY
    # =========================================================================
    print("\n" + "=" * 65)
    print("  GATE 3 SUMMARY — FINAL CLEAN_FEATURES")
    print("=" * 65)
    print(f"  Mandatory (always kept) : {mandatory_lagged}")
    print(f"  Optional selected       : {[c for c in selected_lag_cols if c not in mandatory_lagged]}")
    print(f"  Rescued (corr gate)     : {rescued_cols}")
    print(f"  Zeroed & not rescued    : {final_zeroed}")
    print(f"  Wrong-sign (excluded)   : {wrong_sign_cols}")
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
        s     = train_lagged[col] if col in train_lagged.columns else pd.Series(dtype=float)
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
        row = df_lagged.loc[yr, clean_features_lagged] if yr in df_lagged.index \
              else pd.Series(np.nan, index=clean_features_lagged)
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
    print(f"  CLEAN_FEATURES  = {clean_features_lagged}")
    print(f"  FEATURE_LAG_MAP = {feature_lag_map}")
    print(f"  LONG_KPIS       = {long_kpis}")
    print(f"  SHORT_KPIS      = {short_kpis}")
    print(f"  COUNTRY         = '{COUNTRY}'")
    print(f"  TARGET_PCT      = '{TARGET_PCT}'")
    print(f"  TARGET_LVL      = '{TARGET_LVL}'")
    print(f"  TRAIN_END       = {TRAIN_END}")
    print(f"  TEST_END        = {TEST_END}")
    print("\n  KPI Selection complete.\n")

    # Trim buffer rows
    df_lagged    = df_lagged[df_lagged.index <= TEST_END]
    hist_clean   = hist_clean[hist_clean.index <= TEST_END]
    train_lagged = train_lagged[train_lagged.index <= TEST_END]

    return {
        "df":              df_lagged,
        "hist_clean":      hist_clean,
        "train_lagged":    train_lagged,
        "lag_col_map":     lag_col_map,
        "FEATURE_LAG_MAP": feature_lag_map,
        "CLEAN_FEATURES":  clean_features_lagged,
        "LONG_KPIS":       long_kpis,
        "SHORT_KPIS":      short_kpis,
        "COUNTRY":         COUNTRY,
        "TARGET_PCT":      TARGET_PCT,
        "TARGET_LVL":      TARGET_LVL,
        "TRAIN_END":       TRAIN_END,
        "TEST_END":        TEST_END,
    }