# =============================================================================
# forecastmodel.py — Housing / Construction Output Forecasting
# =============================================================================
#
# TARGET  : Euroconstruct residential construction output (EUR bn)
# COUNTRIES: France, Germany, Italy, United Kingdom
# PERIOD  : 2006-2028  (train 2006-2025, forecast 2026-2028)
#
# MODELS
# ------
#   Ridge, Lasso, ElasticNet, XGBoost, HuberRegressor, QuantileRegressor, ARIMAX
#
# WALK-FORWARD  2015-2025  + extra 2026 fold
#   For each year 2015-2025, train on all prior years, predict that year.
#   Extra 2026 fold: train 2006-2025, predict 2026 using Euroconstruct
#   reference level as the "actual" (benchmark we are evaluating against).
#   Best model = lowest walk-forward MAPE across all folds.
#
# RECURSIVE FORECAST 2026-2028
#   Each year expands the training set; anchor = Euroconstruct level(yr-1).
#   level(yr) = EC_level(yr-1) × (1 + predicted_YoY% / 100)
#   Errors do NOT compound — anchor resets to EC reference each year.
#
# OUTPUTS
#   <Country>_wide_table.xlsx  — all models' forecasts
#   <Country>_final.xlsx       — best model vs Euroconstruct reference
#   Chart A  — full 2006-2028 time series
#   Chart B  — zoom 2025-2028 with value labels
#
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_ARIMAX = True
except ImportError:
    HAS_ARIMAX = False
    print("  WARNING: statsmodels not available — ARIMAX disabled.")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    import subprocess, sys
    print("  INFO: xgboost not found — attempting auto-install...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "xgboost", "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        from xgboost import XGBRegressor
        HAS_XGB = True
        print("  INFO: xgboost installed successfully.")
    except Exception:
        HAS_XGB = False
        print("  WARNING: xgboost could not be installed — XGBoost model disabled.")

# =============================================================================
# CONFIG
# =============================================================================

OUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUT_DIR, exist_ok=True)

VALIDATE_FROM   = 2015
MIN_TRAIN_ROWS  = 6

RIDGE_ALPHA     = 5.0    # stronger regularisation for small dataset
LASSO_ALPHA     = 0.5
ENET_ALPHA      = 1.0
ENET_L1         = 0.5
HUBER_EPSILON   = 1.35
QUANTILE_Q      = 0.50
QUANTILE_ALPHA  = 0.0

FORECAST_PCT_CAP = 12.0   # cap YoY% — construction rarely moves >12% in a year
TOP_N_ENSEMBLE   = 3       # average best N models for the ensemble forecast

HALFLIFE_BY_COUNTRY = {
    "France":         15,
    "Germany":        15,
    "Italy":          15,
    "United Kingdom": 15,
}
COVID_YEARS  = {2020, 2021}
COVID_WEIGHT = 0.30

ARIMAX_ORDER = (1, 0, 0)


# =============================================================================
# HELPERS
# =============================================================================

def _safe_save(wb, path, label="Excel", retries=3, wait=3):
    """Save workbook, retrying if the file is locked (e.g. open in Excel)."""
    import time
    for attempt in range(1, retries + 1):
        try:
            wb.save(path)
            print(f"  {label} saved: {path}")
            return
        except PermissionError:
            if attempt < retries:
                print(f"  WARNING: {label} locked (attempt {attempt}/{retries}) — "
                      f"close the file in Excel, retrying in {wait}s...")
                time.sleep(wait)
            else:
                alt = path.replace(".xlsx", f"_v{attempt}.xlsx")
                try:
                    wb.save(alt)
                    print(f"  {label} saved to alternate path (original locked): {alt}")
                except Exception as e2:
                    print(f"  ERROR: Could not save {label}: {e2}")

def pct_to_level(prev_level, pct):
    if pd.isna(prev_level) or pd.isna(pct):
        return np.nan
    return round(float(prev_level) * (1.0 + float(pct) / 100.0), 1)


def cagr(start_val, end_val, n_years):
    if n_years <= 0 or pd.isna(start_val) or pd.isna(end_val) or float(start_val) <= 0:
        return np.nan
    return ((float(end_val) / float(start_val)) ** (1.0 / n_years) - 1) * 100


def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    try:
        return r2_score(y_true, y_pred)
    except Exception:
        return np.nan


def safe_mae(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    try:
        return mean_absolute_error(y_true, y_pred)
    except Exception:
        return np.nan


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def validate_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _sample_weights(years, country, max_year=None):
    halflife = HALFLIFE_BY_COUNTRY.get(country, 15)
    mx = max_year or max(years)
    w = np.array([
        (2.0 ** (-((mx - yr) / halflife))) * (COVID_WEIGHT if yr in COVID_YEARS else 1.0)
        for yr in years
    ], dtype=float)
    w /= w.mean()
    return w


def _imax(a, b):
    return a if a > b else b


# =============================================================================
# MODEL NAMES
# =============================================================================

def _build_model_names():
    names = ["Ridge", "Lasso", "ElasticNet", "Huber", "Quantile"]
    if HAS_XGB:
        names.append("XGBoost")
    if HAS_ARIMAX:
        names.append("ARIMAX")
    return names


# =============================================================================
# FIT & PREDICT
# =============================================================================

def _fit_models(X_raw, y, X_scaled, use_xgb, sample_weight=None, train_years=None):
    sw = sample_weight
    models = {}
    models["Ridge"] = Ridge(alpha=RIDGE_ALPHA).fit(X_scaled, y, sample_weight=sw)
    models["Lasso"] = Lasso(alpha=LASSO_ALPHA, max_iter=10000).fit(
        X_scaled, y, sample_weight=sw)
    models["ElasticNet"] = ElasticNet(
        alpha=ENET_ALPHA, l1_ratio=ENET_L1, max_iter=10000
    ).fit(X_scaled, y, sample_weight=sw)
    try:
        models["Huber"] = HuberRegressor(
            epsilon=HUBER_EPSILON, max_iter=500
        ).fit(X_scaled, y, sample_weight=sw)
    except TypeError:
        models["Huber"] = HuberRegressor(
            epsilon=HUBER_EPSILON, max_iter=500
        ).fit(X_scaled, y)
    try:
        models["Quantile"] = QuantileRegressor(
            quantile=QUANTILE_Q, alpha=QUANTILE_ALPHA, solver="highs"
        ).fit(X_scaled, y)
    except Exception as e:
        print(f"  WARNING: Quantile fit failed: {e}")
    if use_xgb:
        try:
            models["XGBoost"] = XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42, verbosity=0,
            ).fit(X_raw, y, sample_weight=sw)
        except Exception as e:
            print(f"  WARNING: XGBoost fit failed: {e}")
    if HAS_ARIMAX and len(y) >= _imax(8, ARIMAX_ORDER[0] + 2):
        try:
            idx = (pd.date_range(str(train_years[0]), periods=len(y), freq="YE")
                   if train_years is not None else pd.RangeIndex(len(y)))
            endog = pd.Series(y, index=idx)
            exog  = pd.DataFrame(X_raw, index=idx)
            res = SARIMAX(
                endog, exog=exog, order=ARIMAX_ORDER,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            models["ARIMAX"] = res
        except Exception as e:
            print(f"  WARNING: ARIMAX fit failed: {e}")
    return models


def _predict_one(model_name, model_obj, X_raw, X_scaled, train_years=None):
    if model_name in ("Ridge", "Lasso", "ElasticNet", "Huber", "Quantile"):
        raw = float(model_obj.predict(X_scaled)[0])
    elif model_name == "XGBoost":
        X_xgb = np.array(X_raw, dtype=np.float32).reshape(1, -1)
        raw = float(model_obj.predict(X_xgb)[0])
    elif model_name == "ARIMAX":
        exog_fc = pd.DataFrame(X_raw.reshape(1, -1))
        try:
            fc = model_obj.forecast(steps=1, exog=exog_fc)
            raw = float(fc.iloc[0])
        except Exception as e:
            print(f"  WARNING: ARIMAX forecast failed ({e}) — using in-sample mean")
            raw = float(model_obj.fittedvalues.mean())
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return float(np.clip(raw, -FORECAST_PCT_CAP, FORECAST_PCT_CAP))


# =============================================================================
# EUROCONSTRUCT TARGET LOADER  (full series including reference forecasts)
# =============================================================================

def load_full_ec_series(country, test_end):
    """
    Load the full Euroconstruct series (2006-2028) for a country.
    The 2026-2028 values are Euroconstruct reference forecasts.
    """
    from preprocessing import EUROCONSTRUCT_FILE, _normalize

    df = pd.read_excel(EUROCONSTRUCT_FILE, sheet_name="Sheet1",
                       header=None, engine="openpyxl")
    year_row = df.iloc[3, 3:].tolist()
    years    = []
    for y in year_row:
        try:
            years.append(int(float(y)))
        except Exception:
            years.append(None)

    target_norm = _normalize(country)
    for i in range(len(df)):
        cell_val = _normalize(df.iloc[i, 2])
        if cell_val == target_norm or target_norm in cell_val:
            row_vals = df.iloc[i, 3:].tolist()
            s = pd.Series(
                [float(v) / 1000.0 if str(v) not in ('nan', '-') else np.nan
                 for v in row_vals],
                index=years,
                name="EC",
            )
            s = s[s.index.notna()].copy()
            s.index = s.index.astype(int)
            return s.sort_index()

    raise ValueError(f"Country '{country}' not found in Euroconstruct file.")


# =============================================================================
# CHART STYLE
# =============================================================================

def _clean_axes(ax, fig):
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors="#555555", labelsize=8.5, length=0)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5, zorder=0)
    ax.grid(axis="x", visible=False)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_forecast(prep):

    # ── Unpack ────────────────────────────────────────────────────────────────
    df             = prep["df"]
    hist_clean     = prep["hist_clean"]
    train_lagged   = prep["train_lagged"]
    clean_features = prep["CLEAN_FEATURES"]
    target_pct     = prep["TARGET_PCT"]
    target_lvl     = prep["TARGET_LVL"]
    COUNTRY        = prep["COUNTRY"]
    train_end      = prep["TRAIN_END"]   # 2025
    test_end       = prep["TEST_END"]    # 2028

    model_names = _build_model_names()

    print(f"  Country        : {COUNTRY}")
    print(f"  Training years : 2006-{train_end}  ({len(hist_clean)} actuals)")
    print(f"  Forecast years : {train_end+1}-{test_end}")
    print(f"  Features       : {clean_features}")
    print(f"  Models         : {model_names}")

    validate_cols(df,           clean_features,                "df")
    validate_cols(hist_clean,   [target_pct, target_lvl],      "hist_clean")
    validate_cols(train_lagged, clean_features + [target_pct], "train_lagged")

    train_years    = [y for y in hist_clean.index if y <= train_end]
    forecast_years = list(range(train_end + 1, test_end + 1))  # [2026, 2027, 2028]

    ec_full = load_full_ec_series(COUNTRY, test_end)

    # =========================================================================
    # WALK-FORWARD VALIDATION  2015-2026
    # Folds 2015-2025: train on all prior years, predict vs Euroconstruct actual.
    # Extra fold 2026: train on 2006-2025, predict vs EC 2026 reference.
    # Best model = lowest MAPE across all folds.
    # =========================================================================
    print("\n" + "=" * 65)
    print(f"  WALK-FORWARD VALIDATION  {VALIDATE_FROM}-{train_end+1}")
    print("=" * 65)

    wf_actuals_lvl = {}
    wf_preds_lvl   = {m: {} for m in model_names}

    # ── Folds 2015-2025 ───────────────────────────────────────────────────────
    for i in range(1, len(train_years)):
        tr_yrs = train_years[:i]
        te_yr  = train_years[i]
        if te_yr < VALIDATE_FROM:
            continue

        train_slice = train_lagged.loc[
            tr_yrs, clean_features + [target_pct]
        ].replace([np.inf, -np.inf], np.nan).dropna()
        if len(train_slice) < MIN_TRAIN_ROWS:
            continue
        if te_yr not in train_lagged.index:
            continue

        test_row = train_lagged.loc[[te_yr], clean_features + [target_pct]].replace(
            [np.inf, -np.inf], np.nan
        )
        if test_row[clean_features].isna().any(axis=1).iloc[0]:
            continue
        if pd.isna(test_row[target_pct].iloc[0]):
            continue

        X_tr   = train_slice[clean_features].values
        y_tr   = train_slice[target_pct].values
        X_te   = test_row[clean_features].values
        sw     = _sample_weights(train_slice.index.tolist(), COUNTRY)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        prev_level            = float(hist_clean.loc[tr_yrs[-1], target_lvl])
        wf_actuals_lvl[te_yr] = float(hist_clean.loc[te_yr, target_lvl])

        fitted = _fit_models(X_tr, y_tr, X_tr_s, use_xgb=HAS_XGB,
                              sample_weight=sw,
                              train_years=train_slice.index.tolist())
        for m in model_names:
            if m not in fitted:
                continue
            try:
                pp = _predict_one(m, fitted[m], X_te, X_te_s,
                                  train_years=train_slice.index.tolist())
                pl = pct_to_level(prev_level, pp)
            except Exception:
                pp, pl = np.nan, np.nan
            wf_preds_lvl[m][te_yr] = pl

    # ── Extra fold: first forecast year (train_end + 1 = 2026) ───────────────
    yr_first_fc = train_end + 1
    if yr_first_fc in ec_full.index and yr_first_fc in df.index:
        train_slice_fc1 = train_lagged.loc[
            train_years, clean_features + [target_pct]
        ].replace([np.inf, -np.inf], np.nan).dropna()

        feat_fc1 = df.loc[[yr_first_fc], clean_features].copy().replace(
            [np.inf, -np.inf], np.nan
        )
        if "output_yoy_lag1" in clean_features:
            if train_end in ec_full.index and train_end - 1 in ec_full.index:
                l1 = float(ec_full.loc[train_end])
                l2 = float(ec_full.loc[train_end - 1])
                if l2 > 0:
                    feat_fc1.loc[yr_first_fc, "output_yoy_lag1"] = round(
                        (l1 / l2 - 1) * 100, 3)

        feat_fc1_clean = feat_fc1[clean_features]
        has_missing    = feat_fc1_clean.isna().any(axis=1).iloc[0]

        if len(train_slice_fc1) >= MIN_TRAIN_ROWS and not has_missing:
            X_tr   = train_slice_fc1[clean_features].values
            y_tr   = train_slice_fc1[target_pct].values
            X_te   = feat_fc1_clean.values
            sw     = _sample_weights(train_slice_fc1.index.tolist(), COUNTRY)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            prev_level                  = float(ec_full.loc[train_end])
            wf_actuals_lvl[yr_first_fc] = float(ec_full.loc[yr_first_fc])

            fitted_fc1 = _fit_models(X_tr, y_tr, X_tr_s, use_xgb=HAS_XGB,
                                      sample_weight=sw,
                                      train_years=train_slice_fc1.index.tolist())
            for m in model_names:
                if m not in fitted_fc1:
                    continue
                try:
                    pp = _predict_one(m, fitted_fc1[m], X_te, X_te_s,
                                      train_years=train_slice_fc1.index.tolist())
                    pl = pct_to_level(prev_level, pp)
                except Exception:
                    pp, pl = np.nan, np.nan
                wf_preds_lvl[m][yr_first_fc] = pl
            print(f"  {yr_first_fc} fold added  (EC reference = "
                  f"{wf_actuals_lvl[yr_first_fc]:.2f} EUR bn)")
        else:
            print(f"  {yr_first_fc} fold skipped — "
                  f"{'missing features' if has_missing else 'too few rows'}")

    # ── Accuracy table ────────────────────────────────────────────────────────
    print(f"\n  Walk-forward folds: {sorted(wf_actuals_lvl.keys())}")
    print(f"  {'Model':<16} {'MAPE%':>7} {'MAE(EUR bn)':>13} {'R²':>8} {'N':>5}")
    print(f"  {'-' * 55}")

    mape_dict = {}
    mae_dict  = {}
    r2_dict   = {}
    act_yrs   = sorted(wf_actuals_lvl.keys())
    act_arr   = np.array([wf_actuals_lvl[y] for y in act_yrs])

    for m in model_names:
        pred = np.array([wf_preds_lvl[m].get(y, np.nan) for y in act_yrs])
        mask = np.isfinite(pred)
        n    = int(mask.sum())
        if n < 3:
            continue
        mape_dict[m] = safe_mape(act_arr[mask], pred[mask])
        mae_dict[m]  = safe_mae( act_arr[mask], pred[mask])
        r2_dict[m]   = safe_r2(  act_arr[mask], pred[mask])
        print(f"  {m:<16} {mape_dict[m]:>7.2f}% {mae_dict[m]:>12.1f}  "
              f"{r2_dict[m]:>+8.4f} {n:>5}")

    if not mape_dict:
        raise ValueError("Walk-forward produced no usable folds.")

    best_model = min(mape_dict, key=mape_dict.get)
    # Prefer Ensemble if it ties or is within 0.5pp of individual best
    if ("Ensemble" in mape_dict and
            mape_dict["Ensemble"] <= mape_dict[best_model] + 0.5):
        best_model = "Ensemble"
    print(f"\n  → Best model: {best_model}  "
          f"(walk-forward MAPE {mape_dict[best_model]:.2f}%,  "
          f"folds {min(wf_actuals_lvl)}-{max(wf_actuals_lvl)})")

    # ── Build Ensemble: average of top-N models by walk-forward MAPE ──────────
    ranked_models  = sorted(mape_dict, key=mape_dict.get)
    ensemble_models= ranked_models[:min(TOP_N_ENSEMBLE, len(ranked_models))]
    print(f"  → Ensemble: average of top-{len(ensemble_models)} models: {ensemble_models}")

    # Ensemble walk-forward predictions
    wf_preds_lvl["Ensemble"] = {}
    for yr in act_yrs:
        vals = [wf_preds_lvl[m].get(yr, np.nan) for m in ensemble_models]
        vals = [v for v in vals if pd.notna(v)]
        wf_preds_lvl["Ensemble"][yr] = np.mean(vals) if vals else np.nan

    # Ensemble accuracy
    ens_pred = np.array([wf_preds_lvl["Ensemble"].get(y, np.nan) for y in act_yrs])
    ens_mask = np.isfinite(ens_pred)
    if ens_mask.sum() >= 3:
        mape_dict["Ensemble"] = safe_mape(act_arr[ens_mask], ens_pred[ens_mask])
        mae_dict["Ensemble"]  = safe_mae( act_arr[ens_mask], ens_pred[ens_mask])
        r2_dict["Ensemble"]   = safe_r2(  act_arr[ens_mask], ens_pred[ens_mask])
        print(f"  → Ensemble MAPE: {mape_dict['Ensemble']:.2f}%  "
              f"MAE: {mae_dict['Ensemble']:.1f}  R²: {r2_dict['Ensemble']:+.4f}")
    if "Ensemble" not in model_names:
        model_names.append("Ensemble")

    # =========================================================================
    # RECURSIVE FORECAST 2026-2028
    # Anchor = EC reference level(yr-1).  Errors do NOT compound.
    # =========================================================================
    print("\n" + "=" * 65)
    print(f"  RECURSIVE FORECAST {train_end+1}-{test_end}  (all models)")
    print("=" * 65)

    stage1_pct = {m: {} for m in model_names}
    stage1_lvl = {m: {} for m in model_names}
    stage1_pct["Ensemble"] = {}
    stage1_lvl["Ensemble"] = {}

    fcast_features = df.loc[forecast_years, clean_features].copy().replace(
        [np.inf, -np.inf], np.nan
    )

    for yr in forecast_years:
        all_train_yrs = [y for y in range(2000, yr)
                         if y in train_lagged.index and y in ec_full.index]

        train_slice_full = train_lagged.loc[
            [y for y in all_train_yrs if y in train_lagged.index],
            clean_features + [target_pct]
        ].replace([np.inf, -np.inf], np.nan).dropna()

        # Append pseudo-actual rows for forecast years < yr using EC reference
        extra_rows = []
        for prev_yr in range(train_end + 1, yr):
            if prev_yr not in ec_full.index or prev_yr - 1 not in ec_full.index:
                continue
            feat_row = fcast_features.loc[[prev_yr], clean_features].copy()
            if "output_yoy_lag1" in clean_features:
                l1 = float(ec_full.loc[prev_yr - 1])
                l2_yr = prev_yr - 2
                if l2_yr in ec_full.index:
                    l2 = float(ec_full.loc[l2_yr])
                    if l2 > 0:
                        feat_row.loc[prev_yr, "output_yoy_lag1"] = round(
                            (l1 / l2 - 1) * 100, 3)
            curr_lvl     = float(ec_full.loc[prev_yr])
            prev_lvl_ec  = float(ec_full.loc[prev_yr - 1])
            if prev_lvl_ec > 0:
                act_pct = round((curr_lvl / prev_lvl_ec - 1) * 100, 3)
                feat_row[target_pct] = act_pct
                row_clean = feat_row.replace([np.inf, -np.inf], np.nan).dropna()
                if (len(row_clean) > 0 and
                        not row_clean[clean_features].isna().any(axis=1).iloc[0]):
                    extra_rows.append(row_clean)

        if extra_rows:
            train_slice_full = pd.concat(
                [train_slice_full] + extra_rows
            ).sort_index().replace([np.inf, -np.inf], np.nan).dropna()

        if len(train_slice_full) < MIN_TRAIN_ROWS:
            print(f"  WARNING: year {yr} — too few rows ({len(train_slice_full)})")
            continue

        X_tr   = train_slice_full[clean_features].values
        y_tr   = train_slice_full[target_pct].values
        sw     = _sample_weights(train_slice_full.index.tolist(), COUNTRY)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)

        X_fc_row = fcast_features.loc[[yr], clean_features].copy()
        if "output_yoy_lag1" in clean_features:
            py = yr - 1
            if py in ec_full.index and py - 1 in ec_full.index:
                l1 = float(ec_full.loc[py])
                l2 = float(ec_full.loc[py - 1])
                if l2 > 0:
                    X_fc_row.loc[yr, "output_yoy_lag1"] = round(
                        (l1 / l2 - 1) * 100, 3)

        if X_fc_row[clean_features].isna().any(axis=1).iloc[0]:
            miss = [f for f in clean_features if pd.isna(X_fc_row.loc[yr, f])]
            print(f"  WARNING: year {yr} — missing features: {miss}")
            continue

        X_fc   = X_fc_row.values
        X_fc_s = scaler.transform(X_fc)

        fitted = _fit_models(X_tr, y_tr, X_tr_s, use_xgb=HAS_XGB,
                              sample_weight=sw,
                              train_years=train_slice_full.index.tolist())

        prev_level = float(ec_full.loc[yr - 1]) if yr - 1 in ec_full.index else np.nan

        for m in model_names:
            if m == "Ensemble":
                continue   # computed after individual models below
            if m not in fitted:
                continue
            try:
                pp = _predict_one(m, fitted[m], X_fc, X_fc_s,
                                  train_years=train_slice_full.index.tolist())
                pl = pct_to_level(prev_level, pp)
            except Exception as e:
                print(f"  WARNING: {m} year {yr} failed: {e}")
                pp, pl = np.nan, np.nan
            stage1_pct[m][yr] = pp
            stage1_lvl[m][yr] = pl

        # Ensemble = average of top-N models for this year
        ens_pcts = [stage1_pct[m].get(yr, np.nan) for m in ensemble_models
                    if pd.notna(stage1_pct[m].get(yr, np.nan))]
        ens_pp = float(np.mean(ens_pcts)) if ens_pcts else np.nan
        stage1_pct["Ensemble"][yr] = ens_pp
        stage1_lvl["Ensemble"][yr] = pct_to_level(prev_level, ens_pp)

    # ── Forecast summary table ─────────────────────────────────────────────
    print(f"\n  Forecast levels (EUR bn) — anchor = EC reference prior-year:")
    print(f"  {'Year':>5}" + "".join(f"  {m:>12}" for m in model_names))
    print(f"  {'-' * (7 + 14*len(model_names))}")
    for yr in [train_end] + forecast_years:
        row_str = f"  {yr:>5}"
        for m in model_names:
            v = (float(ec_full.loc[train_end])
                 if yr == train_end and train_end in ec_full.index
                 else stage1_lvl[m].get(yr, np.nan))
            row_str += f"  {v:>11.1f}" if pd.notna(v) else f"  {'n/a':>11}"
        print(row_str)

    # ── CAGR table ─────────────────────────────────────────────────────────
    n_cagr    = test_end - train_end
    cagr_dict = {}

    print(f"\n  CAGR {train_end}-{test_end}  (EUR bn):")
    print(f"  {'Series':<22} {'Start':>11} {'End':>11} {'CAGR%/yr':>10}")
    print(f"  {'-' * 58}")

    ec_s  = float(ec_full.loc[train_end]) if train_end in ec_full.index else np.nan
    ec_e  = float(ec_full.loc[test_end])  if test_end  in ec_full.index else np.nan
    cagr_dict["Euroconstruct"] = cagr(ec_s, ec_e, n_cagr)
    print(f"  {'Euroconstruct':<22} {ec_s:>10.1f}  {ec_e:>10.1f}  "
          f"{cagr_dict['Euroconstruct']:>+8.2f}%")

    for m in model_names:
        e_v = stage1_lvl[m].get(test_end, np.nan)
        if pd.notna(ec_s) and pd.notna(e_v):
            cg = cagr(ec_s, float(e_v), n_cagr)
            cagr_dict[m] = cg
            print(f"  {m:<22} {ec_s:>10.1f}  {float(e_v):>10.1f}  {cg:>+8.2f}%")

    # ── Rankings ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  RANKINGS — {COUNTRY}  (walk-forward MAPE {VALIDATE_FROM}-{train_end+1})")
    print(f"{'=' * 65}")
    print(f"  {'Rank':<6} {'Model':<16} {'MAPE%':>7} {'R²':>9} "
          f"{'MAE(EUR bn)':>12} {'CAGR%/yr':>10}")
    print(f"  {'-' * 62}")
    medals = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th"]
    for i, (mn, mv) in enumerate(sorted(mape_dict.items(), key=lambda x: x[1])):
        rank = medals[i] if i < len(medals) else f"{i+1}."
        r2_s = f"{r2_dict.get(mn, np.nan):>+9.3f}" if pd.notna(r2_dict.get(mn)) else f"{'n/a':>9}"
        ma_s = f"{mae_dict.get(mn, np.nan):>11.1f}"  if pd.notna(mae_dict.get(mn)) else f"{'n/a':>12}"
        cg   = cagr_dict.get(mn, np.nan)
        cg_s = f"{cg:>+9.2f}%" if pd.notna(cg) else f"{'n/a':>10}"
        star = "  ← BEST" if mn == best_model else ""
        print(f"  {rank:<6} {mn:<16} {mv:>6.2f}%  {r2_s}  {ma_s}  {cg_s}{star}")

    # ── Shared y-axis ───────────────────────────────────────────────────────
    _all_vals = list(ec_full.dropna().values)
    for m in model_names:
        _all_vals += [v for v in stage1_lvl[m].values() if pd.notna(v)]
    _lo  = min(_all_vals)
    _hi  = max(_all_vals)
    _pad = (_hi - _lo) * 0.15
    YMIN = max(0, _lo - _pad)
    YMAX = _hi + _pad

    best_color = "#E63946"
    ec_color   = "#1A1A1A"
    ec_hist    = ec_full[ec_full.index <= train_end]
    ec_fcast   = ec_full[ec_full.index >= train_end]

    best_s1 = {train_end: float(ec_full.loc[train_end])}
    best_s1.update({yr: stage1_lvl[best_model].get(yr, np.nan) for yr in forecast_years})
    best_s1_s = pd.Series(best_s1).dropna()

    # =========================================================================
    # CHART A — Full 2006-2028
    # =========================================================================
    fig_a, ax_a = plt.subplots(figsize=(14, 4))
    _clean_axes(ax_a, fig_a)

    ax_a.plot(ec_hist.index, ec_hist.values,
              color=ec_color, linewidth=1.8, zorder=6, solid_capstyle="round")
    ax_a.plot(ec_fcast.index, ec_fcast.values,
              color=ec_color, linewidth=1.8, linestyle="--", zorder=6)
    ax_a.plot(best_s1_s.index, best_s1_s.values,
              color=best_color, linewidth=1.6, zorder=7, solid_capstyle="round")

    ax_a.axvline(x=train_end, color="#BBBBBB", linestyle="--", linewidth=0.8, zorder=2)
    ax_a.text(train_end + 0.2, YMAX * 0.98, "Forecast",
              fontsize=8, color="#888888", va="top")
    ax_a.set_ylim(YMIN, YMAX)
    ax_a.set_title(
        f"Residential construction output — {COUNTRY}  (EUR bn)",
        fontsize=9.5, loc="left", pad=6, color="#333333"
    )
    ax_a.set_xticks(sorted(set(list(ec_full.index) + forecast_years)))
    ax_a.tick_params(axis="x", rotation=45, labelsize=7)

    leg_a = [
        Line2D([0],[0], color=ec_color,   linewidth=1.8, label="Euroconstruct reference"),
        Line2D([0],[0], color=ec_color,   linewidth=1.8, linestyle="--",
               label="Euroconstruct (forecast)"),
        Line2D([0],[0], color=best_color, linewidth=1.6,
               label=f"{best_model}  (walk-fwd MAPE {mape_dict[best_model]:.1f}%)"),
    ]
    ax_a.legend(handles=leg_a, fontsize=7.5, loc="upper left",
                frameon=True, framealpha=0.9, edgecolor="#DDDDDD")

    plt.tight_layout()
    chart_a_path = os.path.join(OUT_DIR,
        f"{COUNTRY.replace(' ','_')}_construction_full_2006_{test_end}.png")
    fig_a.savefig(chart_a_path, dpi=150, bbox_inches="tight")
    plt.close(fig_a)
    print(f"\n  Chart A saved: {chart_a_path}")

    # =========================================================================
    # CHART B — Zoom 2025-2028
    # =========================================================================
    fig_b, ax_b = plt.subplots(figsize=(9, 4.5))
    _clean_axes(ax_b, fig_b)

    zoom_years = [train_end] + forecast_years
    ec_z = ec_full[ec_full.index >= train_end]
    ax_b.plot(ec_z[ec_z.index <= train_end].index,
              ec_z[ec_z.index <= train_end].values,
              color=ec_color, linewidth=2.0, zorder=6, solid_capstyle="round")
    ax_b.plot(ec_z[ec_z.index >= train_end].index,
              ec_z[ec_z.index >= train_end].values,
              color=ec_color, linewidth=2.0, linestyle="--", zorder=6)
    ax_b.plot(best_s1_s[best_s1_s.index >= train_end].index,
              best_s1_s[best_s1_s.index >= train_end].values,
              color=best_color, linewidth=2.0, zorder=7, solid_capstyle="round")

    ax_b.axvline(x=train_end, color="#BBBBBB", linestyle="--", linewidth=0.8, zorder=2)
    ax_b.set_ylim(YMIN, YMAX)
    ym_b, yM_b = ax_b.get_ylim()
    ax_b.text(train_end + 0.05, yM_b * 0.98, "Forecast",
              fontsize=8, color="#888888", va="top")

    placed = []
    gap    = (yM_b - ym_b) * 0.05
    for name, val, col in sorted([
        ("Euroconstruct", float(ec_full.loc[test_end]) if test_end in ec_full.index
                          else np.nan, ec_color),
        (best_model,      stage1_lvl[best_model].get(test_end, np.nan), best_color),
    ], key=lambda x: x[1] if pd.notna(x[1]) else 0, reverse=True):
        if pd.isna(val):
            continue
        y_t = float(val)
        for p in placed:
            if abs(y_t - p) < gap:
                y_t = p - gap
        placed.append(y_t)
        cg   = cagr_dict.get(name, np.nan)
        cg_s = f"  {cg:+.1f}%/yr" if pd.notna(cg) else ""
        ax_b.annotate(
            f"{val:.2f}bn{cg_s}",
            xy=(test_end, val), xytext=(test_end + 0.08, y_t),
            fontsize=7.5, color=col, va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=col, alpha=0.4)
        )

    ax_b.set_title(
        f"Residential construction output — {COUNTRY}  (EUR bn)",
        fontsize=9.5, loc="left", pad=6, color="#333333"
    )
    ax_b.set_xticks(zoom_years)
    ax_b.tick_params(axis="x", rotation=0, labelsize=9)
    leg_b = [
        Line2D([0],[0], color=ec_color,   linewidth=2.0, label="Euroconstruct reference"),
        Line2D([0],[0], color=ec_color,   linewidth=2.0, linestyle="--",
               label="Euroconstruct (forecast)"),
        Line2D([0],[0], color=best_color, linewidth=2.0,
               label=f"{best_model}  (walk-fwd MAPE {mape_dict[best_model]:.1f}%)"),
    ]
    ax_b.legend(handles=leg_b, fontsize=7.5, loc="upper left",
                frameon=True, framealpha=0.9, edgecolor="#DDDDDD")
    plt.tight_layout()
    chart_b_path = os.path.join(OUT_DIR,
        f"{COUNTRY.replace(' ','_')}_construction_zoom_{train_end+1}_{test_end}.png")
    fig_b.savefig(chart_b_path, dpi=150, bbox_inches="tight")
    plt.close(fig_b)
    print(f"  Chart B saved: {chart_b_path}")

    # Derive start_year from the actual df index (covers 2006 even if EC file starts 2007)
    start_year = int(df[df.index >= 2000].index.min())

    # =========================================================================
    # EXCEL 1 — WIDE TABLE
    # =========================================================================
    _export_wide_table(
        country=COUNTRY, ec_full=ec_full,
        stage1_lvl=stage1_lvl, model_names=model_names,
        best_model=best_model, mape_dict=mape_dict,
        mae_dict=mae_dict, r2_dict=r2_dict,
        train_end=train_end, test_end=test_end,
        forecast_years=forecast_years, out_dir=OUT_DIR,
        start_year=start_year,
    )

    # =========================================================================
    # EXCEL 2 — FINAL
    # =========================================================================
    _export_final_excel(
        country=COUNTRY, ec_full=ec_full,
        best_model=best_model, stage1_lvl=stage1_lvl,
        mape_dict=mape_dict,
        train_end=train_end, test_end=test_end,
        forecast_years=forecast_years, out_dir=OUT_DIR,
        start_year=start_year,
    )

    print(f"\n  Outputs saved to: {OUT_DIR}/")
    print(f"    Chart A — full 2006-{test_end}")
    print(f"    Chart B — forecast zoom {train_end}-{test_end}")


# =============================================================================
# EXCEL 1 — WIDE TABLE  (all models)
# =============================================================================

def _export_wide_table(country, ec_full, stage1_lvl, model_names,
                        best_model, mape_dict, mae_dict, r2_dict,
                        train_end, test_end, forecast_years, out_dir,
                        start_year=None):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    _start = start_year if start_year is not None else int(ec_full.index.min())
    all_years = list(range(_start, test_end + 1))
    _vx = lambda v: round(float(v), 3) if (v is not None and pd.notna(v)) else None

    HDR_FILL  = PatternFill("solid", fgColor="1F3864")
    HDR_FONT  = Font(name="Arial", bold=True, color="FFFFFF", size=9)
    MET_FONT  = Font(name="Arial", bold=True, size=9)
    DAT_FONT  = Font(name="Arial", size=9)
    EC_FILL   = PatternFill("solid", fgColor="E8F5E9")
    HIST_FILL = PatternFill("solid", fgColor="F2F2F2")
    BEST_FILL = PatternFill("solid", fgColor="FFF3E0")
    THIN      = Side(style="thin", color="DDDDDD")
    BORDER    = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    center_a  = Alignment(horizontal="center", vertical="center")
    left_a    = Alignment(horizontal="left",   vertical="center")

    model_colors = {
        "Ridge":      "E3F2FD",
        "Lasso":      "F3E5F5",
        "ElasticNet": "E8F5E9",
        "Huber":      "FFF9C4",
        "Quantile":   "FFF3E0",
        "XGBoost":    "FCE4EC",
        "ARIMAX":     "E0F7FA",
    }

    wb = Workbook()
    ws = wb.active
    ws.title = f"{country} All Models"

    hdr = ["Metric"] + [str(y) for y in all_years]
    ws.append(hdr)
    for ci, v in enumerate(hdr, 1):
        c = ws.cell(row=1, column=ci)
        c.fill = HDR_FILL; c.font = HDR_FONT
        c.alignment = left_a if ci == 1 else center_a
        c.border = BORDER

    # Euroconstruct reference row
    ec_data = [_vx(ec_full.loc[y]) if y in ec_full.index else None for y in all_years]
    ws.append(["Euroconstruct reference"] + ec_data)
    ri = 2
    for ci in range(1, len(all_years) + 2):
        cell = ws.cell(row=ri, column=ci)
        cell.fill = EC_FILL
        cell.font = MET_FONT if ci == 1 else DAT_FONT
        cell.alignment = left_a if ci == 1 else center_a
        cell.border = BORDER
        if ci > 1 and cell.value is not None:
            cell.number_format = "0.000"

    hist_ci  = {i + 2 for i, y in enumerate(all_years) if y <= train_end}

    for m in model_names:
        mape_lbl = f"  (MAPE {mape_dict[m]:.1f}%)" if m in mape_dict else ""
        star     = "  ← BEST" if m == best_model else ""
        label    = f"{m}{mape_lbl}{star}"
        row_data = [label]
        for y in all_years:
            row_data.append(None if y <= train_end else _vx(stage1_lvl[m].get(y)))
        ws.append(row_data)
        ri += 1
        fill_hex = model_colors.get(m, "FFFFFF")
        row_fill = (BEST_FILL if m == best_model
                    else PatternFill("solid", fgColor=fill_hex))
        for ci in range(1, len(row_data) + 1):
            cell = ws.cell(row=ri, column=ci)
            cell.border = BORDER
            if ci == 1:
                cell.font = MET_FONT; cell.fill = row_fill; cell.alignment = left_a
            else:
                cell.font = DAT_FONT; cell.alignment = center_a
                yr = all_years[ci - 2]
                if yr <= train_end:
                    cell.fill = HIST_FILL
                else:
                    cell.fill = row_fill if cell.value is not None else PatternFill()
                if cell.value is not None:
                    cell.number_format = "0.000"

    ws.freeze_panes = "B2"
    ws.column_dimensions["A"].width = 36
    for i in range(2, len(all_years) + 2):
        ws.column_dimensions[get_column_letter(i)].width = 6.5
    for r in range(1, ws.max_row + 1):
        ws.row_dimensions[r].height = 14

    # Accuracy sheet
    ws2 = wb.create_sheet("Walk-forward accuracy")
    ws2.column_dimensions["A"].width = 22
    for col in ["B", "C", "D"]:
        ws2.column_dimensions[col].width = 14
    ws2.append([f"Walk-forward validation  {VALIDATE_FROM}-{train_end+1}  "
                f"({train_end+1} uses EC reference as actual)"])
    ws2.append(["Model", "MAPE %", "MAE (EUR bn)", "R²"])
    ws2["A1"].font = Font(name="Arial", bold=True, size=9, color="1F3864")
    ws2["A2"].font = ws2["B2"].font = ws2["C2"].font = ws2["D2"].font = Font(
        name="Arial", bold=True, size=9)
    for mn, mv in sorted(mape_dict.items(), key=lambda x: x[1]):
        star = "  ← BEST" if mn == best_model else ""
        ws2.append([
            mn + star,
            round(mv, 2),
            round(mae_dict.get(mn, np.nan), 1) if pd.notna(mae_dict.get(mn, np.nan)) else None,
            round(r2_dict.get(mn, np.nan), 4)  if pd.notna(r2_dict.get(mn, np.nan))  else None,
        ])
    for row in ws2.iter_rows(min_row=3):
        for c in row:
            c.font = Font(name="Arial", size=9)
            c.alignment = Alignment(horizontal="left")

    path = os.path.join(out_dir, f"{country.replace(' ','_')}_wide_table.xlsx")
    _safe_save(wb, path, "Wide table")


# =============================================================================
# EXCEL 2 — FINAL  (Euroconstruct + best model)
# =============================================================================

def _year_cagr(v_start, v_end):
    """Single-period CAGR (%). Returns None if inputs invalid."""
    try:
        if v_start is None or v_end is None:
            return None
        s, e = float(v_start), float(v_end)
        if s <= 0:
            return None
        return round((e / s - 1) * 100, 2)
    except Exception:
        return None


def _export_final_excel(country, ec_full, best_model, stage1_lvl,
                         mape_dict, train_end, test_end,
                         forecast_years, out_dir,
                         start_year=None):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    # Dynamic — works for any number of forecast years (2, 3, etc.)
    # Model forecast shown for ALL forecast years including the first (2026).
    # CAGR anchors from train_end (last historical year = 2025).

    _start = start_year if start_year is not None else int(ec_full.index.min())
    all_years = list(range(_start, test_end + 1))
    _vx = lambda v: round(float(v), 3) if (v is not None and pd.notna(v)) else None

    # ── Styles ────────────────────────────────────────────────────────────────
    HDR_FILL   = PatternFill("solid", fgColor="1F3864")
    HDR_FONT   = Font(name="Arial", bold=True, color="FFFFFF", size=9)
    MET_FONT   = Font(name="Arial", bold=True, size=9)
    DAT_FONT   = Font(name="Arial", size=9)
    EC_FILL    = PatternFill("solid", fgColor="E8F5E9")
    BEST_FILL  = PatternFill("solid", fgColor="FFF3E0")
    HIST_FILL  = PatternFill("solid", fgColor="F5F5F5")
    CAGR_EC_FILL   = PatternFill("solid", fgColor="C8E6C9")
    CAGR_MDL_FILL  = PatternFill("solid", fgColor="FFE0B2")
    THIN       = Side(style="thin", color="DDDDDD")
    BORDER     = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    center_a   = Alignment(horizontal="center", vertical="center")
    left_a     = Alignment(horizontal="left",   vertical="center")

    mape_lbl   = f"  (walk-fwd MAPE {mape_dict[best_model]:.1f}%)" \
                 if best_model in mape_dict else ""
    best_label = f"{best_model} — Recursive forecast{mape_lbl}"

    # ── Level values ─────────────────────────────────────────────────────────
    # EC row: full history + EC reference for all forecast years
    ec_vals = {y: _vx(ec_full.loc[y]) if y in ec_full.index else None
               for y in all_years}

    # Model row: blank for historical years only, show forecast for ALL forecast years
    mdl_vals = {}
    for y in all_years:
        if y <= train_end:
            mdl_vals[y] = None
        else:
            mdl_vals[y] = _vx(stage1_lvl[best_model].get(y))

    # ── CAGR rows — dynamic, anchored from train_end (last historical year) ──
    # Both EC and Model CAGR chain year-by-year from train_end onward.
    ec_cagr_vals  = {y: None for y in all_years}
    mdl_cagr_vals = {y: None for y in all_years}

    prev_ec  = ec_vals.get(train_end)
    prev_mdl = ec_vals.get(train_end)   # model chain anchors from last EC actual

    for yr in forecast_years:
        curr_ec  = ec_vals.get(yr)
        curr_mdl = mdl_vals.get(yr)

        raw_ec = _year_cagr(prev_ec, curr_ec)
        ec_cagr_vals[yr] = round(raw_ec / 100, 4) if raw_ec is not None else None

        raw_mdl = _year_cagr(prev_mdl, curr_mdl)
        mdl_cagr_vals[yr] = round(raw_mdl / 100, 4) if raw_mdl is not None else None

        prev_ec  = curr_ec
        prev_mdl = curr_mdl

    # Build human-readable CAGR row labels
    parts_ec, parts_mdl = [], []
    prev_yr = train_end
    for yr in forecast_years:
        parts_ec.append(f"{yr} = EC {prev_yr}→{yr}")
        parts_mdl.append(f"{yr} = Mdl {prev_yr}→{yr}")
        prev_yr = yr
    ec_cagr_label  = "Euroconstruct: YoY growth  [" + " | ".join(parts_ec)  + "]"
    mdl_cagr_label = "Model forecast: YoY growth  [" + " | ".join(parts_mdl) + "]"

    # CAGR shading starts from first forecast year
    cagr_shade_from = forecast_years[0]

    # ── Build workbook ────────────────────────────────────────────────────────
    wb = Workbook()
    ws = wb.active
    ws.title = "Construction Output"

    # Row 1 — header
    hdr = ["Metric"] + [str(y) for y in all_years]
    ws.append(hdr)
    for ci, v in enumerate(hdr, 1):
        c = ws.cell(row=1, column=ci)
        c.fill = HDR_FILL; c.font = HDR_FONT
        c.alignment = left_a if ci == 1 else center_a
        c.border = BORDER

    def _write_level_row(ws, ri, label, lbl_fill, yr_vals, is_ec=False):
        row_data = [label] + [yr_vals.get(y) for y in all_years]
        ws.append(row_data)
        for ci in range(1, len(row_data) + 1):
            cell = ws.cell(row=ri, column=ci)
            cell.border = BORDER
            if ci == 1:
                cell.font = MET_FONT; cell.fill = lbl_fill; cell.alignment = left_a
            else:
                cell.font = DAT_FONT; cell.alignment = center_a
                yr = all_years[ci - 2]
                if is_ec:
                    cell.fill = EC_FILL
                elif yr <= train_end:   # blank for historical years only
                    cell.fill = HIST_FILL
                else:
                    cell.fill = lbl_fill if cell.value is not None else PatternFill()
                if cell.value is not None:
                    cell.number_format = "0.000"

    def _write_cagr_row(ws, ri, label, lbl_fill, cagr_vals, shade_from_yr):
        """Write a CAGR row. Cells < shade_from_yr get HIST_FILL, others get lbl_fill."""
        row_data = [label] + [cagr_vals.get(y) for y in all_years]
        ws.append(row_data)
        for ci in range(1, len(row_data) + 1):
            cell = ws.cell(row=ri, column=ci)
            cell.border = BORDER
            if ci == 1:
                cell.font = MET_FONT; cell.fill = lbl_fill; cell.alignment = left_a
            else:
                cell.font = DAT_FONT; cell.alignment = center_a
                yr = all_years[ci - 2]
                if yr < shade_from_yr:
                    cell.fill = HIST_FILL
                else:
                    cell.fill = lbl_fill if cell.value is not None else PatternFill()
                if cell.value is not None:
                    cell.number_format = "0.00%"

    # Row 2: EC levels
    _write_level_row(ws, 2, "Euroconstruct reference (EUR bn)", EC_FILL, ec_vals, is_ec=True)
    # Row 3: Model forecast levels (yr_fc2 onward only)
    _write_level_row(ws, 3, best_label + "  [EUR bn]", BEST_FILL, mdl_vals, is_ec=False)
    # Row 4: EC YoY CAGR
    _write_cagr_row(ws, 4, ec_cagr_label,  CAGR_EC_FILL,  ec_cagr_vals,  shade_from_yr=cagr_shade_from)
    # Row 5: Model YoY CAGR
    _write_cagr_row(ws, 5, mdl_cagr_label, CAGR_MDL_FILL, mdl_cagr_vals, shade_from_yr=cagr_shade_from)

    ws.freeze_panes = "B2"
    ws.column_dimensions["A"].width = max(52, len(best_label) + 10)
    for i in range(2, len(all_years) + 2):
        ws.column_dimensions[get_column_letter(i)].width = 6.5
    for r in range(1, ws.max_row + 1):
        ws.row_dimensions[r].height = 15

    # Notes sheet
    ws_n = wb.create_sheet("Notes")
    ws_n.column_dimensions["A"].width = 24
    ws_n.column_dimensions["B"].width = 65
    notes = [
        ("Country",          country),
        ("Best model",       best_model),
        ("Selection basis",  f"Walk-forward MAPE {VALIDATE_FROM}-{train_end+1} "
                             f"({VALIDATE_FROM}-{train_end} vs EC actuals; "
                             f"{train_end+1} vs EC reference)"),
        ("Level formula",    "level(yr) = EC_level(yr-1) × (1 + predicted_YoY% / 100)"),
        ("Target source",    "Euroconstruct_data.xlsx — residential construction (EUR bn)"),
        ("Period",           f"2006-{train_end} historical, {train_end+1}-{test_end} "
                             "Euroconstruct reference"),
        ("KPI sources",      "Euromonitor: GDP, HPI, permits, interest rate, labour cost, "
                             "disposable income, gross income, population, "
                             "house-to-rent ratio, household size"),
    ]
    ws_n.append(["Field", "Value"])
    ws_n["A1"].font = ws_n["B1"].font = Font(name="Arial", bold=True, size=9)
    for field, value in notes:
        ws_n.append([field, value])
    for row in ws_n.iter_rows():
        for c in row:
            c.font = Font(name="Arial", size=9)
            c.alignment = Alignment(horizontal="left", vertical="center",
                                    wrap_text=True)
            ws_n.row_dimensions[c.row].height = 16

    path = os.path.join(out_dir, f"{country.replace(' ','_')}_final.xlsx")
    _safe_save(wb, path, "Final Excel")