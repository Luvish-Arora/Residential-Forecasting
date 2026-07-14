# =============================================================================
# forecastmodel.py — Residential Market Forecasting
# =============================================================================
#
# Styled after the Industrial Output pipeline.
#
# MODELS: Ridge, Lasso, ElasticNet, HuberRegressor, QuantileRegressor,
#         XGBoost, ARIMAX
#
# WORKFLOW
# --------
# WALK-FORWARD VALIDATION  (VALIDATE_FROM → TRAIN_END)
#   For each year in that range: train on all prior years, predict, compare.
#   Best model = lowest MAPE.
#
# STAGE 1 — RECURSIVE FORECAST  (TRAIN_END+1 → TEST_END)
#   level(yr) = actual_level(yr-1) × (1 + predicted_YoY% / 100)
#   Anchor always = actual prior-year level. Errors do NOT compound.
#
# OUTPUTS  (per country)
# ----------------------
# <Country>_wide_table.xlsx    — actuals + ALL models, all years
# <Country>_resi_forecast.xlsx — actuals + best model (+ comparison rows)
# <Country>_yoy_full.png       — full 2007→TEST_END YoY% chart, all models
# <Country>_yoy_zoom.png       — zoom: last 3 actuals + forecast, all models
#
# COMPARISON  (after all countries — call export_comparison())
# ------------------------------------------------------------
# comparison_residential_forecast.xlsx   — All Countries tab + per-country tabs
# comparison_residential_allcountries.png — multi-panel chart
#
# UNIT CONVERSION
# ---------------
# Source (Euroconstruct) is in MILLION €/£.
# ALL level values displayed/saved are in BILLION (÷ 1,000).
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

from sklearn.linear_model import (Ridge, Lasso, ElasticNet,
                                   HuberRegressor, QuantileRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_ARIMAX = True
except ImportError:
    HAS_ARIMAX = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# =============================================================================
# CONFIG
# =============================================================================

VALIDATE_FROM  = 2015
MIN_TRAIN_ROWS = 5
FORECAST_CAP   = 20.0
ARIMAX_ORDER   = (1, 0, 0)
MILLION_TO_BN  = 1_000.0     # source in million → display in billion

HALFLIFE_BY_COUNTRY = {
    "Germany":15, "France":15, "Italy":15, "United Kingdom":15,
}
COVID_YEARS  = {2020, 2021}
COVID_WEIGHT = 0.30

# Chart line colours per model
MODEL_LINE_COLORS = {
    "Ridge":      "#1565C0",
    "Lasso":      "#6A1B9A",
    "ElasticNet": "#C00000",
    "Huber":      "#E65100",
    "Quantile":   "#2E7D32",
    "XGBoost":    "#00838F",
    "ARIMAX":     "#AD1457",
}
# Excel fill colours per model (hex, no #)
MODEL_COLORS_HEX = {
    "Ridge":      "DDEEFF",
    "Lasso":      "EDE7F6",
    "ElasticNet": "E8F5E9",
    "Huber":      "FFF9C4",
    "Quantile":   "FFF3E0",
    "XGBoost":    "FCE4EC",
    "ARIMAX":     "E0F7FA",
}

_COUNTRY_RESULTS = {}   # accumulated for cross-country comparison


# =============================================================================
# METRIC HELPERS
# =============================================================================

def safe_mape(yt, yp):
    yt=np.array(yt,dtype=float); yp=np.array(yp,dtype=float)
    mask=np.isfinite(yt)&np.isfinite(yp)&(yt!=0)
    return float(np.mean(np.abs((yt[mask]-yp[mask])/yt[mask]))*100) if mask.sum()>0 else np.nan

def safe_mae(yt,yp):
    try: return mean_absolute_error(yt,yp)
    except: return np.nan

def safe_r2(yt,yp):
    if len(yt)<2: return np.nan
    try: return r2_score(yt,yp)
    except: return np.nan

def cagr(sv,ev,n):
    if n<=0 or pd.isna(sv) or pd.isna(ev) or float(sv)<=0: return np.nan
    return ((float(ev)/float(sv))**(1/n)-1)*100

def pct_to_level(prev,pct):
    if pd.isna(prev) or pd.isna(pct): return np.nan
    return round(float(prev)*(1+float(pct)/100),3)

def to_bn(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return None
    return round(float(v)/MILLION_TO_BN,4)

def _sw(years,country):
    hl=HALFLIFE_BY_COUNTRY.get(country,15); mx=max(years)
    w=np.array([(2**(-((mx-yr)/hl)))*(COVID_WEIGHT if yr in COVID_YEARS else 1.)
                for yr in years],dtype=float)
    w/=w.mean(); return w

def _build_model_names():
    n=["Ridge","Lasso","ElasticNet","Huber","Quantile"]
    if HAS_XGB:    n.append("XGBoost")
    if HAS_ARIMAX: n.append("ARIMAX")
    return n

def _clean_axes(ax,fig):
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for s in ["top","right","left","bottom"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors="#555555",labelsize=8.5,length=0)
    ax.grid(axis="y",color="#E8E8E8",linewidth=0.5,zorder=0)
    ax.grid(axis="x",visible=False)


# =============================================================================
# FIT & PREDICT
# =============================================================================

def _fit_models(X_raw,y,X_scaled,use_xgb,sample_weight=None,train_years=None):
    sw=sample_weight; models={}
    models["Ridge"]      = Ridge(alpha=1.0).fit(X_scaled,y,sample_weight=sw)
    models["Lasso"]      = Lasso(alpha=0.1,max_iter=10000).fit(X_scaled,y,sample_weight=sw)
    models["ElasticNet"] = ElasticNet(alpha=0.3,l1_ratio=0.5,max_iter=10000).fit(
                               X_scaled,y,sample_weight=sw)
    try:
        models["Huber"] = HuberRegressor(epsilon=1.35,max_iter=500).fit(
            X_scaled,y,sample_weight=sw)
    except TypeError:
        models["Huber"] = HuberRegressor(epsilon=1.35,max_iter=500).fit(X_scaled,y)
    try:
        models["Quantile"] = QuantileRegressor(quantile=0.5,alpha=0.0,
                                               solver="highs").fit(X_scaled,y)
    except Exception as e:
        print(f"  WARNING Quantile: {e}")
    if use_xgb:
        try:
            models["XGBoost"] = XGBRegressor(n_estimators=100,max_depth=3,
                learning_rate=0.1,subsample=0.8,random_state=42,
                verbosity=0).fit(X_raw,y,sample_weight=sw)
        except Exception as e:
            print(f"  WARNING XGBoost: {e}")
    if HAS_ARIMAX and len(y)>=8 and train_years is not None:
        try:
            idx=pd.date_range(str(train_years[0]),periods=len(y),freq="YE")
            res=SARIMAX(pd.Series(y,index=idx),
                        exog=pd.DataFrame(X_raw,index=idx),
                        order=ARIMAX_ORDER,
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)
            models["ARIMAX"]=res
        except Exception as e:
            print(f"  WARNING ARIMAX: {e}")
    return models

def _predict_one(mn,mo,X_raw,X_scaled,train_years=None):
    if mn in ("Ridge","Lasso","ElasticNet","Huber","Quantile"):
        raw=float(mo.predict(X_scaled)[0])
    elif mn=="XGBoost":
        raw=float(mo.predict(np.array(X_raw,dtype=np.float32).reshape(1,-1))[0])
    elif mn=="ARIMAX":
        try:   raw=float(mo.forecast(steps=1,exog=pd.DataFrame(X_raw.reshape(1,-1))).iloc[0])
        except: raw=float(mo.fittedvalues.mean())
    else: raise ValueError(f"Unknown model: {mn}")
    return float(np.clip(raw,-FORECAST_CAP,FORECAST_CAP))

def _fill_ar(fc_row,clean_features,ar_term,target_series,yr,prev_fc_pct):
    if ar_term not in clean_features: return fc_row
    py=yr-1
    if py in target_series.index and pd.notna(target_series.loc[py]):
        py2=py-1
        if py2 in target_series.index and pd.notna(target_series.loc[py2]):
            fc_row.loc[yr,ar_term]=round(
                (float(target_series.loc[py])/float(target_series.loc[py2])-1)*100,3)
    elif prev_fc_pct is not None:
        fc_row.loc[yr,ar_term]=round(prev_fc_pct,3)
    return fc_row


# =============================================================================
# EXCEL STYLE HELPERS
# =============================================================================

def _xl_styles():
    THIN=Side(style="thin",color="CCCCCC"); MED=Side(style="medium",color="888888")
    BORDER=Border(left=THIN,right=THIN,top=THIN,bottom=THIN)
    THICK_B=Border(left=THIN,right=THIN,top=THIN,bottom=MED)
    def fill(h):  return PatternFill("solid",fgColor=h)
    def font(bold=False,color="000000",size=9):
        return Font(name="Arial",bold=bold,color=color,size=size)
    def align(h="center",v="center",wrap=False):
        return Alignment(horizontal=h,vertical=v,wrap_text=wrap)
    def vv(val,dec=3):
        if val is None or (isinstance(val,float) and np.isnan(val)): return None
        return round(float(val),dec)
    return BORDER,THICK_B,fill,font,align,vv


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_forecast(prep):
    df             = prep["df"]
    hist_clean     = prep["hist_clean"]
    train_lagged   = prep["train_lagged"]
    clean_features = prep["CLEAN_FEATURES"]
    target_pct     = prep["TARGET_PCT"]
    target_lvl     = prep["TARGET_LVL"]
    COUNTRY        = prep["COUNTRY"]
    train_end      = prep["TRAIN_END"]
    test_end       = prep["TEST_END"]
    out_dir        = prep.get("OUT_DIR","output")
    ar_term        = "res_yoy_lag1"
    model_names    = _build_model_names()
    currency       = "£bn" if COUNTRY=="United Kingdom" else "€bn"

    print(f"\n  Country        : {COUNTRY}")
    print(f"  Train window   : {hist_clean.index.min()}–{train_end}")
    print(f"  Forecast years : {train_end+1}–{test_end}")
    print(f"  Features       : {clean_features}")
    print(f"  Models         : {model_names}")
    print(f"  Level unit     : {currency}  (source million ÷ {int(MILLION_TO_BN)})")

    target_series  = hist_clean[target_lvl].dropna()   # MILLION
    train_years    = [y for y in hist_clean.index if y<=train_end]
    forecast_years = list(range(train_end+1,test_end+1))
    all_years      = list(range(int(hist_clean.index.min()),test_end+1))

    # =========================================================================
    # WALK-FORWARD VALIDATION
    # =========================================================================
    print(f"\n{'='*65}\n  WALK-FORWARD VALIDATION  {VALIDATE_FROM}–{train_end}\n{'='*65}")

    wf_actuals_lvl={}
    wf_preds_lvl={m:{} for m in model_names}

    for i in range(1,len(train_years)):
        tr_yrs=train_years[:i]; te_yr=train_years[i]
        if te_yr<VALIDATE_FROM: continue
        train_slice=train_lagged.loc[tr_yrs,clean_features+[target_pct]].replace(
            [np.inf,-np.inf],np.nan).dropna()
        if len(train_slice)<MIN_TRAIN_ROWS: continue
        if te_yr not in train_lagged.index: continue
        test_row=train_lagged.loc[[te_yr],clean_features+[target_pct]].replace(
            [np.inf,-np.inf],np.nan)
        if test_row[clean_features].isna().any(axis=1).iloc[0]: continue
        if pd.isna(test_row[target_pct].iloc[0]): continue
        X_tr=train_slice[clean_features].values; y_tr=train_slice[target_pct].values
        X_te=test_row[clean_features].values
        sw=_sw(train_slice.index.tolist(),COUNTRY)
        scaler=StandardScaler(); X_tr_s=scaler.fit_transform(X_tr); X_te_s=scaler.transform(X_te)
        prev_level=float(hist_clean.loc[tr_yrs[-1],target_lvl])
        wf_actuals_lvl[te_yr]=float(hist_clean.loc[te_yr,target_lvl])
        fitted=_fit_models(X_tr,y_tr,X_tr_s,use_xgb=HAS_XGB,sample_weight=sw,
                           train_years=train_slice.index.tolist())
        for m in model_names:
            if m not in fitted: continue
            try:
                pp=_predict_one(m,fitted[m],X_te,X_te_s,train_years=train_slice.index.tolist())
                wf_preds_lvl[m][te_yr]=pct_to_level(prev_level,pp)
            except: wf_preds_lvl[m][te_yr]=np.nan

    print(f"\n  Validation folds: {sorted(wf_actuals_lvl.keys())}")
    print(f"\n  {'Model':<16} {'MAPE%':>7} {'MAE ({})'.format(currency):>12} {'R²':>9} {'N':>5}")
    print(f"  {'─'*52}")

    mape_dict={}; mae_dict={}; r2_dict={}
    act_yrs=sorted(wf_actuals_lvl.keys())
    act_arr=np.array([wf_actuals_lvl[y] for y in act_yrs])

    for m in model_names:
        pred=np.array([wf_preds_lvl[m].get(y,np.nan) for y in act_yrs])
        mask=np.isfinite(pred)
        if mask.sum()<3: continue
        mape_dict[m]=safe_mape(act_arr[mask],pred[mask])
        mae_dict[m] =safe_mae(act_arr[mask],pred[mask])/MILLION_TO_BN   # display bn
        r2_dict[m]  =safe_r2(act_arr[mask],pred[mask])
        print(f"  {m:<16} {mape_dict[m]:>7.2f}%  {mae_dict[m]:>11.3f}  "
              f"{r2_dict[m]:>+9.4f} {int(mask.sum()):>5}")

    if not mape_dict: raise ValueError("Walk-forward produced no usable folds.")
    best_model=min(mape_dict,key=mape_dict.get)
    print(f"\n  → Best model: {best_model}  (MAPE {mape_dict[best_model]:.2f}%)")

    # =========================================================================
    # STAGE 1 — RECURSIVE FORECAST
    # =========================================================================
    print(f"\n{'='*65}\n  STAGE 1 — Forecast {train_end+1}–{test_end}  (all models)\n{'='*65}")

    stage1_pct={m:{} for m in model_names}
    stage1_lvl={m:{} for m in model_names}   # MILLION internally

    fcast_features=(df.loc[forecast_years,clean_features].copy().replace(
                        [np.inf,-np.inf],np.nan)
                    if forecast_years else pd.DataFrame())
    prev_pct_by_model={m:None for m in model_names}

    for yr in forecast_years:
        all_tr_yrs=[y for y in range(int(hist_clean.index.min()),yr)
                    if y in train_lagged.index and y in target_series.index]
        train_slice=train_lagged.loc[
            [y for y in all_tr_yrs if y in train_lagged.index],
            clean_features+[target_pct]
        ].replace([np.inf,-np.inf],np.nan).dropna()
        if len(train_slice)<MIN_TRAIN_ROWS:
            print(f"  WARNING: {yr} — too few rows, skipped."); continue
        X_tr=train_slice[clean_features].values; y_tr=train_slice[target_pct].values
        sw=_sw(train_slice.index.tolist(),COUNTRY)
        scaler=StandardScaler(); X_tr_s=scaler.fit_transform(X_tr)
        fc_row=fcast_features.loc[[yr],clean_features].copy()
        prior_mean=(float(np.nanmean([v for v in prev_pct_by_model.values() if v is not None]))
                    if any(v is not None for v in prev_pct_by_model.values()) else None)
        fc_row=_fill_ar(fc_row,clean_features,ar_term,target_series,yr,prior_mean)
        if fc_row[clean_features].isna().any(axis=1).iloc[0]:
            miss=[f for f in clean_features if pd.isna(fc_row.loc[yr,f])]
            print(f"  WARNING: {yr} — missing features: {miss}"); continue
        X_fc=fc_row.values; X_fc_s=scaler.transform(X_fc)
        fitted=_fit_models(X_tr,y_tr,X_tr_s,use_xgb=HAS_XGB,sample_weight=sw,
                           train_years=train_slice.index.tolist())
        prev_level=(float(target_series.loc[yr-1]) if yr-1 in target_series.index else np.nan)
        for m in model_names:
            if m not in fitted: continue
            try:
                pp=_predict_one(m,fitted[m],X_fc,X_fc_s,train_years=train_slice.index.tolist())
                pl=pct_to_level(prev_level,pp)
                stage1_pct[m][yr]=pp; stage1_lvl[m][yr]=pl
                prev_pct_by_model[m]=pp
            except Exception as e:
                print(f"  WARNING {m} yr {yr}: {e}")
                stage1_pct[m][yr]=np.nan; stage1_lvl[m][yr]=np.nan

    # Console summary
    print(f"\n  YoY% forecasts:")
    hdr="  {:>5}".format("Year")+"".join(f"  {m:>12}" for m in model_names)
    print(hdr); print("  "+"─"*(5+14*len(model_names)+2))
    for yr in forecast_years:
        row=f"  {yr:>5}"
        for m in model_names:
            v=stage1_pct[m].get(yr,np.nan)
            row+=f"  {v:>+11.2f}%" if pd.notna(v) else f"  {'n/a':>11}"
        print(row)

    print(f"\n  Level forecasts ({currency}):")
    print(hdr); print("  "+"─"*(5+14*len(model_names)+2))
    for yr in forecast_years:
        row=f"  {yr:>5}"
        for m in model_names:
            v=stage1_lvl[m].get(yr,np.nan)
            row+=f"  {to_bn(v):>11.3f}" if pd.notna(v) else f"  {'n/a':>11}"
        print(row)

    # CAGR
    n_cagr=test_end-train_end
    cagr_dict={}
    act_start_m=(float(target_series.loc[train_end]) if train_end in target_series.index else np.nan)
    print(f"\n  CAGR {train_end}–{test_end} ({currency}):")
    print(f"  {'Series':<22} {'Start':>10} {'End':>10} {'CAGR%/yr':>10}")
    print(f"  {'─'*56}")
    for m in model_names:
        ev=stage1_lvl[m].get(test_end,np.nan)
        if pd.notna(act_start_m) and pd.notna(ev):
            cg=cagr(act_start_m,float(ev),n_cagr)
            cagr_dict[m]=cg
            print(f"  {m:<22} {to_bn(act_start_m):>10.3f} {to_bn(ev):>10.3f}  {cg:>+8.2f}%")

    # Rankings
    print(f"\n{'='*65}\n  RANKINGS — {COUNTRY}\n{'='*65}")
    print(f"  {'Rank':<6} {'Model':<16} {'MAPE%':>7} {'R²':>9} {'MAE':>9} {'CAGR%':>9}")
    print(f"  {'─'*58}")
    medals=["1st","2nd","3rd","4th","5th","6th","7th"]
    for i,(mn,mv) in enumerate(sorted(mape_dict.items(),key=lambda x:x[1])):
        rank=medals[i] if i<len(medals) else f"{i+1}."
        r2s=f"{r2_dict.get(mn,np.nan):>+9.3f}" if pd.notna(r2_dict.get(mn)) else f"{'n/a':>9}"
        mas=f"{mae_dict.get(mn,np.nan):>8.3f}"  if pd.notna(mae_dict.get(mn)) else f"{'n/a':>9}"
        cgs=f"{cagr_dict.get(mn,np.nan):>+8.2f}%" if pd.notna(cagr_dict.get(mn)) else f"{'n/a':>9}"
        star="  ← BEST" if mn==best_model else ""
        print(f"  {rank:<6} {mn:<16} {mv:>6.2f}%  {r2s}  {mas}  {cgs}{star}")

    # Accumulate for comparison
    _COUNTRY_RESULTS[COUNTRY]=dict(
        target_series=target_series,stage1_pct=stage1_pct,stage1_lvl=stage1_lvl,
        model_names=model_names,best_model=best_model,mape_dict=mape_dict,
        mae_dict=mae_dict,r2_dict=r2_dict,cagr_dict=cagr_dict,
        train_end=train_end,test_end=test_end,forecast_years=forecast_years,
        all_years=all_years,hist_clean=hist_clean,
        wf_actuals_lvl=wf_actuals_lvl,wf_preds_lvl=wf_preds_lvl,out_dir=out_dir,
    )

    # ── EXCEL 1 — WIDE TABLE (all models) ─────────────────────────────────────
    _export_wide_table(COUNTRY,hist_clean,target_series,stage1_lvl,model_names,
                       best_model,mape_dict,mae_dict,r2_dict,
                       train_end,test_end,forecast_years,all_years,
                       wf_actuals_lvl,wf_preds_lvl,out_dir,currency)

    # ── EXCEL 2 — FINAL (best model + actuals, clean) ─────────────────────────
    _export_final_excel(COUNTRY,hist_clean,target_series,stage1_lvl,stage1_pct,
                        best_model,mape_dict,cagr_dict,
                        train_end,test_end,forecast_years,all_years,out_dir,currency)

    # ── CHARTS ────────────────────────────────────────────────────────────────
    _export_charts(COUNTRY,hist_clean,target_pct,stage1_pct,stage1_lvl,
                   best_model,mape_dict,cagr_dict,model_names,
                   train_end,test_end,forecast_years,out_dir,currency)

    print(f"\n  All outputs saved to: {out_dir}/")


# =============================================================================
# EXCEL 1 — WIDE TABLE  (all models, styled like industrial wide_table.xlsx)
# =============================================================================

def _export_wide_table(country,hist_clean,target_series,stage1_lvl,model_names,
                        best_model,mape_dict,mae_dict,r2_dict,
                        train_end,test_end,forecast_years,all_years,
                        wf_actuals_lvl,wf_preds_lvl,out_dir,currency):
    BORDER,THICK_B,fill,font,align,vv=_xl_styles()
    C_HDR="1F3864"; C_FG="FFFFFF"
    C_ACT="E8F5E9"; C_HIST="F2F2F2"; C_FC="EBF3FB"; C_BEST="FFF3E0"
    THIN=Side(style="thin",color="DDDDDD"); BRD=Border(left=THIN,right=THIN,top=THIN,bottom=THIN)

    fc_set=set(forecast_years)
    hist_ci={i+2 for i,y in enumerate(all_years) if y<=train_end}

    wb=Workbook()
    ws=wb.active; ws.title=f"{country} All Models"
    ws.sheet_view.showGridLines=False

    # Header row
    hdr=["Metric"]+[str(y) for y in all_years]
    ws.append(hdr)
    for ci,v in enumerate(hdr,1):
        c=ws.cell(1,ci,v)
        c.fill=fill(C_HDR); c.font=font(bold=True,color=C_FG)
        c.alignment=align("left") if ci==1 else align()
        c.border=BRD

    # Actuals level row (billion)
    act_data={y:(vv(to_bn(float(target_series.loc[y])),3)
                 if y in target_series.index and pd.notna(target_series.loc[y]) else None)
              for y in all_years}
    ws.append([f"Euroconstruct Actuals — Level ({currency})"]+[act_data[y] for y in all_years])
    ri=2
    for ci in range(1,len(all_years)+2):
        c=ws.cell(ri,ci)
        c.fill=fill(C_ACT); c.border=BRD
        c.font=font(bold=(ci==1))
        c.alignment=align("left") if ci==1 else align()
        if ci>1 and c.value is not None: c.number_format="0.000"

    # Actuals YoY% row
    act_pct_data={}
    for y in all_years:
        l=act_data.get(y); p=act_data.get(y-1)
        act_pct_data[y]=round((l/p-1)*100,2) if l and p and p!=0 else None
    ws.append([f"Euroconstruct Actuals — YoY%"]+[act_pct_data[y] for y in all_years])
    ri+=1
    for ci in range(1,len(all_years)+2):
        c=ws.cell(ri,ci)
        c.fill=fill(C_ACT); c.border=BRD
        c.font=font(bold=(ci==1))
        c.alignment=align("left") if ci==1 else align()
        if ci>1 and c.value is not None: c.number_format="0.00"

    # One row per model — level, forecast years only
    for m in model_names:
        mape_s=f"  (MAPE {mape_dict[m]:.1f}%)" if m in mape_dict else ""
        star="  ← BEST" if m==best_model else ""
        label=f"{m}{mape_s}{star}"
        row_data=[label]+[vv(to_bn(stage1_lvl[m].get(y)),3) if y in fc_set else None
                          for y in all_years]
        ws.append(row_data)
        ri+=1
        bg=C_BEST if m==best_model else MODEL_COLORS_HEX.get(m,"FFFFFF")
        for ci in range(1,len(row_data)+1):
            c=ws.cell(ri,ci)
            c.border=BRD
            if ci==1:
                c.fill=fill(bg); c.font=font(bold=(m==best_model)); c.alignment=align("left")
            else:
                yr=all_years[ci-2]
                c.fill=fill(bg if yr in fc_set else C_HIST)
                c.font=font(bold=(m==best_model and yr in fc_set))
                c.alignment=align()
                if c.value is not None: c.number_format="0.000"

    ws.freeze_panes="B2"
    ws.column_dimensions["A"].width=40
    for ci in range(2,len(all_years)+2):
        ws.column_dimensions[get_column_letter(ci)].width=7.5

    # Walk-forward accuracy sheet
    ws2=wb.create_sheet("Walk-forward Accuracy")
    ws2.sheet_view.showGridLines=False
    acc_yrs=sorted(wf_actuals_lvl.keys())
    hdr2=["Model","MAPE%",f"MAE ({currency})","R²"]+[str(y) for y in acc_yrs]
    ws2.append(hdr2)
    for ci,v in enumerate(hdr2,1):
        c=ws2.cell(1,ci,v)
        c.fill=fill(C_HDR); c.font=font(bold=True,color=C_FG)
        c.alignment=align("left") if ci==1 else align(); c.border=BRD
    # Actual row
    act_r2=["Actual (Euroconstruct)","","",""]+\
            [vv(to_bn(wf_actuals_lvl.get(y)),3) for y in acc_yrs]
    ws2.append(act_r2)
    for ci,v in enumerate(act_r2,1):
        c=ws2.cell(2,ci,v); c.fill=fill(C_ACT); c.font=font(bold=True)
        c.alignment=align("left") if ci==1 else align()
        c.border=BRD
        if ci>4 and c.value is not None: c.number_format="0.000"
    for ri2,(mn,mv) in enumerate(sorted(mape_dict.items(),key=lambda x:x[1]),3):
        star=" ★" if mn==best_model else ""
        bg=C_BEST if mn==best_model else MODEL_COLORS_HEX.get(mn,"FFFFFF")
        row_d=([mn+star,round(mv,2),
                round(mae_dict.get(mn,np.nan),3) if pd.notna(mae_dict.get(mn)) else None,
                round(r2_dict.get(mn,np.nan),4) if pd.notna(r2_dict.get(mn)) else None]
               +[vv(to_bn(wf_preds_lvl[mn].get(y)),3) for y in acc_yrs])
        for ci,v in enumerate(row_d,1):
            c=ws2.cell(ri2,ci,v); c.fill=fill(bg); c.font=font(bold=(mn==best_model))
            c.alignment=align("left") if ci==1 else align(); c.border=BRD
            if ci>1 and c.value is not None: c.number_format="0.000"
    ws2.column_dimensions["A"].width=24
    for ci in range(2,len(acc_yrs)+5): ws2.column_dimensions[get_column_letter(ci)].width=11

    path=os.path.join(out_dir,f"{country.replace(' ','_')}_wide_table.xlsx")
    wb.save(path); print(f"\n  ✓ Wide table Excel : {path}")


# =============================================================================
# EXCEL 2 — FINAL  (actuals + best model, styled like industrial _final.xlsx)
# =============================================================================

def _export_final_excel(country,hist_clean,target_series,stage1_lvl,stage1_pct,
                         best_model,mape_dict,cagr_dict,
                         train_end,test_end,forecast_years,all_years,out_dir,currency):
    BORDER,THICK_B,fill,font,align,vv=_xl_styles()
    C_HDR="1F3864"; C_FG="FFFFFF"
    C_ACT="E8F5E9"; C_BEST="FFF3E0"; C_HIST="F5F5F5"
    THIN=Side(style="thin",color="DDDDDD"); BRD=Border(left=THIN,right=THIN,top=THIN,bottom=THIN)

    fc_set=set(forecast_years)
    mape_s=f"  (walk-fwd MAPE {mape_dict[best_model]:.1f}%)" if best_model in mape_dict else ""
    best_label=f"{best_model} — Stage 1 Forecast{mape_s}"

    act_lvl_bn={y:vv(to_bn(float(target_series.loc[y])),3)
                 if y in target_series.index and pd.notna(target_series.loc[y]) else None
                for y in all_years}
    best_lvl_bn={y:(None if y<=train_end else vv(to_bn(stage1_lvl[best_model].get(y)),3))
                 for y in all_years}
    best_pct={y:(None if y<=train_end else vv(stage1_pct[best_model].get(y),2))
              for y in all_years}

    # Build YoY% actuals
    act_pct_bn={}
    for y in all_years:
        l=act_lvl_bn.get(y); p=act_lvl_bn.get(y-1)
        act_pct_bn[y]=round((l/p-1)*100,2) if l and p and p!=0 else None

    row_defs=[
        (f"Euroconstruct Actuals — Level ({currency})", C_ACT, act_lvl_bn, "0.000"),
        (f"Euroconstruct Actuals — YoY%",               C_ACT, act_pct_bn, "0.00"),
        (best_label+f" — Level ({currency})",           C_BEST, best_lvl_bn,"0.000"),
        (best_label+f" — YoY%",                         C_BEST, best_pct,   "0.00"),
    ]

    wb=Workbook(); ws=wb.active
    ws.title="Residential Forecast"; ws.sheet_view.showGridLines=False

    hdr=["Metric"]+[str(y) for y in all_years]
    ws.append(hdr)
    for ci,v in enumerate(hdr,1):
        c=ws.cell(1,ci,v); c.fill=fill(C_HDR); c.font=font(bold=True,color=C_FG)
        c.alignment=align("left") if ci==1 else align(); c.border=BRD

    THIN2=Side(style="medium",color="888888")
    for ri,(label,bg,data,nfmt) in enumerate(row_defs,2):
        ws.append([label]+[data.get(y) for y in all_years])
        is_act=(bg==C_ACT)
        brd=Border(left=BRD.left,right=BRD.right,top=BRD.top,
                   bottom=Side(style="medium",color="888888") if ri%2==1 else BRD.bottom)
        for ci in range(1,len(all_years)+2):
            c=ws.cell(ri,ci)
            c.border=BRD
            if ci==1:
                c.fill=fill(bg); c.font=font(bold=True); c.alignment=align("left")
            else:
                yr=all_years[ci-2]
                if is_act: c.fill=fill(C_ACT)
                elif yr<=train_end: c.fill=fill(C_HIST)
                else: c.fill=fill(bg)
                c.font=font(bold=(not is_act and yr in fc_set))
                c.alignment=align()
                if c.value is not None: c.number_format=nfmt

    ws.freeze_panes="B2"
    ws.column_dimensions["A"].width=max(42,len(best_label)+4)
    for ci in range(2,len(all_years)+2):
        ws.column_dimensions[get_column_letter(ci)].width=7.5

    # Notes sheet
    ws_n=wb.create_sheet("Notes")
    ws_n.column_dimensions["A"].width=26; ws_n.column_dimensions["B"].width=70
    notes=[("Country",country),
           ("Level unit",f"{currency}  (source ÷ {int(MILLION_TO_BN)})"),
           ("Best model",best_model),
           ("Selection basis",f"Walk-forward MAPE {VALIDATE_FROM}–{train_end}"),
           ("Level formula",f"level({currency}) = actual_level(yr-1)×(1+YoY%/100)÷1000"),
           ("Training period",f"{hist_clean.index.min()}–{train_end}"),
           ("Forecast period",f"{train_end+1}–{test_end}"),
           ("MANDATORY KPIs","interest_rate_chg, housing_permits_yoy"),
           ("AR lag term","res_yoy_lag1"),
           ("COVID weight",f"{COVID_WEIGHT}x for {sorted(COVID_YEARS)}"),
           ("CAGR info",", ".join([f"{m}:{cagr_dict[m]:+.2f}%/yr"
                                   for m in [best_model] if m in cagr_dict]))]
    ws_n.append(["Field","Value"])
    ws_n.cell(1,1).font=ws_n.cell(1,2).font=Font(name="Arial",bold=True,size=9)
    for f_,v in notes: ws_n.append([f_,v])
    for row in ws_n.iter_rows():
        for c in row:
            c.font=Font(name="Arial",size=9)
            c.alignment=Alignment(horizontal="left",vertical="center",wrap_text=True)
            ws_n.row_dimensions[c.row].height=16

    path=os.path.join(out_dir,f"{country.replace(' ','_')}_resi_forecast.xlsx")
    wb.save(path); print(f"  ✓ Final Excel      : {path}")


# =============================================================================
# CHARTS  — full + zoom, ALL models (styled like industrial charts)
# =============================================================================

def _export_charts(country,hist_clean,target_pct,stage1_pct,stage1_lvl,
                   best_model,mape_dict,cagr_dict,model_names,
                   train_end,test_end,forecast_years,out_dir,currency):

    act_color="#1A1A1A"
    hist_pct_s=hist_clean[target_pct].dropna()

    all_fc_vals=[v for m in model_names for v in stage1_pct[m].values() if pd.notna(v)]
    all_vals=list(hist_pct_s.values)+all_fc_vals
    ymin=(min(all_vals)-2) if all_vals else -10
    ymax=(max(all_vals)+2) if all_vals else 10

    last_yr_h=hist_pct_s.index[-1]; last_vl_h=hist_pct_s.iloc[-1]

    # ── Chart A — full 2007→TEST_END, ALL models ──────────────────────────────
    fig_a,ax_a=plt.subplots(figsize=(14,4.5))
    _clean_axes(ax_a,fig_a)
    ax_a.plot(hist_pct_s.index,hist_pct_s.values,color=act_color,linewidth=2.0,
              label="Euroconstruct actuals YoY%",zorder=10,solid_capstyle="round")
    ax_a.axhline(0,color="#BBBBBB",linewidth=0.7,linestyle="--",zorder=1)
    for m in model_names:
        fc_y=sorted([yr for yr in forecast_years if pd.notna(stage1_pct[m].get(yr))])
        fc_v=[stage1_pct[m][yr] for yr in fc_y]
        if not fc_y: continue
        lw=2.2 if m==best_model else 1.2; alpha=1. if m==best_model else 0.65
        ls="-" if m==best_model else "--"
        mape_s=f"  MAPE {mape_dict[m]:.1f}%" if m in mape_dict else ""
        star=" ★" if m==best_model else ""
        ax_a.plot([last_yr_h]+fc_y,[last_vl_h]+fc_v,
                  color=MODEL_LINE_COLORS.get(m,"#888888"),
                  linewidth=lw,alpha=alpha,linestyle=ls,
                  label=f"{m}{mape_s}{star}",
                  zorder=8 if m==best_model else 6,solid_capstyle="round")
    ax_a.axvline(x=train_end,color="#BBBBBB",linestyle="--",linewidth=0.8,zorder=2)
    ax_a.text(train_end+0.15,ymax*0.97,"Forecast",fontsize=8,color="#888888",va="top")
    ax_a.set_ylim(ymin,ymax)
    ax_a.set_title(f"Total Residential Construction — YoY% Growth — {country}  (all models)",
                   fontsize=9.5,loc="left",pad=6,color="#333333")
    ax_a.set_xticks(sorted(set(list(hist_pct_s.index)+forecast_years)))
    ax_a.tick_params(axis="x",rotation=45,labelsize=7)
    ax_a.legend(fontsize=7,loc="upper left",ncol=2,frameon=True,
                framealpha=0.92,edgecolor="#DDDDDD")
    plt.tight_layout()
    fa=os.path.join(out_dir,f"{country.replace(' ','_')}_yoy_full.png")
    fig_a.savefig(fa,dpi=150,bbox_inches="tight"); plt.close(fig_a)
    print(f"\n  ✓ Chart A (full, all models)  : {fa}")

    # ── Chart B — zoom: last 3 actuals + all models + value labels + CAGR ─────
    zoom_start=train_end-3
    hist_zoom=hist_pct_s[hist_pct_s.index>=zoom_start]
    all_zoom_vals=list(hist_zoom.values)+all_fc_vals
    valid_zoom=[v for v in all_zoom_vals if pd.notna(v)]
    yz_min=(min(valid_zoom)-1.5) if valid_zoom else -5
    yz_max=(max(valid_zoom)+4.0) if valid_zoom else 12

    fig_b,ax_b=plt.subplots(figsize=(10,5.5))
    _clean_axes(ax_b,fig_b)
    ax_b.plot(hist_zoom.index,hist_zoom.values,color=act_color,linewidth=2.2,
              label="Euroconstruct actuals YoY%",zorder=10,solid_capstyle="round")
    ax_b.axhline(0,color="#BBBBBB",linewidth=0.7,linestyle="--",zorder=1)

    last_yr_z=hist_zoom.index[-1]; last_vl_z=hist_zoom.iloc[-1]
    offsets=[1.8,-1.8,2.8,-2.8,0.9,-0.9,3.5]

    # Sort models for annotation: best first so it's least crowded
    sorted_models=[best_model]+[m for m in model_names if m!=best_model]
    placed=[]

    for mi,m in enumerate(sorted_models):
        fc_y=sorted([yr for yr in forecast_years if pd.notna(stage1_pct[m].get(yr))])
        fc_v=[stage1_pct[m][yr] for yr in fc_y]
        if not fc_y: continue
        lw=2.3 if m==best_model else 1.4; alpha=1. if m==best_model else 0.70
        ls="-" if m==best_model else "--"
        mape_s=f"  MAPE {mape_dict[m]:.1f}%" if m in mape_dict else ""
        star=" ★" if m==best_model else ""
        color=MODEL_LINE_COLORS.get(m,"#888888")
        ax_b.plot([last_yr_z]+fc_y,[last_vl_z]+fc_v,color=color,
                  linewidth=lw,alpha=alpha,linestyle=ls,
                  label=f"{m}{mape_s}{star}",
                  zorder=8 if m==best_model else 6,solid_capstyle="round")

        # Annotate last-year value + CAGR for each model
        last_yr_fc=fc_y[-1]; last_val=fc_v[-1]
        if pd.notna(last_val):
            off=offsets[mi%len(offsets)]
            # Avoid overlap
            y_t=float(last_val)+off
            for p in placed:
                if abs(y_t-p)<0.9: y_t=p-1.1
            placed.append(y_t)
            cg=cagr_dict.get(m,np.nan)
            cg_s=f"  {cg:+.1f}%/yr" if pd.notna(cg) else ""
            lbl_text=f"{m}: {last_val:+.1f}%{cg_s}"
            ax_b.annotate(lbl_text,
                          xy=(last_yr_fc,last_val),
                          xytext=(last_yr_fc+0.08,y_t),
                          fontsize=7 if m==best_model else 6.5,
                          color=color,ha="left",alpha=alpha,va="center",
                          arrowprops=dict(arrowstyle="-",color=color,alpha=0.4,lw=0.7))

    ax_b.axvline(x=train_end,color="#BBBBBB",linestyle="--",linewidth=0.8,zorder=2)
    ax_b.set_ylim(yz_min,yz_max)
    ax_b.set_title(f"Total Residential Construction — {country}  (Forecast zoom, all models)",
                   fontsize=9.5,loc="left",pad=6,color="#333333")
    ax_b.set_xticks(list(hist_zoom.index)+forecast_years)
    ax_b.tick_params(axis="x",rotation=0,labelsize=9)
    ax_b.legend(fontsize=7,loc="upper left",ncol=2,frameon=True,
                framealpha=0.92,edgecolor="#DDDDDD")
    plt.tight_layout()
    fb=os.path.join(out_dir,
       f"{country.replace(' ','_')}_yoy_zoom_{train_end+1}_{test_end}.png")
    fig_b.savefig(fb,dpi=150,bbox_inches="tight"); plt.close(fig_b)
    print(f"  ✓ Chart B (zoom, all models) : {fb}")


# =============================================================================
# CROSS-COUNTRY COMPARISON
# =============================================================================

def export_comparison(out_dir=None):
    if not _COUNTRY_RESULTS:
        print("  No results — run run_forecast() first."); return
    countries=list(_COUNTRY_RESULTS.keys())
    if out_dir is None: out_dir=_COUNTRY_RESULTS[countries[0]]["out_dir"]
    print(f"\n{'='*65}\n  CROSS-COUNTRY COMPARISON  ({', '.join(countries)})\n{'='*65}")
    _export_comparison_excel(countries,out_dir)
    _export_comparison_chart(countries,out_dir)


def _export_comparison_excel(countries,out_dir):
    BORDER,THICK_B,fill,font,align,vv=_xl_styles()
    C_HDR="1F3864"; C_FG="FFFFFF"; C_ACT="E8F5E9"
    C_BEST="FFF3E0"; C_SEC="C00000"; C_SEC_FG="FFFFFF"
    THIN=Side(style="thin",color="DDDDDD"); BRD=Border(left=THIN,right=THIN,top=THIN,bottom=THIN)
    clr_map={"Germany":"DDEEFF","France":"E8F5E9",
              "Italy":"FFF9C4","United Kingdom":"EDE7F6"}

    all_yrs_u=sorted(set(yr for co in countries for yr in _COUNTRY_RESULTS[co]["all_years"]))
    wb=Workbook(); wb.remove(wb.active)

    # ── All Countries summary sheet (first) ───────────────────────────────────
    ws_all=wb.create_sheet(title="All Countries")
    ws_all.sheet_view.showGridLines=False
    ws_all.cell(1,1,"Country / Metric").fill=fill(C_HDR)
    ws_all.cell(1,1).font=font(bold=True,color=C_FG)
    ws_all.cell(1,1).alignment=align("left"); ws_all.cell(1,1).border=BRD
    for ci,yr in enumerate(all_yrs_u,2):
        c=ws_all.cell(1,ci,yr); c.fill=fill(C_HDR); c.font=font(bold=True,color=C_FG)
        c.alignment=align(); c.border=BRD; c.number_format="0"

    ri=2
    for country in countries:
        r=_COUNTRY_RESULTS[country]; ts=r["target_series"]
        fc_set_c=set(r["forecast_years"]); best=r["best_model"]
        mape_v=r["mape_dict"].get(best,np.nan)
        s1_pct=r["stage1_pct"]; s1_lvl=r["stage1_lvl"]
        cur="£bn" if country=="United Kingdom" else "€bn"
        bg=clr_map.get(country,"FFFFFF")
        # Section header
        c=ws_all.cell(ri,1,f"{country} — {best} ★  (MAPE {mape_v:.1f}%)")
        c.fill=fill(C_SEC); c.font=font(bold=True,color=C_SEC_FG)
        c.alignment=align("left"); c.border=BRD
        for ci in range(2,len(all_yrs_u)+2):
            ws_all.cell(ri,ci).fill=fill(C_SEC); ws_all.cell(ri,ci).border=BRD
        ri+=1
        # Actual level (bn)
        ws_all.cell(ri,1,f"  Actual Level ({cur})").fill=fill(bg)
        ws_all.cell(ri,1).font=font(bold=True); ws_all.cell(ri,1).alignment=align("left")
        ws_all.cell(ri,1).border=BRD
        for ci,yr in enumerate(all_yrs_u,2):
            v_bn=to_bn(float(ts.loc[yr])) if yr in ts.index and pd.notna(ts.loc[yr]) else None
            c=ws_all.cell(ri,ci,vv(v_bn,3)); c.fill=fill(bg)
            c.font=font(); c.alignment=align(); c.number_format="0.000"; c.border=BRD
        ri+=1
        # Each model YoY% + Level
        for m in r["model_names"]:
            mbg=C_BEST if m==best else MODEL_COLORS_HEX.get(m,"FFFFFF")
            mape_s=f" (MAPE {r['mape_dict'][m]:.1f}%)" if m in r["mape_dict"] else ""
            star=" ★" if m==best else ""
            for row_type in ["YoY%","Level"]:
                lbl=f"  {m} {row_type}{mape_s}{star}"
                is_last=(row_type=="Level")
                ws_all.cell(ri,1,lbl).fill=fill(mbg)
                ws_all.cell(ri,1).font=font(bold=(m==best))
                ws_all.cell(ri,1).alignment=align("left")
                ws_all.cell(ri,1).border=THICK_B if is_last else BRD
                for ci,yr in enumerate(all_yrs_u,2):
                    if row_type=="YoY%":
                        v=vv(s1_pct[m].get(yr),2) if yr in fc_set_c else None
                        nfmt="0.00"
                    else:
                        v_m=s1_lvl[m].get(yr) if yr in fc_set_c else None
                        v=vv(to_bn(v_m),3)
                        nfmt="0.000"
                    c=ws_all.cell(ri,ci,v)
                    c.fill=fill(mbg if yr in fc_set_c else bg)
                    c.font=font(bold=(m==best and yr in fc_set_c))
                    c.alignment=align(); c.number_format=nfmt
                    c.border=THICK_B if is_last else BRD
                ri+=1

    ws_all.column_dimensions["A"].width=46
    for ci in range(2,len(all_yrs_u)+2): ws_all.column_dimensions[get_column_letter(ci)].width=9
    ws_all.freeze_panes="B2"

    # ── Per-country sheets ────────────────────────────────────────────────────
    for country in countries:
        r=_COUNTRY_RESULTS[country]; ts=r["target_series"]
        fc_yrs=r["forecast_years"]; all_yrs=r["all_years"]
        s1_pct=r["stage1_pct"]; s1_lvl=r["stage1_lvl"]
        mnames=r["model_names"]; best=r["best_model"]
        mape_d=r["mape_dict"]; fc_set=set(fc_yrs)
        cur="£bn" if country=="United Kingdom" else "€bn"
        bg=clr_map.get(country,"FFFFFF")
        ws=wb.create_sheet(title=country[:28]); ws.sheet_view.showGridLines=False
        ws.cell(1,1,f"{country}  [{cur}]").fill=fill(C_HDR)
        ws.cell(1,1).font=font(bold=True,color=C_FG)
        ws.cell(1,1).alignment=align("left"); ws.cell(1,1).border=BRD
        for ci,yr in enumerate(all_yrs,2):
            c=ws.cell(1,ci,yr); c.fill=fill(C_HDR); c.font=font(bold=True,color=C_FG)
            c.alignment=align(); c.border=BRD; c.number_format="0"
        ri=2
        act_bn={yr:vv(to_bn(float(ts.loc[yr])),3)
                if yr in ts.index and pd.notna(ts.loc[yr]) else None for yr in all_yrs}
        for row_type,data,nfmt in [
            (f"Actual Level ({cur})",act_bn,"0.000"),
            (f"Actual YoY%",
             {yr:(round((act_bn[yr]/act_bn.get(yr-1)-1)*100,2)
                  if act_bn.get(yr) and act_bn.get(yr-1) else None) for yr in all_yrs},
             "0.00"),
        ]:
            ws.cell(ri,1,row_type).fill=fill(C_ACT)
            ws.cell(ri,1).font=font(bold=True); ws.cell(ri,1).alignment=align("left")
            ws.cell(ri,1).border=BRD
            for ci,yr in enumerate(all_yrs,2):
                c=ws.cell(ri,ci,data.get(yr)); c.fill=fill(C_ACT)
                c.font=font(); c.alignment=align(); c.number_format=nfmt; c.border=BRD
            ri+=1
        for m in mnames:
            mbg=C_BEST if m==best else MODEL_COLORS_HEX.get(m,"FFFFFF")
            mape_s=f" (MAPE {mape_d[m]:.1f}%)" if m in mape_d else ""
            star=" ★" if m==best else ""
            for row_type in ["YoY%","Level"]:
                lbl=f"{m} {row_type}{mape_s}{star}"
                ws.cell(ri,1,lbl).fill=fill(mbg)
                ws.cell(ri,1).font=font(bold=(m==best))
                ws.cell(ri,1).alignment=align("left"); ws.cell(ri,1).border=BRD
                for ci,yr in enumerate(all_yrs,2):
                    if row_type=="YoY%":
                        v=vv(s1_pct[m].get(yr),2) if yr in fc_set else None; nf="0.00"
                    else:
                        v=vv(to_bn(s1_lvl[m].get(yr)),3) if yr in fc_set else None; nf="0.000"
                    c=ws.cell(ri,ci,v)
                    c.fill=fill(mbg if yr in fc_set else "FFFFFF")
                    c.font=font(bold=(m==best and yr in fc_set))
                    c.alignment=align(); c.number_format=nf; c.border=BRD
                ri+=1
        ws.column_dimensions["A"].width=42
        for ci in range(2,len(all_yrs)+2): ws.column_dimensions[get_column_letter(ci)].width=9
        ws.freeze_panes="B2"

    fname=os.path.join(out_dir,"comparison_residential_forecast.xlsx")
    wb.save(fname); print(f"\n  ✓ Comparison Excel : {fname}")


def _export_comparison_chart(countries,out_dir):
    n=len(countries); fig,axes=plt.subplots(1,n,figsize=(6*n,5),sharey=False)
    if n==1: axes=[axes]
    act_color="#1A1A1A"
    for ax,country in zip(axes,countries):
        r=_COUNTRY_RESULTS[country]; hist_c=r["hist_clean"]
        fc_yrs=r["forecast_years"]; s1_pct=r["stage1_pct"]
        mnames=r["model_names"]; best=r["best_model"]
        mape_d=r["mape_dict"]; train_end=r["train_end"]
        pct_col=[c for c in hist_c.columns if "yoy_pct" in c]
        if not pct_col: continue
        hist_pct=hist_c[pct_col[0]].dropna()
        _clean_axes(ax,fig)
        zoom_start=train_end-4
        hist_zoom=hist_pct[hist_pct.index>=zoom_start]
        ax.plot(hist_zoom.index,hist_zoom.values,color=act_color,linewidth=2.0,
                label="Actuals",zorder=10,solid_capstyle="round")
        ax.axhline(0,color="#BBBBBB",linewidth=0.7,linestyle="--")
        ax.axvline(x=train_end,color="#BBBBBB",linestyle="--",linewidth=0.8)
        last_yr_z=hist_zoom.index[-1]; last_vl_z=hist_zoom.iloc[-1]
        for mi,m in enumerate(mnames):
            fc_y=sorted([yr for yr in fc_yrs if pd.notna(s1_pct[m].get(yr))])
            fc_v=[s1_pct[m][yr] for yr in fc_y]
            if not fc_y: continue
            lw=2.2 if m==best else 1.2; alpha=1. if m==best else 0.60
            ls="-" if m==best else "--"
            mape_s=f" {mape_d[m]:.1f}%" if m in mape_d else ""
            star="★" if m==best else ""
            color=MODEL_LINE_COLORS.get(m,"#888888")
            ax.plot([last_yr_z]+fc_y,[last_vl_z]+fc_v,color=color,
                    linewidth=lw,alpha=alpha,linestyle=ls,
                    label=f"{m}{mape_s}{star}",solid_capstyle="round")
            if fc_y and pd.notna(fc_v[-1]):
                ax.annotate(f"{fc_v[-1]:+.1f}%",xy=(fc_y[-1],fc_v[-1]),
                            xytext=(fc_y[-1]+0.05,fc_v[-1]),
                            fontsize=6.5,color=color,alpha=alpha,ha="left")
        ax.set_title(country,fontsize=9,loc="left",color="#333333",pad=4)
        ax.set_xticks(list(hist_zoom.index)+fc_yrs)
        ax.tick_params(axis="x",rotation=45,labelsize=7)
        ax.legend(fontsize=6,loc="best",frameon=True,framealpha=0.9,edgecolor="#DDDDDD")
    fig.suptitle("Residential Construction — YoY% Growth  (All Countries, All Models)",
                 fontsize=10,y=1.01,color="#333333")
    plt.tight_layout()
    fc_=os.path.join(out_dir,"comparison_residential_allcountries.png")
    fig.savefig(fc_,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ Comparison chart : {fc_}")