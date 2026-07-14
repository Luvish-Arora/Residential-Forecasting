# =============================================================================
# preprocessing.py — Residential Market Forecasting  (KPI Selection)
# =============================================================================
# Values from Euroconstruct are in MILLION (source units).
# forecastmodel.py converts to BILLION for all outputs.
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

INTEREST_FILE = os.path.join(DATA_DIR, "Interest_rate.xls")
PERMITS_FILE  = os.path.join(DATA_DIR, "Housing_permits.xls")
GDP_FILE      = os.path.join(DATA_DIR, "GDP.xls")
HPI_FILE      = os.path.join(DATA_DIR, "House_price_index.xls")
HTR_FILE      = os.path.join(DATA_DIR, "House_to_rent_ratio.xls")
LCI_FILE      = os.path.join(DATA_DIR, "Labor_cost_index.xls")
GROSS_FILE    = os.path.join(DATA_DIR, "Gross_income.xls")
DISP_FILE     = os.path.join(DATA_DIR, "Disposable_income_per_household.xls")
POP_FILE      = os.path.join(DATA_DIR, "Population.xls")
HSIZE_FILE    = os.path.join(DATA_DIR, "Household_Size.xls")
TARGET_FILE   = os.path.join(DATA_DIR, "Euroconstruct_data.xlsx")

START_YEAR = 2007
TRAIN_END  = 2025
TEST_END   = 2027

MANDATORY   = ["interest_rate_chg", "housing_permits_yoy"]
AR_LAG_TERM = "res_yoy_lag1"

ALL_OPTIONAL = [
    "gdp_yoy", "house_price_index_yoy", "house_to_rent_ratio_yoy",
    "labor_cost_yoy", "gross_income_yoy", "disposable_income_yoy",
    "population_yoy", "household_size_chg",
]

SIGN_MAP = {
    "interest_rate_chg":            "negative",
    "housing_permits_yoy":          "positive",
    "res_yoy_lag1":                 "positive",
    "gdp_yoy":                      "positive",
    "house_price_index_yoy":        "positive",
    "house_to_rent_ratio_yoy":      "positive",
    "labor_cost_yoy":               "negative",
    "gross_income_yoy":             "positive",
    "disposable_income_yoy":        "positive",
    "population_yoy":               "positive",
    "household_size_chg":           "negative",
    "gdp_yoy_lag1":                 "positive",
    "house_price_index_yoy_lag1":   "positive",
    "house_to_rent_ratio_yoy_lag1": "positive",
    "labor_cost_yoy_lag1":          "negative",
    "gross_income_yoy_lag1":        "positive",
    "disposable_income_yoy_lag1":   "positive",
    "population_yoy_lag1":          "positive",
    "household_size_chg_lag1":      "negative",
}

COUNTRY_KPI_CONFIG = {
    "Germany": {
        "priority":  ["house_price_index_yoy","gdp_yoy","disposable_income_yoy","gross_income_yoy"],
        "secondary": ["labor_cost_yoy","population_yoy","house_to_rent_ratio_yoy"],
        "exclude":   ["household_size_chg"],
    },
    "France": {
        "priority":  ["house_price_index_yoy","house_to_rent_ratio_yoy","gdp_yoy","disposable_income_yoy"],
        "secondary": ["gross_income_yoy","labor_cost_yoy","population_yoy"],
        "exclude":   ["household_size_chg"],
    },
    "Italy": {
        "priority":  ["gdp_yoy","disposable_income_yoy","gross_income_yoy","house_price_index_yoy"],
        "secondary": ["labor_cost_yoy","house_to_rent_ratio_yoy","population_yoy"],
        "exclude":   ["household_size_chg"],
    },
    "United Kingdom": {
        "priority":  ["house_price_index_yoy","gdp_yoy","gross_income_yoy","disposable_income_yoy"],
        "secondary": ["house_to_rent_ratio_yoy","population_yoy","labor_cost_yoy"],
        "exclude":   ["household_size_chg"],
    },
}

PRIORITY_RESCUE_THRESHOLD  = 0.12
SECONDARY_RESCUE_THRESHOLD = 0.18
COLLINEARITY_RESCUE_THRESH = 0.72
ZERO_THRESHOLD             = 0.0001
COLLINEARITY_FLAG_THRESH   = 0.75

TARGET_PCT = "residential_yoy_pct"
TARGET_LVL = "residential_value"   # stored in MILLION; converted to bn in forecastmodel

HALFLIFE_BY_COUNTRY = {"Germany":15,"France":15,"Italy":15,"United Kingdom":15}
COVID_YEARS  = {2020,2021}
COVID_WEIGHT = 0.30


def read_euroconstruct_target(country):
    if not os.path.exists(TARGET_FILE):
        raise FileNotFoundError(f"Euroconstruct_data.xlsx not found at:\n  {TARGET_FILE}")
    df = pd.read_excel(TARGET_FILE, header=None, engine="openpyxl")
    header_row = None
    for i in range(min(10,len(df))):
        row_vals = pd.to_numeric(df.iloc[i], errors="coerce").dropna()
        year_like = row_vals[(row_vals>=2000)&(row_vals<=2035)]
        if len(year_like)>=5:
            header_row=i; break
    if header_row is None:
        raise ValueError("Could not find year header row in Euroconstruct_data.xlsx")
    year_row = df.iloc[header_row]; year_cols={}
    for col_idx,val in enumerate(year_row):
        try:
            yr=int(float(val))
            if 2000<=yr<=2035: year_cols[col_idx]=yr
        except: pass
    data_rows=df.iloc[header_row+1:].reset_index(drop=True)
    norm=lambda v:str(v).strip().lower(); tgt=norm(country)
    country_col=None; match=pd.DataFrame()
    for try_col in range(min(6,df.shape[1])):
        geo=data_rows.iloc[:,try_col].map(norm)
        exact=data_rows.loc[geo==tgt]
        if len(exact)>0: country_col=try_col; match=exact; break
        partial=data_rows.loc[geo.str.startswith(tgt[:6],na=False)]
        if len(partial)>0: country_col=try_col; match=partial; break
    if len(match)==0:
        available=[]
        for c in range(min(6,df.shape[1])):
            available+=[v for v in data_rows.iloc[:,c].dropna().tolist() if isinstance(v,str)]
        raise ValueError(f"Country '{country}' not found. Available: {available}")
    row=match.iloc[0]; records={}
    for col_idx,yr in year_cols.items():
        val=row.iloc[col_idx] if col_idx<len(row) else np.nan
        try: records[yr]=float(val)
        except: records[yr]=np.nan
    s=pd.Series(records,name=TARGET_LVL).sort_index()
    s=s[s.index.notna()]; s.index=s.index.astype(int)
    valid=s.dropna()
    if len(valid)==0: raise ValueError(f"No valid values for '{country}' in Euroconstruct file.")
    print(f"  Target loaded: {country}  range={valid.min():.0f}–{valid.max():.0f} million")
    return s

def _norm(v): return str(v).strip().lower()

def _find_header_row(df, keyword="geography"):
    for i in range(min(15,len(df))):
        if keyword in _norm(str(df.iloc[i,0])): return i
    raise ValueError(f"Header row with '{keyword}' not found.")

def _match_country(df_data, country):
    geo=df_data.iloc[:,0].astype(str).map(_norm); tgt=_norm(country)
    exact=df_data.loc[geo==tgt]
    return exact if len(exact)>0 else df_data.loc[geo.str.contains(tgt,na=False)]

def _read_xls(filepath):
    import xlrd
    wb=xlrd.open_workbook(filepath); ws=wb.sheet_by_index(0); rows=[]
    for r in range(ws.nrows):
        row=[]
        for c in range(ws.ncols):
            cell=ws.cell(r,c)
            if cell.ctype==0: row.append(None)
            elif cell.ctype==1: row.append(cell.value)
            elif cell.ctype==2:
                v=cell.value; row.append(int(v) if v==int(v) else v)
            elif cell.ctype==3:
                row.append(xlrd.xldate_as_datetime(v,wb.datemode).year)
            else: row.append(cell.value)
        rows.append(row)
    max_len=max(len(r) for r in rows) if rows else 0
    rows=[r+[None]*(max_len-len(r)) for r in rows]
    return pd.DataFrame(rows)

def _parse_annual(df, hr, country, col):
    years=df.iloc[hr,5:].tolist(); df_data=df.iloc[hr+1:].reset_index(drop=True)
    matches=_match_country(df_data,country)
    if len(matches)==0:
        print(f"  WARNING: '{country}' not found for '{col}'")
        return pd.Series(np.nan,index=range(START_YEAR-1,TEST_END+1),name=col)
    row=matches.iloc[0,5:].tolist(); s=pd.Series(row,index=years,name=col)
    s=pd.to_numeric(s,errors="coerce"); s.index=pd.to_numeric(s.index,errors="coerce")
    s=s[s.index.notna()].copy(); s.index=s.index.astype(int)
    return s.sort_index()

def _parse_quarterly_to_annual(df, hr, country, col, agg="mean"):
    qtrs=df.iloc[hr,5:].tolist(); df_data=df.iloc[hr+1:].reset_index(drop=True)
    matches=_match_country(df_data,country)
    if len(matches)==0:
        print(f"  WARNING: '{country}' not found for '{col}'")
        return pd.Series(np.nan,index=range(START_YEAR-1,TEST_END+1),name=col)
    row=matches.iloc[0,5:].tolist(); years_q=[]
    for q in qtrs:
        qs=str(q).strip(); years_q.append(int(qs.split()[-1]) if " " in qs else np.nan)
    s=pd.Series(pd.to_numeric(row,errors="coerce").tolist(),index=years_q)
    s=s[~pd.isna(s.index)].copy(); s.index=[int(y) for y in s.index]
    annual=s.groupby(s.index).mean() if agg=="mean" else s.groupby(s.index).sum()
    return annual.rename(col).sort_index()

def read_interest_rate(c):
    df=_read_xls(INTEREST_FILE); return _parse_quarterly_to_annual(df,_find_header_row(df),c,"interest_rate_raw","mean")
def read_housing_permits(c):
    df=_read_xls(PERMITS_FILE); return _parse_quarterly_to_annual(df,_find_header_row(df),c,"housing_permits_raw","sum")
def read_gdp(c):
    df=_read_xls(GDP_FILE); return _parse_quarterly_to_annual(df,_find_header_row(df),c,"gdp_raw","mean")
def read_hpi(c):
    df=_read_xls(HPI_FILE); return _parse_quarterly_to_annual(df,_find_header_row(df),c,"hpi_raw","mean")
def read_labor_cost(c):
    df=_read_xls(LCI_FILE); return _parse_quarterly_to_annual(df,_find_header_row(df),c,"lci_raw","mean")
def read_house_to_rent(c):
    df=_read_xls(HTR_FILE); return _parse_annual(df,_find_header_row(df),c,"htr_raw")
def read_gross_income(c):
    df=_read_xls(GROSS_FILE); return _parse_annual(df,_find_header_row(df),c,"gross_income_raw")
def read_disposable_income(c):
    df=_read_xls(DISP_FILE); return _parse_annual(df,_find_header_row(df),c,"disposable_income_raw")
def read_population(c):
    df=_read_xls(POP_FILE); return _parse_annual(df,_find_header_row(df),c,"population_raw")
def read_household_size(c):
    df=_read_xls(HSIZE_FILE); return _parse_annual(df,_find_header_row(df),c,"hsize_raw")

def safe_pct(s):
    return s.pct_change(fill_method=None).mul(100).replace([np.inf,-np.inf],np.nan).round(3)
def safe_diff(s,periods=1): return s.diff(periods).round(4)
def safe_merge(base,series,name):
    return base.merge(series.rename(name),left_index=True,right_index=True,how="left")

def impute_trend(series, target_years):
    s=series.dropna().sort_index()
    if len(s)<4:
        last=s.iloc[-1] if len(s)>0 else 0.
        fill=pd.Series(last,index=target_years)
        return pd.concat([series,fill[~fill.index.isin(series.index)]]).sort_index()
    missing=[y for y in target_years if pd.isna(series.reindex([y]).iloc[0])]
    if not missing: return series
    n_fc=max(missing)-int(s.index.max())
    if n_fc<=0: return series
    try:
        fc=ExponentialSmoothing(s.values,trend="add",damped_trend=True,
            initialization_method="estimated").fit(optimized=True).forecast(n_fc)
    except Exception as e:
        print(f"  HW fallback ({e})")
        coef=np.polyfit(np.arange(len(s)),s.values,1)
        fc=np.polyval(coef,np.arange(len(s),len(s)+n_fc))
    fcast_s=pd.Series(fc,index=range(int(s.index.max())+1,int(s.index.max())+n_fc+1))
    out=pd.concat([series,fcast_s])
    return out[~out.index.duplicated(keep="first")].sort_index()

def compute_lag_correlations(train_df,features,target):
    records,lag_choice=[],{}
    for f in features:
        if f not in train_df.columns: continue
        tmp_c=train_df[[f,target]].dropna()
        corr_c=tmp_c[f].corr(tmp_c[target]) if len(tmp_c)>=5 else np.nan
        tmp_l=train_df[[f,target]].copy(); tmp_l[f]=tmp_l[f].shift(1); tmp_l=tmp_l.dropna()
        corr_l1=tmp_l[f].corr(tmp_l[target]) if len(tmp_l)>=5 else np.nan
        abs_c=abs(corr_c) if pd.notna(corr_c) else 0.
        abs_l1=abs(corr_l1) if pd.notna(corr_l1) else 0.
        best_lag,best_corr=(1,corr_l1) if abs_l1>abs_c else (0,corr_c)
        lag_choice[f]=best_lag
        records.append({"Feature":f,"Corr_contemp":round(corr_c,3) if pd.notna(corr_c) else np.nan,
            "Corr_lag1":round(corr_l1,3) if pd.notna(corr_l1) else np.nan,
            "Best_lag":best_lag,"Best_corr":round(best_corr,3) if pd.notna(best_corr) else np.nan,
            "AbsBestCorr":round(abs(best_corr),3) if pd.notna(best_corr) else 0.})
    if not records: return pd.DataFrame(),{}
    return pd.DataFrame(records).sort_values("AbsBestCorr",ascending=False),lag_choice

def apply_lags(df,lag_choice):
    df_out,lag_col_map=df.copy(),{}
    for f,lag in lag_choice.items():
        if f not in df_out.columns: continue
        if lag==1:
            nc=f"{f}_lag1"; df_out[nc]=df_out[f].shift(1); lag_col_map[f]=nc
        else: lag_col_map[f]=f
    return df_out,lag_col_map

def pairwise_collinearity_report(train_df,features,n_rows):
    avail=[f for f in features if f in train_df.columns]
    if len(avail)<2: return
    cm=train_df[avail].corr(); high=[]
    for i,f1 in enumerate(avail):
        for f2 in avail[i+1:]:
            r=float(cm.loc[f1,f2])
            if abs(r)>COLLINEARITY_FLAG_THRESH: high.append((f1,f2,r))
    print(f"\n  PAIRWISE COLLINEARITY (|r|>{COLLINEARITY_FLAG_THRESH}, n={n_rows})")
    if not high: print("  None exceed threshold.")
    else:
        for f1,f2,r in sorted(high,key=lambda x:abs(x[2]),reverse=True):
            print(f"    {f1:<38} @ {f2:<38}  r={r:+.3f}")

def _sign_ok(corr_val,fname):
    if pd.isna(corr_val): return False
    exp=SIGN_MAP.get(fname,"?")
    return (corr_val>0 if exp=="positive" else corr_val<0 if exp=="negative" else True)

def _slbl(corr_val,fname): return "v" if _sign_ok(corr_val,fname) else "x"


def run_preprocessing(COUNTRY, user_optional=None):
    print(f"\n{'='*70}")
    print(f"  RESIDENTIAL FORECASTING — KPI SELECTION")
    print(f"  Country : {COUNTRY}   |   Train {START_YEAR}–{TRAIN_END}   Forecast {TRAIN_END+1}–{TEST_END}")
    print(f"{'='*70}")

    cfg=COUNTRY_KPI_CONFIG.get(COUNTRY,{})
    priority_kpis=set(cfg.get("priority",[])); secondary_kpis=set(cfg.get("secondary",[]))
    excluded_kpis=set(cfg.get("exclude",[])); allowed=(priority_kpis|secondary_kpis)-excluded_kpis
    active_optional=[k for k in ALL_OPTIONAL if k in allowed] if cfg else list(ALL_OPTIONAL)
    if user_optional is not None:
        active_optional=[k for k in active_optional if k in set(user_optional)]

    print(f"\n  Priority : {sorted(priority_kpis-excluded_kpis)}")
    print(f"  Secondary: {sorted(secondary_kpis-excluded_kpis)}")
    print(f"  Excluded : {sorted(excluded_kpis)}")

    print(f"\n{'─'*70}\n  STEP 1 — Loading KPI files\n{'─'*70}")
    interest_raw=read_interest_rate(COUNTRY); permits_raw=read_housing_permits(COUNTRY)
    gdp_raw=read_gdp(COUNTRY); hpi_raw=read_hpi(COUNTRY); htr_raw=read_house_to_rent(COUNTRY)
    lci_raw=read_labor_cost(COUNTRY); gross_raw=read_gross_income(COUNTRY)
    disp_raw=read_disposable_income(COUNTRY); pop_raw=read_population(COUNTRY)
    hsize_raw=read_household_size(COUNTRY)
    print("  KPI files loaded.")

    print(f"\n{'─'*70}\n  STEP 2 — Loading Euroconstruct target (million)\n{'─'*70}")
    target_raw=read_euroconstruct_target(COUNTRY)

    print(f"\n{'─'*70}\n  STEP 3 — Building master dataframe\n{'─'*70}")
    all_years=list(range(START_YEAR-1,TEST_END+1))
    df=pd.DataFrame(index=all_years); df.index.name="year"
    for s,n in [(interest_raw,"interest_rate_raw"),(permits_raw,"housing_permits_raw"),
                (gdp_raw,"gdp_raw"),(hpi_raw,"hpi_raw"),(htr_raw,"htr_raw"),
                (lci_raw,"lci_raw"),(gross_raw,"gross_income_raw"),
                (disp_raw,"disposable_income_raw"),(pop_raw,"population_raw"),(hsize_raw,"hsize_raw")]:
        df=safe_merge(df,s,n)
    df["interest_rate_chg"]=safe_diff(df["interest_rate_raw"])
    df["housing_permits_yoy"]=safe_pct(df["housing_permits_raw"])
    df["gdp_yoy"]=safe_pct(df["gdp_raw"]); df["house_price_index_yoy"]=safe_pct(df["hpi_raw"])
    df["house_to_rent_ratio_yoy"]=safe_pct(df["htr_raw"]); df["labor_cost_yoy"]=safe_pct(df["lci_raw"])
    df["gross_income_yoy"]=safe_pct(df["gross_income_raw"]); df["disposable_income_yoy"]=safe_pct(df["disposable_income_raw"])
    df["population_yoy"]=safe_pct(df["population_raw"]); df["household_size_chg"]=safe_diff(df["hsize_raw"])
    print("  KPI features computed.")

    print(f"\n{'─'*70}\n  STEP 5 — Attaching target\n{'─'*70}")
    df[TARGET_LVL]=target_raw
    df.loc[df.index>TRAIN_END,TARGET_LVL]=np.nan
    df[TARGET_PCT]=safe_pct(df[TARGET_LVL]); df[AR_LAG_TERM]=df[TARGET_PCT].shift(1).round(3)
    df=df[df.index>=START_YEAR].copy()
    hist_clean=df[df[TARGET_LVL].notna()].copy()
    if hist_clean.empty: raise ValueError(f"No valid actuals for '{COUNTRY}'.")
    print(f"  Target rows: {len(hist_clean)}  ({hist_clean.index.min()}–{hist_clean.index.max()})")

    train_hist=hist_clean[hist_clean.index<=TRAIN_END].copy()
    candidate=MANDATORY+[AR_LAG_TERM]+active_optional
    available=[f for f in candidate if f in train_hist.columns and train_hist[f].notna().sum()>=5]
    mandatory_avail=[f for f in MANDATORY if f in available]
    ar_avail=[AR_LAG_TERM] if AR_LAG_TERM in available else []
    optional_avail=[f for f in active_optional if f in available]
    mandatory_dropped=[f for f in MANDATORY if f not in available]
    if mandatory_dropped:
        print(f"\n  NOTE: Mandatory KPIs dropped: {mandatory_dropped}")

    # GATE 0
    print(f"\n{'─'*70}\n  GATE 0 — SIGN PRE-SCREEN\n{'─'*70}")
    sign_pass,sign_fail=[],[]
    for f in optional_avail:
        if f not in train_hist.columns: continue
        exp=SIGN_MAP.get(f,"?")
        tmp_c=train_hist[[f,TARGET_PCT]].dropna()
        corr_c=tmp_c[f].corr(tmp_c[TARGET_PCT]) if len(tmp_c)>=5 else np.nan
        tmp_l=train_hist[[f,TARGET_PCT]].copy(); tmp_l[f]=tmp_l[f].shift(1); tmp_l=tmp_l.dropna()
        corr_l1=tmp_l[f].corr(tmp_l[TARGET_PCT]) if len(tmp_l)>=5 else np.nan
        ok_c=_sign_ok(corr_c,f); ok_l1=_sign_ok(corr_l1,f)
        both_wrong=not ok_c and not ok_l1
        decision="EXCLUDED (both lags wrong sign)" if both_wrong else "PASS"
        (sign_fail if both_wrong else sign_pass).append(f)
        c_str  = f"{corr_c:+.3f}"  if pd.notna(corr_c)  else "n/a"
        l1_str = f"{corr_l1:+.3f}" if pd.notna(corr_l1) else "n/a"
        print(f"  {f:<34} {exp:<9} r(t)={c_str}  r(t-1)={l1_str}  {decision}")
    optional_avail=sign_pass

    # GATE 1a
    print(f"\n{'─'*70}\n  GATE 1a — LAG SELECTION\n{'─'*70}")
    lag_df,lag_choice_opt=compute_lag_correlations(train_hist,optional_avail,TARGET_PCT)
    for _,r in lag_df.iterrows():
        f=r["Feature"]; bc=r["Best_corr"]; ll="contemp" if r["Best_lag"]==0 else "lag-1"
        print(f"  {f:<34} r_contemp={r['Corr_contemp']:+.3f}  r_lag1={r['Corr_lag1']:+.3f}  chosen={ll}  |r|={abs(bc):.3f}")

    lag_choice_all={**{f:0 for f in mandatory_avail},**({AR_LAG_TERM:0} if ar_avail else {}),**lag_choice_opt}
    df_lagged,lag_col_map=apply_lags(df,lag_choice_all)
    train_lagged=df_lagged.loc[train_hist.index].copy()
    mandatory_lagged=[lag_col_map.get(f,f) for f in mandatory_avail]
    ar_lagged=[lag_col_map.get(AR_LAG_TERM,AR_LAG_TERM)] if ar_avail else []
    optional_lagged=[lag_col_map.get(f,f) for f in optional_avail if lag_col_map.get(f,f) in train_lagged.columns]

    # GATE 1b
    all_cands=[c for c in mandatory_lagged+ar_lagged+optional_lagged if c in train_lagged.columns]
    model_df=train_lagged[all_cands+[TARGET_PCT]].replace([np.inf,-np.inf],np.nan).dropna()
    corr_s=model_df[all_cands+[TARGET_PCT]].corr()[TARGET_PCT].drop(TARGET_PCT)
    print(f"\n{'─'*70}\n  GATE 1b — CORRELATION TABLE\n{'─'*70}")
    for col in corr_s.index:
        cv=corr_s[col]; exp=SIGN_MAP.get(col,SIGN_MAP.get(col.replace("_lag1",""),"?"))
        ll="lag-1" if col.endswith("_lag1") else "contemp"
        print(f"  {col:<38} r={cv:+.3f}  |r|={abs(cv):.3f}  exp={exp}  {_slbl(cv,col)}  {ll}")

    # GATE 2
    print(f"\n{'─'*70}\n  GATE 2 — ElasticNetCV\n{'─'*70}")
    enet_cands=[c for c in ar_lagged+optional_lagged if c in model_df.columns]
    selected_lag_cols,zeroed_cols,wrong_sign_cols,rescued_cols=[],[],[],[]
    if enet_cands:
        opt_df=model_df[enet_cands+[TARGET_PCT]].dropna()
        if len(opt_df)<6: raise ValueError(f"Too few rows ({len(opt_df)}) for ElasticNetCV.")
        X=opt_df[enet_cands].values; y=opt_df[TARGET_PCT].values
        hl=HALFLIFE_BY_COUNTRY.get(COUNTRY,15); mx=max(opt_df.index.tolist())
        sw=np.array([(2.**(-((mx-yr)/hl)))*(COVID_WEIGHT if yr in COVID_YEARS else 1.) for yr in opt_df.index.tolist()],dtype=float)
        sw/=sw.mean()
        Xs=StandardScaler().fit_transform(X)
        enet=ElasticNetCV(l1_ratio=[0.1,0.2,0.3,0.5],cv=5,max_iter=50000,alphas=200,random_state=42)
        enet.fit(Xs,y,sample_weight=sw)
        print(f"  alpha={enet.alpha_:.4f}  l1_ratio={enet.l1_ratio_:.2f}")
        for col,coef in zip(enet_cands,enet.coef_):
            cv=corr_s.get(col,0.); exp=SIGN_MAP.get(col,SIGN_MAP.get(col.replace("_lag1",""),"?"))
            ok=((exp=="positive" and coef>0) or (exp=="negative" and coef<0) or exp=="?")
            if abs(coef)<=ZERO_THRESHOLD: status="ZEROED"; zeroed_cols.append(col)
            elif ok: status="KEPT"; selected_lag_cols.append(col)
            else: status="WRONG_SIGN"; wrong_sign_cols.append(col)
            print(f"  {col:<38} coef={coef:+.4f}  {status}")

        # GATE 3 rescue
        print(f"\n{'─'*70}\n  GATE 3 — RESCUE\n{'─'*70}")
        all_opt_cols=[c for c in enet_cands if c in model_df.columns]
        pair_corr=model_df[all_opt_cols].corr() if len(all_opt_cols)>=2 else pd.DataFrame()
        for col in list(zeroed_cols):
            cv=corr_s.get(col,np.nan); base=col.replace("_lag1","")
            ok=_sign_ok(cv,col); is_p=base in priority_kpis
            thr=PRIORITY_RESCUE_THRESHOLD if is_p else SECONDARY_RESCUE_THRESHOLD
            tier="PRIORITY" if is_p else "secondary"
            if pd.isna(cv) or not(abs(cv)>thr and ok):
                print(f"  SKIP  [{tier}] {col:<36} |r|={abs(cv):.3f} < {thr}"); continue
            blocked,note=False,""
            if not pair_corr.empty and col in pair_corr.index:
                for sel in selected_lag_cols:
                    if sel in pair_corr.columns:
                        rp=abs(pair_corr.loc[col,sel])
                        if rp>COLLINEARITY_RESCUE_THRESH: blocked=True; note=f"collinear w/ '{sel}' r={rp:.3f}"; break
            if blocked: print(f"  BLOCK [{tier}] {col:<36} {note}")
            else:
                print(f"  RESCUE [{tier}] {col:<36} r={cv:+.3f}")
                rescued_cols.append(col); selected_lag_cols.append(col)
        if not rescued_cols: print("  (no features rescued)")

    seen=set(); clean_features=[]
    for f in mandatory_lagged+ar_lagged+selected_lag_cols:
        if f not in seen: seen.add(f); clean_features.append(f)
    final_zeroed=[c for c in zeroed_cols if c not in rescued_cols]
    feature_lag_map={}
    for f in mandatory_avail+(ar_avail or [])+optional_avail:
        mapped=lag_col_map.get(f,f)
        if mapped in clean_features: feature_lag_map[f]=mapped

    long_kpis,short_kpis,drop_kpis=[],[],[]
    print(f"\n{'─'*70}\n  KPI COVERAGE\n{'─'*70}")
    for col in clean_features:
        s=train_lagged[col] if col in train_lagged.columns else pd.Series(dtype=float)
        n=int(s.notna().sum())
        if n>=12: long_kpis.append(col); flag,tier="v","LONG"
        elif n>=5: short_kpis.append(col); flag,tier="~","SHORT"
        else: drop_kpis.append(col); flag,tier="x","DROP"
        print(f"  {col:<38} n={n}  {flag} {tier}")

    pairwise_collinearity_report(
        train_lagged[[c for c in clean_features if c in train_lagged.columns]].dropna(),
        clean_features,
        len(train_lagged[[c for c in clean_features if c in train_lagged.columns]].dropna()))

    forecast_years=list(range(TRAIN_END+1,TEST_END+1))
    print(f"\n{'─'*70}\n  FORECAST COVERAGE CHECK\n{'─'*70}")
    for yr in forecast_years:
        if yr not in df_lagged.index: print(f"  WARNING: {yr} missing"); continue
        row=df_lagged.loc[[yr],clean_features]
        miss=[f for f in clean_features if pd.isna(row.loc[yr,f])]
        if miss:
            for f in miss:
                imp=impute_trend(df_lagged[f].copy(),[yr])
                if yr in imp.index and pd.notna(imp.loc[yr]):
                    df_lagged.loc[yr,f]=round(float(imp.loc[yr]),4)
            still=[f for f in clean_features if pd.isna(df_lagged.loc[yr,f])]
            print(f"  {'OK (imputed)' if not still else 'WARNING'}: {yr}  {still if still else ''}")
        else:
            print(f"  OK: {yr}")

    print(f"\n  CLEAN_FEATURES = {clean_features}")

    df_lagged=df_lagged[df_lagged.index<=TEST_END]
    hist_clean=hist_clean[hist_clean.index<=TEST_END]
    train_lagged=train_lagged[train_lagged.index<=TEST_END]

    return {"df":df_lagged,"hist_clean":hist_clean,"train_lagged":train_lagged,
            "lag_col_map":lag_col_map,"FEATURE_LAG_MAP":feature_lag_map,
            "CLEAN_FEATURES":clean_features,"LONG_KPIS":long_kpis,
            "SHORT_KPIS":short_kpis,"COUNTRY":COUNTRY,
            "TARGET_PCT":TARGET_PCT,"TARGET_LVL":TARGET_LVL,
            "TRAIN_END":TRAIN_END,"TEST_END":TEST_END,"OUT_DIR":OUT_DIR}