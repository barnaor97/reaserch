import os
"""30 m vs 10 m migration comparison. Reads only finished artifacts from both analyses."""
import json, os
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.environ.get("THESIS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F30=os.path.join(ROOT,"outputs","final_polygon_thesis")
F10=os.path.join(ROOT,"outputs","polygon_10m_candidate")
OUT=os.path.join(F10,"migration"); os.makedirs(OUT,exist_ok=True)
c30=pd.read_csv(os.path.join(F30,"table_polygon_population_comparison.csv"))
def g30(m,col): 
    r=c30[c30.measure.str.strip()==m]
    return float(r[col].iloc[0]) if len(r) else np.nan
r30=pd.read_csv(os.path.join(F30,"reference_units_final.csv")); m30=pd.read_csv(os.path.join(F30,"model_units_final.csv"))
r10=pd.read_csv(os.path.join(F10,"reference_units_final.csv")); m10=pd.read_csv(os.path.join(F10,"model_units_final.csv"))
A10=json.load(open(os.path.join(F10,"AGREEMENT_10m.json")))
rows=[
 ("reference raw area (ha)",            g30("burned footprint before the minimum-area rule","reference"), A10["reference_raw_ha"]),
 ("model raw area (ha)",                g30("burned footprint before the minimum-area rule","model"),     A10["model_raw_ha"]),
 ("overlap area (ha)",                  np.nan,                                                           A10["overlap_ha"]),
 ("footprint Jaccard",                  g30("Jaccard of the burned footprints","reference"),              A10["footprint_jaccard"]),
 ("unit-footprint Jaccard",             g30("Jaccard of the analysed unit footprints","reference"),       A10["unit_footprint_jaccard"]),
 ("% reference detected",               g30("reference-defined burned area also detected by the model","reference"), A10["pct_reference_detected"]),
 ("% model outside inventory",          g30("model-detected area that did not spatially overlap the fire-reference inventory","model"), A10["pct_model_outside_inventory"]),
 ("reference final units",              g30("spatial units","reference"),                                 A10["reference_final_units"]),
 ("model final units",                  g30("spatial units","model"),                                     A10["model_final_units"]),
 ("reference analysed area (ha)",       g30("area analysed as units","reference"),                        A10["reference_analysed_ha"]),
 ("model analysed area (ha)",           g30("area analysed as units","model"),                            A10["model_analysed_ha"]),
 ("mean RIS balanced (reference)",      g30("mean RIS (Balanced) of units","reference"),                  float(r10.mean_RIS_balanced.mean())),
 ("mean RIS balanced (model)",          g30("mean RIS (Balanced) of units","model"),                      float(m10.mean_RIS_balanced.mean())),
 ("mean observed NBR change (reference)",g30("mean observed NBR change 2017 minus 2025","reference"),     float(r10.observed_nbr_change_2017_minus_2025.mean())),
 ("mean observed NBR change (model)",   g30("mean observed NBR change 2017 minus 2025","model"),          float(m10.observed_nbr_change_2017_minus_2025.mean())),
]
h30=pd.read_csv(os.path.join(F30,"table_global_vs_within_landcover_counts.csv"))
h30=h30[~h30.small_n_class_flag]
for pop in ("reference","model"):
    rows.append(("hidden within-LC units (%s)"%pop,
                 float(h30[h30.population==pop].high_within_class_but_below_global_cut_n.sum()),
                 A10["hidden_units_%s"%pop]))
MIG=pd.DataFrame(rows,columns=["measure","m30","m10"])
MIG["change"]=MIG.m10-MIG.m30
MIG["pct_change"]=100.0*(MIG.m10-MIG.m30)/MIG.m30.replace(0,np.nan)
MIG.to_csv(os.path.join(OUT,"migration_headline.csv"),index=False)
print("HEADLINE MIGRATION TABLE\n"+"-"*96)
print("%-38s %14s %14s %12s %9s"%("measure","30 m","10 m","change","%"))
for _,r in MIG.iterrows():
    f=lambda v:"—" if pd.isna(v) else ("%14.4f"%v if abs(v)<10 else "%14.1f"%v)
    print("%-38s %s %s %s %8s"%(r.measure[:38],f(r.m30),f(r.m10),f(r["change"]),
          "—" if pd.isna(r["pct_change"]) else "%+.1f%%"%r["pct_change"]))

# ---- by land-cover class
def cls(df,pop):
    g=df.groupby("land_cover_name").agg(n_units=("unit_id","size"),area_ha=("area_ha","sum"),
        sev=("mean_severity_model","mean"),per=("mean_persistence_model","mean"),
        rg=("mean_recovery_gap_predicted","mean"),rec=("mean_recurrence_model","mean"),
        ris=("mean_RIS_balanced","mean"),obs=("observed_nbr_change_2017_minus_2025","mean")).reset_index()
    g["population"]=pop; return g
for pop,a,b in (("reference",r30,r10),("model",m30,m10)):
    x=cls(a,pop).merge(cls(b,pop),on="land_cover_name",suffixes=("_30","_10"),how="outer")
    x.to_csv(os.path.join(OUT,"migration_by_landcover_%s.csv"%pop),index=False)
    print("\nBY LAND COVER — %s units\n"%pop+"-"*112)
    print("%-20s %11s %11s %11s %11s %11s %11s %11s"%("class","units 30/10","area 30/10","severity","persist","rec.gap","recur","RIS"))
    for _,r in x.sort_values("area_ha_10",ascending=False).iterrows():
        n30,n10=r.n_units_30,r.n_units_10
        print("%-20s %5s/%-5s %5.0f/%-5.0f %5.3f/%-5.3f %5.3f/%-5.3f %5.3f/%-5.3f %5.3f/%-5.3f %5.3f/%-5.3f"%(
            str(r.land_cover_name)[:20], "—" if pd.isna(n30) else int(n30), "—" if pd.isna(n10) else int(n10),
            r.area_ha_30 if pd.notna(r.area_ha_30) else -1, r.area_ha_10 if pd.notna(r.area_ha_10) else -1,
            r.sev_30,r.sev_10,r.per_30,r.per_10,r.rg_30,r.rg_10,r.rec_30,r.rec_10,r.ris_30,r.ris_10))
    ok=x.dropna(subset=["ris_30","ris_10"])
    if len(ok)>2:
        print("  class-ordering Spearman rho 30 m vs 10 m: RIS %.4f | recovery gap %.4f | severity %.4f | observed NBR change %.4f"%(
            stats.spearmanr(ok.ris_30,ok.ris_10).statistic, stats.spearmanr(ok.rg_30,ok.rg_10).statistic,
            stats.spearmanr(ok.sev_30,ok.sev_10).statistic,
            stats.spearmanr(ok.obs_30,ok.obs_10).statistic))
print("\nwrote ->",OUT)
