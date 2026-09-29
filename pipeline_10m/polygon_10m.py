import os
"""Polygon / spatial-unit analysis rebuilt on the aligned 10 m grid.

Authoritative logic follows 15_polygon_workflow.py (unit formation, rankings) and
19_finalize_polygon_thesis_outputs.py (final CSV schema, agreement metrics), with the
grid changed to 10 m. PX_HA = 0.01 and the PHYSICAL floor of 0.36 ha is preserved as
MIN_UNIT_PX = 36 cells. Every statistic is recomputed from 10 m cells.
"""
import json, os, sys, warnings
from datetime import datetime, timezone
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio import features
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from scipy import ndimage, stats
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = os.environ.get("THESIS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAST = os.path.join(ROOT, "outputs", "inference_10m_candidate")
OUTD = os.path.join(ROOT, "outputs", "polygon_10m_candidate")
TAB  = os.path.join(OUTD, "tables"); os.makedirs(TAB, exist_ok=True)
P    = os.path.join(ROOT, "data", "processed")
FIRES = os.path.join(ROOT, "data", "thesis_maps", "fig_gt_fires.shp")
LC_TIF  = os.path.join(RAST, "landcover_2018_10m.tif")
OBS_TIF = os.path.join(RAST, "nbr_2017_2025_10m.tif")

GRID_T = Affine(10.0, 0.0, 620610.0, 0.0, -10.0, 3514560.0)
H, W, CRS = 7794, 4875, "EPSG:32636"
PX_HA, THR, MIN_UNIT_HA = 0.01, 0.502, 0.36
MIN_UNIT_PX = int(round(MIN_UNIT_HA / PX_HA))
SMALL_N_UNITS, NODATA = 20, -9999.0
RANK_MIN_HA = 1.0
LC_NAMES = {10:"forest",11:"orchards",20:"shrubland",30:"grassland",40:"cropland",
            41:"covered agriculture",50:"built-up",60:"bare",61:"fallow",80:"water"}
LC_ORDER = [10,11,20,30,40,41,50,60,61,80]
CLS_ORDER = [LC_NAMES[c] for c in LC_ORDER] + ["unclassified / other code"]
CLS_IDX = {c:i for i,c in enumerate(CLS_ORDER)}
code_of = {CLS_IDX[LC_NAMES[c]]: c for c in LC_ORDER}
PROFILES = {"ecological":(0.20,0.20,0.50,0.10),"fire":(0.35,0.35,0.20,0.10),"balanced":(0.25,0.25,0.25,0.25)}
POP_LABEL = {"reference":"A. Reference-defined burned areas","model":"B. Model-detected burned areas"}
RUN = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
assert MIN_UNIT_PX == 36
print("10 m grid %dx%d | PX_HA %.2f | MIN_UNIT_PX %d cells = %.2f ha (physical floor preserved)"
      % (W,H,PX_HA,MIN_UNIT_PX,MIN_UNIT_PX*PX_HA))

def save(name, df):
    df.to_csv(os.path.join(TAB,"p10_%s.csv"%name), index=False)
    print("  [table] %-44s %6d rows"%(name,len(df)))

# ------------------------------------------------------------- 1. layers
A, align = {}, []
for role, fn in [("severity","severity_10m.tif"),("persistence","persistence_10m.tif"),
                 ("recovery_gap_predicted","recovery_gap_10m.tif"),
                 ("model_recurrence","model_recurrence_10m.tif"),("burned_prob","burned_prob_10m.tif")]:
    with rasterio.open(os.path.join(RAST,fn)) as s:
        assert str(s.crs)==CRS and (s.height,s.width)==(H,W) and s.transform.almost_equals(GRID_T), fn
        A[role]=s.read(1).astype(np.float64)
        align.append(dict(layer=role,file=fn,shape="%dx%d"%(s.height,s.width),crs=str(s.crs),ok=True))
with rasterio.open(LC_TIF) as s:
    assert str(s.crs)==CRS and (s.height,s.width)==(H,W) and s.transform.almost_equals(GRID_T)
    LC_GRID=s.read(1); align.append(dict(layer="land_cover_2018",file=os.path.basename(LC_TIF),
                                         shape="%dx%d"%(s.height,s.width),crs=str(s.crs),ok=True))
with rasterio.open(OBS_TIF) as s:
    assert str(s.crs)==CRS and (s.height,s.width)==(H,W) and s.transform.almost_equals(GRID_T)
    OB17,OB25,NV17,NV25 = (s.read(i).astype(np.float64) for i in (1,2,3,4))
    align.append(dict(layer="observed_NBR",file=os.path.basename(OBS_TIF),
                      shape="%dx%d"%(s.height,s.width),crs=str(s.crs),ok=True))
for a in (OB17,OB25): a[a==NODATA]=np.nan
OBS_CHANGE = OB17-OB25
ELIG = np.all([np.isfinite(A[k]) for k in A],axis=0)
cls_i = np.full((H,W), CLS_IDX["unclassified / other code"], dtype=np.int16)
for code,nm in LC_NAMES.items(): cls_i[LC_GRID==code]=CLS_IDX[nm]
RIS = {p: np.where(ELIG, ws*A["severity"]+wp*A["persistence"]+wr*A["recovery_gap_predicted"]
                   +wc*A["model_recurrence"], np.nan) for p,(ws,wp,wr,wc) in PROFILES.items()}
save("t00_grid_alignment", pd.DataFrame(align))
print("eligible 10 m cells: %s (%.2f%%)"%(f"{int(ELIG.sum()):,}",100*ELIG.mean()))

# sampled 10 m support (native resolution of the training sample)
meta_co = pd.read_csv(os.path.join(ROOT,"data","raw","pixel_coords_10m.csv"))
RG_ANALYTICAL = np.load(os.path.join(P,"targets_aux.npz"))["recovery_gap"].astype(float)
n_s = len(RG_ANALYTICAL)
burned_s = np.asarray(np.load(os.path.join(P,"y_burned.npy"),mmap_mode="r")).astype(bool).any(1)[:n_s]
nbr_s = np.load(os.path.join(P,"nbr_raw.npy"),mmap_mode="r")
IMP17_S = np.nanmean(np.asarray(nbr_s[:n_s,:12],dtype=np.float64),axis=1)
xs,ys = warp_transform("EPSG:4326",CRS,meta_co.lon.to_numpy()[:n_s],meta_co.lat.to_numpy()[:n_s])
COLp=np.floor((np.array(xs)-GRID_T.c)/GRID_T.a).astype(int)
ROWp=np.floor((np.array(ys)-GRID_T.f)/GRID_T.e).astype(int)
ONp=(ROWp>=0)&(ROWp<H)&(COLp>=0)&(COLp<W)
RCp=(np.clip(ROWp,0,H-1),np.clip(COLp,0,W-1))

# ------------------------------------------------------------- 2. footprints
def polygonal(g):
    if g is None or g.is_empty: return None
    g=make_valid(g)
    if isinstance(g,(Polygon,MultiPolygon)): return g
    if isinstance(g,GeometryCollection):
        parts=[p for p in g.geoms if isinstance(p,(Polygon,MultiPolygon))]
        return unary_union(parts) if parts else None
    return None
fires=gpd.read_file(FIRES)
geoms=[g for g in (polygonal(x) for x in fires[fires.fire_year>=2017].to_crs(CRS).geometry)
       if g is not None and not g.is_empty]
INSIDE=features.rasterize(((g,1) for g in geoms),out_shape=(H,W),transform=GRID_T,fill=0,
                          all_touched=False,dtype="uint8").astype(bool)
REF=INSIDE&ELIG
DET=ELIG&(A["burned_prob"]>THR)
print("reference-defined %s cells (%.0f ha) | model-detected %s cells (%.0f ha)"
      %(f"{int(REF.sum()):,}",REF.sum()*PX_HA,f"{int(DET.sum()):,}",DET.sum()*PX_HA))

# ------------------------------------------------------------- 3. units
STRUCT=np.ones((3,3),dtype=int)
def components(mask,min_px=0):
    lab,n=ndimage.label(mask,structure=STRUCT)
    if min_px>1 and n:
        sizes=np.bincount(lab.ravel()); drop=np.where(sizes<min_px)[0]; drop=drop[drop>0]
        if len(drop): lab[np.isin(lab,drop)]=0
        lab,n=ndimage.label(lab>0,structure=STRUCT)
    return lab,n
UNITS,ZONES,acct={}, {}, []
for pop,M in (("reference",REF),("model",DET)):
    lab_all,n_all=components(M)
    sizes_all=np.bincount(lab_all.ravel())[1:] if n_all else np.array([],dtype=int)
    small=sizes_all[sizes_all<MIN_UNIT_PX]
    lab,n=components(M,MIN_UNIT_PX); ZONES[pop]=lab
    sel=lab>0; rr,cc=np.nonzero(sel)
    d=pd.DataFrame({"zone":lab[sel],"cls":cls_i[sel],
                    "east":GRID_T.c+(cc+0.5)*GRID_T.a,"north":GRID_T.f+(rr+0.5)*GRID_T.e,
                    "severity":A["severity"][sel],"persistence":A["persistence"][sel],
                    "rg_pred":A["recovery_gap_predicted"][sel],"recurrence":A["model_recurrence"][sel],
                    "ob17":OB17[sel],"ob25":OB25[sel],"chg":OBS_CHANGE[sel],
                    "nv17":NV17[sel],"nv25":NV25[sel],
                    **{"RIS_"+p:RIS[p][sel] for p in PROFILES}})
    u=d.groupby(["zone","cls"]).agg(
        pixel_count=("zone","size"), east=("east","mean"), north=("north","mean"),
        mean_severity_model=("severity","mean"), mean_persistence_model=("persistence","mean"),
        mean_recovery_gap_predicted=("rg_pred","mean"), mean_recurrence_model=("recurrence","mean"),
        mean_RIS_balanced=("RIS_balanced","mean"), mean_RIS_ecological=("RIS_ecological","mean"),
        mean_RIS_fire=("RIS_fire","mean"), observed_mean_NBR_2017=("ob17","mean"),
        observed_mean_NBR_2025=("ob25","mean"), observed_nbr_change_2017_minus_2025=("chg","mean"),
        valid_months_2017_mean=("nv17","mean"), valid_months_2025_mean=("nv25","mean")).reset_index()
    cand_n,cand_px=len(u),int(u.pixel_count.sum())
    u=u[u.pixel_count>=MIN_UNIT_PX].reset_index(drop=True)
    u["unit_id"]=["%s_%d_%d"%({"reference":"doc","model":"mod"}[pop],z,c) for z,c in zip(u.zone,u.cls)]
    u["burned_area_definition"]=POP_LABEL[pop]; u["population"]=pop
    u["land_cover_name"]=[CLS_ORDER[c] for c in u.cls]
    u["land_cover_code"]=[code_of.get(c,-1) for c in u.cls]
    u["area_ha"]=u.pixel_count*PX_HA
    zz,ccx=lab[RCp],cls_i[RCp]; ins=ONp&(zz>0)&burned_s; idx=np.where(ins)[0]
    samp=pd.DataFrame({"unit_id":["%s_%d_%d"%({"reference":"doc","model":"mod"}[pop],zz[i],ccx[i]) for i in idx],
                       "rg":RG_ANALYTICAL[idx],"b17":IMP17_S[idx]})
    agg=samp.groupby("unit_id").agg(analytical_recovery_gap_sampled_mean=("rg","mean"),
                                    analytical_recovery_gap_sampled_n_px=("rg","size"),
                                    baseline_NBR_2017_imputed_sampled_mean=("b17","mean")).reset_index()
    u=u.merge(agg,on="unit_id",how="left")
    u["analytical_recovery_gap_support"]=np.where(u.analytical_recovery_gap_sampled_n_px.notna(),
        "sampled 10 m pixels inside the unit","no sampled pixel in this unit")
    u["global_RIS_rank"]=u.mean_RIS_balanced.rank(ascending=False,method="min").astype(int)
    u["global_RIS_percentile"]=100.0*u.mean_RIS_balanced.rank(pct=True)
    u["within_landcover_RIS_rank"]=u.groupby("land_cover_name").mean_RIS_balanced.rank(ascending=False,method="min").astype(int)
    u["within_landcover_RIS_percentile"]=100.0*u.groupby("land_cover_name").mean_RIS_balanced.rank(pct=True)
    u["n_units_in_landcover_class"]=u.groupby("land_cover_name").mean_RIS_balanced.transform("size").astype(int)
    u["small_n_class_flag"]=u.n_units_in_landcover_class<SMALL_N_UNITS
    UNITS[pop]=u
    acct+=[dict(population=pop,step="0. burned mask",n_items=int(n_all),n_px=int(M.sum()),area_ha=M.sum()*PX_HA),
           dict(population=pop,step="1a. components below floor REMOVED",n_items=int(len(small)),n_px=int(small.sum()),area_ha=small.sum()*PX_HA),
           dict(population=pop,step="1b. components kept",n_items=int(n),n_px=int(sel.sum()),area_ha=sel.sum()*PX_HA),
           dict(population=pop,step="2a. zone x class candidates",n_items=cand_n,n_px=cand_px,area_ha=cand_px*PX_HA),
           dict(population=pop,step="2b. candidates below floor REMOVED",n_items=cand_n-len(u),n_px=cand_px-int(u.pixel_count.sum()),area_ha=(cand_px-int(u.pixel_count.sum()))*PX_HA),
           dict(population=pop,step="3. FINAL units analysed",n_items=len(u),n_px=int(u.pixel_count.sum()),area_ha=float(u.area_ha.sum()))]
    print("   %-9s %d components -> %d kept -> %d candidates -> %5d FINAL units | %8.0f ha -> %8.0f ha | RG support %.1f%%"
          %(pop,n_all,n,cand_n,len(u),M.sum()*PX_HA,u.area_ha.sum(),
            100.0*u.analytical_recovery_gap_sampled_n_px.notna().mean()))
save("t01_unit_formation_accounting",pd.DataFrame(acct))

COLS=["unit_id","population","burned_area_definition","land_cover_code","land_cover_name","area_ha",
      "pixel_count","mean_severity_model","mean_persistence_model","mean_recovery_gap_predicted",
      "mean_recurrence_model","mean_RIS_balanced","mean_RIS_ecological","mean_RIS_fire",
      "observed_mean_NBR_2017","observed_mean_NBR_2025","observed_nbr_change_2017_minus_2025",
      "valid_months_2017_mean","valid_months_2025_mean","analytical_recovery_gap_sampled_mean",
      "analytical_recovery_gap_sampled_n_px","analytical_recovery_gap_support",
      "baseline_NBR_2017_imputed_sampled_mean","global_RIS_rank","global_RIS_percentile",
      "within_landcover_RIS_rank","within_landcover_RIS_percentile","n_units_in_landcover_class",
      "small_n_class_flag","east","north"]
for pop,f in (("reference","reference_units_final.csv"),("model","model_units_final.csv")):
    UNITS[pop][COLS+["zone","cls"]].to_csv(os.path.join(OUTD,f),index=False)
    print("   wrote %s (%d units)"%(f,len(UNITS[pop])))

# ------------------------------------------------------------- 4. agreement
inter=int((REF&DET).sum()); union=int((REF|DET).sum())
uf={}
for pop in ("reference","model"):
    keep=set(UNITS[pop].zone); lab=ZONES[pop]
    uf[pop]=np.isin(lab,list(keep))&(lab>0)
ju=float((uf["reference"]&uf["model"]).sum()/max((uf["reference"]|uf["model"]).sum(),1))
AG=dict(reference_raw_ha=REF.sum()*PX_HA, model_raw_ha=DET.sum()*PX_HA, overlap_ha=inter*PX_HA,
        reference_only_ha=(int(REF.sum())-inter)*PX_HA, model_only_ha=(int(DET.sum())-inter)*PX_HA,
        footprint_jaccard=inter/union if union else np.nan, unit_footprint_jaccard=ju,
        pct_reference_detected=100.0*inter/max(int(REF.sum()),1),
        pct_model_outside_inventory=100.0*(int(DET.sum())-inter)/max(int(DET.sum()),1),
        reference_final_units=len(UNITS["reference"]), model_final_units=len(UNITS["model"]),
        reference_analysed_ha=float(UNITS["reference"].area_ha.sum()),
        model_analysed_ha=float(UNITS["model"].area_ha.sum()))
# hidden within-class units
# Hidden-unit rule, identical to the 30 m analysis: within-class top decile is
# ceil(n/10) units by within-class RIS rank; a unit is "hidden" if it is in that set but
# below the population's global top-decile cut. Classes flagged small-n are excluded.
import math
hidden={}; hid_rows=[]
for pop in ("reference","model"):
    u=UNITS[pop]; cut=u.mean_RIS_balanced.quantile(0.9); ids=[]
    for cname,sub in u.groupby("land_cover_name"):
        n=len(sub); small = n < SMALL_N_UNITS
        tw=sub.nsmallest(int(math.ceil(n/10.0)),"within_landcover_RIS_rank")
        below=tw[tw.mean_RIS_balanced<cut]
        hid_rows.append(dict(population=pop,land_cover_name=cname,n_units=n,small_n_class_flag=small,
                             median_RIS=float(sub.mean_RIS_balanced.median()),
                             within_class_top_decile_n=len(tw),
                             also_in_global_top_decile_n=int((tw.mean_RIS_balanced>=cut).sum()),
                             high_within_class_but_below_global_cut_n=len(below)))
        if not small: ids+=list(below.unit_id)
    hidden[pop]=ids; AG["hidden_units_%s"%pop]=len(ids)
save("t09_global_vs_within_landcover_counts",pd.DataFrame(hid_rows))
save("t10_footprint_agreement",pd.DataFrame([AG]).T.reset_index().rename(columns={"index":"measure",0:"value"}))
json.dump({k:(float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in AG.items()},
          open(os.path.join(OUTD,"AGREEMENT_10m.json"),"w"),indent=1)
pd.DataFrame({"population":sum([[p]*len(hidden[p]) for p in hidden],[]),
              "unit_id":sum(hidden.values(),[])}).to_csv(os.path.join(TAB,"p10_t08_hidden_units.csv"),index=False)
print("footprint Jaccard %.4f | unit Jaccard %.4f | reference detected %.1f%% | model outside inventory %.1f%% | hidden %d/%d"
      %(AG["footprint_jaccard"],AG["unit_footprint_jaccard"],AG["pct_reference_detected"],
        AG["pct_model_outside_inventory"],AG["hidden_units_reference"],AG["hidden_units_model"]))

# ------------------------------------------------------------- 5. land-cover class table
rows=[]
for pop in ("reference","model"):
    u=UNITS[pop]
    for cname in CLS_ORDER:
        s=u[u.land_cover_name==cname]
        if not len(s): continue
        rows.append(dict(population=pop,land_cover=cname,n_units=len(s),area_ha=float(s.area_ha.sum()),
            mean_severity=float(s.mean_severity_model.mean()),mean_persistence=float(s.mean_persistence_model.mean()),
            mean_recovery_gap_predicted=float(s.mean_recovery_gap_predicted.mean()),
            mean_recurrence=float(s.mean_recurrence_model.mean()),
            mean_RIS_balanced=float(s.mean_RIS_balanced.mean()),median_RIS_balanced=float(s.mean_RIS_balanced.median()),
            observed_nbr_change=float(s.observed_nbr_change_2017_minus_2025.mean()),
            observed_NBR_2017=float(s.observed_mean_NBR_2017.mean()),
            analytical_recovery_gap_sampled_mean=float(s.analytical_recovery_gap_sampled_mean.mean()),
            n_units_with_RG_support=int(s.analytical_recovery_gap_sampled_n_px.notna().sum()),
            small_n_class=bool(len(s)<SMALL_N_UNITS)))
LCT=pd.DataFrame(rows); save("t11_landcover_indicator_table",LCT)
ref=LCT[LCT.population=="reference"].dropna(subset=["analytical_recovery_gap_sampled_mean"])
if len(ref)>2:
    print("class-ordering rho (analytical vs predicted recovery gap, reference units): %.4f"
          %stats.spearmanr(ref.analytical_recovery_gap_sampled_mean,ref.mean_recovery_gap_predicted).statistic)
json.dump(dict(run=RUN,grid=dict(crs=CRS,H=H,W=W,res=10,origin=[620610,3514560]),px_ha=PX_HA,
               min_unit_px=MIN_UNIT_PX,min_unit_ha=MIN_UNIT_HA,threshold=THR,
               eligible_cells=int(ELIG.sum()),**{k:(float(v) if isinstance(v,(int,float,np.floating)) else v)
               for k,v in AG.items()}),open(os.path.join(OUTD,"SUMMARY_10m.json"),"w"),indent=1)
print("\npolygon 10 m analysis complete ->",OUTD)
