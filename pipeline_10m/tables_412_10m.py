import os
"""Regenerate Tables 21 and 22 of Section 4.12 on the final 10 m grid.
Definitions are unchanged; only the grid changes (cells of 10 m, unit floor 36 cells = 0.36 ha)."""
import json, os, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
ROOT = os.environ.get("THESIS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAST=os.path.join(ROOT,"outputs","inference_10m_candidate")
OUT=os.path.join(ROOT,"outputs","polygon_10m_candidate","tables"); os.makedirs(OUT,exist_ok=True)
GRID_T=Affine(10.,0.,620610.,0.,-10.,3514560.); H,W,CRS=7794,4875,"EPSG:32636"
PX_HA, THR, MIN_UNIT_PX = 0.01, 0.502, 36
LC_NAMES={10:"forest",11:"orchards",20:"shrubland",30:"grassland",40:"cropland",
          41:"covered agriculture",50:"built-up",60:"bare",61:"fallow",80:"water"}
LC_ORDER=[10,11,20,30,40,41,50,60,61,80]
CLS_ORDER=[LC_NAMES[c] for c in LC_ORDER]+["unclassified / other code"]
PROFILES={"ecological":(.20,.20,.50,.10),"fire":(.35,.35,.20,.10),"balanced":(.25,.25,.25,.25)}
A={}
for k,f in [("severity","severity_10m.tif"),("persistence","persistence_10m.tif"),
            ("recovery_gap","recovery_gap_10m.tif"),("model_recurrence","model_recurrence_10m.tif"),
            ("burned_prob","burned_prob_10m.tif")]:
    with rasterio.open(os.path.join(RAST,f)) as s: A[k]=s.read(1).astype(np.float64)
with rasterio.open(os.path.join(RAST,"landcover_2018_10m.tif")) as s: LC=s.read(1)
ELIG=np.all([np.isfinite(A[k]) for k in A],axis=0)
RIS={p:np.where(ELIG,ws*A["severity"]+wp*A["persistence"]+wr*A["recovery_gap"]+wc*A["model_recurrence"],np.nan)
     for p,(ws,wp,wr,wc) in PROFILES.items()}
CLS=np.full((H,W),len(CLS_ORDER)-1,dtype=np.int16)
for i,c in enumerate(LC_ORDER): CLS[LC==c]=i
def polygonal(g):
    if g is None or g.is_empty: return None
    g=make_valid(g)
    if isinstance(g,(Polygon,MultiPolygon)): return g
    if isinstance(g,GeometryCollection):
        pp=[x for x in g.geoms if isinstance(x,(Polygon,MultiPolygon))]
        return unary_union(pp) if pp else None
    return None
fires=gpd.read_file(os.path.join(ROOT,"data","thesis_maps","fig_gt_fires.shp"))
geoms=[g for g in (polygonal(x) for x in fires[fires.fire_year>=2017].to_crs(CRS).geometry) if g is not None and not g.is_empty]
REF=features.rasterize(((g,1) for g in geoms),out_shape=(H,W),transform=GRID_T,fill=0,all_touched=False,dtype="uint8").astype(bool)&ELIG
DET=ELIG&(A["burned_prob"]>THR)
def fmt(v): return "%.4f"%v
rows=[]
for popname,M in (("Documented-burned cells",REF),("Model-detected cells",DET)):
    sub=[]
    for i,cn in enumerate(CLS_ORDER):
        m=M&(CLS==i); n=int(m.sum())
        if not n: continue
        sub.append(dict(population=popname,land_cover=cn,n=n,
            eco_mean=np.nanmean(RIS["ecological"][m]), eco_med=np.nanmedian(RIS["ecological"][m]),
            fire_mean=np.nanmean(RIS["fire"][m]),      fire_med=np.nanmedian(RIS["fire"][m]),
            bal_mean=np.nanmean(RIS["balanced"][m]),   bal_med=np.nanmedian(RIS["balanced"][m])))
    sub.sort(key=lambda r:-r["bal_mean"])
    rows+=sub
    n=int(M.sum())
    rows.append(dict(population=popname,land_cover="All classes",n=n,
        eco_mean=np.nanmean(RIS["ecological"][M]),eco_med=np.nanmedian(RIS["ecological"][M]),
        fire_mean=np.nanmean(RIS["fire"][M]),fire_med=np.nanmedian(RIS["fire"][M]),
        bal_mean=np.nanmean(RIS["balanced"][M]),bal_med=np.nanmedian(RIS["balanced"][M])))
t21=pd.DataFrame(rows); t21.to_csv(os.path.join(OUT,"p10_table21_RIS_by_landcover.csv"),index=False)
print("TABLE 21 (10 m)\n"+"-"*96)
for _,r in t21.iterrows():
    if r.land_cover=="All classes" or r.name==0 or True:
        print("%-24s %-26s %9s  %s  %s  %s"%(r.population if r.land_cover=="All classes" else "",
              r.land_cover,f"{int(r.n):,}",
              "%s (%s)"%(fmt(r.eco_mean),fmt(r.eco_med)),
              "%s (%s)"%(fmt(r.fire_mean),fmt(r.fire_med)),
              "%s (%s)"%(fmt(r.bal_mean),fmt(r.bal_med))))
# ---- Table 22 : pixel-level vs area-level over documented-burned cells
STRUCT=np.ones((3,3),int)
lab,_=ndimage.label(REF,structure=STRUCT)
sizes=np.bincount(lab.ravel()); drop=np.where(sizes<MIN_UNIT_PX)[0]; drop=drop[drop>0]
if len(drop): lab[np.isin(lab,drop)]=0
lab,ncomp=ndimage.label(lab>0,structure=STRUCT)
sel=lab>0
df=pd.DataFrame({"zone":lab[sel],"cls":CLS[sel],"bal":RIS["balanced"][sel],
                 "rg":A["recovery_gap"][sel],"sev":A["severity"][sel],"per":A["persistence"][sel]})
u=df.groupby(["zone","cls"]).agg(n=("zone","size"),bal=("bal","mean"),rg=("rg","mean"),
                                 sev=("sev","mean"),per=("per","mean")).reset_index()
u=u[u.n>=MIN_UNIT_PX]
rows=[]
for i,cn in enumerate(CLS_ORDER):
    m=REF&(CLS==i); n=int(m.sum())
    if not n: continue
    uu=u[u.cls==i]
    rows.append(dict(land_cover=cn,cells=n,units=len(uu),
        bal_px=np.nanmean(RIS["balanced"][m]), bal_ar=uu.bal.mean() if len(uu) else np.nan,
        rg_px=np.nanmean(A["recovery_gap"][m]), rg_ar=uu.rg.mean() if len(uu) else np.nan,
        sev_px=np.nanmean(A["severity"][m]),    sev_ar=uu.sev.mean() if len(uu) else np.nan,
        per_px=np.nanmean(A["persistence"][m]), per_ar=uu.per.mean() if len(uu) else np.nan))
rows.sort(key=lambda r:-r["bal_px"])
rows.append(dict(land_cover="All classes",cells=int(REF.sum()),units=int(ncomp),
    bal_px=np.nanmean(RIS["balanced"][REF]),bal_ar=u.bal.mean(),
    rg_px=np.nanmean(A["recovery_gap"][REF]),rg_ar=u.rg.mean(),
    sev_px=np.nanmean(A["severity"][REF]),sev_ar=u.sev.mean(),
    per_px=np.nanmean(A["persistence"][REF]),per_ar=u.per.mean()))
t22=pd.DataFrame(rows); t22.to_csv(os.path.join(OUT,"p10_table22_pixel_vs_area.csv"),index=False)
print("\nTABLE 22 (10 m)\n"+"-"*112)
print("%-24s %9s %6s %8s %8s %8s %8s %8s %8s %8s %8s"%("class","cells","units","balRIS_px","balRIS_ar","rg_px","rg_ar","sev_px","sev_ar","per_px","per_ar"))
for _,r in t22.iterrows():
    print("%-24s %9s %6d %8s %8s %8s %8s %8s %8s %8s %8s"%(r.land_cover,f"{int(r.cells):,}",int(r.units),
          fmt(r.bal_px),fmt(r.bal_ar),fmt(r.rg_px),fmt(r.rg_ar),fmt(r.sev_px),fmt(r.sev_ar),fmt(r.per_px),fmt(r.per_ar)))
print("\nwrote -> %s"%OUT)
