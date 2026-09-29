"""Regenerate the Section 4.11 severity-surface diagnostic (Table 20) on the 10 m grid.
Definition unchanged: cells whose estimated severity >= 0.5 that do not intersect a KKL fire
polygon (source == 'layer1'), tabulated by fixed 2018 land-cover class.
This is a diagnostic of the severity surface, NOT the fire-detection criterion."""
import os, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio import features
from rasterio.transform import Affine
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
ROOT = os.environ.get("THESIS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAST=os.path.join(ROOT,"outputs","inference_10m_candidate")
OUT=os.path.join(ROOT,"outputs","polygon_10m_candidate","tables")
GRID_T=Affine(10.,0.,620610.,0.,-10.,3514560.); H,W,CRS=7794,4875,"EPSG:32636"
LC_NAMES={10:"forest",11:"orchards",20:"shrubland",30:"grassland",40:"cropland",
          41:"covered agriculture",50:"built-up",60:"bare",61:"fallow",80:"water"}
with rasterio.open(os.path.join(RAST,"severity_10m.tif")) as s: SEV=s.read(1)
with rasterio.open(os.path.join(RAST,"landcover_2018_10m.tif")) as s: LC=s.read(1)
ELIG=np.isfinite(SEV)
fires=gpd.read_file(os.path.join(ROOT,"data","thesis_maps","fig_gt_fires.shp"))
kkl=fires[fires.source=="layer1"].to_crs(CRS)
print("KKL polygons (source == 'layer1'): %d of %d total"%(len(kkl),len(fires)))
def polygonal(g):
    if g is None or g.is_empty: return None
    g=make_valid(g)
    if isinstance(g,(Polygon,MultiPolygon)): return g
    if isinstance(g,GeometryCollection):
        pp=[x for x in g.geoms if isinstance(x,(Polygon,MultiPolygon))]
        return unary_union(pp) if pp else None
    return None
geoms=[g for g in (polygonal(x) for x in kkl.geometry) if g is not None and not g.is_empty]
# "do not intersect a KKL polygon": all_touched=True so any cell touched by a polygon is excluded
KKL=features.rasterize(((g,1) for g in geoms),out_shape=(H,W),transform=GRID_T,fill=0,
                       all_touched=True,dtype="uint8").astype(bool)
FLAG=ELIG&(SEV>=0.5)&(~KKL)
tot=int(FLAG.sum())
print("cells with severity >= 0.5 outside KKL polygons: %s"%f"{tot:,}")
rows=[]
for code,name in LC_NAMES.items():
    m=FLAG&(LC==code); n=int(m.sum())
    inclass=int((ELIG&(LC==code)).sum())
    if not n: continue
    rows.append(dict(land_cover=name,code=code,flagged=n,pct_of_flagged=100.0*n/tot,
                     in_class_rate=100.0*n/inclass if inclass else np.nan,
                     class_eligible_cells=inclass))
df=pd.DataFrame(rows).sort_values("flagged",ascending=False).reset_index(drop=True)
df.to_csv(os.path.join(OUT,"p10_table20_severity_diagnostic.csv"),index=False)
unclass=int((FLAG&(LC==0)).sum())
print("\nTABLE 20 (10 m)\n"+"-"*74)
print("%-22s %5s %12s %12s %12s"%("LC class","code","Flagged","% of flagged","in-class rate"))
for _,r in df.iterrows():
    print("%-22s %5d %12s %11.1f%% %11.1f%%"%(r.land_cover,r.code,f"{int(r.flagged):,}",r.pct_of_flagged,r.in_class_rate))
print("%-22s %5s %12s %11.1f%%"%("(unclassified)","-",f"{unclass:,}",100.0*unclass/tot))
print("\nsum of listed %% = %.1f%%"%df.pct_of_flagged.sum())
print("total flagged = %s ; eligible domain = %s"%(f"{tot:,}",f"{int(ELIG.sum()):,}"))
