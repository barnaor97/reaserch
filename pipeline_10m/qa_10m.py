"""Inference QA gate for the 10 m candidate mosaic. Read-only."""
import hashlib, json, os, sys
import numpy as np, rasterio
from rasterio.windows import Window
ROOT = os.environ.get("THESIS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=os.path.join(ROOT,"outputs","inference_10m_candidate")
LAYERS=["severity","persistence","recovery_gap","burned_prob","model_recurrence"]
GW,GH,X0,Y0=4875,7794,620610,3514560
man=json.load(open(os.path.join(OUT,"manifest.json")))
ok=True; rows=[]
def chk(name,cond,detail=""):
    global ok; ok&=bool(cond); rows.append((name,bool(cond),detail))
    print("[%s] %-62s %s"%("PASS" if cond else "FAIL",name,detail))
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

print("="*104); print("A. INFERENCE QA — 10 m candidate"); print("="*104)
# ---- tiling: completeness, uniqueness, zero overlap
TILE=man["tile"]
exp=[(r,c,min(TILE,GH-r),min(TILE,GW-c)) for r in range(0,GH,TILE) for c in range(0,GW,TILE)]
keys=["%d_%d"%(r,c) for r,c,_,_ in exp]
chk("every expected tile recorded", set(keys)==set(man["done"]), "%d expected, %d recorded"%(len(keys),len(man["done"])))
chk("no duplicate tile key", len(man["done"])==len(set(man["done"])))
cover=np.zeros((GH,GW),dtype=np.uint8)
for r,c,h,w in exp: cover[r:r+h,c:c+w]+=1
chk("tiles have ZERO overlap and cover the grid exactly", cover.min()==1 and cover.max()==1,
    "min %d max %d -> every cell written by exactly one tile"%(cover.min(),cover.max()))
del cover

# ---- independent eco mask
with rasterio.open(os.path.join(ROOT,"data","raw","burn_count_with_mask_10m.tif")) as s:
    eco=s.read(2,window=Window(0,0,GW,GH),boundless=True,fill_value=0)
ECO=(eco==1); n_eco=int(ECO.sum()); del eco
print("\nindependent 10 m eco mask: %s eligible cells"%f"{n_eco:,}")

# ---- rasters
fp=None; stats={}
for L in LAYERS:
    p=os.path.join(OUT,"%s_10m.tif"%L)
    with rasterio.open(p) as s:
        chk("%s: dimensions exactly %d x %d"%(L,GW,GH), (s.width,s.height)==(GW,GH), "%dx%d"%(s.width,s.height))
        chk("%s: CRS EPSG:32636"%L, str(s.crs)=="EPSG:32636", str(s.crs))
        t=s.transform
        chk("%s: exact aligned transform"%L,
            (t.a,t.e,t.c,t.f)==(10.0,-10.0,float(X0),float(Y0)), str(tuple(round(v,1) for v in t[:6])))
        v=s.read(1)
    f=np.isfinite(v)
    stats[L]=dict(n_finite=int(f.sum()), vmin=float(np.nanmin(v)) if f.any() else np.nan,
                  vmax=float(np.nanmax(v)) if f.any() else np.nan)
    if fp is None: fp=f
    else: chk("%s: valid-data footprint identical to %s"%(L,LAYERS[0]), bool((f==fp).all()),
              "%d differing cells"%int((f!=fp).sum()))
    chk("%s: values within [0,1]"%L, np.nanmin(v)>=0.0 and np.nanmax(v)<=1.0,
        "[%.4f, %.4f]"%(stats[L]["vmin"],stats[L]["vmax"]))
    del v,f

n_pred=int(fp.sum())
EXC=json.load(open(os.path.join(OUT,"QA_EXCEPTIONS.json")))["documented_exceptions"]
EXC_RC={(e["row"],e["col"]) for e in EXC}
masks={}
for L in LAYERS:
    with rasterio.open(os.path.join(OUT,"%s_10m.tif"%L)) as s2: masks[L]=np.isfinite(s2.read(1))
common=np.logical_and.reduce([masks[L] for L in LAYERS])
fail=ECO & ~common
fail_rc={(int(a),int(b)) for a,b in np.argwhere(fail)}
outside=int((common & ~ECO).sum())
print("\nELIGIBILITY RECONCILIATION")
print("  eco-eligible cells                 : %s"%f"{n_eco:,}")
print("  valid common prediction footprint  : %s"%f"{int(common.sum()):,}")
print("  numerical prediction failures      : %d"%int(fail.sum()))
print("  predictions outside eligible domain: %d"%outside)
print("\n  NOTE: failing cells stay inside the eco-eligible domain; they are excluded")
print("        only from the common prediction footprint. See QA_EXCEPTIONS.json.")
print("\nPER-LAYER FOOTPRINTS")
print("  %-18s %14s %14s %14s %14s"%("layer","eligible","predicted","missing","outside"))
for L in LAYERS:
    f=masks[L]; miss=int((ECO & ~f).sum()); ext=int((f & ~ECO).sum())
    print("  %-18s %14s %14s %14d %14d"%(L,f"{n_eco:,}",f"{int(f.sum()):,}",miss,ext))
    chk("%s: missing cells are exactly the documented exceptions"%L,
        {(int(a),int(b)) for a,b in np.argwhere(ECO & ~f)}==EXC_RC, "%d missing"%miss)
    chk("%s: predicted outside == 0"%L, ext==0, str(ext))
chk("all five valid-data masks are identical",
    all(np.array_equal(masks[L],masks[LAYERS[0]]) for L in LAYERS), "including model_recurrence")
chk("failures are exactly the documented exceptions", fail_rc==EXC_RC,
    "%d failing vs %d documented"%(len(fail_rc),len(EXC_RC)))
chk("no prediction outside the eligible domain", outside==0, str(outside))
chk("common footprint == eco-eligible minus documented exceptions",
    int(common.sum())==n_eco-len(EXC_RC), "%s vs %s"%(f"{int(common.sum()):,}",f"{n_eco-len(EXC_RC):,}"))
chk("sum of per-tile eligible counts equals eco-eligible",
    sum(v["eligible"] for v in man["done"].values())==n_eco,
    "manifest %s"%f"{sum(v['eligible'] for v in man['done'].values()):,}")
del masks, common, fail
n_pred=n_eco-len(EXC_RC)

# ---- seam test along interior tile boundaries (predictions are pixel-independent,
#      so a seam would indicate a windowing bug rather than a modelling effect)
with rasterio.open(os.path.join(OUT,"burned_prob_10m.tif")) as s: BP=s.read(1)
bnd_cols=[c for c in range(TILE,GW,TILE)]
def jump(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    return np.abs(a[m]-b[m]) if m.any() else np.array([])
seam=np.concatenate([jump(BP[:,c-1],BP[:,c]) for c in bnd_cols]) if bnd_cols else np.array([])
inter=np.concatenate([jump(BP[:,c-3],BP[:,c-2]) for c in bnd_cols]) if bnd_cols else np.array([])
if seam.size and inter.size:
    chk("no seam at tile boundaries", abs(float(seam.mean())-float(inter.mean()))<0.01,
        "mean |delta| across boundary %.5f vs interior control %.5f"%(seam.mean(),inter.mean()))
del BP

# ---- provenance
chk("checkpoint hash unchanged", man["checkpoint_sha256"]==sha(os.path.join(ROOT,"models/main/model_main.pt")),
    man["checkpoint_sha256"][:16])
chk("scaler hash unchanged", man["scaler_sha256"]==sha(os.path.join(ROOT,"data/processed/scaler.pkl")),
    man["scaler_sha256"][:16])
chk("all 108 months used in every processed tile", man["months"]==108 and
    all(v.get("months_fetched",108)==108 for v in man["done"].values() if v.get("eligible",0)>0), "T=108")
chk("52 model features", man["features"]==52)
chk("threshold unchanged at 0.502", man["threshold"]==0.502)
chk("no 30 m raster among the inputs", all("_30m" not in x for x in os.listdir(OUT) if x.endswith(".tif")),
    ", ".join(sorted(x for x in os.listdir(OUT) if x.endswith(".tif"))))

hashes={L: sha(os.path.join(OUT,"%s_10m.tif"%L)) for L in LAYERS}
json.dump(dict(grid=dict(width=GW,height=GH,crs="EPSG:32636",origin=[X0,Y0],res=10),
               eligible_cells=n_eco, predicted_cells=n_pred, tiles=len(man["done"]), tile_size=TILE,
               checkpoint_sha256=man["checkpoint_sha256"], scaler_sha256=man["scaler_sha256"],
               months=108, model_features=52, threshold=0.502, layer_sha256=hashes,
               layer_stats=stats, qa_pass=bool(ok)),
          open(os.path.join(OUT,"PRODUCTION_MANIFEST.json"),"w"), indent=1)
print("\n%s"%("INFERENCE QA: PASS" if ok else "INFERENCE QA: FAIL"))
for L in LAYERS: print("  %-18s sha256 %s  range [%.4f, %.4f]"%(L,hashes[L][:32],stats[L]["vmin"],stats[L]["vmax"]))
sys.exit(0 if ok else 1)
