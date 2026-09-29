import os
"""Mosaic the per-tile predictions into the final 10 m rasters in a SINGLE pass.
Direct aligned window placement: no resampling, no interpolation, no seam averaging.
Tiles are disjoint by construction, so every cell is written exactly once."""
import json, os, numpy as np, rasterio
from rasterio.windows import Window
ROOT = os.environ.get("THESIS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=os.path.join(ROOT,"outputs","inference_10m_candidate"); TD=os.path.join(OUT,"tiles")
LAYERS=["severity","persistence","recovery_gap","burned_prob","model_recurrence"]
GW,GH,X0,Y0=4875,7794,620610,3514560
man=json.load(open(os.path.join(OUT,"manifest.json"))); TILE=man["tile"]
exp=[(r,c,min(TILE,GH-r),min(TILE,GW-c)) for r in range(0,GH,TILE) for c in range(0,GW,TILE)]
missing=[(r,c) for r,c,_,_ in exp if not os.path.exists(os.path.join(TD,"t_%d_%d.npz"%(r,c)))]
assert not missing, "missing %d tile files, e.g. %s" % (len(missing), missing[:5])
print("all %d tile files present" % len(exp))
cover=np.zeros((GH,GW),dtype=np.uint8)
for r,c,h,w in exp: cover[r:r+h,c:c+w]+=1
assert cover.min()==1 and cover.max()==1, "tiles overlap or leave gaps"
print("tile windows cover the grid exactly once (min=max=1)"); del cover
prof=dict(driver="GTiff",height=GH,width=GW,count=1,dtype="float32",crs="EPSG:32636",
          transform=rasterio.Affine(10,0,X0,0,-10,Y0),nodata=np.nan,tiled=True,
          blockxsize=256,blockysize=256,compress="deflate")
handles={L:rasterio.open(os.path.join(OUT,"%s_10m.tif"%L),"w",**prof) for L in LAYERS}
try:
    for n,(r,c,h,w) in enumerate(exp,1):
        z=np.load(os.path.join(TD,"t_%d_%d.npz"%(r,c)))
        for L in LAYERS:
            a=z[L]
            assert a.shape==(h,w), "tile %d_%d shape %s != %s"%(r,c,a.shape,(h,w))
            handles[L].write(a.astype(np.float32),1,window=Window(c,r,w,h))
        if n%100==0: print("  mosaicked %d/%d tiles"%(n,len(exp)),flush=True)
finally:
    for d in handles.values(): d.close()
print("mosaic written in a single pass")
for L in LAYERS:
    with rasterio.open(os.path.join(OUT,"%s_10m.tif"%L)) as s:
        v=s.read(1); f=np.isfinite(v)
        print("  %-18s %dx%d  finite %s  range [%.4f, %.4f]"%(L,s.width,s.height,f"{int(f.sum()):,}",
              np.nanmin(v),np.nanmax(v)))
