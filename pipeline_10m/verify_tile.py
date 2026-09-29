"""Verify the regenerated tile is finite on all five outputs, before exceptions are applied."""
import json, numpy as np, rasterio, sys
from rasterio.windows import Window
OUT="outputs/inference_10m_candidate"
r,c,h,w=4352,768,256,256
z=np.load("%s/tiles/t_%d_%d.npz"%(OUT,r,c))
with rasterio.open("data/raw/burn_count_with_mask_10m.tif") as s:
    eco=(s.read(2,window=Window(c,r,w,h),boundless=True,fill_value=0)==1)
EXC=json.load(open("%s/QA_EXCEPTIONS.json"%OUT))["documented_exceptions"]
nexc=sum(1 for e in EXC if r<=e["row"]<r+h and c<=e["col"]<c+w)
n=int(eco.sum()); expect=n-nexc; ok=True
print("tile 4352_768 regenerated: eco-eligible %d | documented exceptions in tile %d | expected finite %d"%(n,nexc,expect))
for L in ("severity","persistence","recovery_gap","burned_prob","model_recurrence"):
    f=int(np.isfinite(z[L]).sum()); good=(f==expect); ok&=good
    print("   %-18s finite %6d   %s"%(L,f,"= eco-eligible minus %d documented exception(s)"%nexc
          if good else "DIFFERS by %d"%(f-expect)))
v=z["severity"][4414-r,1021-c]
print("   cell (4414,1021) severity before exception: %s"%v)
print("VERIFY: %s"%("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
