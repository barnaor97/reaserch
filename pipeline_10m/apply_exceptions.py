"""Force documented exception cells to NoData in ALL five layers of their tile file.
This keeps the five valid-data footprints identical and removes the spurious
model_recurrence = 0.0 that arises from (NaN > threshold) being False."""
import json, os, numpy as np
ROOT = os.environ.get("THESIS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=os.path.join(ROOT,"outputs","inference_10m_candidate"); TD=os.path.join(OUT,"tiles")
LAYERS=["severity","persistence","recovery_gap","burned_prob","model_recurrence"]
TILE=256
exc=json.load(open(os.path.join(OUT,"QA_EXCEPTIONS.json")))["documented_exceptions"]
for e in exc:
    r,c=e["row"],e["col"]; tr,tc=TILE*(r//TILE),TILE*(c//TILE)
    p=os.path.join(TD,"t_%d_%d.npz"%(tr,tc))
    z=dict(np.load(p)); lr,lc=r-tr,c-tc
    before={L:float(z[L][lr,lc]) for L in LAYERS}
    for L in LAYERS: z[L][lr,lc]=np.nan
    tmp=p+".tmp.npz"; np.savez_compressed(tmp,**z); os.replace(tmp,p)
    print("exception applied at row %d col %d (tile %d_%d)"%(r,c,tr,tc))
    print("   before: %s"%{k:("nan" if np.isnan(v) else round(v,6)) for k,v in before.items()})
    print("   after : all five layers = NoData")
print("documented exceptions applied: %d"%len(exc))
