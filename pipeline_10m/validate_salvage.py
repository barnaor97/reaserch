"""Validate SALVAGED tile values, not just their footprints.

The 423 salvaged tiles came out of a mosaic left uncleanly closed by kill -9. This selects a
reproducible sample spanning the ROI, recomputes each one from the original 10 m GEE inputs using
the FINAL patched inference code, and requires bit-identical agreement on all five layers.

It drives the real inference script (not a reimplementation) by removing the sampled tiles from the
manifest and letting run_inference_10m.py recompute exactly those. Passing tiles are left in place.
"""
import json, os, shutil, subprocess, sys
import numpy as np, rasterio
from rasterio.windows import Window
ROOT = os.environ.get("THESIS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=os.path.join(ROOT,"outputs","inference_10m_candidate")
TD=os.path.join(OUT,"tiles"); BK=os.path.join(OUT,"_salvage_check"); os.makedirs(BK,exist_ok=True)
PY="python"
LAYERS=["severity","persistence","recovery_gap","burned_prob","model_recurrence"]
GW,GH=4875,7794
# Unaffected set = every tile that did NOT contain a cell with a missing month. Recomputing
# these under the isfinite fix must reproduce the stored values bit-for-bit, which demonstrates
# that the fix repairs only the affected cells and changes nothing else.
aff=set(json.load(open(os.path.join(OUT,"affected_tiles.json"))))
SAL=[k for k in json.load(open(os.path.join(OUT,"manifest.json")))["done"] if k not in aff]
print("unaffected tiles available for value validation: %d"%len(SAL))
man=json.load(open(os.path.join(OUT,"manifest.json"))); TILE=man["tile"]
with rasterio.open(os.path.join(ROOT,"data","raw","burn_count_with_mask_10m.tif")) as s:
    eco=s.read(2,window=Window(0,0,GW,GH),boundless=True,fill_value=0)==1

cand=[]
for k in SAL:
    r,c=map(int,k.split("_")); h=min(TILE,GH-r); w=min(TILE,GW-c)
    n=int(eco[r:r+h,c:c+w].sum()); cand.append((k,r,c,h,w,n,n/float(h*w)))
full=[x for x in cand if x[6]>=0.999]
part=[x for x in cand if 0.05<x[6]<0.999]
# deterministic, ROI-spanning selection: sort by (row, col) and take a fixed stride
full.sort(key=lambda x:(x[1],x[2])); part.sort(key=lambda x:(x[1],x[2]))
pick=[]
if full: pick += [full[0], full[len(full)//2], full[-1]]
if part: pick += [part[i] for i in (0, len(part)//4, len(part)//2, 3*len(part)//4, len(part)-1)]
seen=set(); SAMPLE=[]
for x in pick:
    if x[0] not in seen: seen.add(x[0]); SAMPLE.append(x)
print("salvaged tiles: %d | fully eligible: %d | partially eligible: %d"%(len(cand),len(full),len(part)))
print("reproducible sample (%d tiles, sorted by grid position, fixed stride):"%len(SAMPLE))
for k,r,c,h,w,n,frac in SAMPLE:
    print("   tile %-11s row %-5d col %-5d  %dx%d  eligible %6d (%.1f%% of tile)"%(k,r,c,h,w,n,100*frac))

for k,_,_,_,_,_,_ in SAMPLE:
    shutil.copy(os.path.join(TD,"t_%s.npz"%k), os.path.join(BK,"t_%s.npz"%k))
    os.remove(os.path.join(TD,"t_%s.npz"%k)); man["done"].pop(k,None)
json.dump(man, open(os.path.join(OUT,"manifest.json"),"w"))
print("\nrecomputing %d sampled tiles with the final patched code ..."%len(SAMPLE))
r=subprocess.run([PY,"-u",os.path.join(ROOT,"notebooks","prod10m","run_inference_10m.py")],
                 capture_output=True,text=True,cwd=ROOT)
if r.returncode!=0:
    print(r.stdout[-2000:]); print(r.stderr[-2000:]); sys.exit(1)
print("   "+ "\n   ".join(l for l in r.stdout.strip().split("\n") if "tile" in l or "COMPLETE" in l))

ok=True
print("\n%-12s %-18s %-10s %-12s %s"%("tile","layer","mask equal","max |diff|","verdict"))
for k,r_,c_,h,w,n,_ in SAMPLE:
    a=np.load(os.path.join(BK,"t_%s.npz"%k)); b=np.load(os.path.join(TD,"t_%s.npz"%k))
    for L in LAYERS:
        x,y=a[L],b[L]
        meq=bool(np.array_equal(np.isnan(x),np.isnan(y)))
        m=~np.isnan(x)
        md=float(np.abs(x[m]-y[m]).max()) if m.any() else 0.0
        good=meq and md==0.0
        ok&=good
        print("%-12s %-18s %-10s %-12.3e %s"%(k,L,meq,md,"PASS" if good else "FAIL"))
print("\nUNAFFECTED-TILE VALUE VALIDATION: %s"%("PASS — unaffected tiles are bit-identical under the fixed code" if ok else "FAIL"))
json.dump({"sample":[s[0] for s in SAMPLE],"pass":bool(ok)}, open(os.path.join(OUT,"SALVAGE_VALIDATION.json"),"w"), indent=1)
sys.exit(0 if ok else 1)
