"""Build the two 10 m source layers the polygon workflow needs, directly on the
inference grid. Land cover is NATIVE 10 m (no aggregation of any kind); observed NBR
is the valid-only annual mean of the same monthly composites, at native 10 m.
Neither layer is derived from any 30 m product."""
import os, sys, time, json, warnings, numpy as np, rasterio, ee, google.auth
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")
ROOT = os.environ.get("THESIS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,ROOT)
import config
OUT=os.path.join(ROOT,"outputs","inference_10m_candidate")
X0,Y0,W,H=620610,3514560,4875,7794
TILE=1024
cred,_=google.auth.default(scopes=['https://www.googleapis.com/auth/earthengine',
                                   'https://www.googleapis.com/auth/cloud-platform'])
ee.Initialize(credentials=cred, project=config.GEE["project"],
              opt_url='https://earthengine-highvolume.googleapis.com')
roi=ee.FeatureCollection(config.GEE["roi"]).geometry()
lcc=ee.ImageCollection(config.GEE["landcover_yearly"])
s2 =ee.ImageCollection(config.GEE["ic_s2"])
CLEAR=None  # the monthly composites are already cloud-masked upstream (01_data_to_gee)

def grid_of(x0,y0,w,h):
    return {"crsCode":"EPSG:32636","affineTransform":{"scaleX":10,"shearX":0,"translateX":x0,
            "shearY":0,"scaleY":-10,"translateY":y0},"dimensions":{"width":w,"height":h}}
def compute(img,names,g,tries=6):
    for a in range(tries):
        try:
            arr=ee.data.computePixels({"expression":img,"fileFormat":"NUMPY_NDARRAY","grid":g})
            return np.stack([arr[n] for n in names],axis=0)
        except Exception as e:
            if a==tries-1: raise
            time.sleep(min(2**a,60))

lc2018=ee.Image(lcc.filter(ee.Filter.eq("system:index","2018")).first()).select("b1")
print("land cover native scale: %.1f m (no reduceResolution, no reprojection)"
      % lc2018.projection().nominalScale().getInfo(), flush=True)

def nbr_year(y):
    ic=s2.filter(ee.Filter.eq("year",y))
    def nbr(img): return img.normalizedDifference(["B8","B12"]).rename("NBR")
    m=ic.map(nbr)
    return m.mean().rename("NBR_%d_mean"%y), m.count().rename("n_valid_months_%d"%y)
NODATA=-9999.0
n17,c17=nbr_year(2017); n25,c25=nbr_year(2025)
# band order and nodata follow data/raw/nbr_2017_2025_30m.tif exactly, which is what
# 19_finalize_polygon_thesis_outputs.py reads as (OB17, OB25, NV17, NV25)
NBR=ee.Image.cat([n17.unmask(NODATA), n25.unmask(NODATA),
                  c17.unmask(0), c25.unmask(0)]).clip(roi).toFloat()
NBRB=["NBR_2017_mean","NBR_2025_mean","n_valid_months_2017","n_valid_months_2025"]

specs=[("landcover_2018_10m.tif",lc2018.clip(roi),["b1"],"int16",0),
       ("nbr_2017_2025_10m.tif",NBR,NBRB,"float32",-9999.0)]
for fname,img,bands,dt,nod in specs:
    p=os.path.join(OUT,fname)
    if os.path.exists(p): print("exists, skipping:",fname, flush=True); continue
    prof=dict(driver="GTiff",height=H,width=W,count=len(bands),dtype=dt,crs="EPSG:32636",
              transform=rasterio.Affine(10,0,X0,0,-10,Y0),nodata=nod,tiled=True,
              blockxsize=256,blockysize=256,compress="deflate")
    tiles=[(r,c,min(TILE,H-r),min(TILE,W-c)) for r in range(0,H,TILE) for c in range(0,W,TILE)]
    t0=time.time()
    with rasterio.open(p,"w",**prof) as d:
        def job(t):
            r,c,h,w=t
            a=compute(img,bands,grid_of(X0+c*10,Y0-r*10,w,h))
            return t,a
        with ThreadPoolExecutor(max_workers=8) as ex:
            for (r,c,h,w),a in ex.map(job,tiles):
                if dt=="int16": a=np.nan_to_num(a,nan=0).astype(np.int16)
                else: a=a.astype(np.float32)
                for b in range(len(bands)): d.write(a[b],b+1,window=Window(c,r,w,h))
        d.descriptions=tuple(bands)
    print("wrote %s  (%d tiles, %.1f min)"%(fname,len(tiles),(time.time()-t0)/60), flush=True)
    with rasterio.open(p) as s:
        print("   %dx%d %s transform %s"%(s.width,s.height,s.crs,tuple(round(v,1) for v in s.transform[:6])), flush=True)
print("10 m sources complete", flush=True)
