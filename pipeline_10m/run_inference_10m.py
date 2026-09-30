"""Full-ROI 10 m inference, streamed and resumable.

GEE source assets -> computePixels tile -> 108x52 sequences -> existing scaler ->
existing L-TAE checkpoint -> five prediction rasters. Nothing is exported to Drive,
nothing becomes a GEE asset, nothing large is staged on disk.

Feature construction is character-identical to notebooks/07a (build_monthly_feature_image);
the only change is transport: the 20 land-cover one-hot bands are constant within a year,
so the categorical class raster is fetched once per year per tile and one-hot encoded
locally. Verified bit-exact against the fetched bands on the pilot tile.
"""
import os, sys, json, time, hashlib, threading, warnings, socket
socket.setdefaulttimeout(300)   # no request may hang forever
import numpy as np, torch, rasterio, ee, google.auth
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")
ROOT = os.environ.get("THESIS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import config
from model_defs import FireRecoveryModel

OUT   = os.path.join(ROOT, "outputs", "inference_10m_candidate")
GRID  = dict(crs="EPSG:32636", x0=620610, y0=3514560, W=4875, H=7794, res=10)
TILE  = 256
WORKERS = int(os.environ.get("WORKERS","24"))
LAYERS = ["severity", "persistence", "recovery_gap", "burned_prob", "model_recurrence"]
os.makedirs(OUT, exist_ok=True)
MAN = os.path.join(OUT, "manifest.json")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

CKPT = os.path.join(ROOT, "models/main/model_main.pt")
SCAL = os.path.join(ROOT, "data/processed/scaler.pkl")
ECO  = os.path.join(ROOT, "data/raw/burn_count_with_mask_10m.tif")

cred, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/earthengine',
                                      'https://www.googleapis.com/auth/cloud-platform'])
ee.Initialize(credentials=cred, project=config.GEE["project"],
              opt_url='https://earthengine-highvolume.googleapis.com')
s2    = ee.ImageCollection(config.GEE["ic_s2"])
bmean = ee.ImageCollection(config.GEE["ic_baseline_mean"])
bstd  = ee.ImageCollection(config.GEE["ic_baseline_std"])
lcc   = ee.ImageCollection(config.GEE["landcover_yearly"])
roi   = ee.FeatureCollection(config.GEE["roi"]).geometry()
MONTHS = config.get_month_list(); T = len(MONTHS)
GEE_FEATURES = [f for f in config.MODEL_FEATURES
                if f not in config.DELTA_FEATURES and f not in config.HARMONIC_FEATURES]
SPEC = [f for f in GEE_FEATURES if not f.startswith(("lc_prev", "lc_curr"))]
NF = len(config.MODEL_FEATURES)

def add_indices(i):                                        # verbatim, 07a cell 13
    eps = ee.Image.constant(1e-6)
    b4, b8, b8a = i.select("B4"), i.select("B8"), i.select("B8A")
    b11, b12 = i.select("B11"), i.select("B12")
    ndvi = i.normalizedDifference(["B8","B4"]).rename("NDVI")
    ndmi = i.normalizedDifference(["B8","B11"]).rename("NDMI")
    nbr  = i.normalizedDifference(["B8","B12"]).rename("NBR")
    evi  = (b8.subtract(b4).multiply(2.5).divide(b8.add(b4.multiply(2.4)).add(1).add(eps))).rename("EVI")
    mirbi= (b12.multiply(10).subtract(b11.multiply(9.8)).add(2)).rename("MIRBI")
    bsi  = ((b11.add(b4).subtract(b8.add(b8a))).divide(b11.add(b4).add(b8).add(b8a).add(eps))).rename("BSI")
    t1 = ee.Image(1).subtract(b11.multiply(b12).multiply(b8a).divide(b4.add(eps)).sqrt())
    t2 = b12.subtract(b8a).divide(b12.add(b8a).add(eps).sqrt()).add(1)
    return i.select(config.RAW_BANDS).addBands([ndvi,evi,ndmi,nbr,mirbi,bsi,
                                                t1.multiply(t2).rename("BAIS2")]).toFloat()

def spectral_image(y, m):
    wi = add_indices(ee.Image(s2.filter(ee.Filter.eq("year", y))
                                .filter(ee.Filter.eq("month", m)).first()))
    bm = ee.Image(bmean.filter(ee.Filter.eq("month", m)).first())
    bs = ee.Image(bstd.filter(ee.Filter.eq("month", m)).first())
    o = wi.select(config.RAW_BANDS + config.INDICES)
    for b in config.ANOMALY_SOURCE_INDICES:
        o = o.addBands(wi.select(b).subtract(bm.select(b)).divide(bs.select(b).max(1e-6)).rename(b+"_zscore"))
        o = o.addBands(wi.select(b).subtract(bm.select(b)).rename(b+"_anomaly"))
    return o.select(SPEC).clip(roi).toFloat()

def lc_image(year):
    return ee.Image(lcc.filter(ee.Filter.eq("system:index", str(max(year, 2018)))).first()).select("b1")

def grid_of(x0, y0, w, h):
    return {"crsCode": GRID["crs"],
            "affineTransform": {"scaleX":10,"shearX":0,"translateX":x0,
                                "shearY":0,"scaleY":-10,"translateY":y0},
            "dimensions": {"width": w, "height": h}}

def compute(img, names, g, tries=6):
    for a in range(tries):
        try:
            arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": g})
            return np.stack([arr[n] for n in names], axis=0).astype(np.float32)
        except Exception as e:
            if a == tries - 1: raise
            print("      retry %d after %s" % (a+1, str(e)[:70]), flush=True)
            time.sleep(min(2 ** a, 60))                     # conservative backoff

# ---- outputs: preallocated full-size rasters, windowed writes, no mosaic step
TD = os.path.join(OUT, "tiles"); os.makedirs(TD, exist_ok=True)
# Per-tile outputs are written as standalone files and mosaicked in a single pass afterwards.
# Incremental in-place updates of a compressed GeoTIFF are not crash-safe: a hard kill during
# Write each tile to a temporary file and rename it into place, so an interrupted
# run cannot leave a partially written tile behind.

man = json.load(open(MAN)) if os.path.exists(MAN) else {
    "grid": GRID, "tile": TILE, "months": T, "features": NF,
    "checkpoint_sha256": sha(CKPT), "scaler_sha256": sha(SCAL),
    "gee_features": GEE_FEATURES, "model_features": config.MODEL_FEATURES,
    "threshold": 0.502, "done": {}}
assert man["checkpoint_sha256"] == sha(CKPT), "checkpoint changed mid-run"
assert man["scaler_sha256"] == sha(SCAL), "scaler changed mid-run"

import pickle
sb = pickle.load(open(SCAL, "rb")); scaler = sb["scaler"]; SPEC_COLS = np.array(sb["spectral_cols"])
model = FireRecoveryModel(n_features=NF, d_model=config.MODEL["d_model"], n_head=config.MODEL["n_head"],
                          d_k=config.MODEL["d_k"], max_len=T, tau=config.MODEL["positional_tau"], dropout=0.0)
model.load_state_dict(torch.load(CKPT, map_location="cpu")); model.eval()
cal = np.array([m for _, m in MONTHS], dtype=np.float32)
MCOS, MSIN = np.cos(2*np.pi*cal/12.), np.sin(2*np.pi*cal/12.)
DSRC = {b: SPEC.index(b) for b in config.DELTA_SOURCE_INDICES}
DTGT = {b: config.MODEL_FEATURES.index("delta_"+b) for b in config.DELTA_SOURCE_INDICES}
STGT = np.array([config.MODEL_FEATURES.index(n) for n in SPEC])
LPREV = [config.MODEL_FEATURES.index(n) for n in config.LC_FEATURES_PREV]
LCURR = [config.MODEL_FEATURES.index(n) for n in config.LC_FEATURES_CURR]

tiles = [(r, c, min(TILE, GRID["H"]-r), min(TILE, GRID["W"]-c))
         for r in range(0, GRID["H"], TILE) for c in range(0, GRID["W"], TILE)]
todo = [t for t in tiles if "%d_%d" % (t[0], t[1]) not in man["done"]]
LIMIT = int(os.environ.get("TILE_LIMIT", "0"))
if LIMIT: todo = todo[:LIMIT]
print("tiles total %d | done %d | remaining %d" % (len(tiles), len(man["done"]), len(todo)), flush=True)
lock = threading.Lock(); t_start = time.time()

for n, (r, c, h, w) in enumerate(todo, 1):
    x0 = GRID["x0"] + c*10; y0 = GRID["y0"] - r*10
    g = grid_of(x0, y0, w, h)
    # The eligibility mask is a local 10 m raster, so tiles with no eligible cell are
    # skipped before any data is requested. They stay NaN, exactly as initialised.
    with rasterio.open(ECO) as s:
        eco_pre = s.read(2, window=Window(c, r, w, h), boundless=True, fill_value=0)
    if not (eco_pre == 1).any():
        man["done"]["%d_%d" % (r, c)] = {"h": h, "w": w, "eligible": 0, "skipped": "no eligible cell"}
        json.dump(man, open(MAN, "w"))
        continue
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        spec = list(ex.map(lambda i: compute(spectral_image(*MONTHS[i]), SPEC, g), range(T)))
        _yrs = sorted({max(yy,2018) for y_, _ in MONTHS for yy in (y_-1, y_)})
        lcy  = dict(zip(_yrs, ex.map(lambda y: compute(lc_image(y), ["b1"], g)[0], _yrs)))
    # Memory: the full-tile tensor is never materialised. Eligibility is resolved on the
    # spectral stack first, and only eligible pixels are expanded to the 52-feature tensor.
    S = np.stack(spec, 0); del spec                                  # (T, 26, h, w)
    N = h*w
    Xg = S.transpose(2,3,0,1).reshape(N, T, len(SPEC)).copy(); del S
    # A month whose composite does not cover the pixel comes back from computePixels as -inf,
    # not NaN. isnan() misses that, so such months were being marked VALID and -inf reached the
    # network as -3.4e38 via nan_to_num. Validity must therefore be tested with isfinite.
    valid_all = np.isfinite(Xg).all(axis=-1)
    elig = valid_all.any(axis=1) & (eco_pre.reshape(-1) == 1)
    ne = int(elig.sum())
    out = {L: np.full(N, np.nan, np.float32) for L in LAYERS}
    if ne:
        Xg = Xg[elig]; Me = valid_all[elig]
        del valid_all
        X = np.full((ne, T, NF), np.nan, dtype=np.float32)
        X[:, :, STGT] = Xg
        for b in config.DELTA_SOURCE_INDICES:
            v = Xg[:, :, DSRC[b]]; X[:, 1:, DTGT[b]] = v[:, :-1] - v[:, 1:]
        X[:, :, config.MODEL_FEATURES.index("month_cos")] = MCOS[None, :]
        X[:, :, config.MODEL_FEATURES.index("month_sin")] = MSIN[None, :]
        ei = np.nonzero(elig)[0]
        for t, (yy, _) in enumerate(MONTHS):
            pv = lcy[max(yy-1, 2018)].reshape(-1)[ei]; cu = lcy[max(yy, 2018)].reshape(-1)[ei]
            for k, code in enumerate(config.LC_CODES):
                X[:, t, LPREV[k]] = (pv == code); X[:, t, LCURR[k]] = (cu == code)
        del Xg, ei
        np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        fl = X.reshape(-1, NF); fl[:, SPEC_COLS] = scaler.transform(fl[:, SPEC_COLS])
        Xs = X
        buf = {L: np.zeros(ne, np.float32) for L in LAYERS}
        with torch.no_grad():
            for i in range(0, ne, 8192):
                j = min(i+8192, ne)
                o = model(torch.from_numpy(Xs[i:j]), torch.from_numpy(Me[i:j]))
                buf["severity"][i:j]     = o["severity"].numpy()
                buf["persistence"][i:j]  = o["persistence"].numpy()
                buf["recovery_gap"][i:j] = o["recovery_gap"].numpy()
                bp = torch.sigmoid(o["burned_logits"]).numpy()
                buf["burned_prob"][i:j]  = bp.max(axis=1)
                buf["model_recurrence"][i:j] = np.clip((bp > 0.502).sum(axis=1)/5.0, 0, 1)
        for L in LAYERS: out[L][elig] = buf[L]
        del Xs, buf, fl, X, Me
    else:
        del Xg, valid_all
    tmp = os.path.join(TD, "_t_%d_%d.npz" % (r, c))
    np.savez_compressed(tmp, **{L: out[L].reshape(h, w) for L in LAYERS})
    os.replace(tmp, os.path.join(TD, "t_%d_%d.npz" % (r, c)))       # atomic publish
    man["done"]["%d_%d" % (r, c)] = {"h": h, "w": w, "eligible": ne, "months_fetched": T}
    with lock: json.dump(man, open(MAN, "w"))
    import gc; gc.collect()
    el = time.time()-t_start
    print("tile %4d/%4d  r%-5d c%-5d  elig %7d  %5.1f s/tile  elapsed %5.1f min  ETA %5.1f min"
          % (n, len(todo), r, c, ne, el/n, el/60, (len(todo)-n)*el/n/60), flush=True)
print("INFERENCE COMPLETE in %.1f min" % ((time.time()-t_start)/60), flush=True)
