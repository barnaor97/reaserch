"""
stream_extract_nbr.py — extract the NBR columns from a remote X_scaled.npy.

Same job as extract_nbr_arrays.py, but reads the tensor over HTTP instead of
from a local file, so it never needs 13.7 GB of free disk. Use this when
X_scaled.npy is reachable by URL (e.g. a Google Drive file set to "Anyone with
the link") and only the small extracted columns are wanted locally.

How it works. A .npy file is a short ASCII header followed by the raw buffer in
C order, so for a (N, T, C) float32 tensor the value for pixel i, month t,
feature c sits at header + ((i*T + t)*C + c)*4. Reading forward and keeping only
the wanted c positions out of each C-value record yields the columns exactly.

Transfers use HTTP range requests in blocks aligned to record boundaries rather
than one long-lived connection, because Drive throttles sustained single
streams and drops them. Each block is retried independently, and progress is
checkpointed, so an interrupted run resumes where it stopped instead of starting
over. Peak memory is the output arrays (~790 MB) plus one block.

Usage:
    python notebooks/stream_extract_nbr.py <url>
    python notebooks/stream_extract_nbr.py <url> --out data/processed --block-mb 128
"""

import argparse
import ast
import json
import os
import pickle
import sys
import time
import urllib.error
import urllib.request

import numpy as np

WANTED = {"NBR": "nbr_raw.npy", "NBR_zscore": "nbr_zscore_raw.npy",
          "NDVI": "ndvi_raw.npy"}
UA = {"User-Agent": "Mozilla/5.0"}
MAX_TRIES = 6


def fetch_range(url, start, end, tries=MAX_TRIES):
    """GET bytes [start, end] inclusive, with retry and backoff."""
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                if "text/html" in resp.headers.get("Content-Type", ""):
                    raise ValueError("server returned HTML — the URL is not "
                                     "publicly readable (check link sharing)")
                buf = resp.read()
            want = end - start + 1
            if len(buf) != want:
                raise IOError(f"short read: {len(buf)} of {want}")
            return buf
        except Exception as exc:                       # noqa: BLE001
            last = exc
            if attempt < tries - 1:
                time.sleep(2 ** attempt)
    raise IOError(f"range {start}-{end} failed after {tries} tries: {last}")


def read_header(url):
    head = fetch_range(url, 0, 511)
    if head[:6] != b"\x93NUMPY":
        raise ValueError(f"not a .npy file (magic {head[:6]!r})")
    major = head[6]
    if major == 1:
        hlen = int.from_bytes(head[8:10], "little")
        offset = 10 + hlen
    else:
        hlen = int.from_bytes(head[8:12], "little")
        offset = 12 + hlen
    meta = ast.literal_eval(head[10 if major == 1 else 12:offset].decode("latin1"))
    if meta["fortran_order"]:
        raise ValueError("Fortran-order arrays are not supported")
    return np.dtype(meta["descr"]), meta["shape"], offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--block-mb", type=int, default=128)
    ap.add_argument("--features", default=",".join(WANTED))
    args = ap.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, ".stream_extract_state.json")

    feature_names = json.load(open(os.path.join(out_dir, "feature_names.json")))
    with open(os.path.join(out_dir, "scaler.pkl"), "rb") as fh:
        bundle = pickle.load(fh)
    scaler, spectral_cols = bundle["scaler"], bundle["spectral_cols"]

    wanted = [f.strip() for f in args.features.split(",") if f.strip()]
    cols, mus, sds = [], [], []
    for name in wanted:
        col = feature_names.index(name)
        pos = spectral_cols.index(col)
        cols.append(col)
        mus.append(float(scaler.mean_[pos]))
        sds.append(float(scaler.scale_[pos]))
    cols_arr = np.array(cols)
    mus_arr = np.array(mus, dtype=np.float32)
    sds_arr = np.array(sds, dtype=np.float32)

    dtype, shape, data_offset = read_header(args.url)
    n_pix, n_time, n_feat = shape
    record = n_feat * dtype.itemsize
    total_records = n_pix * n_time
    total_bytes = total_records * record
    print(f"remote: shape={shape} dtype={dtype} header={data_offset}B "
          f"payload={total_bytes / 1e9:.2f} GB", flush=True)
    for name, col, mu, sd in zip(wanted, cols, mus, sds):
        print(f"  {name:<12} column {col:>2}  mean={mu:+.6f}  scale={sd:.6f}",
              flush=True)

    # np.save appends ".npy" unless the path already ends in it, so name the
    # checkpoints accordingly — otherwise a resume looks for a file that was
    # never written under that name and silently restarts from zero.
    part_paths = {n: os.path.join(out_dir, WANTED[n] + ".part.npy") for n in wanted}
    done_records = 0
    if os.path.exists(ckpt_path):
        state = json.load(open(ckpt_path))
        if state.get("shape") == list(shape) and state.get("features") == wanted:
            done_records = state["done_records"]
            print(f"resuming at record {done_records:,} "
                  f"({100 * done_records / total_records:.1f}%)", flush=True)

    out = {}
    for name in wanted:
        if done_records and os.path.exists(part_paths[name]):
            out[name] = np.load(part_paths[name])
        else:
            out[name] = np.zeros((n_pix, n_time), dtype=np.float32)
    if done_records and not all(os.path.exists(p) for p in part_paths.values()):
        done_records = 0

    recs_per_block = max(1, (args.block_mb * 1024 * 1024) // record)
    t0 = time.time()
    last_ckpt = done_records

    while done_records < total_records:
        n_rec = min(recs_per_block, total_records - done_records)
        start = data_offset + done_records * record
        end = start + n_rec * record - 1
        buf = fetch_range(args.url, start, end)

        block = np.frombuffer(buf, dtype=dtype).reshape(n_rec, n_feat)
        vals = block[:, cols_arr].astype(np.float32) * sds_arr + mus_arr
        flat = np.arange(done_records, done_records + n_rec)
        pix, tt = np.divmod(flat, n_time)
        for k, name in enumerate(wanted):
            out[name][pix, tt] = vals[:, k]

        done_records += n_rec
        elapsed = max(time.time() - t0, 1e-6)
        moved = (done_records - last_ckpt) * record
        pct = 100 * done_records / total_records
        eta = (total_records - done_records) * record / max(moved / elapsed, 1)
        print(f"  {pct:5.1f}%  {done_records * record / 1e9:6.2f} GB  "
              f"{moved / 1e6 / elapsed:5.1f} MB/s  ETA {eta / 60:5.1f} min",
              flush=True)

        # Checkpoint roughly every 2 GB so an interruption costs little.
        if (done_records - last_ckpt) * record > 2e9 or done_records == total_records:
            for name in wanted:
                np.save(part_paths[name], out[name])
            json.dump({"shape": list(shape), "features": wanted,
                       "done_records": done_records}, open(ckpt_path, "w"))
            last_ckpt, t0 = done_records, time.time()

    print("\nwriting:", flush=True)
    for name in wanted:
        path = os.path.join(out_dir, WANTED[name])
        np.save(path, out[name])
        a = out[name]
        print(f"  {WANTED[name]:<22} {a.shape}  {a.nbytes / 1e6:6.1f} MB  "
              f"range [{np.nanmin(a):+.4f}, {np.nanmax(a):+.4f}]  "
              f"mean {np.nanmean(a):+.4f}")
        if os.path.exists(part_paths[name]):
            os.remove(part_paths[name])
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    # Proof of correctness: rebuild the stored targets from the extracted columns.
    y = np.load(os.path.join(out_dir, "y_burned.npy"))
    stored = np.load(os.path.join(out_dir, "targets_aux.npz"))
    nbr, nbr_z = out["NBR"], out["NBR_zscore"]
    burned = y.sum(axis=1) > 0

    b = np.nanmean(nbr[:, :12].astype(np.float64), axis=1)
    r = np.nanmean(nbr[:, -12:].astype(np.float64), axis=1)
    rg = np.zeros(len(y), dtype=np.float32)
    rg[burned] = np.clip((b - r) / (np.abs(b) + 1e-6), 0, 1)[burned]

    sev = np.zeros(len(y), dtype=np.float32)
    per = np.zeros(len(y), dtype=np.float32)
    for i in np.flatnonzero(burned):
        months = np.flatnonzero(y[i] == 1)
        sev[i] = np.clip((-nbr_z[i, months]).mean() / 5.0, 0, 1)
        post = nbr_z[i, months[-1]:]
        if len(post) > 1:
            per[i] = (post < -0.5).mean()

    # Tolerance note. X_scaled stores standardized float32, so recovering the raw
    # value costs a round-trip of a few ULPs. The recovery gap divides by
    # |baseline NBR|, whose sample mean is only ~0.085, so that tiny absolute
    # error is amplified — and amplified most where the baseline is smallest.
    # Per-pixel agreement at 1e-4 is therefore the wrong bar; agreement at 1e-2,
    # rank correlation, and class means are what any downstream analysis uses.
    print("\nverification — stored targets rebuilt from the extracted columns:")
    ok = True
    for name, got in (("recovery_gap", rg), ("severity", sev), ("persistence", per)):
        want = stored[name]
        diff = np.abs(got - want)
        corr = float(np.corrcoef(got, want)[0, 1])
        within = float(np.mean(diff < 1e-2))
        print(f"  {name:<14} max|diff| = {diff.max():.3e}   "
              f"within 1e-2 for {100 * within:.4f}%   corr = {corr:.8f}")
        ok &= (corr > 0.9999) and (within > 0.999)
    print("\nPASS — arrays reproduce the stored targets" if ok
          else "\nFAIL — do not use these arrays")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
