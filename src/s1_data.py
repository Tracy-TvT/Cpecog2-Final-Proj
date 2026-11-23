# Cells 1-4 of Rice_Leaf_Progress_Report_2.ipynb, unchanged except DATA_DIR; also builds the test array.
import os, hashlib, time, json, random
import numpy as np, pandas as pd
from PIL import Image
import imagehash
SEED = 42
random.seed(SEED); np.random.seed(SEED)
DATA_DIR = os.environ.get("DATA_DIR", "data")   # folder holding Bacterialblight/, Blast/, Brownspot/, Tungro/
CLASSES = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]
NAMES = ["Bacterial blight", "Blast", "Brown spot", "Tungro"]
IMG = 224
os.makedirs("outputs", exist_ok=True)
records, corrupted = [], []
for ci, c in enumerate(CLASSES):
    for f in sorted(os.listdir(os.path.join(DATA_DIR, c))):
        p = os.path.join(DATA_DIR, c, f)
        try:
            with Image.open(p) as im: im.verify()
            with Image.open(p) as im: mode, size = im.mode, im.size
            md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
            records.append(dict(path=p, label=ci, mode=mode, w=size[0], h=size[1], md5=md5))
        except Exception as e:
            corrupted.append(p)
raw = pd.DataFrame(records)
print("Raw files:", len(raw) + len(corrupted), "| corrupted:", len(corrupted))
groups = raw.groupby("md5")["label"].nunique()
conflict_md5 = set(groups[groups > 1].index)
clean = raw[~raw.md5.isin(conflict_md5)].drop_duplicates("md5").reset_index(drop=True)
print("Exact duplicates removed:", len(raw) - len(raw.drop_duplicates("md5")), "| conflicts:", len(conflict_md5))
print("Unique files kept:", len(clean), "| RGBA:", int((clean["mode"] == "RGBA").sum()))
def load_rgb(p):
    with Image.open(p) as im: return im.convert("RGB")
def dihedral(im):
    out = []
    for k in range(4):
        r = im.rotate(90 * k, expand=True); out += [r, r.transpose(Image.FLIP_LEFT_RIGHT)]
    return out
def hashes(p):
    im = load_rgb(p).resize((64, 64))
    return [int(str(imagehash.phash(t)), 16) for t in dihedral(im)]
t = time.time()
H = np.array([hashes(p) for p in clean.path], dtype=np.uint64); base = H[:, 0]
def popcount(x):
    x = x.astype(np.uint64); c = np.zeros(x.shape, dtype=np.uint8)
    for s in range(0, 64, 8):
        c += np.unpackbits(((x >> np.uint64(s)) & np.uint64(255)).astype(np.uint8)[..., None], axis=-1).sum(-1).astype(np.uint8)
    return c
THRESH = 6; n = len(clean); parent = np.arange(n)
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
for i in range(n):
    d = popcount(H[i][:, None] ^ base[None, :]).min(0)
    for j in np.where(d <= THRESH)[0]:
        if j > i:
            a, b = find(i), find(j)
            if a != b: parent[b] = a
clean["cluster"] = [find(i) for i in range(n)]
cl = clean.groupby("cluster")["label"].agg(lambda s: s.mode()[0])
mixed = int((clean.groupby("cluster")["label"].nunique() > 1).sum())
print("Near-duplicate clusters:", clean.cluster.nunique(), "| mixed:", mixed, f"| {time.time()-t:.0f}s")
rng = np.random.RandomState(SEED); clean["split"] = ""
for c in range(4):
    cids = cl[cl == c].index.values.copy(); rng.shuffle(cids)
    sizes = clean[clean.cluster.isin(cids)].groupby("cluster").size()
    total, run = sizes.sum(), 0
    for cid in cids:
        frac = run / total
        s = "train" if frac < 0.70 else ("val" if frac < 0.85 else "test")
        clean.loc[clean.cluster == cid, "split"] = s; run += sizes[cid]
split_table = clean.groupby("split").agg(clusters=("cluster", "nunique"), images=("path", "size")).loc[["train", "val", "test"]]
print(split_table)
print(pd.crosstab(clean.split, clean.label.map(dict(enumerate(NAMES)))))
assert clean.groupby("cluster").split.nunique().max() == 1
clean.to_csv("outputs/splits.csv", index=False)
def load_resized(p):
    return np.asarray(load_rgb(p).resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8)
arr = {}
for s in ["train", "val", "test"]:
    d = clean[clean.split == s]
    arr["X_" + s] = np.stack([load_resized(p) for p in d.path]); arr["Y_" + s] = d.label.values
np.savez("outputs/arrays.npz", **arr)
print({k: v.shape for k, v in arr.items()})
