# Input tests for the demo, inside and outside its specification. Leaf images come from the VALIDATION split only.
import os, sys, numpy as np
from PIL import Image
os.environ["LEAFCHECK_MODEL"] = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "models", "leafcheck_final.keras")
import app
D = os.environ.get("DATA_DIR", "data")   # folder holding the four class folders
cases = {
 "valid: Blast leaf (JPG, 300x300)": Image.open(f"{D}/Blast/BLAST1_005.jpg"),
 "valid: Tungro leaf (non-square size)": Image.open(f"{D}/Tungro/TUNGRO1_003.jpg"),
 "edge: RGBA PNG": Image.open(f"{D}/Brownspot/BROWNSPOT1_003.jpg").convert("RGBA"),
 "edge: greyscale": Image.open(f"{D}/Bacterialblight/BACTERAILBLIGHT3_003.jpg").convert("L"),
 "edge: very large (4000x3000)": Image.open(f"{D}/Blast/BLAST1_005.jpg").resize((4000, 3000)),
 "edge: very dark photo": Image.fromarray((np.asarray(Image.open(f"{D}/Blast/BLAST1_005.jpg").convert("RGB")) * 0.15).astype(np.uint8)),
 "out of scope: random noise": Image.fromarray(np.random.RandomState(0).randint(0, 255, (300, 300, 3), dtype=np.uint8)),
 "out of scope: plain white image": Image.new("RGB", (300, 300), "white"),
 "invalid: no image": None,
 "invalid: 10x10 image": Image.new("RGB", (10, 10), "green"),
 "invalid: empty array": np.zeros((0, 0, 3), dtype=np.uint8),
}
for k, v in cases.items():
    try:
        verdict, probs, cam = app.predict(v)
        print(f"{k:42s} | {verdict[:95]} | heatmap={'yes' if cam is not None else 'no'}")
    except Exception as e:
        print(f"{k:42s} | CRASH: {type(e).__name__}: {e}")
