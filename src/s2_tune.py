# Reproduce R2/R4 on cached features, then new cached-feature tuning runs (validation set only).
from common import *
X, Y = load_arrays()
backbone, lower, upper = build_backbone()
t = time.time()
MID = {s: lower.predict(make_ds(X[s], bs=64).map(lambda b: pre(tf.cast(b, tf.float32))), verbose=0) for s in ["train", "val", "test"]}
np.save("outputs/MID_test.npy", MID["test"])   # stored for the single final evaluation only
MID.pop("test")
GAP = {s: upper.predict(MID[s], batch_size=64, verbose=0) for s in MID}
print("features", {s: MID[s].shape for s in MID}, f"{time.time()-t:.0f}s", flush=True)
upper_w0 = [w.copy() for w in upper.get_weights()]

def stage1(dr=0.5, lr=1e-3, bs=32, units=128, seed=SEED, name=None):
    keras.utils.set_random_seed(seed)
    m = make_head(dr, units); m.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    name = name or f"S1_units{units}_seed{seed}"
    h = m.fit(GAP["train"], Y["train"], validation_data=(GAP["val"], Y["val"]), epochs=40, batch_size=bs, callbacks=callbacks(name, 5), verbose=0)
    r = evaluate(Y["val"], m.predict(GAP["val"], verbose=0).argmax(1))
    return dict(name=name, units=units, seed=seed, epochs=len(h.history["loss"]), val_acc=r["acc"], val_f1=r["f1"], recall=r["recall"], model=m, hist=h.history)

def stage2(head_src, start="block_15", lr=1e-5, epochs=12, class_weight=None, seed=SEED, name="S2", units=128, dr=0.5, patience=3):
    keras.utils.set_random_seed(seed)
    upper.set_weights(upper_w0); nu = set_trainable_from(upper, start)
    head = make_head(dr, units); head.set_weights(head_src.get_weights())
    inp = keras.Input(MID["train"].shape[1:]); m = keras.Model(inp, head(upper(inp, training=False)))
    m.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    h = m.fit(MID["train"], Y["train"], validation_data=(MID["val"], Y["val"]), epochs=epochs, batch_size=32,
              class_weight=class_weight, callbacks=callbacks(name, patience), verbose=0)
    pred = m.predict(MID["val"], verbose=0).argmax(1); r = evaluate(Y["val"], pred)
    print(f"{name}: epochs={len(h.history['loss'])} val_acc={r['acc']:.4f} val_f1={r['f1']:.4f} recall={np.round(r['recall'],3)}", flush=True)
    return dict(name=name, unfrozen=nu, lr=lr, epochs=len(h.history["loss"]), val_acc=r["acc"], val_f1=r["f1"], recall=r["recall"].tolist(),
                pred=pred, hist=h.history, upper_w=[w.copy() for w in upper.get_weights()], head_w=head.get_weights())

res = {}
# --- reproduction of PR2 ---
b1 = stage1(name="R2_repro"); print(f"R2 repro: epochs={b1['epochs']} val_acc={b1['val_acc']:.4f} val_f1={b1['val_f1']:.4f}", flush=True)
res["R2_repro"] = b1
res["R4_repro"] = stage2(b1["model"], lr=1e-5, epochs=12, name="R4_repro")
# --- new runs ---
res["R6_long"] = stage2(b1["model"], lr=1e-5, epochs=30, patience=4, name="R6_lr1e-5_30ep")       # R4 was still improving at epoch 12
cw = {i: len(Y["train"]) / (4 * c) for i, c in enumerate(np.bincount(Y["train"]))}
print("class weights", {NAMES[k]: round(v, 3) for k, v in cw.items()})
res["R7_cw"] = stage2(b1["model"], lr=1e-5, epochs=30, patience=4, class_weight=cw, name="R7_classweights")
for u in [64, 256]:
    s = stage1(units=u, name=f"S1_units{u}")
    print(f"S1 units={u}: epochs={s['epochs']} val_f1={s['val_f1']:.4f}", flush=True)
    res[f"S1_units{u}"] = s
import pickle
keep = {k: {kk: vv for kk, vv in v.items() if kk not in ("model",)} for k, v in res.items()}
pickle.dump(keep, open("outputs/s2_results.pkl", "wb"))
pickle.dump({"b1_head": b1["model"].get_weights(), "cw": cw}, open("outputs/s2_heads.pkl", "wb"))
print("done")
