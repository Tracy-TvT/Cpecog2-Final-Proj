# Seed variation (validation), final model selection (validation), then ONE evaluation on the held-out test set.
import pickle, glob
from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
X, Y = load_arrays()
backbone, lower, upper = build_backbone()
upper_w0 = [w.copy() for w in upper.get_weights()]
heads = pickle.load(open("outputs/s2_heads.pkl", "rb")); cw = heads["cw"]
MID = {s: lower.predict(make_ds(X[s], bs=64).map(lambda b: pre(tf.cast(b, tf.float32))), verbose=0) for s in ["train", "val"]}
GAP = {s: upper.predict(MID[s], batch_size=64, verbose=0) for s in MID}

# ---- seed variation of the R7 configuration (validation only) ----
def r7_with_seed(seed):
    keras.utils.set_random_seed(seed)
    h1 = make_head(0.5); h1.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy")
    h1.fit(GAP["train"], Y["train"], validation_data=(GAP["val"], Y["val"]), epochs=40, batch_size=32, callbacks=callbacks(f"seed{seed}_s1", 5), verbose=0)
    upper.set_weights(upper_w0); set_trainable_from(upper, "block_15")
    inp = keras.Input(MID["train"].shape[1:]); m = keras.Model(inp, h1(upper(inp, training=False)))
    m.compile(keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy")
    m.fit(MID["train"], Y["train"], validation_data=(MID["val"], Y["val"]), epochs=30, batch_size=32, class_weight=cw, callbacks=callbacks(f"seed{seed}_s2", 4), verbose=0)
    r = evaluate(Y["val"], m.predict(MID["val"], verbose=0).argmax(1))
    print(f"R7 config seed {seed}: val_acc={r['acc']:.4f} val_f1={r['f1']:.4f} blast_recall={r['recall'][1]:.3f}", flush=True)
    return dict(seed=seed, val_acc=r["acc"], val_f1=r["f1"], recall=r["recall"].tolist())
seeds = [r7_with_seed(s) for s in [1, 2]]
pickle.dump(seeds, open("outputs/seeds.pkl", "wb"))

# ---- final model selection on validation macro F1 ----
s2 = pickle.load(open("outputs/s2_results.pkl", "rb"))
cands = {"R7_classweights": s2["R7_cw"]}
for f in glob.glob("outputs/R_aug_*.pkl"):
    d = pickle.load(open(f, "rb")); cands[d["name"]] = d
for k, v in cands.items(): print("candidate", k, round(v["val_f1"], 4))
final_name = max(cands, key=lambda k: cands[k]["val_f1"]); fin = cands[final_name]
print("FINAL MODEL:", final_name, flush=True)
upper.set_weights(fin["upper_w"])
head = make_head(0.5); head.set_weights(fin["head_w"])
inp = keras.Input((IMG, IMG, 3))
x = layers.Rescaling(1 / 127.5, offset=-1.0, name="mnv2_preprocess")(inp)   # identical to mobilenet_v2.preprocess_input
full = keras.Model(inp, head(backbone(x, training=False)), name="leafcheck_mnv2")
# sanity: full model reproduces the validation result
pv = full.predict(make_ds(X["val"], bs=64), verbose=0)
print("full-model val macro F1:", round(f1_score(Y["val"], pv.argmax(1), average="macro"), 4))
full.save("outputs/leafcheck_final.keras")

# ---- uncertainty threshold chosen on validation ----
conf_v = pv.max(1); corr_v = pv.argmax(1) == Y["val"]
thr_rows = []
for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
    keep = conf_v >= t
    thr_rows.append(dict(threshold=t, coverage=keep.mean(), acc_kept=corr_v[keep].mean(), errors_flagged=(~corr_v & ~keep).sum() / max((~corr_v).sum(), 1)))
thr = pd.DataFrame(thr_rows); print(thr.round(3).to_string(index=False)); thr.to_csv("outputs/threshold_val.csv", index=False)

# ---- ONE evaluation on the held-out test set ----
pt = full.predict(make_ds(X["test"], bs=64), verbose=0); yp = pt.argmax(1)
rt = evaluate(Y["test"], yp)
print("TEST:", f"acc={rt['acc']:.4f} macro_f1={rt['f1']:.4f}")
print(classification_report(Y["test"], yp, target_names=NAMES, digits=3))
cm = confusion_matrix(Y["test"], yp); print(cm)
ConfusionMatrixDisplay(cm, display_labels=NAMES).plot(cmap="Greens", colorbar=False)
plt.title("Final model — test set confusion matrix"); plt.tight_layout(); plt.savefig("outputs/test_confusion_matrix.png", dpi=150); plt.close()
conf_t = pt.max(1); corr_t = yp == Y["test"]
# baselines on test
base = keras.models.load_model("outputs/baseline_converged.keras")
yb = base.predict(make_ds(X["test"]), verbose=0).argmax(1); rb = evaluate(Y["test"], yb)
maj = np.bincount(Y["train"]).argmax(); rm = evaluate(Y["test"], np.full_like(Y["test"], maj))
print("TEST baseline CNN:", f"acc={rb['acc']:.4f} macro_f1={rb['f1']:.4f}", "| majority:", f"acc={rm['acc']:.4f} f1={rm['f1']:.4f}")

# ---- size and CPU speed ----
keras_mb = os.path.getsize("outputs/leafcheck_final.keras") / 1e6
tfl = tf.lite.TFLiteConverter.from_keras_model(full).convert(); open("outputs/leafcheck_final.tflite", "wb").write(tfl)
x1 = X["test"][:1].astype("float32"); full(x1, training=False)
t0 = time.time()
for _ in range(30): full(x1, training=False)
ms = (time.time() - t0) / 30 * 1000
itp = tf.lite.Interpreter(model_content=tfl, num_threads=2); itp.allocate_tensors()
i_in, i_out = itp.get_input_details()[0], itp.get_output_details()[0]
itp.set_tensor(i_in["index"], x1); itp.invoke(); t0 = time.time()
for _ in range(30): itp.set_tensor(i_in["index"], x1); itp.invoke()
ms_tfl = (time.time() - t0) / 30 * 1000
print(f"Keras {keras_mb:.1f} MB, {ms:.0f} ms/img | TFLite {len(tfl)/1e6:.1f} MB, {ms_tfl:.0f} ms/img")

# ---- error analysis with Grad-CAM ----
grad_model = keras.Model(backbone.input, [backbone.get_layer("out_relu").output, backbone.output])
def gradcam(img_uint8):
    x = pre(tf.cast(img_uint8[None], tf.float32))
    with tf.GradientTape() as tape:
        conv, pooled = grad_model(x, training=False); probs = head(pooled, training=False); c = tf.argmax(probs[0]); score = probs[0, c]
    g = tape.gradient(score, conv); w = tf.reduce_mean(g, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * w[0], -1)).numpy(); cam /= cam.max() + 1e-8
    return tf.image.resize(cam[..., None], (IMG, IMG)).numpy()[..., 0]
err = np.where(~corr_t)[0]
pairs = pd.Series([f"{NAMES[a]} -> {NAMES[b]}" for a, b in zip(Y["test"][err], yp[err])]).value_counts()
print("Test errors:", len(err)); print(pairs.to_string())
print("Mean confidence: correct", round(conf_t[corr_t].mean(), 3), "| wrong", round(conf_t[~corr_t].mean(), 3))
show = err[np.argsort(-conf_t[err])][:6]
fig, ax = plt.subplots(2, len(show), figsize=(3 * len(show), 6.4))
for k, i in enumerate(show):
    ax[0, k].imshow(X["test"][i]); ax[0, k].set_title(f"True: {NAMES[Y['test'][i]]}\nPred: {NAMES[yp[i]]} ({conf_t[i]:.2f})", fontsize=8)
    ax[1, k].imshow(X["test"][i]); ax[1, k].imshow(gradcam(X["test"][i]), cmap="jet", alpha=0.45)
    for a in ax[:, k]: a.axis("off")
plt.suptitle("Most confident test-set errors (top) and Grad-CAM (bottom)"); plt.tight_layout(); plt.savefig("outputs/error_examples_gradcam.png", dpi=130); plt.close()
ok = np.where(corr_t)[0]; rng = np.random.RandomState(SEED); pick = [rng.choice(ok[Y["test"][ok] == c]) for c in range(4)]
fig, ax = plt.subplots(2, 4, figsize=(12, 6.4))
for k, i in enumerate(pick):
    ax[0, k].imshow(X["test"][i]); ax[0, k].set_title(f"{NAMES[yp[i]]} ({conf_t[i]:.2f})", fontsize=9)
    ax[1, k].imshow(X["test"][i]); ax[1, k].imshow(gradcam(X["test"][i]), cmap="jet", alpha=0.45)
    for a in ax[:, k]: a.axis("off")
plt.suptitle("Correct test-set predictions with Grad-CAM"); plt.tight_layout(); plt.savefig("outputs/correct_examples_gradcam.png", dpi=130); plt.close()
json.dump(dict(final=final_name, test=dict(acc=rt["acc"], f1=rt["f1"], recall=rt["recall"].tolist(), precision=rt["precision"].tolist(), f1_per=rt["f1_per"].tolist()),
               cm=cm.tolist(), baseline_test=dict(acc=rb["acc"], f1=rb["f1"], recall=rb["recall"].tolist()), majority_test=dict(acc=rm["acc"], f1=rm["f1"]),
               keras_mb=keras_mb, tflite_mb=len(tfl) / 1e6, ms_keras=ms, ms_tflite=ms_tfl, errors=pairs.to_dict(),
               conf_correct=float(conf_t[corr_t].mean()), conf_wrong=float(conf_t[~corr_t].mean()),
               test_low_conf=dict((str(t), float((conf_t < t).mean())) for t in [0.6, 0.7]),
               test_err_flagged=dict((str(t), float(((conf_t < t) & ~corr_t).sum() / max((~corr_t).sum(), 1))) for t in [0.6, 0.7]),
               test_acc_kept=dict((str(t), float(corr_t[conf_t >= t].mean())) for t in [0.6, 0.7]),
               seeds=seeds), open("outputs/final_summary.json", "w"), indent=1, default=float)
print("done")
