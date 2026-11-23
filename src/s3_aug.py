# Augmentation-strength tuning: end-to-end on images (augmentation on training set only), stage 2 setup of the best run.
import sys, pickle
from common import *
X, Y = load_arrays(); del X["test"]
backbone, lower, upper = build_backbone()
upper_w0 = [w.copy() for w in upper.get_weights()]
heads = pickle.load(open("outputs/s2_heads.pkl", "rb"))
STRENGTH = {"light":  dict(rot=0.05, zoom=0.0,  bright=0.1, contrast=0.1),
            "strong": dict(rot=0.25, zoom=0.15, bright=0.2, contrast=0.2)}
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 15

def aug_block(rot, zoom, bright, contrast):
    l = [layers.RandomFlip("horizontal_and_vertical", seed=SEED), layers.RandomRotation(rot, fill_mode="reflect", seed=SEED)]
    if zoom: l.append(layers.RandomZoom(zoom, fill_mode="reflect", seed=SEED))
    l += [layers.RandomBrightness(bright, value_range=(0, 255), seed=SEED), layers.RandomContrast(contrast, seed=SEED)]
    return keras.Sequential(l, name="augment")

def run(level, lr=1e-5, start="block_15"):
    keras.utils.set_random_seed(SEED)
    upper.set_weights(upper_w0); nu = set_trainable_from(upper, start)
    lower.trainable = False
    head = make_head(0.5); head.set_weights(heads["b1_head"])
    inp = keras.Input((IMG, IMG, 3))
    x = aug_block(**STRENGTH[level])(inp)            # active only during fit()
    x = layers.Lambda(lambda t: pre(tf.cast(t, tf.float32)), name="mnv2_preprocess")(x)
    x = lower(x, training=False); x = upper(x, training=False)
    m = keras.Model(inp, head(x))
    m.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    name = f"R_aug_{level}"
    t = time.time()
    h = m.fit(make_ds(X["train"], Y["train"], shuffle=True), validation_data=make_ds(X["val"], Y["val"], bs=64),
              epochs=EPOCHS, class_weight=heads["cw"], callbacks=callbacks(name, 4), verbose=2)
    pred = m.predict(make_ds(X["val"], bs=64), verbose=0).argmax(1); r = evaluate(Y["val"], pred)
    print(f"{name}: epochs={len(h.history['loss'])} val_acc={r['acc']:.4f} val_f1={r['f1']:.4f} recall={np.round(r['recall'],3)} {time.time()-t:.0f}s", flush=True)
    out = dict(name=name, level=level, unfrozen=nu, lr=lr, epochs=len(h.history["loss"]), val_acc=r["acc"], val_f1=r["f1"],
               recall=r["recall"].tolist(), pred=pred, hist=h.history, upper_w=[w.copy() for w in upper.get_weights()], head_w=head.get_weights())
    pickle.dump(out, open(f"outputs/{name}.pkl", "wb"))

run(sys.argv[1])
