# Baseline CNN (same architecture as R1) retrained to convergence: early stopping instead of a 6-epoch cap.
import pickle
from common import *
X, Y = load_arrays(); del X["test"]
keras.utils.set_random_seed(SEED)
baseline = keras.Sequential([
    keras.Input((IMG, IMG, 3)), layers.Rescaling(1 / 255.0),
    layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"), layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"), layers.Dropout(0.3), layers.Dense(4, activation="softmax")])
baseline.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
t = time.time()
h = baseline.fit(make_ds(X["train"], Y["train"], shuffle=True), validation_data=make_ds(X["val"], Y["val"]), epochs=40,
                 callbacks=callbacks("R_baseline_conv", 5), verbose=2)
pred = baseline.predict(make_ds(X["val"]), verbose=0).argmax(1); r = evaluate(Y["val"], pred)
print(f"Baseline converged: epochs={len(h.history['loss'])} val_acc={r['acc']:.4f} val_f1={r['f1']:.4f} recall={np.round(r['recall'],3)} {time.time()-t:.0f}s", flush=True)
baseline.save("outputs/baseline_converged.keras")
pickle.dump(dict(epochs=len(h.history["loss"]), val_acc=r["acc"], val_f1=r["f1"], recall=r["recall"].tolist(), hist=h.history), open("outputs/R_baseline.pkl", "wb"))
