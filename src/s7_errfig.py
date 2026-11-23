from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
m = keras.models.load_model("outputs/leafcheck_final.keras")
backbone = next(l for l in m.layers if l.name.startswith("mobilenetv2")); head = m.layers[-1]
gm = keras.Model(backbone.input, [backbone.get_layer("out_relu").output, backbone.output])
def gradcam(img):
    x = pre(tf.cast(img[None], tf.float32))
    with tf.GradientTape() as tape:
        conv, pooled = gm(x, training=False); pr = head(pooled, training=False); s = pr[0, tf.argmax(pr[0])]
    g = tape.gradient(s, conv); cam = tf.nn.relu(tf.reduce_sum(conv[0] * tf.reduce_mean(g, axis=(1, 2))[0], -1)).numpy()
    return tf.image.resize((cam / (cam.max() + 1e-8))[..., None], (IMG, IMG)).numpy()[..., 0]
X, Y = load_arrays(); d = pd.read_csv("outputs/splits.csv"); t = d[d.split == "test"].reset_index(drop=True)
P = m.predict(X["test"], batch_size=64, verbose=0); t["pred"] = P.argmax(1); t["conf"] = P.max(1)
e = t[t.pred != t.label]
pick = e.sort_values("conf", ascending=False).groupby("cluster").head(1).sort_values("conf", ascending=False).head(5)
fig, ax = plt.subplots(2, len(pick), figsize=(3.2 * len(pick), 6.6))
for k, (i, r) in enumerate(pick.iterrows()):
    ax[0, k].imshow(X["test"][i]); ax[0, k].set_title(f"True: {NAMES[r.label]}\nPred: {NAMES[r.pred]} ({r.conf:.2f})", fontsize=9)
    ax[1, k].imshow(X["test"][i]); ax[1, k].imshow(gradcam(X["test"][i]), cmap="jet", alpha=0.45)
    for a in ax[:, k]: a.axis("off")
plt.suptitle("Test-set errors, one per misclassified leaf (top) with Grad-CAM (bottom)"); plt.tight_layout()
plt.savefig("outputs/error_examples_gradcam.png", dpi=130)
print(pick[["cluster", "label", "pred", "conf"]])
