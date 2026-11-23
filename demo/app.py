"""LeafCheck demo: rice leaf disease classifier (MobileNetV2) with Grad-CAM.
Run:  pip install -r requirements.txt   then   python app.py
Specification: one RGB/RGBA/greyscale photo of a single rice leaf (JPG/PNG), any size.
Outputs the top class, class probabilities, and a Grad-CAM heatmap.
Predictions below the confidence threshold are reported as "Uncertain"."""
import os, numpy as np, tensorflow as tf, keras, gradio as gr
from PIL import Image

NAMES = ["Bacterial blight", "Blast", "Brown spot", "Tungro"]
IMG = 224
THRESHOLD = float(os.environ.get("LEAFCHECK_THRESHOLD", 0.7))   # chosen on the validation set
MODEL_PATH = os.environ.get("LEAFCHECK_MODEL", os.path.join(os.path.dirname(__file__), "..", "models", "leafcheck_final.keras"))

model = keras.models.load_model(MODEL_PATH)
backbone = next(l for l in model.layers if l.name.startswith("mobilenetv2"))
head = model.layers[-1]
grad_model = keras.Model(backbone.input, [backbone.get_layer("out_relu").output, backbone.output])
pre = keras.applications.mobilenet_v2.preprocess_input

def to_input(img):
    """Accepts a PIL image or numpy array; returns a 224x224x3 uint8 array."""
    if img is None:
        raise ValueError("No image received. Upload a photo of a single rice leaf.")
    if isinstance(img, np.ndarray):
        if img.ndim not in (2, 3) or img.size == 0:
            raise ValueError("Unsupported image array.")
        img = Image.fromarray(img.astype(np.uint8))
    if min(img.size) < 32:
        raise ValueError(f"Image too small ({img.size[0]}x{img.size[1]}). Use a photo at least 32 px on each side.")
    return np.asarray(img.convert("RGB").resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8)

def gradcam(x_uint8):
    x = pre(tf.cast(x_uint8[None], tf.float32))
    with tf.GradientTape() as tape:
        conv, pooled = grad_model(x, training=False)
        probs = head(pooled, training=False)
        score = probs[0, tf.argmax(probs[0])]
    g = tape.gradient(score, conv)
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * tf.reduce_mean(g, axis=(1, 2))[0], -1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG, IMG)).numpy()[..., 0]
    heat = (keras.utils.array_to_img(np.stack([cam] * 3, -1) * 255)).convert("L")
    import matplotlib.cm as cm
    colored = (cm.jet(np.asarray(heat) / 255.0)[..., :3] * 255).astype(np.uint8)
    return (0.55 * x_uint8 + 0.45 * colored).astype(np.uint8)

def predict(img):
    try:
        x = to_input(img)
    except ValueError as e:
        return str(e), {}, None
    probs = model.predict(x[None], verbose=0)[0]
    top = int(probs.argmax()); conf = float(probs[top])
    verdict = (f"{NAMES[top]} (confidence {conf:.2f})" if conf >= THRESHOLD
               else f"Uncertain: best guess {NAMES[top]} ({conf:.2f}). Retake the photo: one leaf, good light, leaf filling the frame.")
    return verdict, {n: float(p) for n, p in zip(NAMES, probs)}, gradcam(x)

demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil", label="Rice leaf photo"),
                    outputs=[gr.Textbox(label="Result"), gr.Label(num_top_classes=4, label="Class probabilities"),
                             gr.Image(label="Grad-CAM: regions the model used")],
                    title="LeafCheck — Rice Leaf Disease Classifier",
                    description="Classifies bacterial blight, blast, brown spot, or tungro from a single-leaf photo. Not trained on healthy leaves or other diseases.")
if __name__ == "__main__":
    demo.launch()
