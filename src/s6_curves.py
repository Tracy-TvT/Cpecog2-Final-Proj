import pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs = [("R13 Baseline CNN (converged)", pickle.load(open("outputs/R_baseline.pkl", "rb"))["hist"]),
        ("R7 Stage 2 + class weights (no aug.)", pickle.load(open("outputs/s2_results.pkl", "rb"))["R7_cw"]["hist"]),
        ("R11 Light augmentation (selected)", pickle.load(open("outputs/R_aug_light.pkl", "rb"))["hist"]),
        ("R12 Strong augmentation", pickle.load(open("outputs/R_aug_strong.pkl", "rb"))["hist"])]
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, (t, h) in zip(ax, runs):
    e = range(1, len(h["loss"]) + 1)
    a.plot(e, h["loss"], label="train loss"); a.plot(e, h["val_loss"], label="val loss")
    a2 = a.twinx(); a2.plot(e, h["accuracy"], "--", c="tab:green", label="train acc"); a2.plot(e, h["val_accuracy"], "--", c="tab:red", label="val acc")
    a2.set_ylim(0.5, 1.02); a.set_title(t, fontsize=10); a.set_xlabel("Epoch"); a.set_ylabel("Loss"); a2.set_ylabel("Accuracy")
    l1, l2 = a.get_legend_handles_labels(), a2.get_legend_handles_labels()
    a.legend(l1[0] + l2[0], l1[1] + l2[1], fontsize=7, loc="center right")
plt.tight_layout(); plt.savefig("outputs/learning_curves_pr3.png", dpi=140)
