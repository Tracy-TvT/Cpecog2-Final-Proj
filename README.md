# CPECOG2 Final Project — Group 5: Rice Leaf Disease Classification (LeafCheck)

MobileNetV2 transfer-learning classifier for four rice leaf diseases (bacterial blight, blast, brown spot, tungro), with a Gradio demo and Grad-CAM.

**Group 5 (CPECOG2 - E41):** Alfonso Luiz Chua, Bleaux Bastasa, Ethan Vincent Ryan Beleno, Jerico Joir Bernasol, Patrice Tracy Itchon

## Results (held-out test set, evaluated once)

| Model | Test accuracy | Test macro F1 |
| --- | --- | --- |
| Majority-class predictor | 0.264 | 0.104 |
| Baseline CNN from scratch (converged) | 0.824 | 0.819 |
| **Final MobileNetV2 (R11)** | **0.944** | **0.936** |

Per-class recall: Bacterial blight 1.000, Blast 0.757, Brown spot 0.978, Tungro 1.000.
Model size 10.3 MB (.keras) / 9.5 MB (TFLite); 135 ms per image on a 2-core CPU (Keras).
Full write-up: `docs/CPECOG2_E41_Group5_Progress_Report_3.pdf`.

## Dataset

Rice Leaf Disease Image Samples (Sethy, 2020): https://data.mendeley.com/datasets/fwcj7stb8r/1
(mirror: https://github.com/maimunul/Rice-Leaf-Disease-Classification-using-CNN).
Put the four class folders (`Bacterialblight/`, `Blast/`, `Brownspot/`, `Tungro/`) in `data/`, or set `DATA_DIR`.

## Repository layout

```
src/        training and evaluation scripts (Progress Report 3 pipeline)
demo/       Gradio app (app.py), input tests (test_app.py), requirements
models/     final model (.keras, .tflite) and converged baseline CNN
outputs/    figures (learning curves, confusion matrix, Grad-CAM) and logs (per-run CSVs, splits, summary)
docs/       Progress Report 3
```

## Reproduce

Run from the repository root (seed 42; same split as Progress Report 2: 3,379 / 722 / 693 images in 231 / 49 / 47 clusters):

```
pip install tensorflow keras scikit-learn pillow imagehash pandas matplotlib
python src/s1_data.py           # integrity check, de-duplication, pHash clustering, cluster-level split
python src/s2_tune.py           # reproduces R2/R4; R6 (longer training), R7 (class weights), R8 (head width)
python src/s2b_units.py         # R9/R10 (256-unit head in stage 2)
python src/s3_aug.py light 15   # R11 light augmentation (selected)
python src/s3_aug.py strong 15  # R12 strong augmentation
python src/s4_baseline.py       # R13 baseline CNN with early stopping
python src/s5_final.py          # seed repeats, selection on validation, single test evaluation, Grad-CAM, size/speed
python src/s6b.py; python src/s7_errfig.py   # figures
```

## Demo

```
cd demo
pip install -r requirements.txt
python app.py          # opens the Gradio app; predictions below 0.70 confidence are reported as "Uncertain"
python test_app.py     # valid / edge / invalid input tests (needs DATA_DIR for the leaf images)
```

## AI tool use

Claude (Anthropic) was used to write and run the code for runs R6 to R13, the final test evaluation, the Grad-CAM analysis and the demo.
