#!/usr/bin/env python3
"""
Script to evaluate multiple TFLite models on an image dataset organized as:

test_images - Copy/
  ├── Bacterial Blight/
  │    └── images/   (contains images of rice leaves with Bacterial Blight)
  ├── Brown Spot/
  │    └── images/   (contains images of rice leaves with Brown Spot)
  ├── Healthy/
  │    └── images/   (contains images of healthy rice leaves)
  ├── Leaf Blast/
  │    └── images/   (contains images of rice leaves with Leaf Blast)
  └── Not Rice/
       └── images/  (contains images of non-rice leaves)

Now with background removal (rembg) before inference.
"""

import os
import csv
from io import BytesIO

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from rembg import remove, new_session

# ---------------- Default Settings ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT_DIR = os.path.join(BASE_DIR, "test-images-3")

MODEL_PATHS = {
    "ZAMBALI":   os.path.join(BASE_DIR, "ZAMBALI_rice_disease_model_V4.tflite"),
    "HUGFACE":   os.path.join(BASE_DIR, "HUGFACE_rice_disease_model_V4.tflite"),
    "COMBINED":  os.path.join(BASE_DIR, "COMBINED_rice_disease_model_V4-1.tflite"),
    "KAGGLE":    os.path.join(BASE_DIR, "KAGGLE_rice_disease_model_V4-1.tflite"),
}

LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
DEFAULT_LABELS = [
    "Bacterial Blight",
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Not a Rice Leaf"
]

# ---------------- Rembg Session ----------------
# Create a shared session so we don't reload the model every single remove() call.
rembg_session = new_session()

# ---------------- Utility Functions ----------------
def load_labels(label_path):
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels = [line.strip() for line in f if line.strip()]
        if labels:
            return labels
        print("labels.txt is empty. Using default labels.")
    else:
        print("labels.txt not found. Using default labels.")
    return DEFAULT_LABELS


def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return {
        "interpreter": interpreter,
        "input_details": interpreter.get_input_details(),
        "output_details": interpreter.get_output_details(),
    }


def remove_background(raw_path):
    """
    Load the image bytes, remove its background, and return a PIL.Image with alpha channel.
    """
    with open(raw_path, 'rb') as f:
        input_bytes = f.read()
    result = remove(input_bytes, session=rembg_session)
    # `result` is bytes of a PNG with transparency
    return Image.open(BytesIO(result)).convert("RGB")


def preprocess_pil(img, target_size=(224, 224)):
    """
    Take a PIL.Image and convert to normalized numpy tensor shape (1, H, W, 3).
    """
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def classify_tensor(tensor, model_info, labels):
    interp = model_info["interpreter"]
    inp_det = model_info["input_details"][0]["index"]
    out_det = model_info["output_details"][0]["index"]
    interp.set_tensor(inp_det, tensor)
    interp.invoke()
    preds = interp.get_tensor(out_det).copy()[0]
    idx = np.argmax(preds)
    label = labels[idx] if idx < len(labels) else "Unknown"
    confidence = preds[idx] * 100
    all_confs = {lab: float(p * 100) for lab, p in zip(labels, preds)}
    return label, confidence, all_confs


def get_image_files(directory):
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [
        os.path.join(directory, fn)
        for fn in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, fn))
        and os.path.splitext(fn)[1].lower() in valid_ext
    ]


def plot_confusion_matrix(cm, classes, model_name, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], 
                 ha="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metrics(metrics, model_name, save_path):
    names = list(metrics.keys())
    vals  = [metrics[n] for n in names]
    plt.figure(figsize=(6,4))
    bars = plt.bar(names, vals)
    plt.title(f"Metrics - {model_name}")
    plt.ylim(0, 100)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h, f"{h:.1f}", 
                 ha='center', va='bottom')
    plt.ylabel("Percentage")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------- Main Evaluation ----------------
def run_evaluation(root_dir, model_paths, labels_path):
    if not os.path.exists(root_dir):
        print(f"Root directory {root_dir} does not exist.")
        return

    labels = load_labels(labels_path)

    # Load TFLite models
    models = {}
    for name, path in model_paths.items():
        try:
            models[name] = load_model(path)
            print(f"Loaded model: {name}")
        except Exception as e:
            print(f"Failed loading {name}: {e}")

    all_rows = []
    eval_data = {m: {"y_true": [], "y_pred": []} for m in models}

    # Walk ground-truth folders
    for gt in os.listdir(root_dir):
        gt_path = os.path.join(root_dir, gt)
        if not os.path.isdir(gt_path):
            continue
        images_dir = os.path.join(gt_path, "images")
        proc_folder = images_dir if os.path.isdir(images_dir) else gt_path
        imgs = get_image_files(proc_folder)
        if not imgs:
            continue

        print(f"\nProcessing {len(imgs)} images for label '{gt}'")
        for img_path in imgs:
            # 1) remove bg
            pil_no_bg = remove_background(img_path)
            # 2) preprocess
            tensor = preprocess_pil(pil_no_bg)

            row = {"filename": os.path.basename(img_path), "ground_truth": gt}
            # 3) classify with each model
            for model_name, info in models.items():
                pred, conf, allc = classify_tensor(tensor, info, labels)
                row[f"pred_{model_name}"]       = pred
                row[f"conf_{model_name}"]       = f"{conf:.2f}"
                row[f"all_confs_{model_name}"]  = str(allc)
                eval_data[model_name]["y_true"].append(gt)
                eval_data[model_name]["y_pred"].append(pred)

            all_rows.append(row)

    # Write consolidated CSV
    csv_out = os.path.join(root_dir, "all_predictions.csv")
    if all_rows:
        cols = ["filename", "ground_truth"]
        for m in models:
            cols += [f"pred_{m}", f"conf_{m}", f"all_confs_{m}"]
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved predictions to {csv_out}")

    # Compute & plot metrics per model
    for m, data in eval_data.items():
        y_true, y_pred = data["y_true"], data["y_pred"]
        if not y_true:
            print(f"No data for model {m}, skipping.")
            continue

        acc = accuracy_score(y_true, y_pred) * 100
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )
        prec, rec, f1 = prec*100, rec*100, f1*100

        print(f"\n== {m} Metrics ==")
        print(f"Accuracy : {acc:.2f}%")
        print(f"Precision: {prec:.2f}%")
        print(f"Recall   : {rec:.2f}%")
        print(f"F1 Score : {f1:.2f}%")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_path = os.path.join(root_dir, f"confusion_{m}.png")
        plot_confusion_matrix(cm, labels, m, cm_path)
        print(f"Saved confusion matrix: {cm_path}")

        # Metrics bar chart
        met = {"Accuracy":acc, "Precision":prec, "Recall":rec, "F1 Score":f1}
        met_path = os.path.join(root_dir, f"metrics_{m}.png")
        plot_metrics(met, m, met_path)
        print(f"Saved metrics chart: {met_path}")


if __name__ == "__main__":
    print("Starting evaluation with background removal...")
    run_evaluation(DEFAULT_ROOT_DIR, MODEL_PATHS, LABELS_PATH)
    print("Done.")
