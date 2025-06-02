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

For every image, the script:
  - Derives the ground truth (from the parent folder name).
  - Preprocesses the image once and runs inference on it with all models.
  - Writes a consolidated CSV file (all_predictions.csv) with:
       filename, ground truth, and for each model:
           predicted label, confidence, and full confidence distribution.
  - Computes accuracy, recall, precision, and F1 score per model.
  - Generates and saves graphs for the confusion matrix and evaluation metrics.
  
If labels.txt is not found, a default set of labels is used.
"""

import os
import csv
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ---------------- Default Settings ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Set the root directory to the folder containing your ground truth folders.
DEFAULT_ROOT_DIR = os.path.join(BASE_DIR, "test_images - Copy")

# Define model paths for multiple models.
MODEL_PATHS = {
    "ZAMBALI": os.path.join(BASE_DIR, "ZAMBALI_rice_disease_model_V4.tflite"),
    "HUGFACE": os.path.join(BASE_DIR, "HUGFACE_rice_disease_model_V4.tflite"),
    "COMBINED": os.path.join(BASE_DIR, "COMBINED_rice_disease_model_V4-1.tflite"),
    "KAGGLE": os.path.join(BASE_DIR, "KAGGLE_rice_disease_model_V4-1.tflite"),
}

LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
# Fallback labels if labels.txt is not found.
DEFAULT_LABELS = [
    "Bacterial Blight",
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Not a Rice Leaf"
]


# ---------------- Utility Functions ----------------
def load_labels(label_path):
    """Load class labels from a text file; if not found or empty, use defaults."""
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels = [line.strip() for line in f if line.strip()]
        if labels:
            return labels
        else:
            print("labels.txt is empty. Using default labels.")
    else:
        print("labels.txt not found. Using default labels.")
    return DEFAULT_LABELS


def load_model(model_path):
    """Load a TFLite model and allocate tensors."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def preprocess_image(image_path, target_size=(224, 224)):
    """Open and preprocess an image for the model."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    img = img.resize(target_size)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)
    return img


def classify_preprocessed(image_tensor, model_info, labels):
    """
    Run inference on a preprocessed image tensor for a given model.
    Returns:
      - predicted_label: label with highest confidence,
      - confidence: confidence (percentage) for predicted label,
      - all_confidences: dictionary mapping each label to confidence.
    """
    interpreter = model_info["interpreter"]
    input_details = model_info["input_details"]
    output_details = model_info["output_details"]
    
    interpreter.set_tensor(input_details[0]['index'], image_tensor)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index']).copy()[0]
    
    predicted_index = np.argmax(predictions)
    predicted_label = labels[predicted_index] if predicted_index < len(labels) else "Unknown"
    confidence = predictions[predicted_index] * 100
    all_confidences = {label: float(prob * 100) for label, prob in zip(labels, predictions)}
    return predicted_label, confidence, all_confidences


def get_image_files(directory):
    """Return a list of image file paths in a directory (common image extensions only)."""
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and os.path.splitext(f)[1].lower() in valid_ext
    ]


def plot_confusion_matrix(cm, classes, model_name, save_path):
    """Plot and save a confusion matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()


def plot_metrics(metrics, model_name, save_path):
    """Plot and save a bar chart of evaluation metrics for a model."""
    metric_names = list(metrics.keys())
    values = [metrics[m] for m in metric_names]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(metric_names, values, color='skyblue')
    plt.title(f"Evaluation Metrics - {model_name}")
    plt.ylim([0, 100])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.1f}', ha='center', va='bottom')
    plt.ylabel("Percentage")
    plt.savefig(save_path)
    plt.close()


# ---------------- Main Evaluation Function ----------------
def run_evaluation(root_dir, model_paths, labels_path):
    """
    Load all models and evaluate them on images found in the dataset.
    The dataset structure:
       root_dir/
         <GroundTruth Label>/
             images/ (or images might be directly under the ground truth folder)
    For each image, the ground truth is the parent folder name.
    A consolidated CSV file ('all_predictions.csv') is created.
    Evaluation metrics (accuracy, precision, recall, f1) are computed per model,
    and graphs (confusion matrix and bar chart) are saved.
    """
    if not os.path.exists(root_dir):
        print(f"Root directory {root_dir} does not exist.")
        return

    # Load labels.
    labels = load_labels(labels_path)

    # Load all models.
    models = {}
    for model_name, path in model_paths.items():
        try:
            interpreter, input_details, output_details = load_model(path)
            models[model_name] = {
                "interpreter": interpreter,
                "input_details": input_details,
                "output_details": output_details
            }
            print(f"Loaded model {model_name} from {path}")
        except Exception as e:
            print(f"Error loading model {model_name} from {path}: {e}")

    # To accumulate predictions.
    all_rows = []
    # For each model, keep lists of ground truth and predictions.
    eval_results = {model_name: {"y_true": [], "y_pred": []} for model_name in models.keys()}

    # Traverse each ground truth folder in the dataset.
    for gt_folder in os.listdir(root_dir):
        gt_path = os.path.join(root_dir, gt_folder)
        if os.path.isdir(gt_path):
            # Look for "images" subfolder; if absent, process the folder itself.
            images_dir = os.path.join(gt_path, "images")
            folder_to_process = images_dir if (os.path.exists(images_dir) and os.path.isdir(images_dir)) else gt_path

            image_files = get_image_files(folder_to_process)
            if not image_files:
                print(f"No images found in {folder_to_process}")
                continue

            print(f"Processing {len(image_files)} images for ground truth '{gt_folder}' in folder: {folder_to_process}")

            for image_path in image_files:
                preprocessed = preprocess_image(image_path)
                if preprocessed is None:
                    continue

                row = {
                    "filename": os.path.basename(image_path),
                    "ground_truth": gt_folder
                }
                # For each model, get predictions.
                for model_name, model_info in models.items():
                    pred_label, conf, all_confs = classify_preprocessed(preprocessed, model_info, labels)
                    # Save prediction for CSV.
                    row[f"predicted_{model_name}"] = pred_label
                    row[f"confidence_{model_name}"] = f"{conf:.2f}"
                    # Optionally, include full confidence distribution as a string.
                    row[f"all_confidences_{model_name}"] = str(all_confs)
                    # Collect for evaluation.
                    eval_results[model_name]["y_true"].append(gt_folder)
                    eval_results[model_name]["y_pred"].append(pred_label)
                all_rows.append(row)

    # Write overall CSV file.
    csv_file = os.path.join(root_dir, "all_predictions.csv")
    if all_rows:
        # Define CSV columns: filename, ground_truth, then for each model, predicted and confidence.
        fieldnames = ["filename", "ground_truth"]
        for model_name in models.keys():
            fieldnames.extend([f"predicted_{model_name}", f"confidence_{model_name}", f"all_confidences_{model_name}"])
        with open(csv_file, "w", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)
        print(f"All predictions saved to {csv_file}")
    else:
        print("No predictions recorded.")

    # Evaluate each model.
    for model_name, results in eval_results.items():
        y_true = results["y_true"]
        y_pred = results["y_pred"]
        if not y_true:
            print(f"No predictions for model {model_name}.")
            continue

        accuracy = accuracy_score(y_true, y_pred) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        precision *= 100
        recall *= 100
        f1 *= 100

        print(f"\nEvaluation for {model_name}:")
        print(f"  Accuracy : {accuracy:.2f}%")
        print(f"  Precision: {precision:.2f}%")
        print(f"  Recall   : {recall:.2f}%")
        print(f"  F1 Score : {f1:.2f}%")

        # Compute confusion matrix.
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_save_path = os.path.join(root_dir, f"confusion_{model_name}.png")
        plot_confusion_matrix(cm, labels, model_name, cm_save_path)
        print(f"  Confusion matrix saved as {cm_save_path}")

        # Prepare metrics for bar chart.
        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
        metrics_save_path = os.path.join(root_dir, f"metrics_{model_name}.png")
        plot_metrics(metrics, model_name, metrics_save_path)
        print(f"  Evaluation metrics graph saved as {metrics_save_path}")


# ---------------- Main Execution ----------------
if __name__ == "__main__":
    print("Starting evaluation of all models...")
    run_evaluation(DEFAULT_ROOT_DIR, MODEL_PATHS, LABELS_PATH)
    print("Evaluation complete.")