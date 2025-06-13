import os
import json
import glob
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

PREDICTIONS_DIR = "./predictions/prediction-long_dataset-2025_06_13T06_23_35_274Z/"

true_labels = []
predicted_scores = []
predicted_labels = []

files = glob.glob(os.path.join(PREDICTIONS_DIR, "explanation.results-*.jsonl"))

for file_path in files:
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)

            try:
                instance = data["instance"]
                prediction = data["prediction"]

                # Defensive check
                if "long_result" not in instance or instance["long_result"] is None:
                    continue

                long_result = int(instance["long_result"])
                score_for_1 = prediction["scores"][1]

                true_labels.append(long_result)
                predicted_scores.append(score_for_1)
                predicted_labels.append(1 if score_for_1 >= 0.5 else 0)

            except Exception as e:
                # Ignore any broken rows
                continue

if not true_labels:
    print("🚫 No valid data loaded from files.")
else:
    acc = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    cm = confusion_matrix(true_labels, predicted_labels)

    print("✅ Evaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(cm)
