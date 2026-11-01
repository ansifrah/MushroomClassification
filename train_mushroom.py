import csv
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


DATA_FILE = Path(__file__).with_name("agaricus-lepiota.data")

FEATURE_NAMES = [
    "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat",
]


def read_data():
    """Read the comma-separated UCI file; '?' is kept as a category."""
    features, labels = [], []
    with DATA_FILE.open("r", newline="", encoding="utf-8") as data_file:
        for line_number, row in enumerate(csv.reader(data_file), start=1):
            if len(row) != len(FEATURE_NAMES) + 1:
                raise ValueError(f"Row {line_number} should contain 23 values.")
            labels.append(row[0])
            features.append(row[1:])
    if not features:
        raise ValueError(f"No rows found in {DATA_FILE}.")
    if set(labels) - {"e", "p"}:
        raise ValueError("Expected class labels 'e' (edible) and 'p' (poisonous).")
    return features, labels


def make_models():
    """Build two classifiers that can work directly with category codes."""
    return {
        "Decision Tree": make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        ),
        "Random Forest": make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
        ),
    }


def main():
    features, labels = read_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    models = make_models()
    print(f"Rows: {len(features)} (training: {len(x_train)}, test: {len(x_test)})")
    print("Positive class for sensitivity: poisonous (p)\n")

    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        sensitivity = recall_score(y_test, predictions, pos_label="p")
        print(f"{name}")
        print(f"Poisonous sensitivity: {sensitivity:.4f}")
        print("Confusion matrix (rows=true, columns=predicted; e then p):")
        print(confusion_matrix(y_test, predictions, labels=["e", "p"]))
        print(classification_report(
            y_test, predictions, labels=["e", "p"],
            target_names=["edible", "poisonous"], zero_division=0,
        ))
        print("100% poisonous sensitivity on this test split:", "yes" if sensitivity == 1.0 else "no")
        print()


if __name__ == "__main__":
    main()