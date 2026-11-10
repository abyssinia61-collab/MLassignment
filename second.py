

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
)

# ============================================================
# PART 1. CREATE DATASET (60 RECORDS: 5 FEATURES + 3 CLASSES)
# Potential data source: Sensor logs and maintenance inspection records
# ============================================================

data = [
    # ID, Temperature_C, Vibration_mm_s, RPM, Pressure, Operating_Hours, Failure_Risk 
    # (0 = Normal, 1 = Warning, 2 = Failure risk / Fault)

    # Normal machines (0) - 20 records
    [1,  42, 1.1, 1500, 2.1,  500, 0],
    [2,  45, 1.3, 1480, 2.2,  800, 0],
    [3,  49, 1.5, 1520, 2.0, 1100, 0],
    [4,  51, 1.7, 1490, 2.3, 1300, 0],
    [5,  47, 1.4, 1510, 2.1,  900, 0],
    [6,  53, 1.8, 1470, 2.4, 1500, 0],
    [7,  55, 2.0, 1505, 2.2, 1700, 0],
    [8,  44, 1.2, 1530, 2.0,  700, 0],
    [9,  50, 1.6, 1485, 2.3, 1200, 0],
    [10, 46, 1.3, 1515, 2.1,  850, 0],
    [11, 57, 2.1, 1465, 2.5, 1900, 0],
    [12, 52, 1.9, 1495, 2.3, 1400, 0],
    [13, 48, 1.5, 1525, 2.0, 1000, 0],
    [14, 54, 1.8, 1475, 2.4, 1600, 0],
    [15, 56, 2.2, 1500, 2.5, 1800, 0],
    [16, 43, 1.2, 1510, 2.1,  600, 0],
    [17, 46, 1.4, 1490, 2.2,  950, 0],
    [18, 49, 1.6, 1480, 2.3, 1150, 0],
    [19, 52, 1.8, 1500, 2.4, 1350, 0],
    [20, 45, 1.3, 1520, 2.0,  750, 0],

    # Warning machines (1) - 20 records
    [21, 62, 2.8, 1440, 3.8, 2200, 1],
    [22, 65, 3.1, 1430, 4.0, 2500, 1],
    [23, 68, 3.4, 1410, 4.2, 2900, 1],
    [24, 60, 2.6, 1450, 3.6, 2050, 1],
    [25, 66, 3.2, 1420, 4.1, 2700, 1],
    [26, 70, 3.7, 1395, 4.5, 3300, 1],
    [27, 63, 2.9, 1435, 3.9, 2350, 1],
    [28, 67, 3.3, 1415, 4.2, 2800, 1],
    [29, 61, 2.7, 1445, 3.7, 2150, 1],
    [30, 69, 3.5, 1400, 4.3, 3100, 1],
    [31, 64, 3.0, 1425, 4.0, 2450, 1],
    [32, 71, 3.8, 1390, 4.6, 3450, 1],
    [33, 66, 3.2, 1418, 4.1, 2650, 1],
    [34, 68, 3.5, 1405, 4.3, 3000, 1],
    [35, 62, 2.8, 1438, 3.8, 2250, 1],
    [36, 72, 4.0, 1380, 4.8, 3700, 1],
    [37, 65, 3.1, 1422, 4.0, 2550, 1],
    [38, 67, 3.4, 1412, 4.2, 2850, 1],
    [39, 63, 2.9, 1432, 3.9, 2400, 1],
    [40, 70, 3.6, 1398, 4.4, 3200, 1],

    # Failure-risk machines / Fault (2) - 20 records
    [41, 75, 4.8, 1370, 6.2, 4200, 2],
    [42, 81, 5.5, 1340, 6.8, 5000, 2],
    [43, 86, 6.2, 1310, 7.4, 5800, 2],
    [44, 78, 5.1, 1360, 6.5, 4600, 2],
    [45, 92, 7.3, 1270, 8.5, 7200, 2],
    [46, 84, 6.0, 1320, 7.2, 5500, 2],
    [47, 96, 8.0, 1240, 9.1, 8000, 2],
    [48, 73, 4.5, 1380, 5.9, 3900, 2],
    [49, 89, 6.8, 1290, 8.0, 6800, 2],
    [50, 87, 6.5, 1300, 7.7, 6200, 2],
    [51, 99, 8.5, 1220, 9.5, 8500, 2],
    [52, 77, 5.0, 1350, 6.4, 4400, 2],
    [53, 94, 7.6, 1250, 8.8, 7600, 2],
    [54, 82, 5.8, 1330, 7.0, 5300, 2],
    [55, 91, 7.1, 1280, 8.3, 7000, 2],
    [56, 76, 4.9, 1365, 6.3, 4300, 2],
    [57, 83, 5.9, 1325, 7.1, 5400, 2],
    [58, 88, 6.7, 1295, 7.9, 6500, 2],
    [59, 95, 7.8, 1245, 9.0, 7800, 2],
    [60, 92, 7.3, 1285, 8.4, 7150, 2],
]

columns = [
    "Machine_ID", "Temperature_C", "Vibration_mm_s",
    "RPM", "Pressure", "Operating_Hours", "Failure_Risk"
]
df = pd.DataFrame(data, columns=columns)

print(f"\nDATASET: {len(df)} MACHINE RECORDS (Source: Sensor logs & Maintenance records)")
print(df.to_string(index=False))
print("\nClass distribution:")
print(df["Failure_Risk"].value_counts().sort_index().rename(index={0: "Normal", 1: "Warning", 2: "Fault (Failure risk)"}))

# Separate predictors (X) and target (y).
feature_cols = ["Temperature_C", "Vibration_mm_s", "RPM", "Pressure", "Operating_Hours"]
X = df[feature_cols]
y = df["Failure_Risk"]

# Initial data visualization.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="Temperature_C", y="Vibration_mm_s",
    hue="Failure_Risk", palette=["green", "orange", "red"], style="Failure_Risk", s=100
)
plt.title("Machine Temperature vs. Vibration")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration (mm/s)")
plt.legend(title="0=Normal, 1=Warning, 2=Fault")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[feature_cols + ["Failure_Risk"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix of sensor & maintenance variables")
plt.tight_layout()
plt.show()

# ============================================================
# PART 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")

# ============================================================
# PART 3. DEFINE FIVE INDEPENDENT CLASSIFIERS
# ============================================================

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))
])

naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

decision_tree_model = DecisionTreeClassifier(
    max_depth=3, random_state=42
)

random_forest_model = RandomForestClassifier(
    n_estimators=100, max_depth=3, random_state=42
)

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}

# ============================================================
# PART 4. TRAIN, PREDICT, EVALUATE, AND VISUALIZE EACH MODEL
# ============================================================

results = []
predictions = {}

for model_name, model in models.items():
    print("\n" + "=" * 65)
    print(f"{model_name.upper()}")
    print("=" * 65)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    # Extract specific metrics for Fault (class 2)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    fault_precision = report_dict["2"]["precision"]
    fault_recall = report_dict["2"]["recall"]

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Fault Precision": fault_precision,
        "Fault Recall": fault_recall
    })

    print(f"Accuracy       : {accuracy:.3f}")
    print(f"Macro F1       : {macro_f1:.3f}")
    print(f"Fault Precision: {fault_precision:.3f}")
    print(f"Fault Recall   : {fault_recall:.3f}")

    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1, 2],
        target_names=["Normal", "Warning", "Fault"],
        zero_division=0
    ))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Normal", "Warning", "Fault"],
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ============================================================
# PART 5. COMPARE THE FIVE ALGORITHMS
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")
print("\nMODEL COMPARISON")
print(results_df.round(3))

results_df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05))
plt.title("Comparison of Five Supervised Algorithms (Priority Metrics)")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ============================================================
# PART 6. INTERPRET TREE-BASED FEATURE IMPORTANCE
# ============================================================

tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in tree_models.items():
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print(f"\n{name} feature importance:")
    print(importance.round(3))

    plt.figure(figsize=(7, 4))
    plt.barh(importance.index, importance.values)
    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Engineering feature")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(16, 8))
plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=["Normal", "Warning", "Fault"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree learned from the training records")
plt.tight_layout()
plt.show()

# ============================================================
# PART 7. TEST A NEW ENGINEERING OBSERVATION
# ============================================================

new_machine = pd.DataFrame([{
    "Temperature_C": 88,
    "Vibration_mm_s": 6.4,
    "RPM": 1335,
    "Pressure": 7.5,
    "Operating_Hours": 6100
}])

print("\nNEW MACHINE OBSERVATION")
print(new_machine)

class_labels = {0: "Normal", 1: "Warning (Inspect)", 2: "Fault (Failure risk)"}

for name, model in models.items():
    pred_class = int(model.predict(new_machine)[0])
    label = class_labels[pred_class]
    print(f"{name:15s} prediction: {label}")

print("\nEND OF PRACTICAL CASE STUDY")
