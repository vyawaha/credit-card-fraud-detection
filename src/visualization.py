import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import train_test_split

from src.features import create_features

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# ==============================
# DATASET PREVIEW
# ==============================

preview = df.head()

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')

table = ax.table(
    cellText=preview.values,
    colLabels=preview.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(8)

plt.savefig("images/dataset_preview.png", bbox_inches='tight')
plt.close()

# ==============================
# FRAUD DISTRIBUTION
# ==============================

sns.countplot(x='Class', data=df)

plt.title("Fraud Distribution")

plt.savefig("images/fraud_distribution.png")

plt.close()

# ==============================
# CORRELATION HEATMAP
# ==============================

plt.figure(figsize=(14, 10))

corr = df.corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.savefig("images/correlation_heatmap.png")

plt.close()

# ==============================
# FEATURE ENGINEERING
# ==============================

df = create_features(df)

X = df.drop("Class", axis=1)

y = df["Class"]

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# ==============================
# LOAD MODEL
# ==============================

model = joblib.load("models/fraud_model.pkl")

# ==============================
# PREDICTIONS
# ==============================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig("images/confusion_matrix.png")

plt.close()

# Save confusion matrix CSV
pd.DataFrame(cm).to_csv(
    "outputs/confusion_matrix.csv",
    index=False
)

# ==============================
# CLASSIFICATION REPORT
# ==============================

report = classification_report(
    y_test,
    predictions,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

fig, ax = plt.subplots(figsize=(10, 4))

ax.axis('off')

table = ax.table(
    cellText=np.round(report_df.values, 2),
    colLabels=report_df.columns,
    rowLabels=report_df.index,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(8)

plt.savefig(
    "images/classification_report.png",
    bbox_inches='tight'
)

plt.close()

# Save metrics
with open("outputs/metrics.txt", "w") as f:
    f.write(classification_report(y_test, predictions))

# ==============================
# MODEL PERFORMANCE GRAPH
# ==============================

precision = report["1"]["precision"]
recall = report["1"]["recall"]
f1 = report["1"]["f1-score"]

metrics = [precision, recall, f1]

labels = ["Precision", "Recall", "F1-Score"]

plt.bar(labels, metrics)

plt.title("Fraud Model Performance")

plt.savefig("images/model_performance.png")

plt.close()

# ==============================
# FEATURE IMPORTANCE
# ==============================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

# ==============================
# PREDICTIONS CSV
# ==============================

pred_df = X_test.copy()

pred_df["Actual"] = y_test.values

pred_df["Prediction"] = predictions

pred_df["Fraud_Probability"] = probabilities

pred_df.to_csv(
    "outputs/predictions.csv",
    index=False
)

# ==============================
# TRAINING LOG
# ==============================

with open("outputs/training_log.txt", "w") as f:
    f.write("Fraud Detection Model Training Completed Successfully\n")
    f.write("\n")
    f.write(classification_report(y_test, predictions))

print("\n✅ All images and outputs generated successfully!")