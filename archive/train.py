# train_symptom_checker.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib

# --------------------------
# 1. Load the dataset
# --------------------------
data = pd.read_csv("/home/rishi/Downloads/archive/Training.csv")  # Replace with your CSV path
print(data.head())

data = data.dropna(axis=1, how="all") 

# Features = all columns except 'prognosis'
X = data.drop(columns=["prognosis"])
# Target = prognosis column
y = data["prognosis"]

# Encode target labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --------------------------
# 2. Split into train/test
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --------------------------
# 3. Initialize XGBoost classifier
# --------------------------
clf = xgb.XGBClassifier(
    objective='multi:softmax',  # multi-class classification
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# --------------------------
# 4. Train the model
# --------------------------
clf.fit(X_train, y_train)

# --------------------------
# 5. Evaluate
# --------------------------
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Cross-validation check
cv_scores = cross_val_score(clf, X, y_encoded, cv=5)
print("5-Fold CV Accuracy:", cv_scores.mean())

# --------------------------
# 6. Save the model
# --------------------------
joblib.dump(clf, "symptom_checker.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Model and LabelEncoder saved!")

# --------------------------
# 7. Real-time prediction example
# --------------------------
def predict_disease(symptoms_list):
    """symptoms_list: list of 0/1s representing symptoms"""
    clf_loaded = joblib.load("symptom_checker.pkl")
    le_loaded = joblib.load("label_encoder.pkl")
    prediction = clf_loaded.predict([symptoms_list])
    disease = le_loaded.inverse_transform(prediction)
    return disease[0]

# Example usage:
# user_input = [1,0,0,1,...]  # Fill with 0/1 for each symptom
# print("Predicted disease:", predict_disease(user_input))
