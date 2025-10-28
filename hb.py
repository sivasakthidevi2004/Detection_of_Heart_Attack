import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
import joblib

# Load dataset
df = pd.read_csv("C:\\Users\\DELL\\Downloads\\heart.csv")

# Preprocessing
df.drop_duplicates(inplace=True)
X = df.drop('output', axis=1)
y = df['output']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize selected columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use in Flask
joblib.dump(scaler, 'scaler.pkl')

# Define models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(probability=True),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'XGBClassifier': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train models and collect predictions
model_predictions = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    model_predictions[name] = preds
    print(f"{name} Accuracy: {accuracy_score(y_test, preds):.4f}")

# Ensemble voting (majority vote)
ensemble_predictions = []
for i in range(len(X_test_scaled)):
    votes = [model_predictions[m][i] for m in models]
    majority_vote = Counter(votes).most_common(1)[0][0]
    ensemble_predictions.append(majority_vote)

ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print("✅ Ensemble Accuracy:", ensemble_accuracy)

# Save all models for later use
for name, model in models.items():
    joblib.dump(model, f'{name}.pkl')

# Save ensemble metadata (just the model names)
joblib.dump(list(models.keys()), 'ensemble_models.pkl')

print("✅ All models and scaler saved successfully!")
