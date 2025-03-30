import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset (Replace with real medical dataset)
data = {
    'Fever': [1, 0, 1, 0, 1, 0, 1, 0],
    'Cough': [1, 1, 0, 0, 1, 1, 0, 0],
    'Fatigue': [0, 1, 1, 0, 1, 0, 1, 0],
    'Disease': ['Flu', 'Cold', 'COVID-19', 'Healthy', 'Flu', 'Cold', 'COVID-19', 'Healthy']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode Disease labels
df['Disease'] = df['Disease'].astype('category').cat.codes

# Split data
X = df.drop(columns=['Disease'])
y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to predict disease
def predict_disease(fever, cough, fatigue):
    symptoms = np.array([[fever, cough, fatigue]])
    prediction = model.predict(symptoms)
    return prediction[0]

# Example usage
print("Predicted Disease:", predict_disease(1, 1, 0))
