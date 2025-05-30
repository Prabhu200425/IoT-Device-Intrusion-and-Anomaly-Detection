import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("RT_IOT2022_test.csv")  # Replace with your actual dataset path

# Drop unwanted features (if present)
if 'proto' in data.columns and 'service' in data.columns:
    data = data.drop(columns=['proto', 'service'])

# Separate features and target
X = data.drop(columns=['Label'])  # Assuming 'Label' is the target column
y = data['Label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Ensure test dataset has the same columns as training
expected_features = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else X_train.columns.tolist()
X_test = X_test[expected_features]  # Drop extra columns
X_test = X_test[expected_features]  # Ensure correct order

# Transform test data
X_test = scaler.transform(X_test)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
