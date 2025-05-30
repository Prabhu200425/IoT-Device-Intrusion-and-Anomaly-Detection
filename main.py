import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib

# Step 1: Data Preprocessing
# Load dataset
dataset_path = "C:/Users/Sikkandhar Jabbar/Desktop/rt_intrusion/input/RT_IOT2022.csv"
df = pd.read_csv(dataset_path)

# Remove missing or corrupted data
df.dropna(inplace=True)

# Encode categorical variables
Label_encoders = {}
categorical_features = df.select_dtypes(include=['object']).columns
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    Label_encoders[col] = le

# Normalize numerical attributes
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Save the StandardScaler
scaler_path = "C:/Users/Sikkandhar Jabbar/Desktop/rt_intrusion/input/scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# Split dataset
X = df.drop(columns=['Label'])  # Assuming 'Label' is the target variable
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Feature Extraction using CNN
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train CNN model
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Extract features from CNN
feature_extractor = tf.keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# Step 3: Classification using WKNN
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_features, y_train)
y_pred = knn.predict(X_test_features)

# Step 4: ANN Model
ann_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_features.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train ANN model
ann_model.fit(X_train_features, y_train, epochs=10, batch_size=32, validation_data=(X_test_features, y_test), verbose=1)

# Step 5: Performance Evaluation
accuracy_cnn = cnn_model.evaluate(X_test, y_test, verbose=0)[1]
accuracy_ann = ann_model.evaluate(X_test_features, y_test, verbose=0)[1]
accuracy_knn = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, knn.predict_proba(X_test_features), multi_class='ovr')

print(f"CNN Accuracy: {accuracy_cnn}")
print(f"ANN Accuracy: {accuracy_ann}")
print(f"KNN Accuracy: {accuracy_knn}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"ROC-AUC Score: {roc_auc}")

# Step 6: Plot Comparison Graphs
models = ['CNN', 'ANN', 'WKNN']
accuracies = [accuracy_cnn, accuracy_ann, accuracy_knn]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xLabel("Models")
plt.yLabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.show()

# Confusion Matrix Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xtickLabels=np.unique(y), ytickLabels=np.unique(y))
plt.xLabel("Predicted")
plt.yLabel("Actual")
plt.title("Confusion Matrix")
plt.show()
