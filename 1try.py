from google.colab import drive
drive.mount('/content/drive/')

# Install necessary libraries
!pip install -q keras

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os

# Mount Google Drive
drive.mount('/content/drive/')

# Define directory paths
train_dir = '/content/drive/MyDrive/Datasets/Datasets/train'
validate_dir = '/content/drive/MyDrive/Datasets/Datasets/validate'

# Load datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256),
    color_mode='rgb',
    shuffle=True,
    interpolation='bilinear'
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validate_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(128, 128),
    color_mode='rgb',
    shuffle=True,
    interpolation='bilinear'
)

# Load pre-trained CNN model (ResNet50)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Create model for feature extraction
model = tf.keras.Model(inputs=base_model.input, outputs=x)

# Data preprocessing with ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

# Extract features using ResNet50 model
features_train = model.predict(train_generator)

# Assuming you have combined CNN features with metadata features (replace with your actual implementation)
combined_features = features_train  # Replace with your actual combined features

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_train)

# Train SVM using Random Forest output
svm = SVC(probability=True)
svm.fit(rf_probs, y_train)

# Example of using the trained model (you can modify this for your specific use case)
# Assuming you have validation data to predict on
validation_generator = datagen.flow_from_directory(
    validate_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

features_val = model.predict(validation_generator)
rf_val_probs = rf.predict_proba(features_val)
svm_predictions = svm.predict(rf_val_probs)

# Error Analysis: Confusion Matrix and Classification Report
y_true = validation_generator.classes  # True labels
y_pred = np.argmax(svm_predictions, axis=1)  # Predicted labels

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
target_names = validation_generator.class_indices
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))
