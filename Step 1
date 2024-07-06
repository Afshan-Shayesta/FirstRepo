
from keras import applications
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Dropout, BatchNormalization
from keras.models import Model
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import os


# importing dataset
dir = os.listdir('/content/drive/MyDrive/Datasets')
for filenames in dir:
    print(filenames)

datagen = ImageDataGenerator(rescale=1./255)

training_set = datagen.flow_from_directory(
    '/content/drive/MyDrive/Datasets/train',
        target_size=(224, 224),
        batch_size=32,
        shuffle= False,
        class_mode='categorical')

val_set = datagen.flow_from_directory(
        '/content/drive/MyDrive/Datasets/validate',
        target_size=(224,224),
        batch_size=32,
        shuffle= False,
        class_mode="categorical")

test_set = datagen.flow_from_directory(
        '/content/drive/MyDrive/Datasets/test',
        target_size=(224,224),
        batch_size=32,
        shuffle= False,
        class_mode="categorical")

# Restrict GPU usage to 14 GB only

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=14000)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


# Early stopping with patience 10

es = EarlyStopping(monitor='val_loss', patience=10)


# Pre-Trained Deep Learning Models

# 3.InceptionV3

csv_logger = tf.keras.callbacks.CSVLogger('Result/InceptionV3_training.csv', append=True)

model_InceptionV3 = tf.keras.applications.InceptionV3(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

for layer in model_InceptionV3.layers:
    layer.trainable = False

x = Flatten()(model_InceptionV3.layers[-1].output)
output= Dense(18, activation = 'softmax')(x)

model = Model(inputs = model_InceptionV3.inputs, outputs = output)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

r = model.fit(training_set, epochs=1, validation_data = val_set, callbacks = [csv_logger, es])

Y_pred = model.predict_generator(test_set)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
cm = confusion_matrix(test_set.classes, y_pred)
df = pd.DataFrame(cm, index=None)
df.to_excel("Result/InceptionV3_confusionmatrix.xlsx")
print(cm)

print('Classification Report')
print(classification_report(test_set.classes, y_pred))
report = classification_report(test_set.classes, y_pred, output_dict=True)
dff = pd.DataFrame(report).transpose()
dff.to_excel("Result/InceptionV3_classificationreport.xlsx")


import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Custom preprocessing function
def custom_preprocessing(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Otsu's thresholding
    _, thresh_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Define a kernel size for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Apply morphological opening
    opened_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)

    # Convert single channel image back to three channels
    processed_image = cv2.cvtColor(opened_image, cv2.COLOR_GRAY2BGR)

    # Normalize the image
    processed_image = processed_image / 255.0

    return processed_image

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=custom_preprocessing
)

# Only rescaling for validation and test sets
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Creating train, validation, and test generators
train_generator = train_datagen.flow_from_directory(
    downsampled_train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    downsampled_validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    downsampled_test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Function to display a batch of augmented images
def display_augmented_images(generator, num_images=10):
    batch = next(generator)
    images, labels = batch
    plt.figure(figsize=(20, 10))
    for i in range(num_images):
        plt.subplot(2, num_images//2, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()

# Display some augmented images
print("Augmented Train Dataset:")
display_augmented_images(train_generator)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Load the VGG16 model, excluding the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the new model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // 32,
    validation_steps=validation_generator.samples // 32
)
