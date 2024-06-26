from keras.preprocessing import image
from keras import applications
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Dropout, BatchNormalization
from keras.models import Model
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import os


# importing dataset
dir = os.listdir('Dataset')
for filenames in dir:
    print(filenames)

datagen = ImageDataGenerator(rescale=1./255)

training_set = datagen.flow_from_directory(
    'Dataset/train',
        target_size=(224, 224),
        batch_size=32,
        shuffle= False,
        class_mode='categorical')

val_set = datagen.flow_from_directory(
        'Dataset/validate',
        target_size=(224,224),
        batch_size=32,
        shuffle= False,
        class_mode="categorical")

test_set = datagen.flow_from_directory(
        'Dataset/test',
        target_size=(224,224),
        batch_size=32,
        shuffle= False,
        class_mode="categorical")
        


# Restrict GPU usage to 14 GB only        

#gpus = tf.config.list_physical_devices('GPU')
#if gpus:
#  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#  try:
#    tf.config.set_logical_device_configuration(
#        gpus[0],
#        [tf.config.LogicalDeviceConfiguration(memory_limit=14000)])
#    logical_gpus = tf.config.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#  except RuntimeError as e:
#    # Virtual devices must be set before GPUs have been initialized
#    print(e)
    
    
    
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

r = model.fit(training_set, epochs=40, validation_data = val_set, callbacks = [csv_logger, es])

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





# 4.Xception

csv_logger = tf.keras.callbacks.CSVLogger('Result/Xception_training.csv', append=True)

model_Xception = tf.keras.applications.Xception(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

for layer in model_Xception.layers:
    layer.trainable = False

x = Flatten()(model_Xception.layers[-1].output)
output= Dense(18, activation = 'softmax')(x)

model = Model(inputs = model_Xception.inputs, outputs = output)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

r = model.fit(training_set, epochs=40, validation_data = val_set, callbacks = [csv_logger, es])

Y_pred = model.predict_generator(test_set)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
cm = confusion_matrix(test_set.classes, y_pred)
df = pd.DataFrame(cm, index=None)
df.to_excel("Result/Xception_confusionmatrix.xlsx")
print(cm)

print('Classification Report')
print(classification_report(test_set.classes, y_pred))
report = classification_report(test_set.classes, y_pred, output_dict=True)
dff = pd.DataFrame(report).transpose()
dff.to_excel("Result/Xception_classificationreport.xlsx")





# 5.MobileNetV2

csv_logger = tf.keras.callbacks.CSVLogger('Result/MobileNetV2_training.csv', append=True)

model_MobileNetV2 = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

for layer in model_MobileNetV2.layers:
   layer.trainable = False

x = Flatten()(model_MobileNetV2.layers[-1].output)
output= Dense(18, activation = 'softmax')(x)

model = Model(inputs = model_MobileNetV2.inputs, outputs = output)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

r = model.fit(training_set, epochs=40, validation_data = val_set, callbacks = [csv_logger, es])

Y_pred = model.predict_generator(test_set)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
cm = confusion_matrix(test_set.classes, y_pred)
df = pd.DataFrame(cm, index=None)
df.to_excel("Result/MobileNetV2_confusionmatrix.xlsx")
print(cm)

print('Classification Report')
print(classification_report(test_set.classes, y_pred))
report = classification_report(test_set.classes, y_pred, output_dict=True)
dff = pd.DataFrame(report).transpose()
dff.to_excel("Result/MobileNetV2_classificationreport.xlsx")

