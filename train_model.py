import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# Define dataset paths
train_dir = "dataset/train"
test_dir = "dataset/test"

# Define class labels
class_labels = ["Dry", "Wet", "Other"]

# Data Augmentation (stronger for Wet waste)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    zoom_range=0.6,
    brightness_range=[0.4, 2.2],  # Increase contrast
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.4
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load Data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Compute Class Weights
class_counts = np.bincount(train_generator.classes)
class_weights = {i: max(class_counts) / c for i, c in enumerate(class_counts)}
class_weights[1] *= 4.0  # Boost Wet waste importance

print("Class Weights:", class_weights)

# Load Pretrained MobileNetV2 Model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers

# Build Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')  # 3 classes: Dry, Wet, Other
])

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks for Fine-tuning
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train Model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=100,  # Train longer for fine-tuning
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

import matplotlib.pyplot as plt
import numpy as np

# 1. Plot Learning Rate Schedule (if you used ReduceLROnPlateau or Scheduler)
# If you don't have learning rate history, you can skip this part
lrs = [0.001 * (0.1 ** (epoch / 30)) for epoch in range(len(history.history['loss']))]
plt.figure(figsize=(8,6))
plt.plot(lrs)
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid()
plt.show()

# 2. Plot Train vs Validation Accuracy
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# 3. Plot Train vs Validation Loss
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# 4. Plot Error Rate
error_rate = [1 - acc for acc in history.history['val_accuracy']]
plt.figure(figsize=(8,6))
plt.plot(error_rate)
plt.title('Validation Error Rate Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.grid()
plt.show()


# Save Model
model.save("waste_classifier_one.h5")
print("Model training complete! 🎉")
