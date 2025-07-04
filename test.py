# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import tensorflow as tf

# # Load the trained model
# model = tf.keras.models.load_model("waste_classifier.h5")  # Replace with your model's path

# # Load an image
# img_path = "other.png"  # Update with your image path
# img = image.load_img(img_path, target_size=(224, 224))  # Match model input size
# img_array = image.img_to_array(img) / 255.0  # Normalize
# img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model

# # Make prediction
# predictions = model.predict(img_array)
# class_labels = ["Dry", "Other", "wet"]
# predicted_class = np.argmax(predictions)

# print(f"Raw Predictions (Probabilities): {predictions[0]}")  
# for i, label in enumerate(class_labels):
#     print(f"{label}: {predictions[0][i]:.4f}")

# print(f"Predicted category: {class_labels[predicted_class]} (Confidence: {predictions[0][predicted_class]:.4f})")


# # Map class index to label
# print("Final Learning Rate:", model.optimizer.learning_rate.numpy())
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('waste_classifier_one.h5')

# Test directory
test_dir = "dataset/test"  # your test folder

# Preprocess test images (same as during training)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important! Don't shuffle for accurate predictions
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Predict classes
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Plot Confusion Matrix (Optional but useful)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Plot Accuracy and Loss (already done during training, but here's for test set separately)
plt.figure(figsize=(6,4))
plt.bar(["Accuracy", "Loss"], [accuracy, loss], color=['skyblue', 'salmon'])
plt.title('Test Accuracy and Loss')
plt.ylabel('Score')
plt.show()




