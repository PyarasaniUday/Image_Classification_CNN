import tensorflow as tf
from tensorflow.keras import datasets, layers, models #type:ignore
from tensorflow.keras.models import load_model #type:ignore
from tensorflow.keras.preprocessing import image #type:ignore
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# 2. Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Class labels
class_names = ['airplane','car','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# 4. Check if model already exists
if os.path.exists("models/cifar_cnn_model.h5"):
    print("✅ Loading saved model...")
    model = load_model("models/cifar_cnn_model.h5")

else:
    print("🚀 Training model...")

    # Build CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    # Compile
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train ONLY once
    history = model.fit(
        x_train, y_train,
        epochs=10,
        validation_data=(x_test, y_test)
    )

    # Save model
    model.save("models/cifar_cnn_model.h5")
    print("💾 Model saved!")

    # Graphs (only when training happens)
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title("Model Accuracy")
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Model Loss")
    plt.show()

# 5. Evaluate (works for both train/load)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("🎯 Test Accuracy:", test_acc)

# 6. Prediction
predictions = model.predict(x_test)

plt.figure(figsize=(4,4))
plt.imshow(x_test[0])
plt.title("Predicted: " + class_names[np.argmax(predictions[0])])
plt.axis('off')
plt.show()

# Load custom image
img_path = "test.jpg"   # change name if needed

img = image.load_img(img_path, target_size=(32,32))
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

# Show result
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print("🔍 Prediction:", predicted_class)
print("📊 Confidence:", confidence)

# Show image
plt.imshow(img)
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()