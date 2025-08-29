import tensorflow as tf
from keras.callbacks import EarlyStopping

# 1. Load the MNIST Dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Reshape and Normalize the Data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# 3. Define the Data Augmentation Layers
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomRotation(0.1), # Rotate by up to 10% 
  tf.keras.layers.RandomZoom(0.1),     # Zoom by up to 10%  
  tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1), # Translate by up to 10% in height and width
])


# 4. Build a CNN Model
model = tf.keras.models.Sequential([
    # Add the data augmentation layers right at the beginning
    data_augmentation, # Apply data augmentation to input images during training 
    
    # First convolutional block
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)), 
    
    # Second convolutional block to learn more complex features
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)), #
    
    # Classifier part
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'), # Fully connected layer with 128 neurons 
    tf.keras.layers.Dense(10, activation='softmax') # Output layer with 10 neurons for 10 classes (digits 0-9)
])

# 5. Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Set up Early Stopping
# This will monitor the validation accuracy and stop if it doesn't improve for 3 epochs.
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True) # Restore the best weights after stopping


# 7. Train the Model
# We can set epochs to a high number like 50. EarlyStopping will find the real best number for us.
# validation_split=0.1 uses 10% of the data as our "practice exam".
model.fit(x_train, y_train, 
          epochs=50, 
          validation_split=0.1,
          callbacks=[early_stopping])


# 8. Save the final, best model
model.save('handwritten_pro_model.keras')

print("\nProfessional model has been trained and saved as handwritten_pro_model.keras")

# Optional: Evaluate the final model on the test set
print("\nEvaluating final model on test data:")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")