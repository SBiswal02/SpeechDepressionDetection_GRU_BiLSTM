import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast # Used for safely evaluating string literals containing Python structures

# Import necessary Keras and Scikit-learn modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings('ignore')

# --- Data Loading and Preprocessing (from your original notebook) ---
# Assuming 'audio_balanced_downsampled.csv' is available in the environment
df_bdown = pd.read_csv('audio_balanced_downsampled.csv')

# Convert the 'Features' column from string representation of numpy arrays to actual numpy arrays
# Use ast.literal_eval for safe evaluation of the string representation of lists/arrays
df_bdown['Features'] = df_bdown['Features'].apply(
    lambda x: np.fromstring(x.replace('\n', ' ').replace('[', '').replace(']', ''), sep=' ')
)

# Reshape features to (num_samples, 16, 16)
# First, ensure all features are 256 elements long (16*16)
# If some features are not 256, you might need to pad or truncate them.
# For simplicity, we'll assume they are all compatible after ast.literal_eval.
# The original notebook already reshaped to (16, 16) in the loop, so we'll ensure that here.
X_d = np.array([f.reshape(16, 16) for f in df_bdown['Features']])
y_d = df_bdown['PHQ_Binary'].values

# Split data into training and testing sets
Xd_train, Xd_test, yd_train, yd_test = train_test_split(X_d, y_d, test_size=0.2, random_state=44, stratify=y_d)
print(f"Train and Test Split for Downsampled: {Xd_train.shape}, {Xd_test.shape}, {yd_train.shape}, {yd_test.shape}")

# --- Model Definition with Improvements ---
model = Sequential()
# GRU layer with reduced units (e.g., 16) and L2 regularization
# L2 regularization penalizes large weights, helping to prevent overfitting.
model.add(GRU(
    16, # Reduced units from 32 to 16 for a simpler model, especially with small dataset
    input_shape=(Xd_train.shape[1], Xd_train.shape[2]), # Input shape (timesteps, features)
    return_sequences=False, # Output only the last hidden state for classification
    kernel_regularizer=l2(0.001) # L2 regularization on the kernel weights
))
# Increased Dropout rate to 0.5 to further combat overfitting
model.add(Dropout(0.5))

# Dense hidden layer with L2 regularization
model.add(Dense(
    16, # Number of neurons in the dense layer
    activation='relu', # ReLU activation function
    kernel_regularizer=l2(0.001) # L2 regularization on the kernel weights
))
# Another Dropout layer
model.add(Dropout(0.5))

# Output layer for binary classification (sigmoid activation)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    optimizer='adam', # Adam optimizer
    loss='binary_crossentropy', # Binary cross-entropy loss for binary classification
    metrics=['accuracy'] # Monitor accuracy during training
)

model.summary()

# --- Evaluation Function (from your original notebook) ---
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints accuracy, precision, recall, and F1 score.
    """
    y_pred = model.predict(X_test)
    # Convert probabilities to binary predictions (0 or 1) using a 0.5 threshold
    y_pred_classes = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes)
    recall = recall_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# --- Plotting Function (from your original notebook) ---
def plot_history(history):
    """
    Plots training and validation accuracy and loss over epochs.
    """
    plt.figure(figsize=(12, 4))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# --- Training with Early Stopping and Class Weights ---
epochs = 100 # Set a higher number of epochs, as early stopping will manage when to stop

# Calculate class weights to handle potential imbalance in the dataset
# This assigns a higher weight to the minority class, making the model pay more attention to it.
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(yd_train),
    y=yd_train
)
# Convert class_weights to a dictionary format required by Keras
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Calculated Class Weights: {class_weight_dict}")

# Define EarlyStopping callback
# It monitors 'val_loss' and stops training if it doesn't improve for 'patience' epochs.
# restore_best_weights=True ensures the model uses the weights from the best epoch.
early_stop = EarlyStopping(
    monitor='val_loss', # Metric to monitor
    patience=10, # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True, # Restores model weights from the epoch with the best monitored value
    verbose=1 # Logs when early stopping is triggered
)

print("\n--- Training the improved model ---")
history = model.fit(
    Xd_train, yd_train,
    epochs=epochs,
    batch_size=16,
    validation_data=(Xd_test, yd_test),
    callbacks=[early_stop], # Add early stopping callback
    class_weight=class_weight_dict, # Apply class weights
    verbose=1 # Show training progress
)

# --- Evaluate and Plot Results ---
print("\n--- Evaluation on Test Set ---")
evaluate_model(model, Xd_test, yd_test)

print("\n--- Plotting Training History ---")
plot_history(history)

# Save the improved model (optional)
# model.save('saved_models/gru_model_improved.keras')
