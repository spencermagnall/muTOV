import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import pickle

# Load and preprocess data
eos_training_data = pd.read_csv('eos_training_set_poly_piece_pool_random_lal_high_mass.csv')
X_data = eos_training_data.drop('lambda', axis=1)
y_data = eos_training_data['lambda']

# Remove tidal deformability > 5,000
indices_to_drop = y_data[y_data > 5000].index
X_data = X_data.drop(indices_to_drop)
y_data = y_data.drop(indices_to_drop)

# Standardize features
for column in X_data.columns:
    mean, std = np.mean(X_data[column]), np.std(X_data[column])
    X_data[column] = (X_data[column] - mean) / std
    print(f'Feature {column} has mean {mean} and std {std}')

# Create uniform distribution of samples
num_bins = 500
target_dataset_size = 100000
target_samples_per_bin = target_dataset_size // num_bins
y_data = np.array(y_data)
hist, bin_edges = np.histogram(y_data, bins=num_bins)

# Subsample data
X_data_subsampled, y_data_subsampled = [], []
for i in range(num_bins):
    indices_in_bin = np.where((y_data >= bin_edges[i]) & (y_data < bin_edges[i+1]))[0]
    subsample_indices = np.random.choice(indices_in_bin, 
                                       size=min(len(indices_in_bin), target_samples_per_bin), 
                                       replace=False)
    X_data_subsampled.extend(X_data.iloc[subsample_indices].values.tolist())
    y_data_subsampled.extend(y_data[subsample_indices].tolist())

X_data_subsampled = np.array(X_data_subsampled)
y_data_subsampled = np.array(y_data_subsampled)

# Split data into train/test/validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X_data_subsampled, y_data_subsampled, 
                                                   test_size=0.1, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, 
                                               test_size=0.5, random_state=42)

# Define and compile model
model = Sequential([
    Dense(128, input_dim=5, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'), 
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_absolute_error')

# Train model
history = model.fit(X_train, y_train,
                   epochs=5000,
                   batch_size=len(X_train)//32,
                   validation_data=(X_val, y_val),
                   callbacks=[early_stopping])

# Save training history
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig("figures/Loss_curves.png", dpi=300)
plt.clf()

with open('muTOV_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Evaluate model
y_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse_test}")

# Calculate and plot errors
relative_error = (np.abs(y_test - y_pred.flatten()) / y_test) * 100
mean_rel_error = np.mean(relative_error)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, relative_error, marker='x', s=5, color='orange', label='Relative Error %')
plt.axhline(y=mean_rel_error, color='red', linestyle='-', 
           label=f'Mean Error ({mean_rel_error:.2f}%)')
plt.title('Prediction Errors')
plt.xlabel('Tidal Deformability ($\Lambda$)')
plt.ylabel('Error (%)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig("figures/Relative_error_mean_error.png", dpi=300)
plt.clf()

actual_error = np.abs(y_test - y_pred.flatten())
mean_abs_error = np.mean(actual_error)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, actual_error, marker='x', s=5, color='orange', label='Actual Error')
plt.axhline(y=mean_abs_error, color='red', linestyle='-', 
           label=f'Mean Error ({mean_abs_error:.2f})')
plt.title('Prediction Errors')
plt.xlabel('Tidal Deformability ($\Lambda$)')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.savefig('figures/Actual_error_mean_error.png', dpi=300)
plt.clf()

# Save model
model.save('muTOV.h5')