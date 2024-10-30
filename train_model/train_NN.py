import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle

# Load the eos_training file
eos_training_data = pd.read_csv('eos_training_set_poly_piece_pool_random_lal_high_mass.csv')

X_data = eos_training_data.drop('lambda', axis=1)
y_data = eos_training_data['lambda']
print("y_data shape:", y_data.shape)

# Drop indices where y_data > 10000
indices_to_drop = y_data[y_data > 5000].index
X_data = X_data.drop(indices_to_drop)
y_data = y_data.drop(indices_to_drop)
print("New y_data shape:", y_data.shape)

# Drop indices where y_data < 0
# Made this 1 because of lambdas of 10^{-9} etc that are not physical  
#indices_to_drop = y_data[y_data <= 10].index
#X_data = X_data.drop(indices_to_drop)
#y_data = y_data.drop(indices_to_drop)
#print("New y_data shape:", y_data.shape)

# Drop indices where log_P1 is greater than 35
#indices_to_drop = X_data[X_data['log_P1'] >= 35].index
#X_data = X_data.drop(indices_to_drop)
#y_data = y_data.drop(indices_to_drop) 

#indices_to_drop = X_data[X_data['Gamma_1'] <= 2.0].index
#X_data = X_data.drop(indices_to_drop)
#y_data = y_data.drop(indices_to_drop)

#indices_to_drop = X_data[X_data['Gamma_2'] <= 1.2].index
#X_data = X_data.drop(indices_to_drop)
#y_data = y_data.drop(indices_to_drop) 

# indices_to_drop = X_data[X_data['Gamma_3'] <= 1.2].index
# X_data = X_data.drop(indices_to_drop)
# y_data = y_data.drop(indices_to_drop) 

# Standardisation for each column of inputs
for column in X_data.columns:
    mean = np.mean(X_data[column])
    std = np.std(X_data[column])
    X_data[column] = (X_data[column] - mean) / std
    print(f'Feature {column} has mean {mean} and std {std}')

#y_data = np.log10(y_data)

print(y_data)
#exit()


# Sample from y_data to construct a uniform distribution aiming for 20000 total samples
# Determine the number of bins for the histogram
num_bins = 500
target_dataset_size = 100000
y_data = np.array(y_data)
# Compute histogram to get the distribution of y_data
hist, bin_edges = np.histogram(y_data, bins=num_bins)

print(hist)
print(bin_edges)
#exit()
# Calculate the target number of samples per bin to achieve 20000 total samples
target_samples_per_bin = target_dataset_size // num_bins

# Initialize empty lists for subsampled data
X_data_subsampled = []
y_data_subsampled = []

# For each bin, sample target_samples_per_bin samples
for i in range(num_bins):
    # Indices of samples in the current bin
    indices_in_bin = np.where((y_data >= bin_edges[i]) & (y_data < bin_edges[i+1]))[0]
    
    # If there are enough samples in the bin, subsample target_samples_per_bin samples, otherwise take what is available
    if len(indices_in_bin) >= target_samples_per_bin:
        subsample_indices = np.random.choice(indices_in_bin, size=target_samples_per_bin, replace=False)
    else:
        subsample_indices = indices_in_bin
    
    # Append subsampled data to the subsampled lists
    X_data_subsampled.extend(X_data.iloc[subsample_indices].values.tolist())
    y_data_subsampled.extend(y_data[subsample_indices].tolist())

# Convert lists to numpy arrays
X_data_subsampled = np.array(X_data_subsampled)
y_data_subsampled = np.array(y_data_subsampled)

print("Subsampled y_data shape:", y_data_subsampled.shape)

# # Plotting the histogram of y_data_subsampled to verify uniform distribution
# plt.figure(figsize=(10, 6))
# plt.hist(y_data_subsampled, bins=num_bins, alpha=0.7, color='blue')  # Use num_bins for consistency in visualization
# plt.title('Histogram of Subsampled y_data')
# plt.xlabel('y_data values')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.savefig('hist_subsample_y.png',dpi=500)
# plt.clf()

# Splitting the dataset into training and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_data_subsampled, y_data_subsampled, test_size=0.1, random_state=42)

# Splitting the testing set into testing and validation sets
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set size: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set size: X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"Validation set size: X_val: {X_val.shape}, y_val: {y_val.shape}")



# Define the model
model = Sequential([
    Dense(128, input_dim=5, activation='relu'),  # Input layer with 5 neurons (since we have 5 inputs: x, a, b, c, d)
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')  # Output layer with 1 neuron (since we're predicting a single value: y)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model
model.summary()


# Define early stopping
samples = X_train.shape[0]
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)
optimizer = Adam(learning_rate=0.001)

# Compile the model with the new learning rate
model.compile(optimizer=optimizer, loss='mean_absolute_error')

# Training the model with early stopping and model checkpoint to restore the best model
history = model.fit(X_train, y_train, epochs=5000, batch_size=samples//32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Plotting the training and validation losses
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig("Loss_curves.png",dpi=500)
plt.clf()

# Save the history
with open('new_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# model_file_path = "/home/smagnall/runs/train_molotov/uTOV-4layer-128-5kcut_high_mass.h5"
# model = keras.models.load_model(model_file_path)
# print(model.history)
# history = model.history
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Yay')
# plt.yscale('log')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.savefig("Loss_curves.png",dpi=500)
# plt.clf()
# # Predicting the test set results
# y_pred = model.predict(X_test)

# # Calculating the Mean Squared Error on the test set
# mse_test = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error on Test Set: {mse_test}")



# # Selecting 100 samples from the test set
# sample_indices = np.random.choice(X_test.shape[0], 100, replace=False)
# X_test_samples = X_test[sample_indices]
# y_test_samples = y_test[sample_indices]

# # Predicting the selected samples
# y_pred_samples = model.predict(X_test_samples)

# # Un log10 the predictions and actual values
# #y_pred_samples_unlogged = 10**y_pred_samples.flatten()
# #y_test_samples_unlogged = 10**y_test_samples

# # Calculating the Mean Absolute Error on the test set for error bars
# #mae_test = mean_absolute_error(np.power(10, y_test), np.power(10, model.predict(X_test)))

# # # Plotting the actual vs predicted values with error bars
# # plt.figure(figsize=(10, 6))
# # plt.plot(range(100), y_pred_samples, 'r.', label='Predicted')
# # plt.scatter(range(100), y_test_samples, color='blue', label='Actual', marker='x')
# # plt.title('Comparison of Actual and Predicted Values for 100 Random Test Samples')
# # plt.xlabel('Sample Index')
# # plt.ylabel('Value')
# # #plt.ylim([500, 1500])
# # plt.legend()
# # plt.savefig("Actual_vs_predicted.png",dpi=500)
# # plt.clf()

# # plt.figure(figsize=(10, 6))
# # # Scale mass 
# # plt.plot(X_test[:,0]*0.2806059558770456 + 1.5168556935090685 , y_pred, 'r.', label='Predicted')
# # plt.scatter(X_test[:,0]*0.2806059558770456 + 1.5168556935090685, y_test, color='blue', label='Actual', marker='x')
# # #plt.title('Comparison of Actual and Predicted Values for 100 Random Test Samples')
# # plt.xlabel('Mass ($M_\odot$)')
# # plt.ylabel('$\Lambda$')
# # plt.legend()
# # plt.savefig("Actual_vs_predicted_standardised.png",dpi=500)
# # plt.clf()

# # # Calculating the relative error (percentage) between the predicted and actual values
# # relative_error = (np.abs(y_test - y_pred.flatten()) / y_test) * 100

# # # Plotting the relative error as a function of the actual values (y_test) and the mean error
# # mean_error = np.mean(relative_error)
# # plt.figure(figsize=(10, 6))
# # plt.scatter(y_test, relative_error, marker='x', s=5, color='orange', label='Relative Error %')
# # plt.axhline(y=mean_error, color='red', linestyle='-', label=f'Mean Error ({mean_error:.2f}%)')
# # plt.title('Relative Error and Mean Error of Prediction vs Actual Values')
# # plt.xlabel('Actual Values (y_test)')
# # plt.ylabel('Error (%)')
# # plt.yscale('log')
# # plt.legend()
# # plt.grid(True)
# # plt.savefig("Relative_error_mean_error.png",dpi=500)
# # plt.clf()

# # # Calculating the actual error between the predicted and actual values
# # actual_error = np.abs(y_test - y_pred.flatten())

# # # Plotting the actual error as a function of the actual values (y_test) and the mean error
# # mean_error = np.mean(actual_error)
# # plt.figure(figsize=(10, 6))
# # plt.xscale('log')
# # plt.scatter(y_test, actual_error, marker='x', s=5, color='orange', label='Actual Error')
# # plt.axhline(y=mean_error, color='red', linestyle='-', label=f'Mean Error ({mean_error:.2f})')
# #plt.title('Actual Error and Mean Error of Prediction vs Actual Values')
# #plt.xlabel('Actual Values (y_test)')
# # plt.xlabel('Tidal Deformability ($\Lambda$)')
# # plt.ylabel('Error')
# # plt.legend()
# # plt.grid(True)
# # plt.savefig('Actual_error_mean_error.png',dpi=500)
# # plt.clf()

# # plt.figure(figsize=(10, 6))
# # plt.hist(relative_error, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Relative Error % Distribution')
# # plt.title('Histogram of Relative Error %')
# # plt.xlabel('Relative Error %')
# # plt.ylabel('Frequency')
# # plt.legend()
# # plt.savefig('Relative_error_hist.png')

# # Save the model
# #model.save('uTOV-4layer-128-5kcut_high_mass.h5')
