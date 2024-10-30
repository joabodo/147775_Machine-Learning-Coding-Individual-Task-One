import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/Users/jbodo/Documents/word/Nairobi Office Price Ex (1).csv"
data = pd.read_csv(file_path)

# Extract relevant features: SIZE (X) and PRICE (y)
X = data["SIZE"].values  # Feature (office size)
y = data["PRICE"].values  # Target (office price)

# Function to compute Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    """Calculates the Mean Squared Error between true and predicted values."""
    return np.mean((y_true - y_pred) ** 2)

# Function to perform a single step of Gradient Descent
def gradient_descent(X, y, m, c, learning_rate):

    N = len(y)  # Number of data points
    y_pred = m * X + c  # Predictions using current weights

    # Compute the gradients for m and c
    dm = (-2 / N) * np.sum(X * (y - y_pred))
    dc = (-2 / N) * np.sum(y - y_pred)

    # Update the weights using the gradients
    m -= learning_rate * dm
    c -= learning_rate * dc

    return m, c

# Initialize model parameters randomly
np.random.seed(42)  # For reproducibility
m = np.random.rand()  # Random slope
c = np.random.rand()  # Random intercept

# Set hyperparameters
learning_rate = 0.0001  # Controls the step size in gradient descent
epochs = 10  # Number of times to iterate over the entire dataset

# Training loop for gradient descent
for epoch in range(epochs):
    # Predict values using the current model
    y_pred = m * X + c

    # Calculate the error (MSE) for the current epoch
    error = mean_squared_error(y, y_pred)
    print(f"Epoch {epoch + 1}: MSE = {error:.4f}")

    # Update the weights using gradient descent
    m, c = gradient_descent(X, y, m, c, learning_rate)

# Plot the final line of best fit
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')  # Scatter plot of original data
plt.plot(X, m * X + c, color='red', label='Line of Best Fit')  # Best fit line
plt.xlabel('Office Size (sq. ft)')
plt.ylabel('Office Price')
plt.title('Linear Regression - Line of Best Fit')
plt.legend()
plt.show()

# Predict the price for an office of 100 sq. ft
predicted_price = m * 100 + c
print(f"The predicted price for an office of 100 sq. ft is: {predicted_price:.2f}")
