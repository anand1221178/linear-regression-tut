import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Sample 150 x-values from a Normal distribution with mean 0 and standard deviation 10
x = np.random.normal(0,10,150)

# Sample true theta values
theta = np.random.uniform(size=3)

# Construct the design matrix with features {1, x, x^2}
X = np.stack((np.ones(150), x, x**2), axis=1)

# Calculate y-values (without noise)
y = X.dot(theta)

# Add random noise to the y-values
noise = np.random.normal(loc=0, scale=8, size=150)
y = y + noise

# Plot the data
plt.scatter(x, y)
plt.xlabel("x-values")
plt.ylabel("y-values")
plt.title("Generated Data for Linear Regression")
plt.grid(True)
plt.show()

# Split the data into training, validation, and test sets (70%/15%/15%)
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

# Print the shapes of the datasets
print("Training data shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

print("\nValidation data shapes:")
print(f"X_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")

print("\nTest data shapes:")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")

theta_hat = np.linalg.pinv(X_train).dot(y_train) # Calculate the estimated theta values
print("Learned parameters closed-form solution:" + str(theta_hat))

diff = theta_hat - theta
print("Difference between learned and true parameters:", diff)

def mean_squared_error(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)

train_error = mean_squared_error(y_train, X_train.dot(theta_hat))
val_error = mean_squared_error(y_val, X_val.dot(theta_hat))

print("Training Error:", train_error)
print("Validation Error:", val_error)

# Sort for cleaner plot
x_train_sorted = X_train[X_train[:,1].argsort()]  
y_predicted = x_train_sorted.dot(theta_hat)

plt.figure()
plt.scatter(X_train[:, 1], y_train)  # Individual data points
plt.plot(x_train_sorted[:, 1], y_predicted, color="red") # Regression line
# ... (add labels, title, etc. as before)
plt.show()

def gradient_descent(X, y, alpha=0.01, max_iterations=1000,  epsilon=1e-6):
    theta = np.zeros(X.shape[1])
    errors = []

    for i in range(max_iterations):
        y_predicted = X.dot(theta)  # Predict y values
        error = y_predicted - y
        cost = np.mean(error**2)  # Mean Squared Error (MSE)
        gradient = (2/len(X)) * X.T.dot(error)

        theta = theta - alpha * gradient  # Update parameters 

        if i % 20 == 0:  # Track error every 20 iterations
            errors.append(cost)

    return theta, errors

theta, errors = gradient_descent(X_train, y_train)
print("Final parameter values (gradient descent):", theta)

plt.plot(range(0, len(errors) * 20, 20), errors)  # Plot errors vs iterations
plt.xlabel("Iterations")
plt.ylabel("Training Error (MSE)")
plt.title("Training Error in Gradient Descent")
plt.grid(True)
plt.show()
