# Step 1: Prepare the Data

import numpy as np
import matplotlib.pyplot as plt 
x = np.array([
    [1500, 3, 20],  # House 1: 1500 sq ft, 3 rooms, 20 years old
    [1800, 4, 15],  # House 2: 1800 sq ft, 4 rooms, 15 years old
    [2400, 3, 10],  # House 3: 2400 sq ft, 3 rooms, 10 years old
    [3000, 5, 5]    # House 4: 3000 sq ft, 5 rooms, 5 years old
])
y = np.array([400, 460, 540, 610])  # Price of the houses

# Step 2: Normalize the Data
x_mean = np.mean(x, axis=0)  # Mean of each column
x_std = np.std(x, axis=0)    # Standard deviation of each column
x_norm = (x - x_mean) / x_std  # Normalize each feature (standard score)



# Step 3: Add Bias Term
x_b = np.c_[np.ones((x_norm.shape[0], 1)), x_norm]  # Add column of ones

# Step 4: Initialize Parameters
theta = np.zeros(x_b.shape[1])  # Initialize all parameters to 0

# Step 5: Define the Cost Function
def cost_function(x, y, theta):
    m = len(y)  # Number of training examples
    predictions = x.dot(theta)  # Predicted prices
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)  # Mean squared error
    return cost

# Step 6: Perform Gradient Descent
def gradient_descent(x, y, theta, learning_rate, iterations):
    cost_history=[]
    m = len(y)  # Number of training examples
    for _ in range(iterations):
        predictions = x.dot(theta)  # Predicted prices
        errors = predictions - y  # Errors between predictions and actual values
        gradients = (1 / m) * x.T.dot(errors)  # Partial derivatives for each parameter
        theta -= learning_rate * gradients  # Update parameters
        cost_history.append(cost_function(x,y,theta))
    return theta , cost_history

# Step 7: Train the Model
learning_rate = 0.01  # Step size for gradient descent
iterations = 1000  # Number of iterations for training
optimized_theta , cost_history = gradient_descent(x_b, y, theta, learning_rate, iterations)

# Step 8: Make Predictions
new_house = np.array([2000, 4, 10])
new_house_norm = (new_house - x_mean) / x_std  # Normalize new data
new_house_b = np.r_[1, new_house_norm]  # Add bias term (1)
predicted_price = new_house_b.dot(theta)  # Predict using the trained model

# Step 9: Print Results
print(f'Final Cost : {cost_history[-1]}')
print(f"Trained Parameters (theta): {theta}")
print(f"Predicted price for the new house: ${predicted_price * 1000:.2f}")

plt.plot(range(iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Over Iterations")
plt.show()