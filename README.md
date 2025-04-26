---

# Linear Regression from Scratch using NumPy

This project implements a simple **Linear Regression** model **from scratch** (without using scikit-learn), to predict house prices based on features like size, number of rooms, and age.

---

## 🛠️ Project Details

- **Dataset:**  
  A small synthetic dataset of 4 houses with features:
  - Area (square feet)
  - Number of rooms
  - Age of the house (years)

- **Preprocessing:**
  - Feature normalization (Standard Score)
  - Adding a bias term manually.

- **Model:**
  - Linear Regression using **Gradient Descent**.
  - Cost function: **Mean Squared Error (MSE)**.

- **Libraries Used:**
  - NumPy
  - Matplotlib

---

## 📈 Workflow

1. Prepare and normalize the data.
2. Add bias term to the input features.
3. Initialize model parameters (theta).
4. Define the cost function.
5. Perform gradient descent to optimize parameters.
6. Train the model for a number of iterations.
7. Make predictions on new data.
8. Visualize cost reduction over training iterations.

---

## 🧪 Results

- Final cost after training.
- Trained model parameters (theta).
- Predicted price for a new house.
- A plot showing the decrease in the cost function over time.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/OsamaAt/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```

3. Run the script:
   ```bash
   python linear_regression_from_scratch.py
   ```

---

## 🔥 Sample Prediction

For a house with:
- 2000 sq ft area
- 4 rooms
- 10 years old

The model predicts the house price as approximately **\$XXX,XXX**.

---

## 📚 Learning Outcomes

- Understanding how Linear Regression works internally.
- Implementing Gradient Descent manually.
- Importance of feature normalization.
- Hands-on experience with NumPy array operations.

---

## ✍️ Author

- [OsamaAt](https://github.com/OsamaAt)

---
