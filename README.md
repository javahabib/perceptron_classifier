# Perceptron Binary Classifier

This project demonstrates a basic implementation of the **Perceptron algorithm**, a foundational machine learning model used for binary classification. It uses synthetic data, a custom Perceptron class, and matplotlib for visualization, including the decision boundary.

---

## Features

- Synthetic data generation (2D binary classification)
- Visualization of raw data and decision boundary
- Custom Perceptron class built from scratch (no ML libraries)
- Training with learning rate and epochs
- Test set evaluation and accuracy calculation

---

## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install dependencies using pip:

```bash
pip install numpy matplotlib
```
##  How It Works
-Generate Data
Random 2D points with binary labels, linearly separable.

-Visualize Data
Scatter plot of Class 0 (red) and Class 1 (blue).

-Train-Test Split
Simple 80/20 split of the dataset.

-Perceptron Class
A basic neural unit with a step activation function.

-Training Loop
Adjusts weights using the Perceptron update rule over multiple epochs.

-Decision Boundary Plot
Visualizes how the model separates the classes.

-Accuracy on Test Set
Prints percentage accuracy of predictions.

##  Example Output
Epoch-wise training error printed on console










