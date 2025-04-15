import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate Synthetic Data
def generate_data(n_samples=500):
    np.random.seed(0)  # For reproducibility
    X = np.random.randn(n_samples, 2)  # Two features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Binary labels (0 or 1)
    return X, y

# Step 2: Visualize the Data
def plot_data(X, y, title="Dataset"):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title(title)
    plt.show()

# Step 3: Split the Data
def train_test_split(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

# Step 4: Implement Perceptron Class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.random.randn(input_size + 1)  # Including bias

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        return np.array([self.step_function(np.dot(x, self.weights)) for x in X])

    def train(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        errors = []
        for epoch in range(self.epochs):
            total_error = 0
            for xi, target in zip(X, y):
                output = self.step_function(np.dot(xi, self.weights))
                error = target - output
                total_error += abs(error)
                self.weights += self.lr * error * xi  # Weight update
            errors.append(total_error)
            print(f"Epoch {epoch+1}, Error: {total_error}")
        return errors

# Step 5: Plot Decision Boundary
def plot_decision_boundary(X, y, perceptron):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.bwr)
    plot_data(X, y, title="Decision Boundary")

# Run the Perceptron
X, y = generate_data()
plot_data(X, y, "Generated Data")
X_train, y_train, X_test, y_test = train_test_split(X, y)
perceptron = Perceptron(input_size=2, learning_rate=0.01, epochs=10)
errors = perceptron.train(X_train, y_train)
plot_decision_boundary(X_test, y_test, perceptron)

# Evaluate on Test Set
y_pred = perceptron.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
