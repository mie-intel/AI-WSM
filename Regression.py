from mpl_toolkits.mplot3d import Axes3D
import itertools
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd


class PolynomialRegression:
    def fit(self, x, y, order=2):
        # Jika input berupa DataFrame, ambil nilai sebagai array 2D
        if isinstance(x, pd.DataFrame):
            x = x.values.tolist()
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.flatten().tolist()

        self.x_raw = x  # List of tuples: [(x1, x2, ...), ...]
        self.y = y
        self.order = order
        self.len = len(x)
        self.num_features = len(x[0]) if len(x) > 0 else 0
        self.buildFeatures()
        self.getCoefficients()

    def buildFeatures(self):
        # Buat semua kombinasi eksponen total <= order
        self.exponents = [
            exp for exp in itertools.product(range(self.order + 1), repeat=self.num_features)
            if sum(exp) <= self.order
        ]

        self.X_poly = []  # Fitur polinomial dari x_raw
        for x in self.x_raw:
            row = [self.applyExponent(x, exps) for exps in self.exponents]
            self.X_poly.append(row)

    def applyExponent(self, x, exps):
        result = 1
        for xi, ei in zip(x, exps):
            # print("xi:", xi, "ei:", ei)
            result *= xi ** ei
        return result

    def getMatrix(self, printMatrix=False):
        m = len(self.X_poly)
        n = len(self.exponents)

        self.A = [[0.0] * n for _ in range(n)]
        self.b = [0.0] * n

        for i in range(n):
            for j in range(n):
                self.A[i][j] = sum(self.X_poly[k][i] *
                                   self.X_poly[k][j] for k in range(m))
            self.b[i] = sum(self.X_poly[k][i] * self.y[k] for k in range(m))

        if printMatrix:
            print("Matrix A (X^T X):")
            for row in self.A:
                print(row)
            print("Vector b (X^T y):", self.b)

        return self.A, self.b

    def getCoefficients(self):
        self.getMatrix()
        self.coefficients = np.linalg.solve(self.A, self.b)
        return self.coefficients

    def predict(self, x):
        features = [self.applyExponent(x, exps) for exps in self.exponents]
        return sum(c * f for c, f in zip(self.coefficients, features))

    def getAllPrediction(self, data=None):
        if data is None:
            data = self.x_raw
        else:
            data = data.to_numpy().tolist()
        print("Data untuk prediksi:", data)
        print([xi for xi in data])
        return pd.Series([self.predict(xi) for xi in data])

    def getSy(self, x_data=None, y_data=None):
        if x_data is None:
            x_data = self.x_raw
        if y_data is None:
            y_data = self.y
        y_avg = sum(y_data) / len(y_data)
        res = sum((yi - y_avg) ** 2 for yi in y_data)
        return math.sqrt(res / (len(y_data) - 1))

    def getSypx(self, x_data=None, y_data=None):
        if x_data is None:
            x_data = self.x_raw
        if y_data is None:
            y_data = self.y
        res = sum((y_data[i] - self.predict(x_data[i]))
                  ** 2 for i in range(len(y_data)))
        return math.sqrt(res / (len(x_data) - len(self.coefficients)))

    def eval(self, x_data=None, y_data=None):
        if x_data is None:
            x_data = self.x_raw
        else:
            x_data = x_data.to_numpy().tolist()
        if y_data is None:
            y_data = self.y
        else:
            y_data = y_data.to_numpy().tolist()
        y_mean = sum(y_data) / len(y_data)
        ss_tot = sum((yi - y_mean) ** 2 for yi in y_data)
        ss_res = sum((yi - self.predict(xi)) **
                     2 for xi, yi in zip(x_data, y_data))
        r2 = 1 - (ss_res / ss_tot)
        return r2


class SinusoidalRegression:
    def __init__(self, max_freq=3):
        self.max_freq = max_freq

    def fit(self, x, y):
        if isinstance(x, pd.DataFrame):
            x = x.values.tolist()
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.flatten().tolist()

        self.x_raw = x
        self.y = y
        self.num_features = len(x[0]) if len(x) > 0 else 0
        self.buildFeatures()
        self.getCoefficients()

    def buildFeatures(self):
        self.X_transformed = []
        self.feature_names = []

        for row in self.x_raw:
            transformed = [1]  # Bias
            for i, val in enumerate(row):
                for freq in range(1, self.max_freq + 1):
                    transformed.append(np.sin(freq * val))
                    transformed.append(np.cos(freq * val))
                    self.feature_names.append(f"sin({freq}x{i+1})")
                    self.feature_names.append(f"cos({freq}x{i+1})")
            self.X_transformed.append(transformed)

    def getMatrix(self):
        m = len(self.X_transformed)
        n = len(self.X_transformed[0])
        self.A = [[0.0] * n for _ in range(n)]
        self.b = [0.0] * n

        for i in range(n):
            for j in range(n):
                self.A[i][j] = sum(self.X_transformed[k][i]
                                   * self.X_transformed[k][j] for k in range(m))
            self.b[i] = sum(self.X_transformed[k][i] * self.y[k]
                            for k in range(m))

        return self.A, self.b

    def getCoefficients(self):
        self.getMatrix()
        self.coefficients = np.linalg.solve(self.A, self.b)
        return self.coefficients

    def predict(self, x):
        transformed = [1]
        for i, val in enumerate(x):
            for freq in range(1, self.max_freq + 1):
                transformed.append(np.sin(freq * val))
                transformed.append(np.cos(freq * val))
        return sum(c * f for c, f in zip(self.coefficients, transformed))

    def getAllPrediction(self, data=None):
        if data is None:
            data = self.x_raw
        else:
            data = data.to_numpy().tolist()
        return pd.Series([self.predict(xi) for xi in data])

    def eval(self, x_data=None, y_data=None):
        if x_data is None:
            x_data = self.x_raw
        else:
            x_data = x_data.to_numpy().tolist()
        if y_data is None:
            y_data = self.y
        else:
            y_data = y_data.to_numpy().tolist()
        y_mean = sum(y_data) / len(y_data)
        ss_tot = sum((yi - y_mean) ** 2 for yi in y_data)
        ss_res = sum((yi - self.predict(xi)) **
                     2 for xi, yi in zip(x_data, y_data))
        r2 = 1 - (ss_res / ss_tot)
        return r2


def plot_regression_surface(model, X_df, y_true):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot titik data asli
    ax.scatter(X_df["A"], X_df["B"], y_true, color='blue', label='Data Asli')

    # Buat grid prediksi permukaan
    A_range = np.linspace(X_df["A"].min(), X_df["A"].max(), 30)
    B_range = np.linspace(X_df["B"].min(), X_df["B"].max(), 30)
    A_grid, B_grid = np.meshgrid(A_range, B_range)

    # Flatten dan prediksi
    A_flat = A_grid.ravel()
    B_flat = B_grid.ravel()
    X_pred = list(zip(A_flat, B_flat))
    Y_pred = np.array([model.predict(xi) for xi in X_pred])

    # Bentuk ulang permukaan prediksi ke grid 2D
    Y_grid = Y_pred.reshape(A_grid.shape)

    # Plot permukaan prediksi
    ax.plot_surface(A_grid, B_grid, Y_grid, color='orange',
                    alpha=0.5, label="Regresi")

    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.set_zlabel("Y")
    ax.set_title("Regresi Polinomial 3D (2 Fitur)")
    plt.legend()
    plt.show()


# Buat DataFrame
# data = pd.DataFrame({
#     "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     "B": [2, 1, 4, 2, 5, 3, 6, 4, 7, 5],
#     "Y": [10, 9, 30, 25, 50, 45, 70, 65, 90, 85]
# })

# # Pisahkan fitur dan label
# X_train = data[["A", "B"]][:7]
# y_train = data["Y"][:7]
# X_test = data[["A", "B"]][7:]
# y_test = data["Y"][7:]

# # print(X_train)
# # print(y_train)
# # print(X_test)
# # print(y_test)

# model = PolynomialRegression()
# model.fit(X_train, y_train, order=2)

# # print("Koefisien:", model.coefficients)
# print("Prediksi semua:", model.getAllPrediction(X_test))
# print("R²:", model.eval(X_test, y_test))

# X = data[["A", "B"]]
# y = data["Y"]
# ypred = model.getAllPrediction(X)
# print("\nPRED", ypred)
# print("\nY", y)
# print("R²:", model.eval(X, y))
# print(type(y))

# plot_regression_surface(model, X, y)


# Data dummy (mempunyai pola sinusoidal)
# Data dibuat agar Y = 10 * sin(A) + 5 * cos(B) + noise

def plot_regression_surface_sinusoidal(model, X_df, y_true):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot titik data asli
    ax.scatter(X_df["A"], X_df["B"], y_true, color='blue', label='Data Asli')

    # Buat grid prediksi permukaan
    A_range = np.linspace(X_df["A"].min(), X_df["A"].max(), 30)
    B_range = np.linspace(X_df["B"].min(), X_df["B"].max(), 30)
    A_grid, B_grid = np.meshgrid(A_range, B_range)

    # Flatten dan prediksi
    A_flat = A_grid.ravel()
    B_flat = B_grid.ravel()
    X_pred = list(zip(A_flat, B_flat))
    Y_pred = np.array([model.predict(xi) for xi in X_pred])

    # Bentuk ulang permukaan prediksi ke grid 2D
    Y_grid = Y_pred.reshape(A_grid.shape)

    # Plot permukaan prediksi
    ax.plot_surface(A_grid, B_grid, Y_grid, color='orange', alpha=0.6)

    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.set_zlabel("Y")
    ax.set_title("Regresi Sinusoidal 3D")
    plt.show()


# np.random.seed(42)
# A = np.linspace(0, 2 * np.pi, 30)
# B = np.linspace(0, 2 * np.pi, 30)
# A_grid, B_grid = np.meshgrid(A, B)
# A_flat = A_grid.ravel()
# B_flat = B_grid.ravel()

# noise = np.random.normal(0, 1, len(A_flat))
# Y = 10 * np.sin(A_flat) + 5 * np.cos(B_flat) + noise

# data = pd.DataFrame({
#     "A": A_flat,
#     "B": B_flat,
#     "Y": Y
# })

# # Pisahkan data training dan testing
# train_data = data.sample(frac=0.8, random_state=1)
# test_data = data.drop(train_data.index)

# X_train = train_data[["A", "B"]]
# y_train = train_data["Y"]
# X_test = test_data[["A", "B"]]
# y_test = test_data["Y"]

# model = SinusoidalRegression(max_freq=3)
# model.fit(X_train, y_train)

# print("R² (train):", model.eval(X_train, y_train))
# print("R² (test):", model.eval(X_test, y_test))

# plot_regression_surface(model, X_test, y_test)
