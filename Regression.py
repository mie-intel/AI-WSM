import itertools
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

    def mean_squared_error(self, x_data=None, y_data=None):
        if x_data is None:
            x_data = self.x_raw
        else:
            x_data = x_data.to_numpy().tolist()
        if y_data is None:
            y_data = self.y
        else:
            y_data = y_data.to_numpy().tolist()
        mse = sum((yi - self.predict(xi))**2 for xi,
                  yi in zip(x_data, y_data)) / len(y_data)
        return mse

    def mean_absolute_error(self, x_data=None, y_data=None):
        if x_data is None:
            x_data = self.x_raw
        else:
            x_data = x_data.to_numpy().tolist()
        if y_data is None:
            y_data = self.y
        else:
            y_data = y_data.to_numpy().tolist()
        mae = sum(abs(yi - self.predict(xi))
                  for xi, yi in zip(x_data, y_data)) / len(y_data)
        return mae

    def r2_score(self, x_data=None, y_data=None):
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

    def evaluate_all(self, x_data=None, y_data=None):
        mse = self.mean_squared_error(x_data, y_data)
        mae = self.mean_absolute_error(x_data, y_data)
        r2 = self.r2_score(x_data, y_data)
        return {"MSE": mse, "MAE": mae, "R2": r2}
