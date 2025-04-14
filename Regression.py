import itertools
import math
import numpy as np
import pandas as pd


class PolynomialRegression:
    def fit(self, x, y, order=2):
        """
        Melatih model regresi polinomial dengan data input x dan target y.
        Parameter:
            x : array-like, bentuknya berupa DataFrame
            y : array-like, target (Series/DataFrame/list)
            order : derajat dari polinomial model
        """
        # Jika x berupa DataFrame, ubah ke list of lists
        if isinstance(x, pd.DataFrame):
            x = x.values.tolist()
        # Jika y berupa Series/DataFrame, ubah ke list 1D
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.flatten().tolist()

        # Simpan data input mentah
        self.x_raw = x
        self.y = y
        self.order = order
        self.len = len(x)
        self.num_features = len(x[0]) if len(x) > 0 else 0

        # Bangun fitur polinomial dan hitung koefisien
        self.buildFeatures()
        self.getCoefficients()

    def buildFeatures(self):
        """
        Membangun fitur polinomial dari x_raw berdasarkan kombinasi pangkat total <= order.
        Misalnya x = [x1, x2], order=2 akan mencakup: x1^0 x2^0, x1^1 x2^0, x1^0 x2^1, x1^2 x2^0, x1^1 x2^1, x1^0 x2^2
        """
        # Semua kombinasi pangkat dari fitur hingga order tertentu
        self.exponents = [
            exp for exp in itertools.product(range(self.order + 1), repeat=self.num_features)
            if sum(exp) <= self.order
        ]

        self.X_poly = []  # Tempat penyimpanan fitur hasil pemangkatan
        for x in self.x_raw:
            # Untuk tiap data x, buat satu baris hasil pemangkatan
            row = [self.applyExponent(x, exps) for exps in self.exponents]
            self.X_poly.append(row)

    def applyExponent(self, x, exps):
        """
        Mengalikan setiap elemen x dengan pangkat sesuai dengan exponents.
        Misal x = [x1, x2], exps = (2, 1) => return x1^2 * x2^1
        """
        result = 1
        for xi, ei in zip(x, exps):
            result *= xi ** ei
        return result

    def getMatrix(self, printMatrix=False):
        """
        Menghitung matriks A dan vektor b untuk menyelesaikan persamaan normal (X^T X)w = X^T y.
        Matriks A = X^T * X, vektor b = X^T * y
        """
        m = len(self.X_poly)  # Jumlah data
        n = len(self.exponents)  # Jumlah fitur polinomial

        # Inisialisasi matriks A dan vektor b dengan 0
        self.A = [[0.0] * n for _ in range(n)]
        self.b = [0.0] * n

        for i in range(n):
            for j in range(n):
                # Hitung elemen A[i][j] = sum dari X_k_i * X_k_j
                self.A[i][j] = sum(self.X_poly[k][i] * self.X_poly[k][j] for k in range(m))
            # Hitung elemen b[i] = sum dari X_k_i * y_k
            self.b[i] = sum(self.X_poly[k][i] * self.y[k] for k in range(m))

        if printMatrix:
            print("Matrix A (X^T X):")
            for row in self.A:
                print(row)
            print("Vector b (X^T y):", self.b)

        return self.A, self.b

    def getCoefficients(self):
        """
        Menyelesaikan sistem persamaan linear untuk mendapatkan koefisien regresi.
        Menggunakan np.linalg.solve untuk menyelesaikan Ax = b.
        """
        self.getMatrix()
        self.coefficients = np.linalg.solve(self.A, self.b)
        return self.coefficients

    def predict(self, x):
        """
        Melakukan prediksi output untuk satu data input x.
        Mengalikan fitur polinomial x dengan koefisien model.
        """
        features = [self.applyExponent(x, exps) for exps in self.exponents]
        return sum(c * f for c, f in zip(self.coefficients, features))

    def getAllPrediction(self, data=None):
        """
        Mengembalikan hasil prediksi untuk semua data.
        Jika data tidak diberikan, maka menggunakan data latih.
        """
        if data is None:
            data = self.x_raw
        else:
            data = data.to_numpy().tolist()
        return pd.Series([self.predict(xi) for xi in data])

    def getSy(self, x_data=None, y_data=None):
        """
        Menghitung simpangan baku dari data target (sebagai pembanding total variasi data).
        """
        if x_data is None:
            x_data = self.x_raw
        if y_data is None:
            y_data = self.y
        y_avg = sum(y_data) / len(y_data)
        res = sum((yi - y_avg) ** 2 for yi in y_data)
        return math.sqrt(res / (len(y_data) - 1))

    def getSypx(self, x_data=None, y_data=None):
        """
        Menghitung galat standar prediksi model (residual error).
        """
        if x_data is None:
            x_data = self.x_raw
        if y_data is None:
            y_data = self.y
        res = sum((y_data[i] - self.predict(x_data[i])) ** 2 for i in range(len(y_data)))
        return math.sqrt(res / (len(x_data) - len(self.coefficients)))

    def mean_squared_error(self, x_data=None, y_data=None):
        """
        Menghitung Mean Squared Error (MSE) antara prediksi dan nilai asli.
        """
        if x_data is None:
            x_data = self.x_raw
        else:
            x_data = x_data.to_numpy().tolist()
        if y_data is None:
            y_data = self.y
        else:
            y_data = y_data.to_numpy().tolist()
        mse = sum((yi - self.predict(xi)) ** 2 for xi, yi in zip(x_data, y_data)) / len(y_data)
        return mse

    def mean_absolute_error(self, x_data=None, y_data=None):
        """
        Menghitung Mean Absolute Error (MAE) antara prediksi dan nilai asli.
        """
        if x_data is None:
            x_data = self.x_raw
        else:
            x_data = x_data.to_numpy().tolist()
        if y_data is None:
            y_data = self.y
        else:
            y_data = y_data.to_numpy().tolist()
        mae = sum(abs(yi - self.predict(xi)) for xi, yi in zip(x_data, y_data)) / len(y_data)
        return mae

    def r2_score(self, x_data=None, y_data=None):
        """
        Menghitung koefisien determinasi R²: proporsi variasi target yang dijelaskan model.
        """
        if x_data is None:
            x_data = self.x_raw
        else:
            x_data = x_data.to_numpy().tolist()
        if y_data is None:
            y_data = self.y
        else:
            y_data = y_data.to_numpy().tolist()

        y_mean = sum(y_data) / len(y_data)
        ss_tot = sum((yi - y_mean) ** 2 for yi in y_data)  # Total sum of squares
        ss_res = sum((yi - self.predict(xi)) ** 2 for xi, yi in zip(x_data, y_data))  # Residual sum
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def evaluate_all(self, x_data=None, y_data=None):
        """
        Mengembalikan semua metrik evaluasi utama: MSE, MAE, dan R²
        """
        mse = self.mean_squared_error(x_data, y_data)
        mae = self.mean_absolute_error(x_data, y_data)
        r2 = self.r2_score(x_data, y_data)
        return {"MSE": mse, "MAE": mae, "R2": r2}
