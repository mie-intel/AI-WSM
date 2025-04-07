import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('data_clean.csv')
print(data.head())


class Regression:
    def fit(self, X, y, pangkat):
        self.pangkat = pangkat
        self.feature_names = X.columns
        self.y = y.values.reshape(-1, 1)

        # Bangun fitur polinomial (tanpa interaksi antar fitur)
        poly_features = [np.ones((len(X), 1))]  # Bias (x^0)

        for col in X.columns:
            x_col = X[col].values.reshape(-1, 1)
            for p in range(1, pangkat + 1):
                poly_features.append(x_col ** p)

        self.X_poly = np.hstack(poly_features)

        # Hitung theta (koefisien regresi)
        self.theta = np.linalg.pinv(self.X_poly) @ self.y

    def predict(self, X_input):
        poly_features = [np.ones((len(X_input), 1))]  # Bias term

        for col in X_input.columns:
            x_col = X_input[col].values.reshape(-1, 1)
            for p in range(1, self.pangkat + 1):
                poly_features.append(x_col ** p)

        X_input_poly = np.hstack(poly_features)
        return np.dot(X_input_poly, self.theta)

    def get_equation(self):
        eq = "y ="
        idx = 0
        for col in self.feature_names:
            for p in range(1, self.pangkat + 1):
                idx += 1
                eq += f" + ({self.theta[idx][0]:.4f})*{col}^{p}"
        eq = f"{eq} + ({self.theta[0][0]:.4f})"
        return eq


data = data[:1000]

model = Regression()
model.fit(data[['max_wind_speed', 'maximum_temperature']],
          data['daily_rainfall_total'], 1)

# Sekarang bisa prediksi kapan saja:
y_pred = model.predict(data[['max_wind_speed', 'maximum_temperature']])
print(y_pred[:5])
print(model.get_equation())


plt.figure(figsize=(12, 6))

# Pastikan data diurutkan berdasarkan tanggal
data['date'] = pd.to_datetime(data['date'])
sorted_data = data.sort_values(by='date')

# Plot data aktual sebagai garis biru
plt.scatter(sorted_data['date'], sorted_data['daily_rainfall_total'], s=10,
            color='blue', label='Actual')

# Plot prediksi sebagai titik merah
plt.plot(sorted_data['date'], y_pred, color='red',
         alpha=0.6, label='Predicted')

# Format sumbu x (tanggal) agar tidak tumpang tindih
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.xlabel('Date')
plt.ylabel('Daily Rainfall Total')
plt.title('Actual vs Predicted Daily Rainfall Total')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
