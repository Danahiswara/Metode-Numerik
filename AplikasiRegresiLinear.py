import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Memuat data dari file CSV
file_path = 'Student_Performance.csv'  # Ganti dengan path file CSV Anda
data = pd.read_csv(file_path)

# Memeriksa kolom data
print(data.columns)

# Menggunakan kolom ke-5 dan ke-6 (indeks 4 dan 5)
X = data.iloc[:, 4].values.reshape(-1, 1)  # Mengambil kolom ke-5 sebagai fitur (X)
y = data.iloc[:, 5].values  # Mengambil kolom ke-6 sebagai target (y)

# Membuat model regresi linear
model = LinearRegression()
model.fit(X, y)

# Prediksi nilai
y_pred = model.predict(X)

# Menghitung galat RMS
rms_linear = np.sqrt(mean_squared_error(y, y_pred))

# Menampilkan koefisien dan intercept
print('Koefisien: ', model.coef_)
print('Intercept: ', model.intercept_)
print('RMS Error: ', rms_linear)

# Plot hasil regresi
plt.scatter(X, y, color='blue', label='Data Asli')
plt.plot(X, y_pred, color='red', label='Regresi Linear')
plt.xlabel('Kolom ke-5')
plt.ylabel('Kolom ke-6')
plt.legend()
plt.show()
