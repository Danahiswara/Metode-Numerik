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

# Transformasi y dengan logaritma natural
log_y = np.log(y)

# Membuat model regresi linear pada data yang telah ditransformasi
model = LinearRegression()
model.fit(X, log_y)

# Prediksi nilai log_y
log_y_pred = model.predict(X)

# Konversi kembali prediksi log_y ke skala asli dengan eksponensial
y_pred = np.exp(log_y_pred)

# Menghitung galat RMS
rms_exp = np.sqrt(mean_squared_error(y, y_pred))

# Menampilkan koefisien dan intercept dari model pada skala log
print('Koefisien (dalam skala log): ', model.coef_)
print('Intercept (dalam skala log): ', model.intercept_)
print('RMS Error: ', rms_exp)

# Plot hasil regresi eksponensial
plt.scatter(X, y, color='blue', label='Data Asli')
plt.plot(X, y_pred, color='red', label='Regresi Eksponensial')
plt.xlabel('Kolom ke-5')
plt.ylabel('Kolom ke-6')
plt.legend()
plt.show()
