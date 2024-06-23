import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('linear_regression_dataset.csv', sep = ';')

# plt.scatter(df.deneyim, df.maas)
# plt.xlabel("Deneyim (Yıl)")
# plt.ylabel("Maaş (TL)")
# plt.show()

from sklearn.linear_model import LinearRegression as lr

linear_reg = lr()
x=df.deneyim.values.reshape(-1,1)
y=df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

b0 = linear_reg.intercept_
print("b0: ", b0)

b1 = linear_reg.coef_
print("b1: ", b1)

# def predict(x):
#     return (b0 + b1 * x)

array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).reshape(-1, 1) #deneyim

plt.scatter(df.deneyim, df.maas)
plt.xlabel("Deneyim (Yıl)")
plt.ylabel("Maaş (TL)")
plt.show()

y_head = linear_reg.predict(array) # maaş

plt.plot(array, y_head, color='red') # linear regression line


