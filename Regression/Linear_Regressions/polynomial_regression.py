import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr

#data
df = pd.read_csv('polynomial_regression.csv', sep = ';')

#data visualiztion
linear_reg = lr()

y=df.araba_max_hiz.values.reshape(-1,1)
x=df.araba_fiyat.values.reshape(-1,1)
linear_reg.fit(x,y)

b0=linear_reg.intercept_
print("b0: ", b0)

b1=linear_reg.coef_
print("b1: ", b1)

plt.scatter(x,y,color='blue', label="data")
# plt.xlabel("Araba Maximum Hız (Km/h)")
# plt.ylabel("Araba Fiyat (Bin TL)")
plt.show()

#linear reg b0 + b1*x
lr = lr()
lr.fit(x,y)
y_head = lr.predict(x)

plt.plot(x,y_head,color='red', label="linear reg")
plt.legend()
plt.show()

#polynomial reg => b0+b1*x + b2*x**2 + b3*x**3 + ... + bn*x**n
#parabolic

from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import PolynomialFeatures as pr

polynomial_regression = pr(degree=2)

x_polynomial = polynomial_regression.fit_transform(x)

linear_reg2 = lr()
linear_reg2.fit(x_polynomial, y)

y_head2 = linear_reg2.predict(x_polynomial)

plt.plot(x, y_head2, color='orange', label="Polynomial Regression")
plt.legend()
plt.show()




