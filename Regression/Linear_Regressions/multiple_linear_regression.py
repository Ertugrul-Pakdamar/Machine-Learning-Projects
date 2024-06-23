import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr

df = pd.read_csv('multiple_linear_regression_dataset.csv', sep = ';')

multiple_linear_reg = lr()

x=df.iloc[:, [0,2]].values
y=df.maas.values.reshape(-1,1)
multiple_linear_reg.fit(x,y)

print("b0: ", multiple_linear_reg.intercept_)
print("b1, b2: ", multiple_linear_reg.coef_)

print("35 yaş, 10-5 yıl deneyimler: ", multiple_linear_reg.predict(np.array([[10,35],[5,35]])))


