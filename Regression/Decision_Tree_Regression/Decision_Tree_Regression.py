import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data import
df = pd.read_csv('decision_tree_regression_dataset.csv', sep = ';', header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#desicion tree regression
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

x_ = np.arange(min(x), max(x), 0.001).reshape(-1,1)
y_head = tree_reg.predict(x_)

#visualization
plt.scatter(x, y, color='blue', label="Data")
plt.plot(x_, y_head, color='red', label="Decision Tree Regres")

plt.xlabel("Tribun seviyesi")
plt.ylabel("Ücret")
plt.legend()
plt.show()




