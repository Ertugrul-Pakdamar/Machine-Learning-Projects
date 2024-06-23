import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data import
df = pd.read_csv('random_forest_regression_dataset.csv', sep = ';', header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#ensemble- Random Forest Regression
from sklearn.ensemble import RandomForestRegressor as rfr

rf = rfr(n_estimators=100, random_state=42)
rf.fit(x,y)

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

#visualize
plt.scatter(x, y, color='blue')
plt.plot_date(x_, y_head, color='red')
plt.show()



