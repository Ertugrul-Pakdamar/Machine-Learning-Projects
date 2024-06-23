# %% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% read csv
data = pd.read_csv("data.csv")
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

M=data[data.diagnosis=="M"]
B=data[data.diagnosis=="B"]


#%% basic visualization

plt.scatter(M.radius_mean, M.texture_mean, color='red', label="Kötü")
plt.scatter(B.radius_mean, B.texture_mean, color='green', label="İyi")
plt.legend()
plt.show()

#%%
data.diagnosis = [1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
x_data=data.drop(["diagnosis"], axis=1)

#%% normalization

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

#%% train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

#%% knn algorithm

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

#%% test prediction
prediction = knn.predict(x_test)
print("{}nn score: {}".format(3, knn.score(x_test, y_test)))

#%% best k value finder it is 20
k_list = []
for each in range(1,100):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train, y_train)
    k_list.append(knn2.score(x_test, y_test))
    print("{}nn score: {}".format(each, knn2.score(x_test, y_test)))

# plt.plot(range(1,100), k_list, color='blue')
# plt.show()








