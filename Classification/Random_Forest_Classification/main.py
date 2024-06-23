import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% data

data = pd.read_csv("data.csv")
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
data.diagnosis = [1 if each=="M" else 0 for each in data.diagnosis]

y=data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

#%% normalization

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

#%% train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=3)

#%% decision tree classifier
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

print("decision tree score: {}".format(dt.score(x_test, y_test)))

#%% random forest classification ensamble learning

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=4,random_state=1)
rf.fit(x_train, y_train)
print("Random forest algorithm score: {}".format(rf.score(x_test, y_test)))

#%% best estimator it is 4

# rf_list=[]
# for each in range(1,100):
#     rf2=RandomForestClassifier(n_estimators=each, random_state=1)
#     rf2.fit(x_train, y_train)
#     rf_list.append(rf2.score(x_test, y_test))
#     print("{}th Random forest algorithm score: {}".format(each, rf2.score(x_test, y_test)))

# plt.plot(range(1,100), rf_list)
# plt.show()






