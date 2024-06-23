import pandas as pd

# %% import twitter data
data = pd.read_csv(r"gender_classifier.csv",encoding = "latin1")
data = pd.concat([data.gender,data.description],axis=1)
data.dropna(axis = 0,inplace = True)
data.gender = [1 if each == "female" else 0 for each in data.gender]

#%% clening data 
# regular expression RE mesela "[^a-zA-Z]"
import re

first_description = data.description[4]
description = re.sub("[^a-zA-Z]"," ",first_description)  # a dan z ye ve A dan Z ye kadar olan harfleri bulma geri kalanları " " (space) ile degistirme
description = description.lower()   # buyuk harftan kucuk harfe cevirme
