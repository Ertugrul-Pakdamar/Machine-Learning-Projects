import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import nltk as nlp

# %% import twitter data
data = pd.read_csv(r"gender_classifier.csv",encoding = "latin1")
data = pd.concat([data.gender,data.description],axis=1)
data.dropna(axis = 0,inplace = True)
data.gender = [1 if each == "female" else 0 for each in data.gender]

#%% clening data 
# regular expression RE mesela "[^a-zA-Z]"

first_description = data.description[4]
description = re.sub("[^a-zA-Z]"," ",first_description)  # a dan z ye ve A dan Z ye kadar olan harfleri bulma geri kalanları " " (space) ile degistirme
description = description.lower()   # buyuk harftan kucuk harfe cevirme

# %% stopwords (irrelavent words) gereksiz kelimeler
nltk.download("stopwords")
nltk.download('punkt')

# description = description.split()

# split yerine tokenizer kullanabilirim
description = nltk.word_tokenize(description)

# split kullanırsam "shouldn't " gibi kelimeler "should" ve "not" diye ikiye ayrılmaz ama word_tokenize() kullanirsam ayrilir
# %%
# greksiz kelimeleri cikar
description = [ word for word in description if not word in set(stopwords.words("english"))]

# %%             
# lemmatazation loved => love   gitmeyecegim = > git
nltk.download('wordnet')

lemma = nlp.WordNetLemmatizer()
description = [ lemma.lemmatize(word) for word in description] 

description = " ".join(description)

