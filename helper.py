import numpy as np
import pandas as pd
import re
import string
import pickle

from nltk.stem import PorterStemmer
ps = PorterStemmer()

#load model

with open('static/model/model.pickle','rb') as f:
    model = pickle.load(f)

#load stop words

with open('static/model/corpora/stopwords/english', 'r') as file:
    sw=file.read().splitlines()

#load tokens

vocab =pd.read_csv('static/model/vocabulary.txt', header=None)
tokens=vocab[0].tolist()



import string

def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocessing(text):
    data= pd.DataFrame([text], columns=['tweet'])
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data["tweet"] = data['tweet'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', part, flags=re.MULTILINE) for part in x.split()))
    data["tweet"] = data["tweet"].apply(remove_punctuation)
    data["tweet"] = data['tweet'].str.replace(r'\d+', '', regex=True)
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(word for word in x.split() if word not in sw))
    data["tweet"]=data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data["tweet"]

import numpy as np

def vectorizer(ds):
    vectorized_lst = []

    for sentence in ds:
        sentence_lst = np.zeros(len(tokens))

        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i] = 1  # Mark the presence of the word in the sentence

        vectorized_lst.append(sentence_lst)

    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)

    return vectorized_lst_new

def get_prediction(vectorized_text):
    prediction=model.predict(vectorized_text)
    if prediction == 1:
        return 'negative'
    else:
        return 'positive'


