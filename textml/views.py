import csv
from io import FileIO, TextIOWrapper
import os
from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from textml import settings


from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np


# modelFile = FileSystemStorage().open('./model/model-bert.h5', 'rb')

import tensorflow_text as text 

model_path = os.path.join(settings.BASE_DIR, 'model', 'nimai_model.h5')
print(model_path)

model = keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

train_data_path = os.path.join(settings.BASE_DIR, 'train_data')

train_essays = pd.read_csv(train_data_path+'/train_essays.csv')
test_df = pd.read_csv(train_data_path+'/test_essays.csv') 

train_essays = train_essays.dropna(how='all')
df = train_essays[['text','generated']]
X = df['text']
Y = df['generated']
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3)
tokenizer = Tokenizer(oov_token='<00V>')
tokenizer.fit_on_texts(x_train)
vocab_len = len(tokenizer.index_word)
max_len = 1364

test_df = pd.read_csv('train_data//test_essays.csv') 


def index(request):
    if request.method == 'POST':
        try:
            myfile = request.FILES['file-upload']
            print("Received file:", myfile.name)
            csv_data = pd.read_csv(myfile)
            print(csv_data.head(5))
            # tokenize word
            data = csv_data['text']
            text_sequences = tokenizer.texts_to_sequences(data)
            X_test= pad_sequences(text_sequences, maxlen=len(data))
            print("X_TEST: ",X_test)
            
            
            fold_preds = np.zeros(shape=(len(data),), dtype='float32')
            y = model.predict(X_test)
            fold_preds += y.squeeze()

            print("Prdiction ", fold_preds)
            return render(request, 'index.html', {"upload": "true"})
        except Exception as e:
            print("Error uploading file:", e)
    else:
        print("GET request received")
    return render(request, 'index.html', {"upload": "false"})

def home(request):
    return HttpResponse("home Page works")