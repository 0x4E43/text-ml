import csv
from io import FileIO, TextIOWrapper
import os
from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from tensorflow import keras
import pandas as pd
import tensorflow_hub as hub
from textml import settings

from keras import Layer
# modelFile = FileSystemStorage().open('./model/model-bert.h5', 'rb')
model_path = os.path.join(settings.BASE_DIR, 'model', 'model-bert.h5')
print(model_path)

model = keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

def index(request):
    if request.method == 'POST':
        try:
            myfile = request.FILES['file-upload']
            print("Received file:", myfile.name)
            csv_data = pd.read_csv(myfile)
            print(csv_data.head(5))
            pred = model.predict(csv_data['text'])
            print(pred)
            return render(request, 'index.html', {"upload": "true"})
        except Exception as e:
            print("Error uploading file:", e)
    else:
        print("GET request received")
    return render(request, 'index.html', {"upload": "false"})

def home(request):
    return HttpResponse("home Page works")