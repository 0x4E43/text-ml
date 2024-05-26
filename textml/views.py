import csv
from io import FileIO, TextIOWrapper
from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import keras
import pandas as pd


model = keras.models.load_model('model-bert.h5')
# def index(request):
#     try:
#         return render(request, 'index.html')
#     except Exception as e:
#         print(e)
#         print("Error rendering template: {}", e)

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