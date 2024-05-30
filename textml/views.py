import os
from django.shortcuts import render

import pandas as pd
from textml import settings


from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np


# modelFile = FileSystemStorage().open('./model/model-bert.h5', 'rb')


model_path = os.path.join(settings.BASE_DIR, 'model', 'nim_bert_model.h5')
print(model_path)

model = keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})






def index(request):
    if request.method == 'POST':
        try:
            myfile = request.FILES['file-upload']
            print("Received file:", myfile.name)
            data = pd.read_csv(myfile)
            data.dropna(subset=['text'], inplace=True)  # Drop rows where 'text' is NaN

            # Initialize and fit the tokenizer
            tokenizer = Tokenizer(oov_token='<00V>')
            tokenizer.fit_on_texts(data['text'])

            # Define max_len based on the training data
            max_len = 857  # Ensure this matches the input length expected by the model

            # Tokenize and pad the text data
            text_sequences = tokenizer.texts_to_sequences(data['text'])
            X_test = pad_sequences(text_sequences, maxlen=max_len)  # Use max_len to pad sequences

            # Ensure the input shape is correct
            print(f"X_test shape: {X_test.shape}")

            # Load the model
            model2 = keras.models.load_model("./model/nimai_model.h5")
            print("Model Summary: ", model.summary())
            # Initialize an array to store predictions for each fold
            fold_preds = np.zeros(shape=(len(data),), dtype='float32')

            # Predict using the model
            y_probabilities = model2.predict(X_test)
            y_probabilities = y_probabilities.squeeze()  # Remove unnecessary dimensions

            # Apply sigmoid function to ensure probabilities are in the range [0, 1]
            y_probabilities = 1 / (1 + np.exp(-y_probabilities))

            # Calculate percentages
            y_percentages = np.round(y_probabilities * 100, 2)

            # Round percentages to two decimal points
            y_percentages_rounded = np.round(y_percentages, 2)

            # Add the rounded percentages to fold_preds
            fold_preds += y_percentages_rounded
            print(fold_preds)
            # Create a new data frame and associate prediction
            result_df = data.copy()
            result_df['predictions'] = fold_preds
            result_dict = result_df.to_dict(orient='records')
            print("Prediction: \n", result_dict)
            return render(request, 'index.html', {"upload": True, "predictions": result_dict})
        except Exception as e:
            print("Error uploading file:", str(e))
            return render(request, 'index.html', {"upload": False, "predictions": None})
    else:
        print("GET request received")
        return render(request, 'index.html', {"upload": False, "predictions": None})
