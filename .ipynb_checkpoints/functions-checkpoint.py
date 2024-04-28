import streamlit as st
import requests
import chromadb
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import re



def transform_query(query, sent_model):
    embedding = sent_model.encode(query)
    return embedding


def predict_paddy(new_query, model,embeddings_model):
    # X_train_sent_test = np.array([transform_query(query, sent_model) for query in X_train_sent]).reshape(len(X_train_sent_test), 1, 384)
    embedding = transform_query(new_query, embeddings_model)  
    input_vector = embedding.reshape(1, 1, 384) 

    prediction = model.predict(input_vector)
    print(prediction)

    if prediction > 0.5:
        return "Paddy"
    else:
        return "Not Paddy"

# def get_prediction(query, model, sent_model):
#     # model = tf.keras.models.load_model('saved-models/sent_model_lstm.keras')
#     final_prediction = predict_paddy(query,model,sent_model)
#     return final_prediction

    




# def get_weather(place):
#     # Replace with your actual API endpoint and key
#     api_url = "http://api.openweathermap.org/data/2.5/weather?q={}&appid=0e7749bf58681b089e6cc377e0444c60&units=metric" 
#     response = requests.get(api_url.format(place))

#     if response.status_code == 200:
#         data = response.json()
#         # Extract and display the relevant weather information
#         temperature = data['main']['temp']
#         st.success(f"The temperature in {place} is {round(temperature, 2)} celsius.")
#     else:
#         st.error("Error fetching weather data")