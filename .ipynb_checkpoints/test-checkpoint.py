__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from functions import *
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
import chromadb
import pandas as pd

# Load models
model = load_model('saved-models/sent_model_lstm.keras')    
sent_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB 
client = chromadb.PersistentClient(path="vectordbs/")
collection = client.get_collection('my_collection')

# Load CSV data 
df = pd.read_csv('FILTERED_COMBINED.csv')
queries = df['QueryText'].tolist()
answers = df['KccAns'].tolist()

# Streamlit UI elements
query_text = st.text_input('Enter your query:')
predict_button = st.button('Predict') 

# Logic for prediction and results display
if predict_button:  # Only execute prediction logic when the button is pressed
    prediction = predict_paddy(query_text, model, sent_model)

    if prediction == 'Not Paddy':
        st.write('The query provided is not related to paddy.')
    else:
        st.write('The query provided is related to paddy.')

retrieve_button = st.button('Retrieve Similar Queries')  # Button to fetch similar queries

if retrieve_button: 
    results = collection.query(
        query_texts=query_text,
        n_results=3,
        include=['documents', 'distances']
    )
    # st.write(results)  # You can uncomment this to inspect the raw results

    similar_query_indexes = [int(x) for x in results['ids'][0]]

    # st.write(similar_query_indexes)
    new_dict = {
        'Query': [queries[i] for i in similar_query_indexes],
        'KccAns': [answers[i] for i in similar_query_indexes]
    }

    

    # st.write(new_dict)

    newdf = pd.DataFrame(new_dict)
    st.dataframe(newdf)
    st.success('Successfully retrieved similar queries!', icon="✅")

