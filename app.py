import streamlit as st
import re
import PyPDF2
import docx2txt
import pdfplumber
import pandas as pd
import en_core_web_sm
import nltk
import warnings
import os
import requests
import pickle as pk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

nlp = en_core_web_sm.load()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="thinc.compat")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="keras")

# Function to download the model from a URL
def download_model(url, dest_path):
    response = requests.get(url)
    with open(dest_path, 'wb') as file:
        file.write(response.content)

# Download model if not exists
model_url = 'YOUR_MODEL_DOWNLOAD_LINK'
model_path = 'artifacts/modelDT.pkl'
vectorizer_url = 'YOUR_VECTORIZER_DOWNLOAD_LINK'
vectorizer_path = 'artifacts/vector.pkl'

if not os.path.exists(model_path):
    download_model(model_url, model_path)

if not os.path.exists(vectorizer_path):
    download_model(vectorizer_url, vectorizer_path)

# Load the model
with open(model_path, 'rb') as file:
    model = pk.load(file)

with open(vectorizer_path, 'rb') as file:
    Vectorizer = pk.load(file)

# Streamlit app
st.title('RESUME CLASSIFICATION')
st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)
st.subheader('Welcome to Resume Classification App')

# Define functions for text extraction, preprocessing, etc.
# ... (include your existing functions here) ...

file_type = pd.DataFrame([], columns=['Uploaded File', 'Predicted Profile', 'Skills',])
filename = []
predicted = []
skills = []

upload_file = st.file_uploader('Upload Your Resumes', type=['docx', 'pdf'], accept_multiple_files=True)

for doc_file in upload_file:
    if doc_file is not None:
        filename.append(doc_file.name)
        cleaned = preprocess(display(doc_file))
        prediction = model.predict(Vectorizer.transform([cleaned]))[0]
        predicted.append(prediction)
        extText = getText(doc_file)
        skills.append(extract_skills(extText))

if len(predicted) > 0:
    file_type['Uploaded File'] = filename
    file_type['Skills'] = skills
    file_type['Predicted Profile'] = predicted
    st.table(file_type.style.format())

select = ['PeopleSoft', 'SQL Developer', 'React JS Developer', 'Workday']
st.subheader('Select as per Requirement')
option = st.selectbox('Fields', select)

if option == 'PeopleSoft':
    st.table(file_type[file_type['Predicted Profile'] == 'PeopleSoft'])
elif option == 'SQL Developer':
    st.table(file_type[file_type['Predicted Profile'] == 'SQL Developer'])
elif option == 'React JS Developer':
    st.table(file_type[file_type['Predicted Profile'] == 'React JS Developer'])
elif option == 'Workday':
    st.table(file_type[file_type['Predicted Profile'] == 'Workday'])
