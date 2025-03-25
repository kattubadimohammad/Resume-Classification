# IMPORT LIBRARIES
import re
import PyPDF2
import docx2txt
import pdfplumber
import pandas as pd
import streamlit as st
import en_core_web_sm
import pickle as pk
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import warnings

# Ensure Stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

nlp = en_core_web_sm.load()

# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="thinc.compat")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="keras")

#----------------------------------------------------------------------------------------------------
st.title('RESUME CLASSIFICATION')
st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)
st.subheader('Welcome to Resume Classification App')

# FUNCTIONS
def extract_skills(resume_text):
    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks
    tokens = [token.text for token in nlp_text if not token.is_stop]

    try:
        data = pd.read_csv(r"csv files/skills.csv")
        skills = list(data.columns.values)
    except Exception as e:
        st.error(f"Error reading skills CSV: {e}")
        return []

    skillset = []

    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)

    return [i.capitalize() for i in set([i.lower() for i in skillset])]

def getText(filename):
    fullText = ''
    try:
        if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            fullText = docx2txt.process(filename)
        else:
            with pdfplumber.open(filename) as pdf_file:
                page = pdf_file.pages[0]
                fullText = page.extract_text() or ''
    except Exception as e:
        st.error(f"Error extracting text from file: {e}")
    return fullText

def display(doc_file):
    resume = []
    try:
        if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume.append(docx2txt.process(doc_file))
        else:
            with pdfplumber.open(doc_file) as pdf:
                page = pdf.pages[0]
                resume.append(page.extract_text() or '')
    except Exception as e:
        st.error(f"Error displaying file: {e}")
    return resume

def preprocess(sentence):
    sentence = str(sentence).lower().replace('{html}', "")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 and w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)

# DataFrame Setup
file_type = pd.DataFrame([], columns=['Uploaded File', 'Predicted Profile', 'Skills'])
filename = []
predicted = []
skills = []

#-------------------------------------------------------------------------------------------------
# MAIN CODE
try:
    model = pk.load(open('artifacts/modelDT.pkl', 'rb'))
    Vectorizer = pk.load(open('artifacts/vector.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")

upload_file = st.file_uploader('Upload Your Resumes', type=['docx', 'pdf'], accept_multiple_files=True)

if upload_file:
    for doc_file in upload_file:
        if doc_file is not None:
            filename.append(doc_file.name)
            cleaned = preprocess(display(doc_file))
            try:
                prediction = model.predict(Vectorizer.transform([cleaned]))[0]
                predicted.append(prediction)
                extText = getText(doc_file)
                skills.append(extract_skills(extText))
            except Exception as e:
                st.error(f"Error during prediction or extraction: {e}")

if len(predicted) > 0:
    file_type['Uploaded File'] = filename
    file_type['Skills'] = skills
    file_type['Predicted Profile'] = predicted
    st.table(file_type.style.format())

# Selection Dropdown
select = ['PeopleSoft', 'SQL Developer', 'React JS Developer', 'Workday']
st.subheader('Select as per Requirement')
option = st.selectbox('Fields', select)

if option in select:
    st.table(file_type[file_type['Predicted Profile'] == option])
