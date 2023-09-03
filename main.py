import joblib
import numpy as np
import pandas as pd
import re
import streamlit as st 

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def addZeroFeatures(matrix):
    maxFeatures = 18038
    numDocs, numTerms = matrix.shape
    missingFeatures = maxFeatures - numTerms
    if missingFeatures > 0:
        zeroFeatures = csr_matrix((numDocs, missingFeatures), dtype=np.float64)
        matrix = hstack([matrix, zeroFeatures])
    return matrix

@st.cache_data
def classifyResumes(resumeClf):
    resumeClf = combineColumns(resumeClf)
    resumeClf['cleanedResume'] = resumeClf.Resume.apply(lambda x: preprocessing(x))
    resumeText = resumeClf['cleanedResume'].values
    vectorizer = loadTfidfVectorizer()
    wordFeatures = vectorizer.transform(resumeText)
    wordFeaturesWithZeros = addZeroFeatures(wordFeatures)
    finalFeatures = dimensionalityReduction(wordFeaturesWithZeros)
    knn = loadKnnModel()
    predictedCategories = knn.predict(finalFeatures)
    le = loadLabelEncoder()
    resumeClf['Industry Category'] = le.inverse_transform(predictedCategories)
    del resumeClf['cleanedResume']
    return resumeClf

def clean(text):
    text = re.sub(r'http\S+\s*|RT|cc|#\S+|@\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'[{}]'.format(re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")), ' ', text)
    text = re.sub(r'\s+', ' ', text).lower()
    return text 

def clickClassify():
    st.session_state.processClf = True

def clickRank():
    st.session_state.processRank = True

def combineColumns(df):
    df['Description'].fillna(' ', inplace=True)
    df['Profession'].fillna(' ', inplace=True)
    df['Experience'].fillna(' ', inplace=True)
    df['Education'].fillna(' ', inplace=True)
    df['Licenses & Certification'].fillna(' ', inplace=True)
    df['Skills'].fillna(' ', inplace=True)
    df['Resume'] = df[['Description', 'Profession', 'Experience', 'Education', 'Licenses & Certification', 'Skills']].apply(" ".join, axis = 1)
    return df

def convertDfToCsv(df):
    return df.to_csv().encode('utf-8')

def dimensionalityReduction(features):
    nca = joblib.load('nca_model.joblib')
    features = nca.transform(features.toarray())
    return features    

def loadKnnModel():
    knnModelFileName = f'knn_model.joblib'
    return joblib.load(knnModelFileName)

def loadLabelEncoder():
    labelEncoderFileName = f'label_encoder.joblib'
    return joblib.load(labelEncoderFileName)

def loadTfidfVectorizer():
    tfidfVectorizerFileName = f'tfidf_vectorizer.joblib' 
    return joblib.load(tfidfVectorizerFileName)


def preprocessing(text):
    text = re.sub(r'http\S+\s*|RT|cc|#\S+|@\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'[{}]'.format(re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")), ' ', text)
    text = re.sub(r'\s+', ' ', text).lower()
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    text = ' '.join(words)
    return text 

@st.cache_data
def rankResumes(jobDescriptionRnk, resumeRnk):
    resumeRnk = combineColumns(resumeRnk)
    resumeRnk['cleanedResume'] = resumeRnk.Resume.apply(lambda x: clean(x))
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    job_tfidf = tfidf_vectorizer.fit_transform([jobDescriptionRnk])
    resume_similarities = []
    for resume_content in resumeRnk['cleanedResume']:
        resume_tfidf = tfidf_vectorizer.transform([resume_content])
        similarity = cosine_similarity(job_tfidf, resume_tfidf)
        resume_similarities.append(similarity[0][0])
    resumeRnk['Similarity'] = resume_similarities
    resumeRnk = resumeRnk.sort_values(by='Similarity', ascending=False)
    del resumeRnk['cleanedResume']
    return resumeRnk

st.write("""
# Resume Screening & Classification
Using K-Nearest Neighbors (KNN) algorithm and Cosine Similarity
######
""")

tab1, tab2, tab3 = st.tabs(['Getting Started', 'Classify', 'Rank'])

with tab1:
    st.write("""
    ## Hello, Welcome!  
    In today's competitive job market, the process of manually screening resumes has become a daunting task for recruiters and hiring managers. The sheer volume of applications received for a single job posting can make it extremely time-consuming to identify the most suitable candidates efficiently. This often leads to missed opportunities and the potential loss of top-tier talent.

    The **Resume Screening & Classification** website application aims to help alleviate the challenges posed by manual resume screening. The objectives are:
    - Classify submitted resumes into their respective job industries
    - Rank resumes based on their similarity to the provided job description

    ## Input Guide 
    For the **Job Description**: Ensure the job description is saved in a .txt format. 
    Clearly outline the responsibilities, qualifications, and skills associated with the position.

    For the **Resumes**: Resumes must be compiled in an excel file and ensure that the following columns are present: "Description", "Profession", "Experience", "Education", "Licenses & Certification", and "Skills."

    For your convenience, we have included sample input files for demo testing purposes. 
    Fifty-five (55) resumes (5 per category) were collected from LinkedIn with personally identifiable information (PII) removed.
    Then, eleven (11) job descriptions were created with the help of Workable.
    You can download the following files to experience the capabilities of the web app:

    - **Access Job Description files [here](https://drive.google.com/drive/folders/1ncCO1Zplo3bj45ko7ZAKtU8RxzLi54od?usp=sharing)**
    - **Access Resume files [here](https://drive.google.com/drive/folders/1U9vFegvztlJXDlGcnaJ9LlrBvS30vAe0?usp=sharing)**
    """)

with tab2:
    st.header('Input')
    uploadedResumeClf = st.file_uploader('Upload Resumes', type = 'xlsx', key = 'upload-resume-clf')

    if uploadedResumeClf is not None:
        isButtonDisabledClf = False
    else:
        st.session_state.processClf = False 
        isButtonDisabledClf = True

    if 'processClf' not in st.session_state:
        st.session_state.processClf = False

    st.button('Start Processing', on_click=clickClassify, disabled = isButtonDisabledClf, key = 'process-clf')

    if st.session_state.processClf:
        st.divider()
        st.header('Output')
        resumeClf = pd.read_excel(uploadedResumeClf)
        resumeClf = classifyResumes(resumeClf)
        with st.expander('View Bar Chart'):
            st.caption('The chart below shows the total number of resumes per category.')
            st.bar_chart(resumeClf['Industry Category'].value_counts())
        st.dataframe(resumeClf)
        csv = convertDfToCsv(resumeClf)
        st.download_button(
            label = "Download as CSV",
            data = csv,
            file_name = "Resumes_categorized.csv",
            mime = 'text/csv',
        )

with tab3:
    st.header('Input')
    uploadedJobDescriptionRnk = st.file_uploader('Upload Job Description', type = 'txt', key = 'upload-jd-rnk')
    uploadedResumeRnk = st.file_uploader('Upload Resumes', type = 'xlsx', key = 'upload-resume-rnk')

    if all([uploadedJobDescriptionRnk, uploadedResumeRnk]):
        isButtonDisabledRnk = False
    else:
        st.session_state.processRank = False
        isButtonDisabledRnk = True

    if 'processRank' not in st.session_state:
        st.session_state.processRank = False

    st.button('Start Processing', on_click = clickRank, disabled = isButtonDisabledRnk, key = 'process-rnk')

    if st.session_state.processRank:
        st.divider()
        st.header('Output')
        jobDescriptionRnk = uploadedJobDescriptionRnk.read().decode('utf-8')
        resumeRnk = pd.read_excel(uploadedResumeRnk)
        resumeRnk = rankResumes(jobDescriptionRnk, resumeRnk)
        with st.expander('View Job Description'):
            st.write(jobDescriptionRnk)
        st.dataframe(resumeRnk)
        csv = convertDfToCsv(resumeRnk)
        st.download_button(
            label = "Download as CSV",
            data = csv,
            file_name = "Resumes_ranked_categorized.csv",
            mime = 'text/csv',
        ) 

