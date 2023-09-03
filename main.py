import altair as alt
import joblib
import numpy as np
import pandas as pd
import re
import streamlit as st 

from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pandas.api.types import (
    is_categorical_dtype,
    is_numeric_dtype,
)
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

def convertDfToXlsx(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def dimensionalityReduction(features):
    nca = joblib.load('nca_model.joblib')
    features = nca.transform(features.toarray())
    return features    

def filterDataframeClf(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.checkbox("Add filters", key = 'checkbox-Clf')
    if not modify:
        return df
    df = df.copy()
    modificationContainer = st.container()
    with modificationContainer:
        toFilterColumns = st.multiselect("Filter table on", df.columns)
        for column in toFilterColumns:
            left, right = st.columns((1, 20))
            left.write("↳")
            if is_categorical_dtype(df[column]):
                userCatInput = right.multiselect(
                    f'Values for {column}',
                    df[column].unique(),
                    default = list(df[column].unique()),
                )
                df = df[df[column].isin(userCatInput)]
            else:
                userTextInput = right.text_input(
                    f'Substring or regex in {column}',
                )
                if userTextInput:
                    df = df[df[column].astype(str).str.contains(userTextInput)]
    return df

def filterDataframeRnk(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.checkbox("Add filters", key = 'checkbox-rnk')
    if not modify:
        return df
    df = df.copy()
    modificationContainer = st.container()
    with modificationContainer:
        toFilterColumns = st.multiselect("Filter table on", df.columns)
        for column in toFilterColumns:
            left, right = st.columns((1, 20))
            left.write("↳")
            if is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                userNumInput = right.slider(
                    f'Values for {column}',
                    min_value = _min,
                    max_value = _max,
                    value = (_min, _max),
                    step = step,
                )
                df = df[df[column].between(*userNumInput)]
            else:
                userTextInput = right.text_input(
                    f'Substring or regex in {column}',
                )
                if userTextInput:
                    df = df[df[column].astype(str).str.contains(userTextInput)]
    return df

def loadKnnModel():
    knnModelFileName = f'knn_model.joblib'
    return joblib.load(knnModelFileName)

def loadLabelEncoder():
    labelEncoderFileName = f'label_encoder.joblib'
    return joblib.load(labelEncoderFileName)

def loadTfidfVectorizer():
    tfidfVectorizerFileName = f'tfidf_vectorizer.joblib' 
    return joblib.load(tfidfVectorizerFileName)

def preprocessing(text, action):
    text = re.sub(r'http\S+\s*|RT|cc|#\S+|@\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'[{}]'.format(re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")), ' ', text)
    text = re.sub(r'\s+', ' ', text).lower()
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    if action == 'classify':
        words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    text = ' '.join(words)
    return text 

@st.cache_data
def resumesClassify(resumeClf):
    resumeClf = combineColumns(resumeClf)
    resumeClf['cleanedResume'] = resumeClf.Resume.apply(lambda x: preprocessing(x, 'classify'))
    resumeText = resumeClf['cleanedResume'].values
    vectorizer = loadTfidfVectorizer()
    wordFeatures = vectorizer.transform(resumeText)
    wordFeaturesWithZeros = addZeroFeatures(wordFeatures)
    finalFeatures = dimensionalityReduction(wordFeaturesWithZeros)
    knn = loadKnnModel()
    predictedCategories = knn.predict(finalFeatures)
    le = loadLabelEncoder()
    resumeClf['Industry Category'] = le.inverse_transform(predictedCategories)
    resumeClf['Industry Category'] = pd.Categorical(resumeClf['Industry Category'])
    del resumeClf['cleanedResume']
    return resumeClf

@st.cache_data
def resumesRank(jobDescriptionRnk, resumeRnk):
    resumeRnk = combineColumns(resumeRnk)
    resumeRnk['cleanedResume'] = resumeRnk.Resume.apply(lambda x: preprocessing(x, 'rank'))
    tfidfVectorizer = TfidfVectorizer(stop_words='english')
    jobTfidf = tfidfVectorizer.fit_transform([jobDescriptionRnk])
    resumeSimilarities = []
    for resumeContent in resumeRnk['cleanedResume']:
        resumeTfidf = tfidfVectorizer.transform([resumeContent])
        similarity = cosine_similarity(jobTfidf, resumeTfidf)
        resumeSimilarities.append(similarity[0][0])
    resumeRnk['Similarity'] = resumeSimilarities
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

    - **Access Job Description files [here]()**
    - **Access Resume files [here]()**
    """)
    # TODO: remove combine function, format samples to have 'Resume' column before processing
    # TODO: add file links
    # TODO: add web app walkthrough

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
        resumeClf = resumesClassify(resumeClf)
        with st.expander('View Bar Chart'):
            value_counts = resumeClf['Industry Category'].value_counts().reset_index()
            value_counts.columns = ['Industry Category', 'Count']
            new_dataframe = pd.DataFrame(value_counts)
            barChart = alt.Chart(new_dataframe,
            ).mark_bar(
                color = '#56B6C2',
                size = 13 
            ).encode(
                x = alt.X('Count:Q', axis = alt.Axis(format = 'd'), title = 'Number of Resumes'),
                y = 'Industry Category:N',
                tooltip = ['Industry Category', 'Count']
            ).properties(
                title = 'Number of Resumes per Category',
            )
            st.altair_chart(barChart, use_container_width = True)
        currentClf = filterDataframeClf(resumeClf)
        st.dataframe(currentClf)
        xlsxClf = convertDfToXlsx(currentClf)
        st.download_button(label='Save Current Output as XLSX', data=xlsxClf, file_name='Resumes_categorized.xlsx')

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
        resumeRnk = resumesRank(jobDescriptionRnk, resumeRnk)
        with st.expander('View Job Description'):
            st.write(jobDescriptionRnk)
        currentRnk = filterDataframeRnk(resumeRnk)
        st.dataframe(currentRnk)
        xlsxRnk = convertDfToXlsx(currentRnk)
        st.download_button(label='Save Current Output as XLSX', data=xlsxRnk, file_name='Resumes_ranked.xlsx')

