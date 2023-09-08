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

def convertDfToXlsx(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processedData = output.getvalue()
    return processedData

def createBarChart(resumeClf):
    value_counts = resumeClf['Industry Category'].value_counts().reset_index()
    value_counts.columns = ['Industry Category', 'Count']
    new_dataframe = pd.DataFrame(value_counts)
    barChart = alt.Chart(new_dataframe,
    ).mark_bar(
        color = '#56B6C2',
        size = 13 
    ).encode(
        x = alt.X('Count:Q', axis = alt.Axis(format = 'd'), title = 'Number of Resumes'),
        y = alt.Y('Industry Category:N', title = 'Category'),
        tooltip = ['Industry Category', 'Count']
    ).properties(
        title = 'Number of Resumes per Category',
    )
    return barChart

def dimensionalityReduction(features):
    nca = joblib.load('nca_model.joblib')
    features = nca.transform(features.toarray())
    return features    

def filterDataframeClf(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.toggle("Add filters", key = 'toggle-clf-1')
    if not modify:
        return df
    df = df.copy()
    modificationContainer = st.container()
    with modificationContainer:
        toFilterColumns = st.multiselect("Filter table on", df.columns, key = 'toggle-clf-2')
        for column in toFilterColumns:
            left, right = st.columns((1, 20))
            left.write("↳")
            if is_categorical_dtype(df[column]):
                userCatInput = right.multiselect(
                    f'Values for {column}',
                    df[column].unique(),
                    default = list(df[column].unique()),
                    key = 'toggle-clf-3'
                )
                df = df[df[column].isin(userCatInput)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                userNumInput = right.slider(
                    f'Values for {column}',
                    min_value = _min,
                    max_value = _max,
                    value = (_min, _max),
                    step = step,
                    key = 'toggle-clf-4'
                )
                df = df[df[column].between(*userNumInput)]
            else:
                userTextInput = right.text_input(
                    f'Substring or regex in {column}',
                    key = 'toggle-clf-5'
                )
                if userTextInput:
                    df = df[df[column].astype(str).str.contains(userTextInput)]
    return df

def filterDataframeRnk(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.toggle("Add filters", key = 'toggle-rnk-1')
    if not modify:
        return df
    df = df.copy()
    modificationContainer = st.container()
    with modificationContainer:
        toFilterColumns = st.multiselect("Filter table on", df.columns, key = 'toggle-rnk-2')
        for column in toFilterColumns:
            left, right = st.columns((1, 20))
            left.write("↳")
            if is_categorical_dtype(df[column]):
                userCatInput = right.multiselect(
                    f'Values for {column}',
                    df[column].unique(),
                    default = list(df[column].unique()),
                    key = 'toggle-rnk-3'
                )
                df = df[df[column].isin(userCatInput)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                userNumInput = right.slider(
                    f'Values for {column}',
                    min_value = _min,
                    max_value = _max,
                    value = (_min, _max),
                    step = step,
                    key = 'toggle-rnk-4'
                )
                df = df[df[column].between(*userNumInput)]
            else:
                userTextInput = right.text_input(
                    f'Substring or regex in {column}',
                    key = 'toggle-rnk-5'
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

def preprocessing(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    text = re.sub('\s+', ' ', text)
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    text = ' '.join(words)
    return text 

@st.cache_data
def resumesClassify(resumeClf):
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
    resumeClf['Industry Category'] = pd.Categorical(resumeClf['Industry Category'])
    del resumeClf['cleanedResume']
    return resumeClf

def preprocessing2(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    text = re.sub('\s+', ' ', text)
    words = text.split()
    words = [word.lower() for word in words if word.lower() not in stop_words]
    text = ' '.join(words)
    return text 

@st.cache_data
def tfidf_vectorize(texts):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_matrix, tfidf_vectorizer

from gensim.models import KeyedVectors
@st.cache_data
def loadModel():
    model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/wiki-news-300d-1M.vec'
    model = KeyedVectors.load_word2vec_format(model_path)
    return model

word2vec_model = loadModel()

@st.cache_data
def resumesRank(jobDescriptionRnk, resumeRnk):
    job_description_text = preprocessing2(jobDescriptionRnk)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_job = tfidf_vectorizer.fit_transform([job_description_text])
    resumeRnk['cleanedResume'] = resumeRnk['Resume'].apply(lambda x: preprocessing2(x)).tolist()
    cleaned_resumes = resumeRnk['cleanedResume'].tolist()
    all_texts = [job_description_text] + cleaned_resumes
    tfidf_matrix, tfidf_vectorizer = tfidf_vectorize(all_texts)
    
    def get_word_embedding(text):
        words = text.split()
        valid_words = [word for word in text.split() if word in word2vec_model]
        if valid_words:
            return np.mean([word2vec_model[word] for word in valid_words], axis=0)
        else:
            return np.zeros(word2vec_model.vector_size)

    jobDescriptionEmbedding = get_word_embedding(job_description_text)
    resumeSimilarities = []

    for resumeContent in resumeRnk['cleanedResume']:
        resumeEmbedding = get_word_embedding(resumeContent)
        tfidf_embedding_similarity = cosine_similarity(
            tfidf_vectorizer.transform([job_description_text]),
            tfidf_vectorizer.transform([resumeContent])
        )[0][0]
        word_embedding_similarity = cosine_similarity(
            [jobDescriptionEmbedding], [resumeEmbedding]
        )[0][0]
        combined_similarity = (0.6 * tfidf_embedding_similarity) + (0.4 * word_embedding_similarity)
        percentageSimilarity = combined_similarity * 100
        resumeSimilarities.append(percentageSimilarity)

    resumeRnk['Similarity Score (%)'] = resumeSimilarities
    resumeRnk = resumeRnk.sort_values(by='Similarity Score (%)', ascending=False)
    del resumeRnk['cleanedResume']
    return resumeRnk

# FASTTEXT WORD EMBEDDINGS
# @st.cache_data
# def resumesRank(jobDescriptionRnk, resumeRnk):
#     def get_word_embedding(text):
#         # Calculate the mean of word embeddings for words in the text
#         words = text.split()
#         valid_words = [word for word in text.split() if word in word2vec_model]
#         if valid_words:
#             return np.mean([word2vec_model[word] for word in valid_words], axis=0)
#         else:
#             return np.zeros(word2vec_model.vector_size)
#
#     jobDescriptionEmbedding = get_word_embedding(jobDescriptionRnk)
#     resumeRnk['cleanedResume'] = resumeRnk.Resume.apply(lambda x: preprocessing2(x))
#     
#     # Calculate cosine similarities
#     resumeSimilarities = []
#     for resumeContent in resumeRnk['cleanedResume']:
#         resumeEmbedding = get_word_embedding(resumeContent)
#         similarity = cosine_similarity([jobDescriptionEmbedding], [resumeEmbedding])[0][0]
#         percentageSimilarity = similarity * 100
#         resumeSimilarities.append(percentageSimilarity)
#     
#     resumeRnk['Similarity Score (%)'] = resumeSimilarities
#     resumeRnk = resumeRnk.sort_values(by='Similarity Score (%)', ascending=False)
#     del resumeRnk['cleanedResume']
#     return resumeRnk

# TF-IDF + COSINE SIMILARITY
# @st.cache_data
# def resumesRank(jobDescriptionRnk, resumeRnk):
#     jobDescriptionRnk = preprocessing(jobDescriptionRnk)
#     resumeRnk['cleanedResume'] = resumeRnk.Resume.apply(lambda x: preprocessing(x))
#     tfidfVectorizer = TfidfVectorizer(stop_words='english')
#     jobTfidf = tfidfVectorizer.fit_transform([jobDescriptionRnk])
#     resumeSimilarities = []
#     for resumeContent in resumeRnk['cleanedResume']:
#         resumeTfidf = tfidfVectorizer.transform([resumeContent])
#         similarity = cosine_similarity(jobTfidf, resumeTfidf)
#         percentageSimilarity = (similarity[0][0] * 100)
#         resumeSimilarities.append(percentageSimilarity)
#     resumeRnk['Similarity Score (%)'] = resumeSimilarities
#     resumeRnk = resumeRnk.sort_values(by='Similarity Score (%)', ascending=False)
#     del resumeRnk['cleanedResume']
#     return resumeRnk

def writeGettingStarted():
    st.write("""
    ## Hello, Welcome!  
    In today's competitive job market, the process of manually screening resumes has become a daunting task for recruiters and hiring managers. 
    The sheer volume of applications received for a single job posting can make it extremely time-consuming to identify the most suitable candidates efficiently. 
    This often leads to missed opportunities and the potential loss of top-tier talent.

    The ***Resume Screening & Classification*** website application aims to help alleviate the challenges posed by manual resume screening. The objectives are:
    - To classify the resumes into their most suitable job industry category
    - To compare the resumes to the job description and rank them by similarity
    """)
    st.divider()
    st.write("""
    ## Input Guide 
    #### For the Job Description: 
    Ensure the job description is saved in a text (.txt) file. 
    Clearly outline the responsibilities, qualifications, and skills associated with the position.

    #### For the Resumes: 
    Resumes must be compiled in an excel (.xlsx) file. 
    The organization of columns is up to you but ensure that the "Resume" column is present.
    The values under this column should include all the relevant details of the resume.
    You may refer to the sample format below.
    """)
    with st.expander('View sample format'):
        st.write('# Hello world')
    st.divider()
    st.write("""
    ## Demo Walkthrough
    #### Classify Tab:
    The web app will classify the resumes into their most suitable job industry category.
    Currently the Category Scope consists of the following:
    """)
    column1, column2 = st.columns(2)
    with column1:
        st.write("""
        - Aviation
        - Business development
        - Culinary
        - Education
        - Engineering
        - Finance
        """)
    with column2:
        st.write("""
        - Fitness
        - Healthcare
        - HR
        - Information Technology
        - Public relations
        """)
    with st.expander('Classification Steps'):
        st.write("""
        ##### Upload Resumes:
        - Navigate to the "Classify" tab.
        - Click the "Upload Resumes" button.
        - Select the Excel file (.xlsx) containing the resumes you want to classify. Ensure that your Excel file has the "Resume" column with the resume text and any necessary columns for filtering or additional information.
        ######
        """)
        st.write("""
        ##### Start Processing:
        - Click the "Start Processing" button.
        - The app will analyze the resumes and categorize them into job industry categories.
        ######
        """)
        st.write("""
        ##### View Classification Results:
        - A bar chart will appear, showing the number of resumes per category, helping you visualize the distribution.
        - Below the bar chart, there is a dataframe that displays the list of resumes along with their assigned categories.
        ######
        """)
        st.write("""
        ##### Add Filters:
        - You can apply filters to the dataframe to narrow down your results.
        ######
        """)
        st.write("""
        ##### Donwload Results:
        - Once you've applied filters or are satisfied with the results, you can download the current dataframe as an Excel file by clicking the "Save Current Output as XLSX" button.
        ####
        """)
    st.write("""
    #### Rank Tab:
    The web app will rank the resumes based on their similarity to the job description. 
    Keep in mind, the ranking placement does not imply that one applicant is absolutely better than the other.
    Interviews and other tests are still preferred before making final decisions.

    """)
    with st.expander('Ranking Steps'):
        st.write("""
        ##### Upload Job Description and Resumes:
        - Navigate to the "Rank" tab.
        - Upload the job description as a text file. This file should contain the description of the job you want to compare resumes against.
        - Upload the Excel file that contains the resumes you want to rank.
        ######
        """)
        st.write("""
        ##### Start Processing:
        - Click the "Start Processing" button.
        - The app will analyze the job description and rank the resumes based on their similarity to the job description.
        ######
        """)
        st.write("""
        ##### View Classification Results:
        - The output will display the contents of the job description.
        - Below the job description, there is a dataframe that lists the ranked resumes along with their similarity scores.
        ######
        """)
        st.write("""
        ##### Add Filters:
        - You can apply filters to the dataframe to narrow down your results.
        ######
        """)
        st.write("""
        ##### Donwload Results:
        - Once you've applied filters or are satisfied with the results, you can download the current dataframe as an Excel file by clicking the "Save Current Output as XLSX" button.
        ####
        """)
    # TODO: add sample resume dataframe format
    # TODO: add picture per step
    # TODO: add file links

