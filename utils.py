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
    modify = st.toggle("Add filters", key = 'toggle-Clf')
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
    modify = st.toggle("Add filters", key = 'toggle-rnk')
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

@st.cache_data
def resumesRank(jobDescriptionRnk, resumeRnk):
    jobDescriptionRnk = preprocessing(jobDescriptionRnk)
    resumeRnk['cleanedResume'] = resumeRnk.Resume.apply(lambda x: preprocessing(x))
    tfidfVectorizer = TfidfVectorizer(stop_words='english')
    jobTfidf = tfidfVectorizer.fit_transform([jobDescriptionRnk])
    resumeSimilarities = []
    for resumeContent in resumeRnk['cleanedResume']:
        resumeTfidf = tfidfVectorizer.transform([resumeContent])
        similarity = cosine_similarity(jobTfidf, resumeTfidf)
        percentageSimilarity = (similarity[0][0] * 100)
        resumeSimilarities.append(percentageSimilarity)
    resumeRnk['Similarity Score (%)'] = resumeSimilarities
    resumeRnk = resumeRnk.sort_values(by='Similarity Score (%)', ascending=False)
    del resumeRnk['cleanedResume']
    return resumeRnk

