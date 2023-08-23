import joblib
import re
import time
import streamlit as st 
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
from utils import loadKnnModel, loadLabelEncoder, cleanText

st.write("""
# Resume Screening & Classification
Using K-Nearest Neighbors (KNN) algorithm and Cosine Similarity
""")
st.divider()

st.header('Input')
uploadedJobDescription = st.file_uploader('Upload Job Description', type = 'txt')
uploadedResume = st.file_uploader('Upload Resume', type = 'csv')
isButtonDisabled = True

if all([uploadedJobDescription, uploadedResume]):
    isButtonDisabled = False

if st.button('Start Processing', disabled = isButtonDisabled):
    st.divider()
    st.header('Output')
    startTime = time.time()

    # CLASSIFICATION PROCESS
    with st.spinner('Classifying resumes ...'):
        resumeDF = pd.read_csv(uploadedResume)
        # resumeDF['cleanedResume'] = resumeDF.Resume.apply(lambda x: cleanText(x))
        # resumeText = resumeDF['cleanedResume'].values
        # vectorizer = TfidfVectorizer(sublinear_tf = True, stop_words = 'english', max_features=15071)
        # vectorizer.fit(resumeText)
        # wordFeatures = vectorizer.transform(resumeText)
        # knn = loadKnnModel()
        # predictedJobCategories = knn.predict(wordFeatures)
        # le = loadLabelEncoder()
        # resumeDF['Job Category'] = le.inverse_transform(predictedJobCategories)

    # RANKING PROCESS
    with st.spinner('Ranking resumes ...'):
        # TODO: insert ranking algo
        time.sleep(3)

    endTime = time.time()
    executionTime = endTime - startTime
    st.success(f'Finished in {executionTime:.2f} seconds')

    st.dataframe(resumeDF)
