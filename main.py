import time
import streamlit as st 
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from utils import loadKnnModel, loadLabelEncoder, cleanText, dimensionalityReduction

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
        resumeDF['cleanedResume'] = resumeDF.Resume.apply(lambda x: cleanText(x))
        resumeText = resumeDF['cleanedResume'].values
        vectorizer = TfidfVectorizer(sublinear_tf = True, stop_words = 'english')
        vectorizer.fit(resumeText)
        wordFeatures = vectorizer.transform(resumeText)
        features = dimensionalityReduction(wordFeatures)
        knn = loadKnnModel()
        predictedCategories = knn.predict(features)
        le = loadLabelEncoder()
        resumeDF['Industry Category'] = le.inverse_transform(predictedCategories)

    # RANKING PROCESS
    with st.spinner('Ranking resumes ...'):
        # TODO: insert ranking algo
        time.sleep(3)

    endTime = time.time()
    executionTime = endTime - startTime
    st.success(f'Finished in {executionTime:.2f} seconds')

    st.dataframe(resumeDF)
