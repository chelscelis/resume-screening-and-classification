import nltk
import re
import time
import warnings
import streamlit as st 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('punkt')

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

if __name__ == "__main__":
# TODO: load pre-trained model (joblib)
# classifier = joblib.load(model_filename)
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

        with st.spinner('Classifying resumes ...'):
            knn_df = pd.read_csv(uploadedResume)
            # knn_df['cleanedResume'] = knn_df.Resume.apply(lambda x: cleanResume(x))
            # requiredText = knn_df['cleanedResume'].values
            # word_vectorizer = TfidfVectorizer(
            #     sublinear_tf = True,
            #     stop_words = 'english'
            # )
            # word_vectorizer.fit(requiredText)
            # WordFeatures = word_vectorizer.transform(requiredText)
            # TODO: insert loaded model
            # jobCategories = classifier.predict(WordFeatures)

            # TODO: add predicted job category to dataframe
            # knn_df['Category'] = le.inverse_transform(jobCategories)

            # warnings.filterwarnings('ignore')

        with st.spinner('Ranking resumes ...'):
            # TODO: insert ranking algo
            time.sleep(10)

        endTime = time.time()
        executionTime = endTime - startTime
        st.success(f'Finished in {executionTime:.2f} seconds')

        st.dataframe(knn_df)
