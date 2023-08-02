import nltk
import re
import string
import time
import warnings
import streamlit as st 
import pandas as pd
from nltk.corpus import stopwords
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

        with st.spinner('Classifying resumes'):
            knn_df = pd.read_csv(uploadedResume)
            knn_df['cleanedResume'] = knn_df.Resume.apply(lambda x: cleanResume(x))
            new_knn_df = knn_df.copy()
            oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
            totalWords =[]
            Sentences = knn_df['Resume'].values
            cleanedSentences = ""
            for records in Sentences:
                cleanedText = cleanResume(records)
                cleanedSentences += cleanedText
                requiredWords = nltk.word_tokenize(cleanedText)
                for word in requiredWords:
                    if word not in oneSetOfStopWords and word not in string.punctuation:
                        totalWords.append(word)
                wordfreqdist = nltk.FreqDist(totalWords)
                mostcommon = wordfreqdist.most_common(50)
            var_mod = ['Category']
            le = LabelEncoder()
            for i in var_mod:
                knn_df[i] = le.fit_transform(knn_df[i])
            knn_df.Category.value_counts()
            new_knn_df.Category.value_counts()
            del new_knn_df
            requiredText = knn_df['cleanResume'].values
            requiredTarget = knn_df['Category'].values
            word_vectorizer = TfidfVectorizer(
                sublinear_tf = True,
                stop_words = 'english'
            )
            word_vectorizer.fit(requiredText)
            WordFeatures = word_vectorizer.transform(requiredText)
            X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=1, test_size=0.2,shuffle=True, stratify=requiredTarget)
            warnings.filterwarnings('ignore')

        endTime = time.time()
        executionTime = endTime - startTime
        st.success(f'Finished in {executionTime:.2f} seconds')
