import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack

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

def clean(text):
    text = re.sub(r'http\S+\s*|RT|cc|#\S+|@\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'[{}]'.format(re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")), ' ', text)
    text = re.sub(r'\s+', ' ', text).lower()
    return text 

