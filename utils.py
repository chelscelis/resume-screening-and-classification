import joblib
import numpy as np
import pandas as pd
import re
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/wiki-news-300d-1M-subword.vec' # fasttext
# model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/wiki-news-300d-1M.vec' # fasttext
model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/GoogleNews-vectors-negative300.bin' # google
model = KeyedVectors.load_word2vec_format(model_path, binary=True) # for google word2vec
# model = KeyedVectors.load_word2vec_format(model_path) # for fasttext


def addZeroFeatures(matrix):
    maxFeatures = 18038
    numDocs, numTerms = matrix.shape
    missingFeatures = maxFeatures - numTerms
    if missingFeatures > 0:
        zeroFeatures = csr_matrix((numDocs, missingFeatures), dtype=np.float64)
        matrix = hstack([matrix, zeroFeatures])
    return matrix

def cleanText(text):
    text = re.sub(r'http\S+\s*|RT|cc|#\S+|@\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'[{}]'.format(re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")), ' ', text)
    text = re.sub(r'\s+', ' ', text).lower()
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    text = ' '.join(words)
    return text 

def convertDfToCsv(df):
    return df.to_csv().encode('utf-8')

def rankResumes(jobDescription, resumes):
    jobDescriptionEmbedding = getEmbedding(jobDescription)
    resumeEmbeddings = [getEmbedding(resume) for resume in resumes]
    cosineSimilarities = [cosine_similarity([jobDescriptionEmbedding], [resumeEmbedding])[0][0] for resumeEmbedding in resumeEmbeddings]
    rankedResumes = sorted(zip(resumes, cosineSimilarities), key=lambda x: x[1], reverse=True)
    return rankedResumes

def dimensionalityReduction(features):
    nca = joblib.load('nca_model.joblib')
    features = nca.transform(features.toarray())
    return features    

def getEmbedding(text):
    tokens = preprocessText(text)
    validTokens = [token for token in tokens if token in model]
    if validTokens:
        embeddings = model[validTokens]
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

def loadKnnModel():
    knnModelFileName = f'knn_model.joblib'
    return joblib.load(knnModelFileName)

def loadLabelEncoder():
    labelEncoderFileName = f'label_encoder.joblib'
    return joblib.load(labelEncoderFileName)

def loadTfidfVectorizer():
    tfidfVectorizerFileName = f'tfidf_vectorizer.joblib' 
    return joblib.load(tfidfVectorizerFileName)

def preprocessText(text):
    text = re.sub(r'http\S+\s*|RT|cc|#\S+|@\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'[{}]'.format(re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")), ' ', text)
    text = re.sub(r'\s+', ' ', text).lower()
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return tokens
