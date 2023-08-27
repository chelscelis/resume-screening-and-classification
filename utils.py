import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
from sklearn.neighbors import NeighborhoodComponentsAnalysis

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

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
