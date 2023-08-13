import nltk
import re
import string
import time
import warnings
import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from gensim.models import Word2Vec
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)
    resumeText = re.sub('RT|cc', ' ', resumeText)
    resumeText = re.sub('#\S+', '', resumeText)
    resumeText = re.sub('@\S+', '  ', resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText

# file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/kaggle-KNN-962.csv'
file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/kaggle-KNN-11k.csv'

startTime = time.time()

df = pd.read_csv(file_path)
df['cleaned_resume'] = df.Resume.apply(lambda x: cleanResume(x))
oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
totalWords = []
Sentences = df['Resume'].values
cleanedSentences = ""
for records in Sentences:
    cleanedText = cleanResume(records)
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)

requiredText = df['cleaned_resume'].values
requiredTarget = df['Category'].values

word2vec_model = Word2Vec(sentences=[totalWords], vector_size=300, window=10, min_count=1, sg=1, epochs=50)

def get_doc_embedding(doc_tokens, model):
    embeddings = [model.wv[word] for word in doc_tokens if word in model.wv]
    if not embeddings:
        return np.zeros(model.vector_size)
    return np.mean(embeddings, axis=0)

doc_embeddings = [get_doc_embedding(nltk.word_tokenize(text), word2vec_model) for text in df['cleaned_resume']]

X_train, X_test, y_train, y_test = train_test_split(doc_embeddings, requiredTarget, random_state=1, test_size=0.2, shuffle=True, stratify=requiredTarget)
print(np.array(X_train).shape)
print(np.array(X_test).shape)

warnings.filterwarnings('ignore')
clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, weights='distance'))
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set:     {:.2f}'.format(clf.score(X_test, y_test)))
print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))

endTime = time.time()
executionTime = endTime - startTime
print(f'Finished in {executionTime:.2f} seconds')
