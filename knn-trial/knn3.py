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

nltk.download('punkt')
nltk.download('stopwords')

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

# file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/kaggle-KNN-962.csv'
file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/kaggle-KNN-11k.csv'

startTime = time.time()

df = pd.read_csv(file_path)
df['cleaned_resume'] = df.Resume.apply(lambda x: cleanResume(x))
var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
requiredText = df['cleaned_resume'].values
requiredTarget = df['Category'].values

# Tokenize sentences and build Word2Vec model
tokenized_resumes = [nltk.word_tokenize(resume.lower()) for resume in requiredText]
word2vec_model = Word2Vec(tokenized_resumes, vector_size=300, window=10, min_count=1, sg=1, epochs=50)  # You can adjust the parameters as needed

# Generate document embeddings using Word2Vec
WordFeatures = []
for tokens in tokenized_resumes:
    embeddings = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    if embeddings:
        doc_embedding = sum(embeddings) / len(embeddings)  # Average of word embeddings
        WordFeatures.append(doc_embedding)
    else:
        WordFeatures.append([0.0] * 100)  # Default value for empty embeddings

X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=1, test_size=0.2, shuffle=True, stratify=requiredTarget)
print(len(X_train))
print(len(X_test))
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
