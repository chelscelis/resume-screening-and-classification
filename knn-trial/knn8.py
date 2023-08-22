import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer

file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/kaggle-KNN-2482.csv'

resumeDataSet = pd.read_csv(file_path)

# FOR 2482
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'BPO']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'AUTOMOBILE']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'AGRICULTURE']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'DIGITAL-MEDIA']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'APPAREL']

print (resumeDataSet['Category'].value_counts())

import re
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

from sklearn.preprocessing import LabelEncoder

var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(
    stop_words='english',
    sublinear_tf=True,  # Use logarithmic form for frequency
    min_df=1,           # Minimum number of documents a word must appear in to be saved
    norm='l2',          # Ensure all feature vectors have the same Euclidean norm
    ngram_range=(1, 2)  # Consider both unigrams and bigrams
)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=42, test_size=0.2,shuffle=True, stratify=requiredTarget)
print(X_train.shape)
print(X_test.shape)

n_neighbors_values = [15]
for n_neighbors in n_neighbors_values:
    print(f"Testing with n_neighbors = {n_neighbors}")

    clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=n_neighbors, 
                                                   metric='cosine',
                                                   # weights='distance',
                                                   ))
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    
    print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
    print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))
