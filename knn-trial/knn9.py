import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis

# file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/kaggle-KNN-2482.csv'
# file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/resume_dataset.csv'
# file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/Raw_Resume.csv'
file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/linkedin.csv'
resumeDataSet = pd.read_csv(file_path)

resumeDataSet.drop(columns=['linkedin', 'profile_picture', 'Name', 'location', 'clean_skills'], inplace=True)
# resumeDataSet['description'].fillna("", inplace=True)
resumeDataSet['Experience'].fillna("", inplace=True)
resumeDataSet['position'].fillna("", inplace=True)
# resumeDataSet['Resume'] = resumeDataSet[['description', 'Experience', 'position', 'skills']].apply(lambda x: ' '.join(x), axis=1)
resumeDataSet['Resume'] = resumeDataSet[['Experience', 'position', 'skills']].apply(lambda x: ' '.join(x), axis=1)
resumeDataSet.drop(columns=['description', 'Experience', 'position', 'skills'], inplace=True)
resumeDataSet.rename(columns={'category': 'Category'}, inplace=True)
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
    resumeText = resumeText.lower()
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
    sublinear_tf=True,
    stop_words='english',
    # max_features=40000
)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

print ("Feature completed .....")

# X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)
X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=42, test_size=0.2,shuffle=True, stratify=requiredTarget)
print(X_train.shape)
print(X_test.shape)

# clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=89, 
#                                                # weights='distance',
#                                                # p=2,
#                                                # algorithm='ball_tree',
#                                                ))
# clf.fit(X_train, y_train)
# prediction = clf.predict(X_test)
#
# print('Checking Accuracy .....')
# print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
# print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
#
# print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))


n_neighbors_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
# n_neighbors_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
# n_neighbors_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
# n_neighbors_values = [99, 101, 103, 107, 109, 111, 113, 115, 117, 119]
# n_neighbors_values = [121, 123, 125, 127, 129]
# n_neighbors_values = [221, 223, 225, 227, 229]
# n_neighbors_values = [321, 323, 325, 327, 329]

for n_neighbors in n_neighbors_values:
    print(f"Testing with n_neighbors = {n_neighbors}")
    
    clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=n_neighbors,
                                                   metric='cosine',
                                                   weights='distance',
                                                   ))
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    
    print('Checking Accuracy .....')
    print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
    print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))
