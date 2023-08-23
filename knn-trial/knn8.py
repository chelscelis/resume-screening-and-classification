import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/kaggle-KNN-2482.csv'

resumeDataSet = pd.read_csv(file_path)

# FOR 2482
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'ACCOUNTANT']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'ADVOCATE']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'AGRICULTURE']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'APPAREL']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'ARTS']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'AUTOMOBILE']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'AVIATION']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'BANKING']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'BPO']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'BUSINESS-DEVELOPMENT']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'CHEF']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'CONSTRUCTION']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'CONSULTANT']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'DESIGNER']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'DIGITAL-MEDIA']
# resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'ENGINEERING']
# resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'FINANCE']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'FITNESS']
# resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'HEALTHCARE']
# resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'HR']
# resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'INFORMATION-TECHNOLOGY']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'PUBLIC-RELATIONS']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'SALES']
resumeDataSet = resumeDataSet[resumeDataSet['Category'] != 'TEACHER']

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

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

    words = resumeText.split()
    words = [word for word in words if word.lower() not in stop_words]
    words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    resumeText = ' '.join(words)
    return resumeText


resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))
resumeDataSet.to_excel('letsee.xlsx', index=False)

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
    max_features=12000
)

word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=42, test_size=0.2,shuffle=True, stratify=requiredTarget)
print(X_train.shape)
print(X_test.shape)

# n_neighbors_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
# n_neighbors_values = [101,103,105,107,109,201,203,205,207,209]
n_neighbors_values = [41]
for n_neighbors in n_neighbors_values:
    print(f"Testing with n_neighbors = {n_neighbors}")

    clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=n_neighbors, 
                                                   metric='cosine',
                                                   weights='distance',
                                                   ))
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    
    print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
    print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))
