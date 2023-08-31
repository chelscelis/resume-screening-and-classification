import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder

file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/dataset_hr_edited.csv'
# file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/dataset_hr_edited_2.csv'
# file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/Raw_Resume.csv'

resumeDataSet = pd.read_csv(file_path)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

print (resumeDataSet['Category'].value_counts())

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
    resumeText = ' '.join(words)
    return resumeText

resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

le = LabelEncoder()
resumeDataSet['Category'] = le.fit_transform(resumeDataSet['Category'])
le_filename = f'label_encoder.joblib'
joblib.dump(le, le_filename)

requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(
    stop_words='english',
    sublinear_tf=True,
    max_features=18038
)

word_vectorizer.fit(requiredText)
joblib.dump(word_vectorizer, 'tfidf_vectorizer.joblib')
WordFeatures = word_vectorizer.transform(requiredText)

nca = NeighborhoodComponentsAnalysis(n_components=300, random_state=42)
WordFeatures = nca.fit_transform(WordFeatures.toarray(), requiredTarget)
nca_filename = f'nca_model.joblib'
joblib.dump(nca, nca_filename)

X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=42, test_size=0.2,shuffle=True, stratify=requiredTarget)
print(X_train.shape)
print(X_test.shape)

# n_neighbors_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
# weights = ["uniform", "distance"]
# metric = ["euclidean", "manhattan", "minkowski", "cosine"]
# algorithm = ['ball_tree', 'kd_tree', 'brute', 'auto']
# param_grid = dict(n_neighbors=n_neighbors_values, weights=weights, metric=metric, algorithm=algorithm)
# knn = KNeighborsClassifier()
# gs = GridSearchCV(estimator=knn, param_grid=param_grid, scoring="accuracy", verbose=1, cv=10, n_jobs=3)
# grid_search = gs.fit(X_train, y_train)
# best_score = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print("Best Score:", best_score)
# print("Best Parameters:", best_parameters)

knn = KNeighborsClassifier(n_neighbors=1, 
                           metric='manhattan',
                           weights='uniform',
                           algorithm='ball_tree',
                           )
knn.fit(X_train, y_train)

knnModel_filename = f'knn_model.joblib'
joblib.dump(knn, knnModel_filename)

prediction = knn.predict(X_test)
print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))
print("\n Classification report for classifier %s:\n%s\n" % (knn, metrics.classification_report(y_test, prediction)))

confusion_matrix = metrics.confusion_matrix(y_test, prediction)

plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
