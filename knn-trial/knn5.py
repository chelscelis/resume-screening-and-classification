import nltk
import re
import time
import pandas as pd
import warnings
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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
# file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/kaggle-KNN-11k.csv'
file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/kaggle-KNN-2482.csv'

startTime = time.time()

df = pd.read_csv(file_path)
df['cleaned_resume'] = df.Resume.apply(cleanResume)

requiredText = df['cleaned_resume'].values
requiredTarget = df['Category'].values

tokenized_resumes = [nltk.word_tokenize(resume.lower()) for resume in requiredText]

# Prepare tagged documents for Doc2Vec
tagged_resumes = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(tokenized_resumes)]

# Train Doc2Vec model
doc2vec_model = Doc2Vec(vector_size=300, window=10, min_count=5, epochs=100, dm=1)
doc2vec_model.build_vocab(tagged_resumes)
doc2vec_model.train(tagged_resumes, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

DocFeatures = [doc2vec_model.dv[str(i)] for i in range(len(tokenized_resumes))]

X_train, X_test, y_train, y_test = train_test_split(DocFeatures, requiredTarget, random_state=1, test_size=0.2, shuffle=True, stratify=requiredTarget)

warnings.filterwarnings('ignore')
clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=13, 
                                               weights='distance'
                                               ))
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

accuracy_train = clf.score(X_train, y_train)
accuracy_test = clf.score(X_test, y_test)
classification_report = metrics.classification_report(y_test, prediction)

print(f'Accuracy of KNeighbors Classifier on training set: {accuracy_train:.2f}')
print(f'Accuracy of KNeighbors Classifier on test set:     {accuracy_test:.2f}')
print("\nClassification report for classifier:\n", classification_report)

executionTime = time.time() - startTime
print(f'Finished in {executionTime:.2f} seconds')

