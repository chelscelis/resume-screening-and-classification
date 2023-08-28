import nltk
import re
import pandas as pd
# import pickle
import warnings
from nltk.corpus import stopwords
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

# file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/kaggle-KNN-2482.csv'
file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/dataset_hr_edited.csv'

df = pd.read_csv(file_path)
df['cleaned_resume'] = df.Resume.apply(cleanResume)

requiredText = df['cleaned_resume'].values
requiredTarget = df['Category'].values

stop_words = set(stopwords.words('english'))
tokenized_resumes = [nltk.word_tokenize(resume.lower()) for resume in requiredText]
filtered_tokenized_resumes = [[word for word in tokens if word not in stop_words] for tokens in tokenized_resumes]

word2vec_model = Word2Vec(filtered_tokenized_resumes, vector_size=300, window=10, min_count=1, sg=1, epochs=50)
# word2vec_model.save('word2vec_model.model')

WordFeatures = []
for tokens in filtered_tokenized_resumes:
    embeddings = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    doc_embedding = sum(embeddings) / len(embeddings) if embeddings else [0.0] * 300
    WordFeatures.append(doc_embedding)

X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=1, test_size=0.2, shuffle=True, stratify=requiredTarget)

warnings.filterwarnings('ignore')
clf = OneVsRestClassifier(KNeighborsClassifier(
    n_neighbors=13, 
))
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

accuracy_train = clf.score(X_train, y_train)
accuracy_test = clf.score(X_test, y_test)
classification_report = metrics.classification_report(y_test, prediction)

print(f'Accuracy of KNeighbors Classifier on training set: {accuracy_train:.2f}')
print(f'Accuracy of KNeighbors Classifier on test set:     {accuracy_test:.2f}')
print("\nClassification report for classifier:\n", classification_report)
