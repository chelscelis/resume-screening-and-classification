import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
import joblib

# Load dataset
# file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/linkedin.csv'
file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/linkedin_testing.csv'
resumeDataSet = pd.read_csv(file_path)

# Print category counts
print(resumeDataSet['Category'].value_counts())

# Label encoding
le = LabelEncoder()
resumeDataSet['Category'] = le.fit_transform(resumeDataSet['Category'])
le_filename = f'label_encoder.joblib'
joblib.dump(le, le_filename)

# Prepare data for training
requiredText = resumeDataSet['cleanedResume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
WordFeatures = word_vectorizer.fit_transform(requiredText)

X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=42, test_size=0.2, shuffle=True, stratify=requiredTarget)
print(X_train.shape)
print(X_test.shape)

# KNN classifier
# n_neighbors_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
n_neighbors_values = [17]

for n_neighbors in n_neighbors_values:
    print(f"Testing with n_neighbors = {n_neighbors}")
    
    # knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine', weights='distance'))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, 
                               metric='cosine', 
                               # weights='distance',
                               )
    knn.fit(X_train, y_train)

    # Save the model
    knnModel_filename = f'knn_model.joblib'
    joblib.dump(knn, knnModel_filename)

    prediction = knn.predict(X_test)
    
    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)
    classification_report = metrics.classification_report(y_test, prediction)
    
    print('Accuracy of KNeighbors Classifier on training set: {:.4f}'.format(train_accuracy))
    print('Accuracy of KNeighbors Classifier on test set: {:.4f}'.format(test_accuracy))
    print("\nClassification report:\n%s\n" % classification_report)

