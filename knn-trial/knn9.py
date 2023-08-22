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
print (resumeDataSet['category'].value_counts())
resumeDataSet.drop(columns=['linkedin', 'profile_picture', 'Name', 'location', 'clean_skills'], inplace=True)
resumeDataSet['description'].fillna("", inplace=True)
resumeDataSet['Experience'].fillna("", inplace=True)
resumeDataSet['position'].fillna("", inplace=True)
resumeDataSet['Resume'] = resumeDataSet[['description', 'Experience', 'position', 'skills']].apply(lambda x: ' '.join(x), axis=1)
print(resumeDataSet.columns)
print(resumeDataSet.Resume)
