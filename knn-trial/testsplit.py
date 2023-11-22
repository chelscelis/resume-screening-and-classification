import pandas as pd
from sklearn.model_selection import train_test_split

file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/Labeled_LiveCareer_Resumes_1076.xlsx'

resumeDataSet = pd.read_excel(file_path)

requiredText = resumeDataSet['Resume'].values
requiredTarget = resumeDataSet['Actual Category'].values

X_train,X_test,y_train,y_test = train_test_split(requiredText,requiredTarget,random_state=42, test_size=0.2,shuffle=True, stratify=requiredTarget)

test_df = pd.DataFrame({'Resume': X_test, 'Actual Category': y_test})

test_excel_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/Labeled_LiveCareer_Resumes_TestSet_216.xlsx'

test_df.to_excel(test_excel_path, index=False)
