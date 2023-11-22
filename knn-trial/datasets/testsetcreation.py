import pandas as pd
from sklearn.model_selection import train_test_split 
file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/dataset_hr_edited.csv'

resumeDataSet = pd.read_csv(file_path)

requiredText = resumeDataSet['Resume'].values
requiredTarget = resumeDataSet['Category'].values

X_train,X_test,y_train,y_test = train_test_split(requiredText,requiredTarget,random_state=42, test_size=0.2,shuffle=True, stratify=requiredTarget)

demo_df = pd.DataFrame({'Resume': X_test, 'Category': y_test})

# Save dataframes to CSV files
test_csv_path = 'demo_data.csv'
demo_df.to_csv(test_csv_path, index=False)

