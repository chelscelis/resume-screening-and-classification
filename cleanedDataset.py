import re
import pandas as pd

# Load dataset
# file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/knn-trial/datasets/linkedin.csv'
file_path = '~/Projects/hau/csstudy/resume-screening-and-classification/linkedin_cleaned.csv'
resumeDataSet = pd.read_csv(file_path)

# # Clean resume text
def cleanResume(resumeText):
    resumeText = re.sub(r'http\S+\s*|RT|cc|#\S+|@\S+', ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]', ' ', resumeText)
    resumeText = re.sub(r'[{}]'.format(re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")), ' ', resumeText)
    resumeText = re.sub(r'\s+', ' ', resumeText).lower()
    return resumeText

# resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(cleanResume)
resumeDataSet['cleanedResume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

resumeDataSet = resumeDataSet.loc[:, ~resumeDataSet.columns.str.contains('^Unnamed')]

# Save as csv
resumeDataSet.to_csv('linkedin_testing.csv', index=False)
