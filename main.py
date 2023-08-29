import time
from matplotlib.cbook import to_filehandle
import streamlit as st 
import pandas as pd
import streamlit_ext as ste

from utils import *
from jobDescriptionVariables import *

st.write("""
# Resume Screening & Classification
Using K-Nearest Neighbors (KNN) algorithm and Cosine Similarity
######
""")

tab1, tab2, tab3 = st.tabs(['Getting Started', 'Classify', 'Rank'])
to_filehandle = pd.DataFrame()

with tab1:
    st.write("""
    ## Hello, Welcome!  
    In today's competitive job market, the process of manually screening resumes has become a daunting task for recruiters and hiring managers. The sheer volume of applications received for a single job posting can make it extremely time-consuming to identify the most suitable candidates efficiently. This often leads to missed opportunities and the potential loss of top-tier talent.

    The **Resume Screening & Classification** website application aims to help alleviate the challenges posed by manual resume screening. The objectives are:
    - Classify submitted resumes into their respective job industries
    - Rank resumes based on their similarity to the provided job description
    ## Demo Testing Guide 
    For the **Job Description**: Ensure the job description is saved in a .txt format. Clearly outline the skills, qualifications, and responsibilities associated with the position.

    For the **Resumes**: Organize the resumes you wish to evaluate in one column entitled as "Resume" and in a .csv format.    

    For your convenience, we have included sample input files for demo testing purposes. You can download the following files to experience the capabilities of the web app:
    """)

    st.write("""
    #####
    #### Download Job Description sample
    """)

    st.write("""
    #####
    #### Download Resume sample
    """)

with tab2:
    st.header('Input')
    uploadedResumeClf = st.file_uploader('Upload Resumes', type = 'csv', key = '1')
    isButtonDisabledClf = True

    if uploadedResumeClf is not None:
        isButtonDisabledClf = False

    if st.button('Start Processing', disabled = isButtonDisabledClf, key = 'jku2'):
        st.divider()
        st.header('Output')
        resumeDF = pd.read_csv(uploadedResumeClf)

        with st.spinner('Classifying resumes ...'):
            resumeDF['cleanedResume'] = resumeDF.Resume.apply(lambda x: cleanText(x))
            resumeText = resumeDF['cleanedResume'].values
            vectorizer = loadTfidfVectorizer()
            wordFeatures = vectorizer.transform(resumeText)
            wordFeaturesWithZeros = addZeroFeatures(wordFeatures)
            finalFeatures = dimensionalityReduction(wordFeaturesWithZeros)
            knn = loadKnnModel()
            predictedCategories = knn.predict(finalFeatures)
            le = loadLabelEncoder()
            resumeDF['Industry Category'] = le.inverse_transform(predictedCategories)
            # TODO: remove cleanedResume
        st.dataframe(resumeDF)
        # csv = convertDfToCsv(resumeDF)
        ste.download_button('Download Data', resumeDF, 'Resumes_ranked_categorized.xlsx')
        # st.download_button(
        #     label = "Download as CSV",
        #     data = csv,
        #     file_name = "Resumes_ranked_categorized.csv",
        #     mime = 'text/csv',
        # )


with tab3:
    st.header('Input')
    uploadedJobDescriptionRnk = st.file_uploader('Upload Job Description', type = 'txt')
    uploadedResumeRnk = st.file_uploader('Upload Resumes', type = 'csv', key = '3')
    isButtonDisabledRnk = True

    if all([uploadedJobDescriptionRnk, uploadedResumeRnk]):
        isButtonDisabledRnk = False

    if st.button('Start Processing', disabled = isButtonDisabledRnk, key = '2'):
        st.divider()
        st.header('Output')
        resumeDF = pd.read_csv(uploadedResumeRnk)
        # TODO: read job desc txt

        with st.spinner('Ranking resumes ...'):
            # TODO: insert ranking algo
            time.sleep(1)

        st.dataframe(resumeDF)
        # csv = convertDfToCsv(resumeDF)
        ste.download_button('Download Data', resumeDF, 'Resumes_ranked_categorized.xlsx')
        # st.download_button(
        #     label = "Download as CSV",
        #     data = csv,
        #     file_name = "Resumes_ranked_categorized.csv",
        #     mime = 'text/csv',
        # )

