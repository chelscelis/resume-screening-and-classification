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

    - **Access Job Description files [here](https://drive.google.com/drive/folders/1ncCO1Zplo3bj45ko7ZAKtU8RxzLi54od?usp=sharing)**
    - **Access Resume files [here](https://drive.google.com/drive/folders/1U9vFegvztlJXDlGcnaJ9LlrBvS30vAe0?usp=sharing)**
    """)
    # TODO: add official google drive links
    st.info("""
    **Reminder:** It is recommended to save the output before starting another process.
    According to Streamlit's app model, the script is re-executed for every time a user interacts with a widget.
    For this web app, interacting with tabs, expanders, and tables do not trigger a rerun (only file uploaders and process buttons).

    For more information, click [here](https://docs.streamlit.io/library/get-started/main-concepts)
    """)

with tab2:
    st.header('Input')
    uploadedResumeClf = st.file_uploader('Upload Resumes', type = 'xlsx', key = 'upload-resume-clf')
    isButtonDisabledClf = True

    if uploadedResumeClf is not None:
        isButtonDisabledClf = False

    if st.button('Start Processing', disabled = isButtonDisabledClf, key = 'process-clf'):
        st.divider()
        st.header('Output')
        # resumeClf = pd.read_csv(uploadedResumeClf)
        resumeClf = pd.read_excel(uploadedResumeClf)

        with st.spinner('Classifying resumes ...'):
            resumeClf['Description'].fillna(' ', inplace=True)
            resumeClf['Current Position'].fillna(' ', inplace=True)
            resumeClf['Experience'].fillna(' ', inplace=True)
            resumeClf['Education'].fillna(' ', inplace=True)
            resumeClf['Licenses & Certification'].fillna(' ', inplace=True)
            resumeClf['Skills'].fillna(' ', inplace=True)
            resumeClf['Resume'] = resumeClf[['Description', 'Current Position', 'Experience', 'Education', 'Licenses & Certification', 'Skills']].apply(" ".join, axis = 1)
            resumeClf['cleanedResume'] = resumeClf.Resume.apply(lambda x: cleanText(x))
            resumeText = resumeClf['cleanedResume'].values
            vectorizer = loadTfidfVectorizer()
            wordFeatures = vectorizer.transform(resumeText)
            wordFeaturesWithZeros = addZeroFeatures(wordFeatures)
            finalFeatures = dimensionalityReduction(wordFeaturesWithZeros)
            knn = loadKnnModel()
            predictedCategories = knn.predict(finalFeatures)
            le = loadLabelEncoder()
            resumeClf['Industry Category'] = le.inverse_transform(predictedCategories)
            del resumeClf['cleanedResume']

        with st.expander('See Bar Chart'):
            st.caption('The chart below shows the total number of resumes per category.')
            st.bar_chart(resumeClf['Industry Category'].value_counts())
        st.dataframe(resumeClf)
        ste.download_button('Download as XLSX', resumeClf, 'Resumes_categorized.xlsx')
        # csv = convertDfToCsv(resumeClf)
        # st.download_button(
        #     label = "Download as CSV",
        #     data = csv,
        #     file_name = "Resumes_categorized.csv",
        #     mime = 'text/csv',
        # )

with tab3:
    st.header('Input')
    uploadedJobDescriptionRnk = st.file_uploader('Upload Job Description', type = 'txt', key = 'upload-jd-rnk')
    uploadedResumeRnk = st.file_uploader('Upload Resumes', type = 'xlsx', key = 'upload-resume-rnk')
    isButtonDisabledRnk = True

    if all([uploadedJobDescriptionRnk, uploadedResumeRnk]):
        isButtonDisabledRnk = False

    if st.button('Start Processing', disabled = isButtonDisabledRnk, key = 'process-rnk'):
        st.divider()
        st.header('Output')

        with st.spinner('Ranking resumes ...'):
            jobDescriptionRnk = uploadedJobDescriptionRnk.read().decode('utf-8')
            # resumeRnk = pd.read_csv(uploadedResumeRnk)
            resumeRnk = pd.read_excel(uploadedResumeRnk)
            # TODO: add ranking logic

        with st.expander('Job Description Content'):
            st.code(jobDescriptionRnk, language = 'None')
        st.dataframe(resumeRnk)
        ste.download_button('Download as XLSX', resumeRnk, 'Resumes_ranked.xlsx')
        # csv = convertDfToCsv(resumeRnk)
        # st.download_button(
        #     label = "Download as CSV",
        #     data = csv,
        #     file_name = "Resumes_ranked_categorized.csv",
        #     mime = 'text/csv',
        # ) 


