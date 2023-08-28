import time
import streamlit as st 
import pandas as pd

from utils import *
from jobDescriptionVariables import *

st.write("""
# Resume Screening & Classification
Using K-Nearest Neighbors (KNN) algorithm and Cosine Similarity
######
""")

tab1, tab2 = st.tabs(['Getting Started', 'Main Program'])

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

    jobDescriptionOption = st.selectbox(
        'Select which job description to download',
        ('None', 
         'Aviation',
         'Business-Development',
         'Culinary',
         'Education',
         'Engineering',
         'Finance',
         'Fitness',
         'Healthcare',
         'HR',
         'Information-Technology',
         'Public-Relations'
         )
    )

    match jobDescriptionOption:
        case 'Aviation':
            code = aviation
            st.code(code, language='None')
            st.download_button(
                label = "Download Job Description",
                data = code,
                file_name = "aviation_jd.txt",
                mime = 'text',
            )
        case 'Business-Development':
            code = businessDevelopment 
            st.code(code, language='None')
            st.download_button(
                label = "Download Job Description",
                data = code,
                file_name = "businessdev_jd.txt",
                mime = 'text',
            )
        case 'Culinary':
            code = culinary 
            st.code(code, language='None')
            st.download_button(
                label = "Download Job Description",
                data = code,
                file_name = "culinary_jd.txt",
                mime = 'text',
            )
        case 'Education':
            code = education 
            st.code(code, language='None')
            st.download_button(
                label = "Download Job Description",
                data = code,
                file_name = "education_jd.txt",
                mime = 'text',
            )
        case 'Engineering':
            code = engineering 
            st.code(code, language='None')
            st.download_button(
                label = "Download Job Description",
                data = code,
                file_name = "engineering_jd.txt",
                mime = 'text',
            )
        case 'Finance':
            code = finance 
            st.code(code, language='None')
            st.download_button(
                label = "Download Job Description",
                data = code,
                file_name = "finance_jd.txt",
                mime = 'text',
            )
        case 'Fitness':
            code = fitness 
            st.code(code, language='None')
            st.download_button(
                label = "Download Job Description",
                data = code,
                file_name = "fitness_jd.txt",
                mime = 'text',
            )
        case 'Healthcare':
            code = healthcare 
            st.code(code, language='None')
            st.download_button(
                label = "Download Job Description",
                data = code,
                file_name = "healthcare_jd.txt",
                mime = 'text',
            )
        case 'HR':
            code = hr 
            st.code(code, language='None')
            st.download_button(
                label = "Download Job Description",
                data = code,
                file_name = "hr_jd.txt",
                mime = 'text',
            )
        case 'Information-Technology':
            code = informationTechnology 
            st.code(code, language='None')
            st.download_button(
                label = "Download Job Description",
                data = code,
                file_name = "infoTech_jd.txt",
                mime = 'text',
            )
        case 'Public-Relations':
            code = publicRelations 
            st.code(code, language='None')
            st.download_button(
                label = "Download Job Description",
                data = code,
                file_name = "publicRelations_jd.txt",
                mime = 'text',
            )

    st.write("""
    #####
    #### Download Resume sample
    """)

with tab2:
    st.header('Input')
    uploadedJobDescription = st.file_uploader('Upload Job Description', type = 'txt')
    uploadedResume = st.file_uploader('Upload Resume', type = 'csv')
    isButtonDisabled = True

    if all([uploadedJobDescription, uploadedResume]):
        isButtonDisabled = False

    if st.button('Start Processing', disabled = isButtonDisabled):
        st.divider()
        st.header('Output')
        startTime = time.time()

        # CLASSIFICATION PROCESS
        with st.spinner('Classifying resumes ...'):
            resumeDF = pd.read_csv(uploadedResume)
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

        # RANKING PROCESS
        with st.spinner('Ranking resumes ...'):
            # TODO: insert ranking algo
            time.sleep(3)

        endTime = time.time()
        executionTime = endTime - startTime
        st.success(f'Finished in {executionTime:.2f} seconds')

        st.dataframe(resumeDF)
        csv = convertDfToCsv(resumeDF)
        st.download_button(
            label = "Download as CSV",
            data = csv,
            file_name = "Resumes_ranked_categorized.csv",
            mime = 'text/csv',
        )

