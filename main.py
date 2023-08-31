import pandas as pd
import streamlit as st 
import streamlit_ext as ste

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import *

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

    ## Input Guide 
    For the **Job Description**: Ensure the job description is saved in a .txt format. 
    Clearly outline the responsibilities, qualifications, and skills associated with the position.

    For the **Resumes**: Resumes must be compiled in an excel file and ensure that the following columns are present: "Description", "Profession", "Experience", "Education", "Licenses & Certification", and "Skills."

    For your convenience, we have included sample input files for demo testing purposes. 
    Fifty-five (55) resumes (5 per category) were collected from LinkedIn with personally identifiable information (PII) removed.
    Then, eleven (11) job descriptions were created with the help of Workable.
    You can download the following files to experience the capabilities of the web app:

    - **Access Job Description files [here](https://drive.google.com/drive/folders/1ncCO1Zplo3bj45ko7ZAKtU8RxzLi54od?usp=sharing)**
    - **Access Resume files [here](https://drive.google.com/drive/folders/1U9vFegvztlJXDlGcnaJ9LlrBvS30vAe0?usp=sharing)**
    """)

    st.info("""
    **Reminder:** It is recommended to save the output before starting another process.
    According to Streamlit's app model, the script is re-executed for every time a user interacts with a widget.
    For this web app, interacting with tabs, expanders, and tables do not trigger a re-execution (only file uploaders and process buttons).

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
        resumeClf = pd.read_excel(uploadedResumeClf)

        with st.spinner('Classifying resumes ...'):
            resumeClf = combineColumns(resumeClf)
            resumeClf['cleanedResume'] = resumeClf.Resume.apply(lambda x: preprocessing(x))
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

        with st.expander('View Bar Chart'):
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
            resumeRnk = pd.read_excel(uploadedResumeRnk)
            resumeRnk = combineColumns(resumeRnk)
            resumeRnk['cleanedResume'] = resumeRnk.Resume.apply(lambda x: clean(x))

            # resumes = resumeRnk['Resume'].values
            # rankedResumes = rank_resumes(jobDescriptionRnk, resumes)
            # resumeRnk = pd.DataFrame(rankedResumes, columns=['Resume', 'Similarity'])

            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            job_tfidf = tfidf_vectorizer.fit_transform([jobDescriptionRnk])
            resume_similarities = []
            for resume_content in resumeRnk['cleanedResume']:
                resume_tfidf = tfidf_vectorizer.transform([resume_content])
                similarity = cosine_similarity(job_tfidf, resume_tfidf)
                resume_similarities.append(similarity[0][0])
            resumeRnk['Similarity'] = resume_similarities
            resumeRnk = resumeRnk.sort_values(by='Similarity', ascending=False)
            del resumeRnk['cleanedResume']

        with st.expander('View Job Description'):
            st.write(jobDescriptionRnk)
        st.dataframe(resumeRnk)
        ste.download_button('Download as XLSX', resumeRnk, 'Resumes_ranked.xlsx')
        # csv = convertDfToCsv(resumeRnk)
        # st.download_button(
        #     label = "Download as CSV",
        #     data = csv,
        #     file_name = "Resumes_ranked_categorized.csv",
        #     mime = 'text/csv',
        # ) 


